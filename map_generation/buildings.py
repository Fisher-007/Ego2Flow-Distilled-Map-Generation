"""OSM building masses with explicit height assumptions and courtyard holes."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from .coordinates import LocalFrame


def _polygons(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry] if not geometry.is_empty else []
    return [p for g in getattr(geometry, "geoms", []) for p in _polygons(g)]


def _tag_height(tags, level_height):
    if "height" in tags:
        raw = tags["height"].strip().lower()
        values = []
        for item in raw.split(";"):
            item = item.strip()
            feet = item.endswith(("ft", "'"))
            suffix = "ft" if item.endswith("ft") else "'" if feet else "m"
            value = float(item.removesuffix(suffix).strip())
            values.append(value*(0.3048 if feet else 1))
        if len(values) > 1:
            return max(values), "OSM height (maximum of listed values)", [
                f"OSM height={tags['height']!r} lists multiple values; coarse roof envelope uses the maximum {max(values):g} m. Raw source tag is retained."]
        return values[0], "OSM height", []
    if "building:levels" in tags:
        return float(tags["building:levels"])*level_height, "OSM building:levels × assumed storey height", [f"Storey height assumed {level_height:g} m; roof shape is not modelled."]
    return None, None, []


def _ground_samples(polygon, step=2.0):
    samples = []
    for ring in [polygon.exterior, *polygon.interiors]:
        samples.extend(ring.interpolate(t).coords[0] for t in np.linspace(0, ring.length, int(np.ceil(ring.length/step))+1))
    xmin, ymin, xmax, ymax = polygon.bounds
    samples.extend((x, y) for x in np.arange(xmin, xmax, step*2) for y in np.arange(ymin, ymax, step*2)
                   if polygon.covers(Point(x, y)))
    samples.append(polygon.representative_point().coords[0])
    return np.asarray(samples)


def read_building_records(config, data_root, bounds):
    """Read complete source outline/part polygons without height realization."""
    roi = box(*bounds)
    frame = LocalFrame(config["city"], config["coordinates"]["origin_city_xyz"])
    osm = ET.parse(Path(data_root)/config["sources"]["osm"]).getroot()
    nodes = {n.get("id"): frame.lonlat_to_local([float(n.get("lon")), float(n.get("lat"))]) for n in osm.findall("node")}
    ways = {w.get("id"): w for w in osm.findall("way")}

    def tags(element):
        return {t.get("k"): t.get("v") for t in element.findall("tag")}

    def coords(way):
        return [nodes[n.get("ref")] for n in way.findall("nd")]

    records, relation_members = [], set()
    for relation in osm.findall("relation"):
        t = tags(relation)
        if t.get("type") != "multipolygon" or not ("building" in t or "building:part" in t):
            continue
        members = [m for m in relation.findall("member") if m.get("type") == "way"]
        missing = [m.get("ref") for m in members if m.get("ref") not in ways]
        if missing:
            if any(LineString(coords(ways[m.get("ref")])).intersects(roi) for m in members if m.get("ref") in ways):
                raise ValueError(f"Selected building relation {relation.get('id')} lacks ways {missing}")
            continue
        outer = list(polygonize([LineString(coords(ways[m.get("ref")])) for m in members if m.get("role") in {"outer", ""}]))
        inner = unary_union(list(polygonize([LineString(coords(ways[m.get("ref")])) for m in members if m.get("role") == "inner"])))
        shape = unary_union(outer).difference(inner)
        if shape.intersects(roi):
            records.append({"id": f"osm:building:relation:{relation.get('id')}", "shape": shape, "tags": t})
            relation_members.update(m.get("ref") for m in members)
    for ident, way in ways.items():
        t = tags(way)
        if ident in relation_members or not ("building" in t or "building:part" in t):
            continue
        refs = [n.get("ref") for n in way.findall("nd")]
        if refs[0] != refs[-1]:
            continue
        shape = Polygon(coords(way))
        if shape.intersects(roi):
            records.append({"id": f"osm:building:way:{ident}", "shape": shape, "tags": t})
    return records


def build_buildings(scene, config, data_root, terrain, bounds=None,
                    default_height_m=None, level_height_m=None):
    """Return non-overlapping outline/part masses intersecting selection bounds.

    A level foundation starts at the lowest sampled ground and a flat roof is
    placed one nominal building height above the highest sampled ground. Whole
    source footprints are retained; the DEM must cover their complete extent.
    """
    coarse = config["coarse"]
    default_height_m = default_height_m if default_height_m is not None else coarse["default_building_height_m"]
    level_height_m = level_height_m if level_height_m is not None else coarse["building_level_height_m"]
    if bounds is None:
        bounds = np.asarray(scene.get("map_region_bbox_local_xy",scene["areas"][0]["bbox_local_xy"])) + np.array([-1, -1, 1, 1])*coarse["buffer_m"]
    records = read_building_records(config, data_root, bounds)
    parts = [r for r in records if r["tags"].get("building:part") not in (None, "no", "false", "0")]
    outlines = [r for r in records if r["tags"].get("building:part") in (None, "no", "false", "0")]
    outline_index = STRtree([r["shape"] for r in outlines])

    def nearby_outlines(shape):
        return (outlines[int(i)] for i in sorted(outline_index.query(shape, predicate="intersects")))

    for part in parts:
        parents = [r for r in nearby_outlines(part["shape"]) if r["shape"].intersection(part["shape"]).area >= 0.99*part["shape"].area]
        part["parent"] = min(parents, key=lambda r: r["shape"].area) if parents else None
    # Merge nested unknown-height outlines while retaining both source records.
    absorbed = set()
    for record in outlines:
        if any(k in record["tags"] for k in ("height", "building:levels")):
            continue
        containers = [other for other in nearby_outlines(record["shape"]) if other is not record
                      and not any(k in other["tags"] for k in ("height", "building:levels"))
                      and other["shape"].area > record["shape"].area
                      and other["shape"].covers(record["shape"])]
        if containers:
            container = max(containers, key=lambda r: r["shape"].area)
            container.setdefault("absorbed", []).append({"id": record["id"], "tags": record["tags"]})
            absorbed.add(record["id"])
            for part in parts:
                if part.get("parent") is record:
                    part["parent"] = container
    result = []
    for record in records:
        if record["id"] in absorbed:
            continue
        t = record["tags"]
        height, height_source, assumptions = _tag_height(t, level_height_m)
        parent = record.get("parent")
        if height is None and parent is not None:
            height, height_source, inherited = _tag_height(parent["tags"], level_height_m)
            if height is not None:
                height_source = f"Inherited {height_source} from {parent['id']}"
                assumptions.extend(inherited + ["Part lacks height; parent height is inherited, not a measured roof step."])
        if height is None:
            height, height_source = default_height_m, "Assumed default; OSM height and levels unavailable"
            assumptions.append(f"Building height defaults to {default_height_m:g} m where source height is unavailable.")
        if record.get("absorbed"):
            assumptions.append("Nested whole-building outlines without heights share this background envelope; source layer tags do not establish vertical clearances.")
        owned_parts = [p["shape"] for p in parts if p.get("parent") is record]
        footprint = record["shape"].difference(unary_union(owned_parts))
        for index, polygon in enumerate(_polygons(footprint)):
            ground = terrain.height_at(_ground_samples(polygon))
            if not np.isfinite(ground).all():
                raise ValueError(f"DEM lacks ground for {record['id']}")
            low, high = float(ground.min()), float(ground.max())
            result.append({"id": f"{record['id']}:{index}", "source": "OpenStreetMap contributors (ODbL)",
                           "source_id": record["id"], "tags": t,
                           "exterior": [list(p) for p in polygon.exterior.coords],
                           "holes": [[list(p) for p in ring.coords] for ring in polygon.interiors],
                           "base_z_m": low, "height_m": height+high-low,
                           "nominal_height_m": height, "foundation_relief_m": high-low,
                           "roof_z_m": high+height, "height_source": height_source,
                           "ground_range_m": [low, high], "ground_sampling_m": {"boundary": 2, "interior": 4},
                           "ground_source": "Vertically aligned USGS 3DEP coarse terrain",
                           "parent_source_id": parent["id"] if parent is not None else None,
                           "absorbed_source_features": record.get("absorbed", []),
                           "clipped_at_selection_boundary": False,
                           "assumptions": assumptions + ["Flat roof is nominal height above sampled maximum ground; foundation extends to sampled minimum ground.",
                                                         "The added foundation relief is a geometry assumption, not measured building height.",
                                                         "Background mass only; no elevated passage, arcade or under-building driveable space is inferred.",
                                                         "Building parts are removed from their parent mass footprint; courtyard holes are preserved."]})
    return result
