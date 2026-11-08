"""Build the initial SceneIR with separate AV2 and OSM source layers."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap

from .coordinates import LocalFrame


def av2_id(value):
    return f"av2:{value}" if value is not None else None


def build_scene(config, data_root):
    data_root = Path(data_root)
    frame = LocalFrame(config["city"], config["coordinates"]["origin_city_xyz"])
    xmin, ymin, xmax, ymax = config["bbox_city_xy"]
    roi_city = box(xmin, ymin, xmax, ymax)
    bounds = frame.city_to_local([[xmin, ymin], [xmax, ymax]])
    roi = box(*bounds.flatten())
    map_path = next((data_root / config["sources"]["av2_log"] / "map").glob("log_map_archive_*.json"))
    geometry_path = data_root/config["sources"]["av2_assembly"] if config["sources"].get("av2_assembly") else map_path
    static_map = ArgoverseStaticMap.from_json(geometry_path)
    assembly = json.loads(geometry_path.read_text()) if geometry_path != map_path else None
    lanes = []
    for lane in static_map.vector_lane_segments.values():
        center, width = compute_midpoint_line(lane.left_lane_boundary.xyz, lane.right_lane_boundary.xyz)
        if not LineString(center[:, :2]).intersects(roi_city):
            continue
        lanes.append({
            "id": av2_id(lane.id), "source_lane_id": lane.id, "source": "av2",
            "lane_type": lane.lane_type.value, "is_intersection": lane.is_intersection,
            "centerline": frame.city_to_local(center).tolist(),
            "left_boundary": frame.city_to_local(lane.left_lane_boundary.xyz).tolist(),
            "right_boundary": frame.city_to_local(lane.right_lane_boundary.xyz).tolist(),
            "left_mark_type": lane.left_mark_type.value, "right_mark_type": lane.right_mark_type.value,
            "mean_ladder_width_m": float(width), "width_source": "AV2 compute_midpoint_line estimate",
            "successors": [av2_id(i) for i in lane.successors],
            "source_predecessors": [av2_id(i) for i in lane.predecessors],
            "left_neighbor": av2_id(lane.left_neighbor_id), "right_neighbor": av2_id(lane.right_neighbor_id),
        })
        if assembly is not None:
            lanes[-1]["source_observations"] = assembly["lane_segments"][str(lane.id)]["source_observations"]
        else:
            lanes[-1]["source_observations"] = [{
                "source": str(map_path.relative_to(data_root)),
                "source_lane_id": lane.id,
                "definition_differences": {},
                "observed_links": {
                    "predecessors": list(lane.predecessors),
                    "successors": list(lane.successors),
                    "left_neighbor_id": lane.left_neighbor_id,
                    "right_neighbor_id": lane.right_neighbor_id,
                },
            }]
    by_id = {lane["id"]: lane for lane in lanes}
    for lane in lanes:
        lane["internal_predecessors"] = [other["id"] for other in lanes if lane["id"] in other["successors"]]
        lane["external_successors"] = [i for i in lane["successors"] if i not in by_id]

    assets = []
    for crossing in static_map.vector_pedestrian_crossings.values():
        a, b = crossing.edge1.xyz, crossing.edge2.xyz
        if not Polygon(np.vstack([a[:, :2], b[::-1, :2]])).intersects(roi_city):
            continue
        assets.append({"id": f"av2:crossing:{crossing.id}", "kind": "pedestrian_crossing", "source": "av2",
                       "edge1": frame.city_to_local(a).tolist(), "edge2": frame.city_to_local(b).tolist()})
    for area in static_map.vector_drivable_areas.values():
        boundary = area.xyz
        if Polygon(boundary[:, :2]).intersects(roi_city):
            local_boundary=frame.city_to_local(boundary)
            assets.append({"id": f"av2:drivable:{area.id}", "kind": "drivable_area", "source": "av2",
                           "boundary": [[float(v) if np.isfinite(v) else None for v in p] for p in local_boundary],
                           "height_status": "source_contains_unknown_height" if not np.isfinite(local_boundary).all() else "source_measured"})

    osm = ET.parse(data_root / config["sources"]["osm"]).getroot()
    nodes = {}
    for node in osm.findall("node"):
        lonlat = [float(node.get("lon")), float(node.get("lat"))]
        nodes[node.get("id")] = {"lonlat": lonlat, "xy": frame.lonlat_to_local(lonlat).tolist(),
                                "tags": {t.get("k"): t.get("v") for t in node.findall("tag")}}
    roads = []
    for way in osm.findall("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        refs = [n.get("ref") for n in way.findall("nd")]
        xy = [nodes[i]["xy"] for i in refs]
        if "highway" in tags and LineString(xy).intersects(roi):
            roads.append({"id": f"osm:way:{way.get('id')}", "source": "osm", "node_ids": refs,
                          "reference_line": [[x, y, None] for x, y in xy], "tags": tags,
                          "representation": "source way, not a synthesized lane or legal-turn graph"})
        if "building" in tags and refs[0] == refs[-1] and Polygon(xy).intersects(roi):
            assets.append({"id": f"osm:building:{way.get('id')}", "kind": "building_footprint", "source": "osm",
                           "boundary": [[x, y, None] for x, y in xy], "height_m": None, "tags": tags,
                           "collision_role": "candidate_obstacle; volume not built"})
    controls = [{"id": f"osm:node:{i}", "kind": node["tags"]["highway"], "source": "osm",
                 "position": [*node["xy"], None], "tags": node["tags"],
                 "controlled_lane_ids": None, "phase_plan": None}
                for i, node in nodes.items()
                if node["tags"].get("highway") in {"traffic_signals", "stop", "give_way"}
                and roi.covers(Point(node["xy"]))]
    landmarks = []
    for landmark in config["landmarks"]:
        node = nodes[landmark["osm_node_id"]]
        hint = frame.city_to_local(landmark["city_xy"])
        landmarks.append({**landmark, "position": [*node["xy"], None], "lonlat": node["lonlat"],
                          "city_hint_error_m": float(np.linalg.norm(hint - node["xy"]))})

    route_ids = [av2_id(i) for i in config["coordinate_check_route"]["lane_ids"]]
    route_points = []
    for i, lane_id in enumerate(route_ids):
        lane = by_id[lane_id]
        if i and lane_id not in by_id[route_ids[i - 1]]["successors"]:
            raise ValueError(f"Route lacks source successor connection to {lane_id}")
        points = lane["centerline"]
        if i and np.linalg.norm(np.array(route_points[-1]) - points[0]) > 0.05:
            raise ValueError(f"Route has a geometric gap before {lane_id}")
        route_points.extend(points if i == 0 else points[1:])
    probes = [{"id": "origin", "position": [0, 0, 0]},
              {"id": "axes_positive", "position": [12, 7, 3]},
              {"id": "axes_negative", "position": [-8, -13, 1.5]},
              {"id": "route_start", "position": route_points[0]},
              {"id": "route_end", "position": route_points[-1]}]
    probes.extend({"id": p["id"], "position": [*p["position"][:2], 0],
                   "z_note": "reference plane for coordinate check, not measured ground"} for p in landmarks)
    return {
        "map_id": config["prototype_id"], "purpose": "Separate AV2 and OSM source layers before fusion",
        "reference_state": config.get("reference_state"),
        "coordinates": dict(frame.metadata(), origin_z_source=config["coordinates"]["origin_z_source"]),
        "roads": roads, "lanes": lanes, "controls": controls, "assets": assets,
        "protected": {"lane_ids": list(by_id),
                      "preserve": ["known lane boundaries and types", "source successor connections and external references",
                                   "crosswalk geometry", "source control locations and evidence"],
                      "note": "Source evidence to retain during fusion; geometry simplification and corrections remain possible."},
        "source_info": {
            "av2_map": str(map_path.relative_to(data_root)), "osm": config["sources"]["osm"],
            **({"av2_assembly": str(geometry_path.relative_to(data_root)),
                "av2_maps": assembly["source_assembly"]["source_maps"],
                "assembly_review": assembly["source_assembly"]} if assembly is not None else {}),
            "licenses": {"av2": "CC BY-NC-SA 4.0", "osm": "ODbL; OpenStreetMap contributors"},
            "selection": "Full source features intersecting the bbox; geometry is not clipped at its boundary.",
            "assumptions": ["OSM z is unknown (null), not zero elevation.",
                            "OSM way tags and shared node IDs are retained without guessing lanes or legal turns.",
                            "Signal phase plans and controlled lanes remain unknown.",
                            "Building ways are retained; building multipolygon relations are not assembled yet.",
                            "AV2 and OSM retain separate layers for overlap and source-epoch review."],
        },
        "map_region_bbox_local_xy": bounds.flatten().tolist(),
        "areas": [dict(area, bbox_local_xy=(np.asarray(area["core_bbox_city_xy"])-np.tile(frame.origin[:2],2)).tolist(),
                       influence_bbox_local_xy=(np.asarray(area["influence_bbox_city_xy"])-np.tile(frame.origin[:2],2)).tolist())
                  for area in config["areas"]] if config.get("areas") else
                 [{"id": config["prototype_id"], "bbox_local_xy": bounds.flatten().tolist(), "landmarks": landmarks}],
        "frame_validation": {"points": probes, "route": {"id": config["coordinate_check_route"].get("id", "av2_coordinate_check"), "lane_ids": route_ids,
            "positions": route_points, "scope": "Source-connected AV2 route for coordinate checks, not the cross-coverage driving route."}},
    }


def save_scene(scene, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n")
