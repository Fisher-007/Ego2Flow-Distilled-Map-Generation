"""OSM-derived coarse roads; SUMO owns lane geometry and junction inference."""

import json
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
from shapely.geometry import Polygon
import sumolib

from .coordinates import LocalFrame


def write_type_defaults(coarse, sumo_home, output):
    """Adjust declared import defaults; explicit OSM access tags remain inputs."""
    overrides = coarse.get("osm_type_defaults", {})
    if not overrides:
        return None
    base = Path(sumo_home) / "data/typemap/osmNetconvert.typ.xml"
    types = {item.get("id"): item for item in ET.parse(base).getroot()}
    root = ET.Element("types")
    for identity, decision in overrides.items():
        item = deepcopy(types[identity])
        attributes = decision["attributes"]
        if "allow" in attributes:
            item.attrib.pop("disallow", None)
        if "disallow" in attributes:
            item.attrib.pop("allow", None)
        item.attrib.update(attributes)
        root.append(item)
    path = Path(output) / "reviewed_defaults.typ.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return str(base.resolve()) + "," + str(path.resolve())


def convert_osm(config, data_root, output, netconvert, sumo_home):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = LocalFrame(config["city"], config["coordinates"]["origin_city_xyz"])
    xmin, ymin, xmax, ymax = config["bbox_city_xy"]
    pad = config["coarse"]["buffer_m"]
    corners = frame.city_to_local([[xmin-pad, ymin-pad], [xmax+pad, ymax+pad]])
    (x0, y0), (x1, y1) = corners
    lonlat = frame.local_to_lonlat([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
    bounds = [*lonlat.min(axis=0), *lonlat.max(axis=0)]
    imported_path = output / "osm.net.xml"
    net_path = output / "coarse.net.xml"
    command = [str(Path(netconvert).resolve()),
               "--osm-files", str((Path(data_root) / config["sources"]["osm"]).resolve()),
               "--keep-edges.in-geo-boundary", ",".join(map(str, bounds)),
               "--keep-edges.by-vclass", "passenger,bus,bicycle",
               "--osm.turn-lanes", "true", "--osm.lane-access", "true", "--osm.bike-access", "true",
               "--osm.all-attributes", "true", "--osm.annotate-defaults", "true",
               "--output.original-names", "true", "--output.street-names", "true",
               "--tls.default-type", "static", "--tls.guess-signals", "true", "--tls.discard-simple", "false",
               "--default.lanewidth", str(config["coarse"]["default_lane_width_m"]),
               "--output-file", str(imported_path)]
    if config["coarse"].get("turnarounds_require_dedicated_lane"):
        command.extend(["--no-turnarounds.except-turnlane", "true"])
    if config["coarse"].get("join_nearby_junctions"):
        command.extend(["--junctions.join", "true", "--junctions.join-dist",
                        str(config["coarse"].get("junction_join_distance_m", 10.))])
    type_files = write_type_defaults(config["coarse"], sumo_home, output)
    if type_files:
        command.extend(["--type-files", type_files])
    env = dict(os.environ, SUMO_HOME=str(Path(sumo_home).resolve()))
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
        env.pop(name, None)
    with (output / "netconvert.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Patch lane-bus counts using the configured side convention; SUMO imports bus:lanes directly.
    patch = ET.Element("edges")
    patched_lanes = []
    for edge in ET.parse(imported_path).getroot().findall("edge"):
        if edge.get("id").split("#")[0] not in config["coarse"].get("forward_bus_osm_way_ids", []):
            continue
        tags = {p.get("key"): p.get("value") for p in edge.findall("param")}
        if tags.get("lanes:bus:forward") != "1":
            raise ValueError(f"Bus-lane patch needs review for changed source tags on {edge.get('id')}")
        lane = edge.find("lane")
        if lane.get("index") != "0" or len(edge.findall("lane")) < 2:
            raise ValueError(f"Bus-lane patch needs review for changed lane layout on {edge.get('id')}")
        ET.SubElement(ET.SubElement(patch, "edge", id=edge.get("id")), "lane", index="0", allow="bus bicycle")
        patched_lanes.append(lane.get("id"))
    patch_path = output / "bus_lanes.edg.xml"
    ET.indent(patch)
    ET.ElementTree(patch).write(patch_path, encoding="utf-8", xml_declaration=True)
    patch_command = [command[0], "--sumo-net-file", str(imported_path),
                     "--edge-files", str(patch_path), "--output-file", str(net_path)]
    if patched_lanes:
        with (output / "bus_patch.log").open("w") as log:
            subprocess.run(patch_command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        shutil.copyfile(imported_path, net_path)
        patch_command = None
    version = subprocess.check_output([command[0], "--version"], env=env, text=True).splitlines()[0]
    (output / "netconvert.json").write_text(json.dumps({"command": command, "patch_command": patch_command, "version": version,
        "bus_lane_patch": {"lane_ids": patched_lanes, "assumption": config["coarse"].get("bus_lane_assumption", "No manual directional bus-lane override; explicit OSM and SUMO permissions retained.")},
        "reviewed_type_defaults": config["coarse"].get("osm_type_defaults", {}),
        "requested_bounds_local_xy": corners.flatten().tolist(), "requested_bounds_lonlat": bounds,
        "selection": "SUMO retains complete edges intersecting the buffered region; endpoints may extend beyond it."}, indent=2)+"\n")
    return net_path


def read_roads(config, net_path, terrain):
    """Read SUMO's computed lanes and legal connections without flattening junctions."""
    frame = LocalFrame(config["city"], config["coordinates"]["origin_city_xyz"])
    net = sumolib.net.readNet(str(net_path), withInternal=True, withPrograms=True)
    xml = ET.parse(net_path).getroot()

    def local_xyz(shape, densify=False):
        xy = frame.sumo_to_local_xy(np.asarray(shape)[:, :2], net.getLocationOffset())
        if densify:
            # Keep every SUMO vertex and endpoint; add samples on long segments.
            parts = [np.linspace(a, b, max(1, int(np.ceil(np.linalg.norm(b-a)/2)))+1)[:-1]
                     for a, b in zip(xy[:-1], xy[1:])]
            xy = np.vstack([*parts, xy[-1:]])
        z = terrain.height_at(xy)
        if not np.isfinite(z).all():
            raise ValueError("DEM has unavailable height on required OSM road/junction geometry")
        return np.column_stack([xy, z]).tolist()

    edges, lanes = [], []
    for edge in net.getEdges(withInternal=True):
        internal = edge.getFunction() == "internal"
        params = dict(edge.getParams())
        edges.append({"id": edge.getID(), "internal": internal,
                      "from_junction": edge.getFromNode().getID(), "to_junction": edge.getToNode().getID(),
                      "lane_ids": [lane.getID() for lane in edge.getLanes()], "source_tags": params})
        for lane in edge.getLanes():
            lanes.append({"id": lane.getID(), "edge_id": edge.getID(), "internal": internal,
                          "source": "OSM via SUMO lane and junction inference",
                          "centerline": local_xyz(lane.getShape(), densify=True),
                          "width_m": lane.getWidth(), "speed_mps": lane.getSpeed(),
                          "sumo_length_m": lane.getLength(), "allowed_classes": sorted(lane.getPermissions()),
                          "successors": [], "source_tags": params})
    from .source_use_permissions import apply_use_conditions,inherit_internal_permissions
    use_review=apply_use_conditions(lanes,config['coarse'].get('source_use_conditions',[]))
    by_id = {lane["id"]: lane for lane in lanes}
    connections = []
    for element in xml.findall("connection"):
        c = element.attrib
        source, target = c["from"] + "_" + c["fromLane"], c["to"] + "_" + c["toLane"]
        next_lane = c.get("via", target)
        by_id[source]["successors"].append(next_lane)
        connections.append({"from_lane": source, "to_lane": target, "next_lane": next_lane,
                            "direction": c["dir"], "signal_id": c.get("tl"),
                            "signal_link_index": int(c["linkIndex"]) if "linkIndex" in c else None})
    use_connections=inherit_internal_permissions(lanes,[r['lane_id']for r in use_review],include_exits=True) if use_review else []
    junctions, no_surface = [], []
    for node in xml.findall("junction"):
        if node.get("type") == "internal" or not node.get("shape"):
            continue
        shape = [[float(x) for x in point.split(",")] for point in node.get("shape").split()]
        if len(shape) < 3 or Polygon(shape).area < 1e-8:
            # SUMO also describes simple end nodes as a line closed on itself.
            # Their lane connections remain valid; there is no area to render.
            no_surface.append(node.get("id"))
            continue
        junctions.append({"id": node.get("id"), "type": node.get("type"),
                          "boundary": local_xyz(shape),
                          "incoming_lane_ids": node.get("incLanes", "").split(),
                          "internal_lane_ids": node.get("intLanes", "").split(),
                          "source": "SUMO generated junction area"})
    signals = [{"id": p.get("id"), "program_id": p.get("programID"), "type": p.get("type"),
                "phases": [dict(state=q.get("state"), duration_s=float(q.get("duration")))
                           for q in p.findall("phase")],
                "source": "SUMO generated static timing; not observed signal timing"}
               for p in xml.findall("tlLogic")]
    conversion = json.loads(Path(net_path).with_name("netconvert.json").read_text())
    return {"source": "OpenStreetMap contributors via SUMO netconvert", "network": str(net_path),
            "edges": edges, "lanes": lanes, "junctions": junctions, "signal_programs": signals,
            "junctions_without_surface": no_surface,
            "connections": connections, "conversion": conversion,"source_use_review":use_review,
            "source_use_connection_review":use_connections,
            "assumptions": ["Unspecified lane count/width/speed and junction turns use SUMO defaults; osmDefaults tags identify defaults.",
                            "Signal locations have OSM evidence; controller grouping and all phase durations are inferred.",
                            "Road centerline heights follow the calibrated bare-earth DEM, not a surveyed road surface.",
                            "AV2 source geometry remains separate and unchanged; local fusion is the next step.",
                            "This is a coarse topology and geometry model, not yet a CARLA drivable map."]}


def route_check(scene, config):
    """Find a legal lane path including internal junction geometry, with no lane jumps."""
    coarse = scene["coarse_map"]
    lanes = {lane["id"]: lane for lane in coarse["lanes"]}
    graph = nx.DiGraph()
    for lane in lanes.values():
        if "passenger" not in lane["allowed_classes"]:
            continue
        graph.add_node(lane["id"])
        for target in lane["successors"]:
            if "passenger" in lanes[target]["allowed_classes"]:
                graph.add_edge(lane["id"], target, weight=lanes[target]["sumo_length_m"])
    route_config = config["coarse"]["check_route"]
    route = nx.shortest_path(graph, route_config["start_lane"], route_config["end_lane"], weight="weight")
    gaps = [float(np.linalg.norm(np.array(lanes[a]["centerline"][-1]) - lanes[b]["centerline"][0]))
            for a, b in zip(route[:-1], route[1:])]
    programs = {p["id"]: p for p in coarse["signal_programs"]}
    green_checks = []
    for a, b in zip(route[:-1], route[1:]):
        c = next(c for c in coarse["connections"] if c["from_lane"] == a and c["next_lane"] == b)
        if c["signal_id"]:
            phases = programs[c["signal_id"]]["phases"]
            green_checks.append({"from_lane": a, "signal_id": c["signal_id"],
                "has_green": any(p["state"][c["signal_link_index"]] in "gG" for p in phases)})
    passed_junctions = []
    for landmark in config["landmarks"]:
        found = any(landmark["osm_node_id"] in j["id"] and set(route) & set(j["internal_lane_ids"])
                    for j in coarse["junctions"])
        passed_junctions.append({"landmark": landmark["id"], "passed": found})
    positions = [point for lane_id in route for point in lanes[lane_id]["centerline"]]
    return {"lane_ids": route, "positions": positions,
            "length_m": sum(lanes[i]["sumo_length_m"] for i in route),
            "max_join_gap_m": max(gaps, default=0), "junction_checks": passed_junctions,
            "signal_green_checks": green_checks,
            "passed": max(gaps, default=0) <= .1 and all(c["passed"] for c in passed_junctions)
                      and all(c["has_green"] for c in green_checks),
            "scope": "Passenger-permitted continuous source lane path through any configured landmarks; no vehicle simulation yet."}
