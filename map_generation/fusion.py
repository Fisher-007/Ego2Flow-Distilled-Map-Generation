"""Fuse AV2 lanes and coarse roads using polygon cuts and boundary seams."""

from collections import defaultdict
from copy import deepcopy
import heapq
from itertools import count

import networkx as nx
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from shapely.geometry import LineString, Point, Polygon, mapping, box
from shapely.ops import unary_union, substring
from shapely.strtree import STRtree


def is_boundary_seam(lane):
    """A seam keeps its graph identity when its geometry source changes."""
    return lane["id"].startswith("seam:")


def station_xyz(points, station):
    points = np.asarray(points)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
    return np.array([np.interp(station, s, points[:, axis]) for axis in range(3)])


def tangent(points, at_end):
    points = np.asarray(points)
    line = LineString(points[:, :2])
    a, b = ((max(0, line.length-3), line.length) if at_end else (0, min(3, line.length)))
    d = station_xyz(points, b)-station_xyz(points, a)
    return d / np.linalg.norm(d[:2])


def seam_geometry(a, b, direction_a, direction_b):
    """Interpolate only the unknown gap, preserving both source endpoints/headings."""
    length = np.linalg.norm(b[:2]-a[:2])
    curve = CubicHermiteSpline([0, length], np.array([a, b]), np.array([direction_a, direction_b]))
    return curve(np.linspace(0, length, max(3, int(np.ceil(length/.5))+1))).tolist()


def extend_height_transition(outside, fine_lane, incoming, distance):
    """Spread a coarse-height correction along its retained, unmeasured road.

    The fine endpoint and its local grade define the near-boundary height.
    A cubic fade returns to the original coarse heights before the opposite
    endpoint. Neither the source layers nor the coarse XY/topology are edited.
    """
    original = np.asarray(outside["centerline"], dtype=float)
    stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(original[:, :2], axis=0), axis=1))]
    length = stations[-1]
    if not 0 < distance < length:
        raise ValueError("Height transition must fit inside the retained coarse lane: " + outside["id"])
    fine_point = np.asarray(fine_lane["centerline"][0 if incoming else -1])
    forward = tangent(fine_lane["centerline"], not incoming)
    endpoint = original[-1 if incoming else 0]
    projected_distance = float((endpoint[:2]-fine_point[:2]) @ forward[:2])
    target_z = float(fine_point[2] + projected_distance*forward[2])
    delta = target_z-endpoint[2]
    # Include the exact fade boundary while retaining every original station.
    stations = np.unique(np.r_[stations, length-distance if incoming else distance])
    points = np.array([station_xyz(original, s) for s in stations])
    u = np.clip((length-stations if incoming else stations)/distance, 0., 1.)
    points[:, 2] += delta*(1-3*u*u+2*u*u*u)
    outside["centerline"] = points.tolist()
    differences = np.diff(points, axis=0)
    report = {"transition_distance_m": distance, "coarse_endpoint_delta_z_m": float(delta),
              "fine_lane_id": fine_lane["id"], "direction": "entry" if incoming else "exit",
              "max_coarse_grade": float(np.max(np.abs(differences[:, 2])/np.linalg.norm(differences[:, :2], axis=1))),
              "opposite_endpoint_unchanged": bool(np.array_equal(points[0 if incoming else -1], original[0 if incoming else -1])),
              "source": "Fine endpoint height and tangent extrapolated to the coarse cut; cubic fade along retained coarse lane.",
              "scope": "Generated boundary alignment, not measured coarse-road elevation; source XYZ is unchanged."}
    outside["height_transition"] = report
    return report


def _line_parts(geometry):
    if geometry.geom_type == "LineString":
        return [geometry] if not geometry.is_empty else []
    return [line for part in getattr(geometry, "geoms", []) for line in _line_parts(part)]


def _retain_source_intervals(intervals, retained, length, lane_id):
    """Union reviewed source stations with the ordinary ownership cuts."""
    requested = []
    for bounds in retained:
        if len(bounds) != 2:
            raise ValueError(f"Retained source interval needs two stations: {lane_id}")
        start, end = map(float, bounds)
        if not (np.isfinite([start, end]).all() and 0 <= start < end <= length):
            raise ValueError(f"Retained source interval is outside lane {lane_id}: {bounds}")
        requested.append((start, end))
    merged = []
    for start, end in sorted([*intervals, *requested], key=lambda span: span[0]):
        if merged and start <= merged[-1][1] + 1e-7:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def connect_retained_coarse(originals, parts_by_source):
    """Keep a source maneuver when SUMO encodes it with zero-length geometry.

    Removing a geometric point must not disconnect its predecessor and successor.
    Contract only zero-length source connectors with coincident external ports
    and unchanged vehicle permissions; retain the traversed identities as evidence.
    """
    zero={key for key,lane in originals.items() if lane.get('internal') and
          LineString(np.asarray(lane['centerline'])[:,:2]).length < .01}
    contracted=[];unresolved=[]
    for source in originals.values():
        parts=parts_by_source[source['id']]
        if not parts or abs(parts[-1]['source_interval_m'][1]-parts[-1]['source_length_m'])>.001:
            continue
        pending=[(target,[],set(source['allowed_classes'])) for target in source['successors']]
        visited=set()
        while pending:
            target,path,permissions=pending.pop()
            common=permissions & set(originals[target]['allowed_classes'])
            state=(target,frozenset(common))
            if state in visited:continue
            visited.add(state)
            following=parts_by_source[target]
            if following and following[0]['source_interval_m'][0]<.001:
                if path:
                    endpoint_permissions=set(source['allowed_classes']) & set(originals[target]['allowed_classes'])
                    gap=float(np.linalg.norm(np.asarray(parts[-1]['centerline'][-1])-following[0]['centerline'][0]))
                    evidence=dict(from_lane=parts[-1]['id'],to_lane=following[0]['id'],
                                  source_path=[source['id'],*path,target],allowed_classes=sorted(common),gap_m=gap)
                    if gap>.02 or common!=endpoint_permissions:
                        unresolved.append(evidence);continue
                    contracted.append(evidence)
                if following[0]['id'] not in parts[-1]['successors']:
                    parts[-1]['successors'].append(following[0]['id'])
            elif target in zero:
                pending.extend((n,[*path,target],common) for n in originals[target]['successors'])
    return dict(contracted=contracted,unresolved=unresolved)


def _boundary_candidates(fine_lane, incoming, originals, original_predecessors, parts_by_source, options, spatial_index=None):
    """Match at the known endpoint, then follow the original coarse turn graph.

    A crop can cut several turns of one coarse junction. Nearest cut-point
    matching would silently lose those branches or join parallel lanes.
    """
    p = np.asarray(fine_lane["centerline"][0 if incoming else -1])
    forward = tangent(fine_lane["centerline"], not incoming)
    ranked = []
    fine_permissions = set(fine_lane["allowed_classes"])
    bicycle_lane = fine_lane["lane_type"] == "BIKE"
    modes = {"bicycle"} if bicycle_lane else {"passenger", "bus", "delivery", "truck", "taxi"}
    if spatial_index is None:
        nearby = [(lane,LineString(lane["centerline"])) for lane in originals.values()]
    else:
        identities, lines, index = spatial_index
        radius = options["anchor_distance_m"]
        nearby = [(originals[identities[i]],lines[i]) for i in index.query(box(p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius))]
    for lane, line in nearby:
        allowed = set(lane["allowed_classes"])
        if not allowed & fine_permissions & modes:
            continue
        if "passenger" in fine_permissions and "passenger" not in allowed:
            continue
        if line.length < .01:
            continue
        station = line.project(Point(p))
        closest = station_xyz(lane["centerline"], station)
        distance = np.linalg.norm(closest[:2]-p[:2])
        if distance > options["anchor_distance_m"]:
            continue
        a, b = max(0, station-1), min(line.length, station+1)
        d = station_xyz(lane["centerline"], b)-station_xyz(lane["centerline"], a)
        d /= np.linalg.norm(d[:2])
        cosine = np.dot(d[:2], forward[:2])
        if cosine < .8:
            continue
        type_cost = 6 if (fine_lane["lane_type"] == "BUS" or bicycle_lane) and "passenger" in allowed else 0
        ranked.append((distance+8*(1-cosine)+type_cost, lane["id"], station))
    if not ranked:
        return [], None, None
    ranked_by_id = {ident: (score, station) for score, ident, station in ranked}
    _, anchor, station = min(ranked)
    nearest = anchor
    # Use the external approach tangent to distinguish sibling junction movements.
    if originals[anchor]["internal"]:
        pending, seen, external = [anchor], set(), []
        while pending:
            ident = pending.pop()
            if ident in seen:
                continue
            seen.add(ident)
            lane = originals[ident]
            if not lane["internal"]:
                if ident in ranked_by_id:
                    score, projected = ranked_by_id[ident]
                    external.append((score, ident, projected))
                continue
            pending.extend(lane["successors"] if incoming else original_predecessors[ident])
        if external:
            _, anchor, station = min(external)
        elif not fine_lane.get("is_intersection"):
            # Match short internal connectors through their aligned external roads.
            ordinary = [(score, ident, s) for score, ident, s in ranked
                        if not originals[ident]["internal"]]
            if ordinary:
                _, anchor, station = min(ordinary)
    anchor_info = {"nearest_lane_id": nearest, "selected_lane_id": anchor,
                   "method": (("aligned_external_departure" if incoming else "aligned_external_approach")
                              if anchor != nearest else "local_endpoint_match")}
    # Carry the intersection of permissions along each candidate source path.
    serial = count()
    pending = [(0., next(serial), anchor, station, [], fine_permissions)]
    visited, found = set(), {}
    omitted = set(options.get("omit_auxiliary_lane_ids", []))
    while pending:
        travelled, _, ident, s, path, permissions = heapq.heappop(pending)
        if travelled > options["max_seam_distance_m"]:
            continue
        lane = originals[ident]
        common = permissions & set(lane["allowed_classes"])
        state = (ident, frozenset(common))
        if not common or state in visited:
            continue
        visited.add(state)
        outside = [part for part in parts_by_source[ident]
                   if part["source_interval_m"][1] <= s+.001] if incoming else [
                       part for part in parts_by_source[ident] if part["source_interval_m"][0] >= s-.001]
        if outside:
            part = outside[-1] if incoming else outside[0]
            if part["id"] in omitted:
                continue  # Excluded branches are not seam targets.
            distance = travelled + abs(part["source_interval_m"][1 if incoming else 0]-s)
            if distance <= options["max_seam_distance_m"] and part["id"] not in found:
                found[part["id"]] = (part, path+[ident])
            continue
        length = LineString(lane["centerline"]).length
        next_ids = original_predecessors[ident] if incoming else lane["successors"]
        for following in next_ids:
            next_station = LineString(originals[following]["centerline"]).length if incoming else 0.
            heapq.heappush(pending, (travelled+(s if incoming else length-s), next(serial),
                                    following, next_station, path+[ident], common))
    return list(found.values()), anchor, anchor_info


def fuse_scene(scene, config):
    scene = deepcopy(scene)
    options = config["fusion"]
    source_lanes = scene["lanes"]+scene.get("av2_context", [])
    fine_ids = {l["id"] for l in source_lanes}
    footprints = [Polygon([*l["left_boundary"], *reversed(l["right_boundary"])]) for l in source_lanes]
    fine_footprint = unary_union(footprints)
    bicycle_footprint = unary_union([p for p, lane in zip(footprints, source_lanes)
                                    if lane["lane_type"] == "BIKE"])
    ownership = fine_footprint.buffer(options["transition_margin_m"])
    bicycle_ownership = bicycle_footprint.buffer(options["transition_margin_m"])
    evidence_margin = options.get("ownership_evidence_margin_m")
    motor_evidence = fine_footprint.buffer(evidence_margin) if evidence_margin is not None else None
    bike_evidence = bicycle_footprint.buffer(evidence_margin) if evidence_margin is not None else None
    retained_unobserved = []
    ownership_lines=[LineString(np.asarray(lane['centerline'])[:,:2]) for lane in source_lanes]
    ownership_index=STRtree(ownership_lines)

    def observed_parallel_road(lane, line, motor):
        """Test source-road alignment within the AV2 ownership envelope."""
        length=line.length
        if length<.01:return False
        for station in np.linspace(0,length,max(2,int(length/4)+1)):
            p=station_xyz(lane['centerline'],station)
            d=station_xyz(lane['centerline'],min(length,station+1))-station_xyz(lane['centerline'],max(0,station-1))
            d=d[:2]/max(np.linalg.norm(d[:2]),1e-8)
            override=options.get('coarse_lane_overrides',{}).get(lane['id'],{})
            radius=max(5.,override.get('ownership_alignment_m',0.))
            for i in ownership_index.query(box(p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius)):
                fine_lane=source_lanes[i]
                if override.get('ownership_fine_lane_ids') and fine_lane['id'] not in override['ownership_fine_lane_ids']:continue
                if not motor and fine_lane['lane_type']!='BIKE':continue
                if motor and fine_lane['lane_type']=='BIKE':continue
                fine_line=ownership_lines[i]
                fs=fine_line.project(Point(p[:2]))
                q=station_xyz(fine_lane['centerline'],fs)
                tolerance=override.get('ownership_alignment_m',min(radius,fine_lane['mean_ladder_width_m']/2+evidence_margin))
                if np.linalg.norm(q[:2]-p[:2])>tolerance:continue
                fd=station_xyz(fine_lane['centerline'],min(fine_line.length,fs+1))-station_xyz(fine_lane['centerline'],max(0,fs-1))
                if d@fd[:2]/max(np.linalg.norm(fd[:2]),1e-8)>.8:return True
        return False
    coarse_overrides = options.get("coarse_lane_overrides", {})
    fine = []
    # AV2's VEHICLE type does not itself encode OSM access restrictions.
    default_classes = ["passenger", "bus", "bicycle", "delivery", "taxi", "truck", "emergency"]
    permissions = options.get("permission_overrides", {})
    for source in source_lanes:
        lane = deepcopy(source)
        override = permissions.get(source["id"])
        default_allowed = (["bicycle"] if source["lane_type"] == "BIKE" else
                           ["bus", "bicycle"] if source["lane_type"] == "BUS" else default_classes)
        speed = options.get("default_bicycle_speed_mps", 5.56) if source["lane_type"] == "BIKE" else options.get("default_vehicle_speed_mps", 25*.44704)
        lane.update(source="av2", internal=source["is_intersection"],
                    width_m=source["mean_ladder_width_m"], speed_mps=speed,
                    allowed_classes=override["allowed_classes"] if override else
                                    default_allowed,
                    successors=[i for i in source["successors"] if i in fine_ids],
                    permission_source=override["evidence"] if override else
                                      ("AV2 BIKE classification with bicycle access" if source["lane_type"] == "BIKE" else
                                       "Configured vehicle-class default; source lane_type retained separately"))
        fine.append(lane)
    for lane in fine:
        lane["internal_predecessors"] = [l["id"] for l in fine if lane["id"] in l["successors"]]

    coarse, parts_by_source = [], {}
    removed = []
    for lane in scene["coarse_map"]["lanes"]:
        points = np.asarray(lane["centerline"])
        line = LineString(points[:, :2])
        # Retain a mapped bicycle-only path where no AV2 bicycle footprint owns it.
        motor = set(lane["allowed_classes"]) & {"passenger", "bus", "delivery", "truck", "taxi"}
        mask = ownership if motor else bicycle_ownership if "bicycle" in lane["allowed_classes"] else None
        # Check road overlap inside the transition envelope to distinguish adjacent paths.
        if mask is not None and evidence_margin is not None:
            support = motor_evidence if motor else bike_evidence
            supported=line.intersects(support)
            if supported:
                supported=observed_parallel_road(lane,line,motor)
            if not supported:
                if line.intersects(mask):retained_unobserved.append(lane["id"])
                mask = None
        override = coarse_overrides.get(lane["id"], {})
        if "transition_margin_m" in override and mask is not None:
            domain = fine_footprint if motor else bicycle_footprint
            mask = domain.buffer(override["transition_margin_m"])
        pieces = _line_parts(line.difference(mask)) if mask is not None else [line]
        intervals = sorted((line.project(Point(p.coords[0])), line.project(Point(p.coords[-1]))) for p in pieces)
        if override.get("retain_source_intervals_m"):
            intervals = _retain_source_intervals(intervals, override["retain_source_intervals_m"],
                                                line.length, lane["id"])
        if override.get('replacement_fine_paths'):
            observed={l['id']:l for l in source_lanes}
            for path in override['replacement_fine_paths']:
                if any(key not in observed for key in path) or any(b not in observed[a]['successors'] for a,b in zip(path,path[1:])):
                    raise ValueError('Reviewed replacement must follow an observed fine path: '+lane['id'])
            intervals=[]
        parts = []
        for index, (start, end) in enumerate(intervals):
            if end-start < .01:
                continue
            shape = substring(line, start, end)
            xy = np.asarray(shape.coords)
            stations = [line.project(Point(p)) for p in xy]
            xyz = [station_xyz(points, s).tolist() for s in stations]
            part = {**lane, "id": f"sumo:{lane['id']}:{index}", "source": "osm_sumo",
                    "source_lane_id": lane["id"], "centerline": xyz,
                    "source_interval_m": [start, end], "source_length_m": line.length,
                    "successors": []}
            parts.append(part)
        parts_by_source[lane["id"]] = parts
        coarse.extend(parts)
        if not parts:
            removed.append(lane["id"])
    point_connections=connect_retained_coarse({lane['id']:lane for lane in scene['coarse_map']['lanes']},parts_by_source)

    groups = {"entry": defaultdict(list), "exit": defaultdict(list)}
    source_open = []
    for lane in fine:
        if lane["id"] in options.get("source_open_entries", {}):
            source_open.append({"lane_id": lane["id"], "direction": "entry", "position": lane["centerline"][0],
                                "evidence": options["source_open_entries"][lane["id"]]})
        elif not lane["internal_predecessors"]:
            groups["entry"][(tuple(np.round(lane["centerline"][0], 4)), lane["lane_type"] == "BIKE")].append(lane)
        if not lane["successors"]:
            groups["exit"][(tuple(np.round(lane["centerline"][-1], 4)), lane["lane_type"] == "BIKE")].append(lane)
    originals = {l["id"]: l for l in scene["coarse_map"]["lanes"]}
    original_ids = list(originals)
    original_lines = [LineString(originals[key]["centerline"]) for key in original_ids]
    spatial_index = (original_ids, original_lines, STRtree(original_lines))
    original_predecessors = defaultdict(list)
    for lane in originals.values():
        for target in lane["successors"]:
            original_predecessors[target].append(lane["id"])
    seams, matches, unresolved = [], [], []
    for direction in ("entry", "exit"):
        for group in groups[direction].values():
            fine_lane = group[0]
            incoming = direction == "entry"
            p = np.asarray(fine_lane["centerline"][0 if incoming else -1])
            forward = tangent(fine_lane["centerline"], not incoming)
            candidates, anchor, anchor_info = _boundary_candidates(fine_lane, incoming, originals, original_predecessors,
                                                       parts_by_source, options, spatial_index)
            if not candidates:
                unresolved.append({"direction": direction, "lane_ids": [l["id"] for l in group],
                                   "position": p.tolist(), "anchor_selection":anchor_info,
                                   "reason": "No compatible outside cut endpoint in search range"})
                continue
            for index, (outside, path) in enumerate(candidates):
                height_transition = coarse_overrides.get(outside["source_lane_id"], {}).get("height_transition_m")
                height_report = extend_height_transition(outside, fine_lane, incoming, height_transition) if height_transition else None
                q = np.asarray(outside["centerline"][-1 if incoming else 0])
                other_direction = tangent(outside["centerline"], incoming)
                common = set(fine_lane["allowed_classes"]).intersection(
                    *(set(originals[ident]["allowed_classes"]) for ident in path))
                seam_id = f"seam:{direction}:{fine_lane['source_lane_id']}:{index}"
                a, b, da, db = (q, p, other_direction, forward) if incoming else (p, q, forward, other_direction)
                geometry = seam_geometry(a, b, da, db)
                seam = {"id": seam_id, "source": "boundary_interpolation", "internal": False,
                    "centerline": geometry, "width_m": fine_lane["width_m"],
                    "end_widths_m": [outside["width_m"], fine_lane["width_m"]] if incoming else
                                    [fine_lane["width_m"], outside["width_m"]],
                    "speed_mps": min(outside["speed_mps"], fine_lane["speed_mps"]),
                    "allowed_classes": sorted(common),
                    "successors": [l["id"] for l in group] if incoming else [outside["id"]]}
                if incoming:
                    outside["successors"].append(seam_id)
                else:
                    for lane in group:
                        lane["successors"].append(seam_id)
                seams.append(seam)
                matches.append({"direction": direction, "fine_lane_ids": [l["id"] for l in group],
                            "coarse_lane_id": outside["id"], "coarse_source_lane_id": outside["source_lane_id"],
                            "coarse_anchor_lane_id": anchor, "coarse_anchor_selection": anchor_info,
                            "coarse_connection_path": path,
                            "seam_id": seam_id, "xy_gap_m": float(np.linalg.norm(q[:2]-p[:2])),
                            "height_difference_m": float(p[2]-q[2])})
                if height_report:
                    matches[-1]["coarse_height_transition"] = height_report

    # Retain the complete source network for access restoration.
    omitted_ids = set(options.get("omit_auxiliary_lane_ids", []))
    omitted = [l for l in coarse if l["id"] in omitted_ids]
    for lane in omitted:
        if set(lane["allowed_classes"]) & {"passenger", "bus"}:
            raise ValueError(f"Auxiliary omission includes a main traffic lane: {lane['id']}")
    lanes = fine+[l for l in coarse if l["id"] not in omitted_ids]+seams
    for lane in lanes:
        lane["successors"] = [s for s in lane["successors"] if s not in omitted_ids]
    scene["fusion_map"] = {"lanes": lanes, "ownership": mapping(ownership),
        "boundary_matches": matches, "unresolved_boundaries": unresolved,
        "source_open_endpoints": source_open,
        "removed_coarse_lane_ids": removed,
        "retained_unobserved_coarse_lane_ids": retained_unobserved,
        "zero_length_source_connections": point_connections,
        "omitted_auxiliary_lanes": {"lane_ids": [l["id"] for l in omitted],
                                    "reason": options.get("auxiliary_omission_reason")},
        "preserved": {"av2_lane_ids": sorted(fine_ids),
                      "source_internal_connections": [[l["id"], target] for l in source_lanes
                                                      for target in l["successors"] if target in fine_ids],
                      "source_external_references": {l["id"]: [s for s in l["successors"] if s not in fine_ids]
                                                     for l in source_lanes if any(s not in fine_ids for s in l["successors"])},
                      "control_anchors": scene["controls"],
                      "route_length_m": float(LineString(scene["frame_validation"]["route"]["positions"]).length)},
        "assumptions": ["AV2 centerlines, boundaries, lane types and internal successors are preserved.",
                        "Coarse motor lanes inside buffered AV2 footprints are replaced; bicycle-only lanes are replaced only inside AV2 bicycle coverage.",
                        "AV2 BIKE lanes use bicycle access; vehicle and bicycle speed caps are configured operating parameters.",
                        "Internal movement anchors use an adjacent external approach/departure only within the same endpoint distance and heading gates; source path permissions are intersected.",
                        "Boundary seams interpolate unknown geometry and height only; no AV2 vertex is moved.",
                        "The source layer retains outside references and geometric matching assumptions.",
                        "Signal anchors are retained; executable signal bindings are a subsequent builder task."]}
    if coarse_overrides:
        scene["fusion_map"]["coarse_lane_overrides"] = deepcopy(coarse_overrides)
        scene["fusion_map"]["assumptions"].append(
            "The ownership polygon is the default mask; listed coarse-lane overrides use reviewed local margins and optional longitudinal height transitions.")
        if any(override.get("retain_source_intervals_m") for override in coarse_overrides.values()):
            scene["fusion_map"]["assumptions"].append(
                "Explicit retained source station intervals are unioned with ordinary cuts; merged parts use original coarse XYZ, widths and permissions. Part IDs may change when intervals merge.")
    scene["purpose"] = "AV2 fine lanes fused with outside OSM coarse lanes"
    return scene


def lane_graph(lanes, vehicle_class="passenger"):
    graph = nx.DiGraph()
    by_id = {l["id"]: l for l in lanes}
    for lane in lanes:
        if vehicle_class not in lane["allowed_classes"]:
            continue
        graph.add_node(lane["id"])
        for target in lane["successors"]:
            if vehicle_class in by_id[target]["allowed_classes"]:
                graph.add_edge(lane["id"], target, weight=LineString(by_id[target]["centerline"]).length)
    return graph
