"""Bind SUMO phase plans to CARLA OpenDRIVE junctions.

External approach validities define stop triggers. Generated phase plans
and source-observed signal states are represented separately.
"""

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import LineString, Point

from map_generation.fusion import is_boundary_seam


def _tangent(points):
    line = LineString(np.asarray(points)[:, :2])
    direction = np.asarray(line.coords[-1]) - np.asarray(line.interpolate(max(0, line.length - 3)).coords[0])
    return direction / np.linalg.norm(direction)


def _phase_states(program, indices):
    """One physical approach cannot display different states for its turns.

    Keep only common green intervals; this conservatively suppresses a permissive
    turn while another retained movement from that approach has a red signal.
    """
    result = []
    for phase in program["phases"]:
        values = [phase["state"][i] for i in indices]
        if all(value in "Gg" for value in values):
            state = "Green"
        elif all(value in "GgyY" for value in values) and any(value in "yY" for value in values):
            state = "Yellow"
        else:
            state = "Red"
        result.append(state)
    return result


def _source_junction(connection):
    # SUMO internal lane names are :<junction>_<connection>_<lane>.
    value = connection["next_lane"]
    return value[1:].rsplit("_", 2)[0] if value.startswith(":") else None


def _approach_interval(metadata,lane_id,movements=None,roads=None):
    """Resolve external signal intervals through source stations and XML predecessors."""
    spans=[s for s in metadata['lane_to_road_intervals'][lane_id] if s['junction_id']is None]
    if spans:
        return dict(max(spans,key=lambda s:s['source_s_end_m']),external_source_lane_id=lane_id,
                    interval_binding_method='retained external source interval')
    candidates=[]
    for road_id,movement in movements or []:
        xml=roads[int(road_id)];link=xml.find('link/predecessor')
        if link is None or link.get('elementType')!='road':
            raise ValueError('Controlled movement has no exact external predecessor: '+str(road_id))
        predecessor=int(link.get('elementId'))
        native_links={int(v.get('id')) for v in xml.findall('lanes/laneSection/*/lane/link/predecessor')}
        source_id=movement['source_lane_path'][0]
        for span in metadata['lane_to_road_intervals'][source_id]:
            if span['junction_id']is None and int(span['road_id'])==predecessor and int(span['lane_id'])in native_links:
                candidates.append(dict(span,external_source_lane_id=source_id,
                    interval_binding_method='complete source path and exact native movement predecessor'))
    identities={(s['road_id'],s['lane_id'],s['road_s_start_m'],s['road_s_end_m'])for s in candidates}
    if len(identities)!=1:
        raise ValueError('Controlled source label has no unique linked external interval: '+lane_id+': '+str(identities))
    return candidates[0]


def _lane_center_t(native,road,lane_id,station):
    offsets=road.findall('lanes/laneOffset')
    active=[o for o in offsets if float(o.get('s'))<=station]
    if active:
        row=active[-1];s=station-float(row.get('s'))
        offset=sum(float(row.get(k,'0'))*s**i for i,k in enumerate(['a','b','c','d']))
    else:offset=0.
    sign=1 if lane_id>0 else -1;rid=int(road.get('id'))
    width=sum(native.get_waypoint_xodr(rid,sign*i,station).lane_width for i in range(1,abs(lane_id)))
    width+=native.get_waypoint_xodr(rid,lane_id,station).lane_width/2
    return offset+sign*width


def _eligible_controls(junction, connections, programs):
    junction_sources = set(junction["source_sumo_junction_ids"])
    return [c for c in connections if c.get('signal_id') in programs and
            ((c['physical_junction_id'] == int(junction['id']) and c['control_entry'])
             if 'physical_junction_id' in c else _source_junction(c) in junction_sources)]


def _validate_approach_reviews(reviews, connections):
    """Keep reviewed control relations tied to an actual directed source path."""
    edges = {(c['from_lane'], c['next_lane']) for c in connections}
    for lane_id, review in reviews.items():
        path = review['source_path']
        if not path or path[-1] != review['source_lane_id']:
            raise ValueError(f'Reviewed control path must end at its source approach: {lane_id}')
        missing = [(a, b) for a, b in zip(path, path[1:]) if (a, b) not in edges]
        if missing:
            raise ValueError(f'Reviewed control path has absent source connections: {lane_id}: {missing}')


def _match_approach(lane, movements, junction, coarse, connections, programs, fusion, boundary_paths=None,
                    approach_reviews=None):
    eligible = _eligible_controls(junction, connections, programs)
    if not eligible:
        return None, "no source signal at this physical junction"
    path_ids = {key for movement in movements for key in movement["source_lane_path"]}
    source_ids = {fusion[key].get("source_lane_id") for key in path_ids if key in fusion}
    seam_paths = {key: value for key, value in (boundary_paths or {}).items() if key in path_ids}
    source_ids.update(source for path in seam_paths.values() for source in path)
    direct = [c for c in eligible if c["from_lane"] == lane.get("source_lane_id")
              or (c.get('control_entry') and c.get('to_lane') == lane.get('source_lane_id')
                  and lane.get('source_lane_id') is not None)
              or c["next_lane"] in source_ids]
    review = (approach_reviews or {}).get(lane['id'])
    if review:
        controls = [c for c in eligible if c['from_lane'] == review['source_lane_id']
                    and c['signal_id'] == review['source_signal_id']
                    and c['signal_link_index'] in review['source_link_indices']]
        if {c['signal_link_index'] for c in controls} != set(review['source_link_indices']):
            raise ValueError(f"Reviewed source control is absent from this physical junction: {lane['id']}")
        candidates = [dict(source_lane_id=review['source_lane_id'],
                           method='reviewed directed source approach',
                           reviewed_source_relation=review, connections=controls)]
    elif direct:
        candidates = [{"source_lane_id": key, "method": "retained SUMO movement/source lane",
                       "connections": [c for c in direct if c["from_lane"] == key]}
                      for key in sorted({c["from_lane"] for c in direct})]
    else:
        direction = _tangent(lane["centerline"])
        endpoint = np.asarray(lane["centerline"][-1])[:2]
        candidates = []
        for key in sorted({c["from_lane"] for c in eligible}):
            source = coarse[key]
            source_line = LineString(np.asarray(source["centerline"])[:, :2])
            source_direction = _tangent(source['centerline'])
            source_endpoint = np.asarray(source['centerline'][-1])[:2]
            angle = math.degrees(math.acos(np.clip(direction @ source_direction, -1, 1)))
            endpoint_distance = float(np.linalg.norm(endpoint - source_endpoint))
            line_distance = source_line.distance(Point(endpoint))
            beyond = float((endpoint-source_endpoint) @ source_direction)
            # Match the bounded approach continuation and retain the source stop point.
            if 0 < beyond <= 25:
                line_distance = min(line_distance, float(abs(np.cross(source_direction, endpoint-source_endpoint))))
            if angle <= 35 and endpoint_distance <= 25 and line_distance <= 12:
                candidates.append({"source_lane_id": key, "method": "same junction, bounded approach geometry",
                                   "endpoint_distance_m": endpoint_distance, "line_distance_m": line_distance,
                                   "beyond_source_stop_m": beyond, 'source_lane_width_m': source.get('width_m',3.2),
                                   "heading_error_deg": angle,
                                   "connections": [c for c in eligible if c["from_lane"] == key]})
        candidates.sort(key=lambda c: c["line_distance_m"] + 0.1 * c["endpoint_distance_m"])
        occupied = [c for c in candidates if c['line_distance_m'] <= c['source_lane_width_m']/2+.5]
        if occupied:
            candidates = occupied
    if not candidates:
        return None, "no retained source path or same-direction source approach within 25m/12m/35deg"
    signatures = set()
    for candidate in candidates:
        controls = candidate.pop("connections")
        ids = {c["signal_id"] for c in controls}
        if len(ids) != 1:
            return None, "candidate approach crosses multiple source controllers"
        tls = ids.pop()
        indices = sorted({c["signal_link_index"] for c in controls})
        states = _phase_states(programs[tls], indices)
        candidate.update(source_signal_id=tls, source_link_indices=indices, phase_states=states,
                         source_connections=controls)
        signatures.add((tls, tuple(states)))
    if len(signatures) != 1:
        return None, "geometrically plausible source approaches have different phase plans"
    if "Green" not in candidates[0]["phase_states"]:
        return None, "no common green interval for retained approach movements"
    selected = candidates[0]
    return {**selected, "boundary_source_paths": seam_paths,
            "alternative_source_approaches": candidates[1:]}, None


def add_signals(scene, metadata, xodr_path, output_dir, replace_existing=False, approach_reviews=None):
    """Write roads.xodr and signal_bindings.json; leave source road geometry intact."""
    import carla

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xodr_path)
    root = tree.getroot()
    previous_display = {}
    if replace_existing:
        previous_ids = {s.get('id') for s in root.findall('road/signals/signal') if s.get('type') == '1000001'}
        for old_road in root.findall('road'):
            for signal in old_road.findall('signals/signal'):
                position = signal.find('positionInertial')
                if signal.get('id') in previous_ids and position is not None:
                    validities = tuple((v.get('fromLane'),v.get('toLane')) for v in signal.findall('validity'))
                    previous_display[(int(old_road.get('id')),validities)] = deepcopy(position)
        for signals in root.findall('road/signals'):
            for element in list(signals):
                if element.get('id') in previous_ids:
                    signals.remove(element)
        for parent in [root, *root.findall('junction')]:
            for controller in parent.findall('controller'):
                parent.remove(controller)
    elif root.findall("controller") or root.findall("road/signals/signal"):
        raise ValueError("Use the unsigned road-builder XODR as input")
    physical = metadata["physical_junctions"]
    roads = {int(r.get("id")): r for r in root.findall("road")}
    junction_xml = {int(j.get("id")): j for j in root.findall("junction")}
    road_info = metadata["roads"]
    fusion = {lane["id"]: lane for lane in scene["fusion_map"]["lanes"]}
    coarse = {lane["id"]: lane for lane in scene["coarse_map"]["lanes"]}
    programs = {p["id"]: p for p in scene["coarse_map"]["signal_programs"]}
    connections = scene["coarse_map"]["connections"]
    approach_reviews = approach_reviews or {}
    _validate_approach_reviews(approach_reviews, connections)
    by_source_junction, by_physical_junction = defaultdict(list), defaultdict(list)
    for connection in connections:
        if connection.get('signal_id') in programs:
            if 'physical_junction_id' in connection:
                if connection['control_entry']:
                    by_physical_junction[connection['physical_junction_id']].append(connection)
            else:
                by_source_junction[_source_junction(connection)].append(connection)
    boundary_paths = {b["seam_id"]: b["coarse_connection_path"] for b in scene["fusion_map"]["boundary_matches"]}
    # Include source-access paths and boundary matches in control association.
    access_connections = scene['fusion_map'].get('source_access_review', {}).get('connections', [])
    # Use compiled movement paths for ordered control matching.
    boundary_paths.update({b['seam_id']: [step[0] for step in b['source_path']]
        for b in access_connections if 'seam_id' in b and 'source_path' in b})
    carla_map = carla.Map(scene["map_id"], Path(xodr_path).read_text())
    bindings, unknown, unsignalized, groups = [], [], [], defaultdict(list)
    for junction in physical:
        jid = int(junction["id"])
        junction_connections = by_physical_junction[jid] + [c for key in junction['source_sumo_junction_ids']
                                                           for c in by_source_junction[key]]
        if not junction_connections:
            unsignalized.append(jid)
            continue
        movements = [road_info[str(i)] for i in junction["movement_road_ids"]]
        for lane_id in junction["incoming_lane_ids"]:
            lane = fusion[lane_id]
            approach_movements = [m for m in movements if m["from_lane_id"] == lane_id]
            match, reason = _match_approach(lane, approach_movements, junction, coarse, junction_connections, programs, fusion,
                                          boundary_paths, approach_reviews)
            if match is None:
                unknown.append({"junction_id": jid, "lane_id": lane_id, "reason": reason})
                continue
            span=_approach_interval(metadata,lane_id,[(i,road_info[str(i)]) for i in junction['movement_road_ids']
                if road_info[str(i)]['from_lane_id']==lane_id],roads)
            road_id = int(span['road_id']);native_lane=int(span['lane_id'])
            road = roads[road_id]
            if int(road.get("junction")) != -1:
                raise ValueError(f"Signal approach road {road_id} is inside a junction")
            start,end=span['road_s_start_m'],span['road_s_end_m'];length=end-start
            epsilon=min(.01,length/4)
            station = end-min(.5,length/4) if native_lane<0 else start+min(.5,length/4)
            stop_source = "external approach road end"
            if is_boundary_seam(lane):
                # Place the control near the source incoming stop line.
                target = np.asarray(coarse[match["source_lane_id"]]["centerline"][-1])[:2]
                samples = np.linspace(start+epsilon, end-epsilon, max(2, math.ceil(length / 0.5)))
                positions = [carla_map.get_waypoint_xodr(road_id, native_lane, float(s)).transform.location for s in samples]
                distances = [np.linalg.norm(np.array([p.x, -p.y]) - target) for p in positions]
                station = float(np.clip(float(samples[int(np.argmin(distances))])+(-.5 if native_lane<0 else .5),start+epsilon,end-epsilon))
                stop_source = "source SUMO approach endpoint projected onto seam (0.5m samples)"
            waypoint = carla_map.get_waypoint_xodr(road_id, native_lane, station)
            if waypoint is None:
                raise ValueError(f"No CARLA approach waypoint for road {road_id}")
            location, rotation = waypoint.transform.location, waypoint.transform.rotation
            approach = {"junction_id": jid, "lane_id": lane_id, "road_id": road_id,
                        "xodr_lane_id": native_lane, "section_id":waypoint.section_id,
                        "external_source_lane_id":span["external_source_lane_id"],
                        "interval_binding_method":span["interval_binding_method"],
                        "s_m": station, "width_m": waypoint.lane_width,
                        "reference_t_m":_lane_center_t(carla_map,road,native_lane,station),
                        "native_trigger_supported": waypoint.lane_type == carla.LaneType.Driving,
                        "stop_location_source": stop_source,
                        "position_local_xyz": [location.x, -location.y, location.z],
                        "heading_rad": math.radians(-rotation.yaw),
                        "movement_road_ids": [int(i) for i in junction["movement_road_ids"]
                        if road_info[str(i)]["from_lane_id"] == lane_id], **match}
            # One head per source approach edge; parallel fine lanes become references.
            source_edge = match["source_lane_id"].rsplit("_", 1)[0]
            groups[(jid, match["source_signal_id"], source_edge, tuple(match["phase_states"]))].append(approach)
    controller_groups = {}
    for number, (key, approaches) in enumerate(sorted(groups.items())):
        jid, tls, _, phase_states = key
        controller_key = (jid, tls, phase_states)
        if controller_key not in controller_groups:
            cid = str(900000 + len(controller_groups))
            controller_groups[controller_key] = cid
            controller = ET.SubElement(root, "controller", id=cid, name=f"MAP_{tls}", sequence=str(len(controller_groups) - 1))
            ET.SubElement(junction_xml[jid], "controller", id=cid, type="0", sequence=str(len(controller_groups) - 1))
        else:
            cid = controller_groups[controller_key]
            controller = next(c for c in root.findall("controller") if c.get("id") == cid)
        signal_id = str(910000 + number)
        ET.SubElement(controller, "control", signalId=signal_id, type="0")
        heading = approaches[0]["heading_rad"]
        right = np.array([math.sin(heading), -math.cos(heading)])
        approaches.sort(key=lambda a: float(np.asarray(a["position_local_xyz"][:2]) @ right), reverse=True)
        for i, approach in enumerate(approaches):
            road = roads[approach["road_id"]]
            signals = road.find("signals")
            if signals is None:
                signals = ET.SubElement(road, "signals")
            attributes = {"id": signal_id, "s": str(approach["s_m"]), "orientation": "+"if approach['xodr_lane_id']<0 else "-"}
            if i == 0:
                side=-1 if approach['xodr_lane_id']<0 else 1
                attributes.update(name=f"MAP_{tls}_{signal_id}", t=str(approach['reference_t_m']+side*(approach["width_m"] / 2 + 0.7)),
                                  dynamic="yes", zOffset="0", country="OpenDRIVE", type="1000001", subtype="-1",
                                  # TrafficLightManager applies the RHT blueprint's 90-degree rotation.
                                  value="-1", height="1.2", width="0.5", hOffset="0", pitch="0", roll="0")
                element = ET.SubElement(signals, "signal", attributes)
                retained_display = previous_display.get((approach['road_id'],
                    ((str(approach['xodr_lane_id']),str(approach['xodr_lane_id'])),)))
                if retained_display is not None:
                    element.append(deepcopy(retained_display))
            else:
                element = ET.SubElement(signals, "signalReference", {**attributes, "t": "0"})
            ET.SubElement(element, "validity", fromLane=str(approach['xodr_lane_id']), toLane=str(approach['xodr_lane_id']))
        bindings.append({"signal_id": signal_id, "controller_id": cid, "junction_id": jid,
                         "source_signal_id": tls, "phase_states": list(phase_states), "approaches": approaches})
    xml_text = ET.tostring(root, encoding="unicode")
    parsed = carla.Map(scene["map_id"], xml_text)
    parsed_ids = {landmark.id for landmark in parsed.get_all_landmarks() if landmark.type == "1000001"}
    expected = {b["signal_id"] for b in bindings}
    if not expected:
        raise ValueError("No source traffic signals could be bound to the physical junctions")
    if parsed_ids != expected:
        raise ValueError(f"CARLA parsed signal IDs differ: expected {expected}, got {parsed_ids}")
    for binding in bindings:
        references = parsed.get_all_landmarks_from_id(binding["signal_id"])
        actual = {(int(reference.road_id),low,high) for reference in references
                  for low,high in reference.get_lane_validities()}
        expected_lanes = {(a["road_id"],a['xodr_lane_id'],a['xodr_lane_id']) for a in binding["approaches"]}
        if actual != expected_lanes:
            raise ValueError(f"CARLA lane references differ for signal {binding['signal_id']}")
    used_programs = {b["source_signal_id"] for b in bindings}
    manifest = {
        "map_id": scene["map_id"], "signal_count": len(bindings), "controller_count": len(controller_groups),
        "signal_reference_count": sum(len(b["approaches"]) - 1 for b in bindings),
        "native_trigger_approach_count": sum(a["native_trigger_supported"] for b in bindings for a in b["approaches"]),
        "bindings": bindings, "source_programs": [programs[key] for key in sorted(used_programs)],
        "unbound_approaches": unknown,
        "junction_ids_without_source_signal_evidence": unsignalized,
        "source_controls": scene["controls"],
        "source_controller_groups": scene['coarse_map'].get('control_groups'),
        "replaced_existing_traffic_lights": replace_existing,
        "reviewed_source_approaches": approach_reviews,
        "source_access_records_without_paths": [b for b in access_connections
            if 'seam_id' in b and 'source_path' not in b],
        "assumptions": [
            "OSM/SUMO locate signalized intersections; AV2 supplies fine lane geometry, not light pole positions or timing.",
            "Signal heads are synthesized beside incoming lanes; CARLA shifts trigger boxes 3m upstream from signal station.",
            "SUMO generated static timing is an inferred phase plan; observed light states are separate inputs.",
            "One approach shows green only when all its selected source movements permit green; other turn-specific green intervals are suppressed.",
            "SUMO permissive g becomes CARLA Green; yielding priority is not encoded by the three-state CARLA light.",
            "XODR establishes controls and references only. Use SignalProgramRunner to apply exported timing; CARLA native cycles otherwise use its defaults.",
            "Non-driving CARLA lane types do not receive TrafficLightComponent trigger volumes; bicycle/restricted road bindings retain source evidence only.",
        ],
        "validation": {"carla_offline_signal_ids_match": True, "carla_offline_lane_references_match": True,
                       "runtime_actor_and_stop_response_checked": False},
    }
    (output / "roads.xodr").write_text(xml_text + "\n")
    (output / "signal_bindings.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def check_runtime_signal_bindings(world, manifest):
    """Read loaded signal validities and actor control groups without advancing the world."""
    import carla

    actors = {a.get_opendrive_id(): a for a in world.get_actors().filter("traffic.traffic_light*")}
    programs = {p["id"]: p for p in manifest["source_programs"]}
    reports = []
    driving_expected = driving_matched = 0
    for binding in manifest["bindings"]:
        signal_id = binding["signal_id"]
        actor = actors.get(signal_id)
        waypoints = actor.get_affected_lane_waypoints() if actor is not None else []
        by_pair = {(w.road_id, w.lane_id): w for w in waypoints if w is not None}
        expected_pairs = {(a["road_id"], a["xodr_lane_id"]) for a in binding["approaches"]}
        expected_group = {b["signal_id"] for b in manifest["bindings"] if b["junction_id"] == binding["junction_id"]}
        actual_group = {a.get_opendrive_id() for a in actor.get_group_traffic_lights()} if actor is not None else set()
        approaches = []
        for approach in binding["approaches"]:
            pair = (approach["road_id"], approach["xodr_lane_id"])
            waypoint = by_pair.get(pair)
            is_driving = waypoint is not None and waypoint.lane_type == carla.LaneType.Driving
            station_error = abs(waypoint.s - approach["s_m"]) if waypoint is not None else None
            phase_matches = _phase_states(programs[binding["source_signal_id"]], approach["source_link_indices"]) == binding["phase_states"]
            passed = (waypoint is not None and station_error <= 0.05 and phase_matches
                      and is_driving == approach["native_trigger_supported"])
            driving_expected += int(approach["native_trigger_supported"])
            driving_matched += int(approach["native_trigger_supported"] and passed)
            approaches.append({"lane_id": approach["lane_id"], "road_id": pair[0], "xodr_lane_id": pair[1],
                               "expected_s_m": approach["s_m"], "actual_s_m": waypoint.s if waypoint else None,
                               "station_error_m": station_error, "native_trigger_supported": approach["native_trigger_supported"],
                               "actual_lane_type": str(waypoint.lane_type) if waypoint else None,
                               "phase_matches_source_program": phase_matches, "passed": passed})
        unexpected = sorted(set(by_pair) - expected_pairs)
        reports.append({"signal_id": signal_id, "actor_id": actor.id if actor else None,
                        "junction_id": binding["junction_id"], "source_signal_id": binding["source_signal_id"],
                        "approaches": approaches, "unexpected_road_lane_pairs": unexpected,
                        "expected_group_signal_ids": sorted(expected_group), "actual_group_signal_ids": sorted(actual_group),
                        "physical_group_matches": actual_group == expected_group,
                        "passed": actor is not None and all(a["passed"] for a in approaches)
                                  and not unexpected and actual_group == expected_group})
    return {
        "passed": bool(reports) and all(r["passed"] for r in reports),
        "expected_signal_count": len(reports), "matched_actor_count": sum(r["actor_id"] is not None for r in reports),
        "expected_driving_approaches": driving_expected, "matched_driving_approaches": driving_matched,
        "non_driving_approaches": sum(not a["native_trigger_supported"] for r in reports for a in r["approaches"]),
        "signals": reports,
        "scope": "Actor IDs plus loaded-map affected road/lane/station and source phases; server-side physical light group membership.",
        "affected_lanes_are_loaded_map_references": True,
        "physical_trigger_overlap_verified": False, "vehicle_red_stop_green_release_verified": False,
    }


class SignalProgramRunner:
    """Apply a phase plan before each synchronous world tick.

    Call update(elapsed_simulation_seconds) after loading the map. close()
    restores saved light states and frozen flags; the caller controls world ticks.
    """

    def __init__(self, world, manifest):
        import carla

        self.states = {name: getattr(carla.TrafficLightState, name) for name in ("Green", "Yellow", "Red")}
        self.bindings = manifest["bindings"]
        self.programs = {p["id"]: p for p in manifest["source_programs"]}
        for binding in self.bindings:
            phases = self.programs[binding["source_signal_id"]]["phases"]
            if len(phases) != len(binding["phase_states"]) or any(p["duration_s"] <= 0 for p in phases):
                raise ValueError("Signal binding states must match positive-duration source phases")
            if any(state not in self.states for state in binding["phase_states"]):
                raise ValueError("Unsupported traffic light state in bindings")
        actors = {a.get_opendrive_id(): a for a in world.get_actors().filter("traffic.traffic_light*")}
        missing = {b["signal_id"] for b in self.bindings} - actors.keys()
        if missing:
            raise RuntimeError(f"Loaded CARLA map lacks exported traffic lights: {sorted(missing)}")
        self.actors = {b["signal_id"]: actors[b["signal_id"]] for b in self.bindings}
        self.original = {key: (actor.get_state(), actor.is_frozen()) for key, actor in self.actors.items()}
        self.last_states = {}
        for actor in self.actors.values():
            actor.freeze(True)
        self.update(0.0)

    def update(self, elapsed_seconds):
        for binding in self.bindings:
            phases = self.programs[binding["source_signal_id"]]["phases"]
            time = elapsed_seconds % sum(p["duration_s"] for p in phases)
            for index, phase in enumerate(phases):
                if time < phase["duration_s"]:
                    key, state = binding["signal_id"], binding["phase_states"][index]
                    if self.last_states.get(key) != state:
                        self.actors[key].set_state(self.states[state])
                        self.last_states[key] = state
                    break
                time -= phase["duration_s"]

    def close(self):
        for key, actor in self.actors.items():
            state, frozen = self.original[key]
            actor.set_state(state)
            actor.freeze(frozen)
