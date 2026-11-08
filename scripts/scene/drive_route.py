"""Drive a source route with CARLA GlobalRoutePlanner and BasicAgent.

Connect to an existing simulator. Optional SUMO phase plans restore the
original light states when the drive ends.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import queue
import sys

from map_generation.fusion import is_boundary_seam


def xyz(location):
    return [location.x, location.y, location.z]


def enu(location):
    return [location.x, -location.y, location.z]


def synchronous_rgb(images, frame):
    """Require RGB from this exact tick; the camera must use sensor_tick=0."""
    while True:
        try:
            image = images.get(timeout=60)
        except queue.Empty as error:
            raise RuntimeError("RGB timeout waiting for world frame %s (sensor_tick=0)" % frame) from error
        if image.frame >= frame:
            break
    if image.frame != frame:
        raise RuntimeError("RGB frame %s differs from world frame %s" % (image.frame, frame))
    return image


def compatible_waypoint(carla_map, waypoint, mapping, allowed):
    """Resolve only geometrically identical shared-source movement samples.

    GRP's DiGraph can retain another movement's entrance at a common node.
    Match its explicit source station, not just the nearest physical position.
    """
    import numpy as np
    if waypoint.road_id in allowed:
        return waypoint, None
    meta = mapping["roads"].get(str(waypoint.road_id), {})
    intervals = mapping.get("lane_to_road_intervals", {})
    for lane_id in meta.get("source_lane_path", []):
        for original in intervals[lane_id]:
            if int(original["road_id"]) != waypoint.road_id or not original["road_s_start_m"] - 1e-6 <= waypoint.s <= original["road_s_end_m"] + 1e-6:
                continue
            fraction = (waypoint.s - original["road_s_start_m"]) / (original["road_s_end_m"] - original["road_s_start_m"])
            station = original["source_s_start_m"] + fraction * (original["source_s_end_m"] - original["source_s_start_m"])
            for target in intervals[lane_id]:
                rid = int(target["road_id"])
                if rid not in allowed or not target["source_s_start_m"] - 1e-6 <= station <= target["source_s_end_m"] + 1e-6:
                    continue
                if 'source_stations_m' in target:
                    road_s = float(np.interp(station, target['source_stations_m'], target['road_stations_m']))
                else:
                    fraction = (station - target["source_s_start_m"]) / (target["source_s_end_m"] - target["source_s_start_m"])
                    road_s = target["road_s_start_m"] + min(1.0, max(0.0, fraction)) * (target["road_s_end_m"] - target["road_s_start_m"])
                candidate = carla_map.get_waypoint_xodr(rid, target["lane_id"], road_s)
                if candidate is None:
                    continue
                error = candidate.transform.location.distance(waypoint.transform.location)
                a, b = candidate.transform.get_forward_vector(), waypoint.transform.get_forward_vector()
                if error <= 0.001 and a.x*b.x + a.y*b.y + a.z*b.z > 0.9999:
                    return candidate, {"original_road_id": waypoint.road_id, "road_id": rid,
                                       "source_lane_id": lane_id, "source_s_m": station, "position_error_m": error}
    return waypoint, None


def resolve_overlapping_movement(carla_map, waypoint, location, forward, mapping, lanes, allowed):
    """Resolve overlapping junction movements using source entrance, position, and heading."""
    from shapely.geometry import LineString, Point
    original = mapping['roads'].get(str(waypoint.road_id), {})
    if waypoint.road_id in allowed or original.get('kind') != 'movement':
        return waypoint, None
    point = Point(location.x, -location.y)
    original_distance = math.hypot(location.x-waypoint.transform.location.x,
                                   location.y-waypoint.transform.location.y)
    candidates = []
    for rid in sorted(allowed):
        meta = mapping['roads'][str(rid)]
        if (meta.get('kind') != 'movement' or meta.get('junction_id') != original.get('junction_id')
                or meta.get('from_lane_id') != original.get('from_lane_id')):
            continue
        for lane_id in meta['source_lane_path']:
            line = LineString([p[:2] for p in lanes[lane_id]['centerline']])
            station = line.project(point)
            for span in mapping['lane_to_road_intervals'][lane_id]:
                if int(span['road_id']) != rid or not span['source_s_start_m'] <= station <= span['source_s_end_m']:
                    continue
                fraction = (station-span['source_s_start_m'])/(span['source_s_end_m']-span['source_s_start_m'])
                road_s = span['road_s_start_m'] + fraction*(span['road_s_end_m']-span['road_s_start_m'])
                candidate = carla_map.get_waypoint_xodr(rid, span['lane_id'], road_s)
                if candidate is None:
                    continue
                p = candidate.transform.location; direction = candidate.transform.get_forward_vector()
                distance = math.hypot(location.x-p.x, location.y-p.y)
                heading = direction.x*forward.x + direction.y*forward.y
                if distance <= min(original_distance+1e-6, candidate.lane_width/2) and heading >= .9:
                    candidates.append((distance, rid, candidate, lane_id, heading))
    if not candidates:
        return waypoint, None
    distance, rid, candidate, lane_id, heading = min(candidates, key=lambda row: (row[0], row[1]))
    return candidate, {'native_query_road_id': waypoint.road_id, 'resolved_road_id': rid,
        'source_lane_id': lane_id, 'native_query_center_distance_m': original_distance,
        'route_center_distance_m': distance, 'heading_dot': heading,
        'method': 'Closer containing native movement with same source entrance and junction; evaluated at this frame only'}



def trace_source_successors(start, goal, allowed, line, maximum_length, step=1.0):
    """Traverse declared native successors within the ordered source corridor."""
    from collections import deque
    from agents.navigation.local_planner import RoadOption
    from shapely.geometry import Point

    def key(wp):
        return wp.road_id, wp.section_id, wp.lane_id, round(wp.s / (step / 4))

    first = key(start)
    nodes, parents = {first: start}, {first: None}
    queue = deque([(first, 0.0)])
    while queue:
        identity, length = queue.popleft()
        wp = nodes[identity]
        same_lane = (wp.road_id, wp.section_id, wp.lane_id) == (
            goal.road_id, goal.section_id, goal.lane_id)
        if same_lane and abs(wp.s - goal.s) <= step:
            path = []
            while identity is not None:
                path.append((nodes[identity], RoadOption.LANEFOLLOW))
                identity = parents[identity]
            return list(reversed(path)) + [(goal, RoadOption.LANEFOLLOW)]
        if length + step > maximum_length:
            continue
        p = Point(wp.transform.location.x, -wp.transform.location.y)
        progress = line.project(p)
        for nxt in wp.next(step):
            q = Point(nxt.transform.location.x, -nxt.transform.location.y)
            k = key(nxt)
            if (nxt.road_id not in allowed or line.distance(q) > 2.0
                    or line.project(q) < progress - step / 4 or k in parents):
                continue
            nodes[k], parents[k] = nxt, identity
            queue.append((k, length + step))
    raise ValueError("No native successor path follows the supplied source corridor")


def plan_route(carla_map, route, lanes, mapping, line, planner=None, vehicle_class="passenger"):
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption
    from shapely.geometry import Point

    # Keep the full vehicle away from the supplied route's endpoint edges.
    start_s, end_s = 5.0, line.length - 3.0
    if end_s <= start_s:
        raise ValueError("Route is too short for the vehicle endpoint margins")
    route_ids = route["lane_ids"]

    def source_path(meta):
        if meta.get('kind') == 'lane_group':
            return [key for key in route_ids if key in meta['source_lane_ids']]
        if meta.get("source_lane_path"):
            return meta["source_lane_path"]
        if meta.get("scene_lane_id"):
            return [meta["scene_lane_id"]]
        return [meta.get("from_lane_id"), meta.get("to_lane_id")]

    # Match each waypoint to its interval in the ordered source route.
    def ordered_overlap(path):
        # Match all source lanes within the declared route endpoints.
        return any(all(route_ids[i] == path[i-offset]
                       for i in range(max(0, offset), min(len(route_ids), offset+len(path))))
                   for offset in range(1-len(path), len(route_ids))) if path else False

    allowed = {int(rid) for rid, meta in mapping["roads"].items()
               if vehicle_class in meta["allowed_classes"]
               and (any(key in route_ids for key in meta['source_lane_ids'])
                   if meta.get('kind') == 'lane_group' else ordered_overlap(source_path(meta)))}
    if "lane_to_road_intervals" in mapping:
        for lane_id in route_ids:
            if not any(int(interval["road_id"]) in allowed
                       for interval in mapping["lane_to_road_intervals"][lane_id]):
                raise ValueError("No ordered compatible movement interval covers " + lane_id)
    for lane_id in route["lane_ids"]:
        if vehicle_class not in lanes[lane_id]["allowed_classes"]:
            raise ValueError(vehicle_class + " not allowed on route lane " + lane_id)
    def waypoint_at(distance):
        p = line.interpolate(distance)
        if planner is not None and hasattr(planner, 'trace_waypoints'):
            import numpy as np
            from shapely.geometry import LineString
            from map_generation.lane_sections import reference_station
            from map_generation.road_builder import fit_lane
            key = min(route_ids, key=lambda k: LineString(lanes[k]['centerline']).distance(p))
            curve = fit_lane(lanes[key], 1.)[0]
            station = reference_station(curve, np.asarray(p.coords[0]))
            candidates = []
            for span in mapping['lane_to_road_intervals'][key]:
                if int(span['road_id']) not in allowed or not span['source_s_start_m'] - 1e-6 <= station <= span['source_s_end_m'] + 1e-6:
                    continue
                road_s = (float(np.interp(station, span['source_stations_m'], span['road_stations_m']))
                    if 'source_stations_m' in span else span['road_s_start_m'] + station - span['source_s_start_m'])
                wp = carla_map.get_waypoint_xodr(int(span['road_id']), int(span['lane_id']), road_s)
                if wp is not None:
                    candidates.append((wp.transform.location.distance(carla.Location(x=p.x, y=-p.y, z=p.z)), wp))
            if not candidates:
                raise ValueError('Route endpoint has no source-backed native interval: ' + key)
            error, wp = min(candidates, key=lambda row: row[0])
            if error > 2.:
                raise ValueError('Source-backed native endpoint differs by %.3f m' % error)
            return wp
        wp = carla_map.get_waypoint(carla.Location(x=p.x, y=-p.y, z=p.z),
                                    lane_type=carla.LaneType.Driving)
        if wp is None or wp.road_id not in allowed:
            raise ValueError("Route endpoint projects onto an unexpected CARLA road")
        return wp

    start, goal = waypoint_at(start_s), waypoint_at(end_s)
    anchors = [start]
    for lane_id in route_ids:
        # Use external lane centers as planner via points at overlapping junction movements.
        regular = sorted((span for span in mapping['lane_to_road_intervals'][lane_id]
                          if span['junction_id'] is None and int(span['road_id']) in allowed),
                         key=lambda span: span['source_s_start_m'])
        for span in regular:
            wp = carla_map.get_waypoint_xodr(int(span['road_id']), int(span['lane_id']),
                                           (span['road_s_start_m'] + span['road_s_end_m']) / 2)
            p = wp.transform.location
            distance = line.project(Point(p.x, -p.y))
            if start_s < distance < end_s:
                anchors.append(wp)
    anchors.append(goal)
    planner = planner or GlobalRoutePlanner(carla_map, sampling_resolution=1.0)
    # Fill missing LANEFOLLOW net_vector attributes from their endpoint geometry.
    loose_end_vectors = []
    for _, _, edge in planner._graph.edges(data=True):
        if edge["type"] == RoadOption.LANEFOLLOW and edge.get("net_vector") is None:
            delta = edge["exit_waypoint"].transform.location - edge["entry_waypoint"].transform.location
            direction = delta.make_unit_vector()
            edge["net_vector"] = [direction.x, direction.y, direction.z]
            loose_end_vectors.append(edge["entry_waypoint"].road_id)
    source_successors = vehicle_class != 'passenger' and any(
        wp.lane_id not in planner._road_id_to_edge.get(wp.road_id, {}).get(wp.section_id, {})
        for wp in anchors)
    traces = ([trace_source_successors(start, goal, allowed, line, end_s - start_s + 2.0)]
              if source_successors else [
                  planner.trace_waypoints(a, b) if hasattr(planner, 'trace_waypoints') else
                  planner.trace_route(a.transform.location, b.transform.location)
                  for a, b in zip(anchors, anchors[1:])])
    plan, progress, max_error, aliases = [], start_s - 0.1, 0.0, []
    for trace in traces:
        for wp, option in trace:
            wp, alias = compatible_waypoint(carla_map, wp, mapping, allowed)
            if alias is not None:
                aliases.append(alias)
            p = Point(wp.transform.location.x, -wp.transform.location.y)
            distance, error = line.project(p), line.distance(p)
            if wp.road_id not in allowed or error > 2.0:
                raise ValueError("GlobalRoutePlanner left the supplied route: road %s, offset %.3f m" % (wp.road_id, error))
            # The official trace includes the start of each graph edge, so
            # successive via-point traces can repeat already supplied samples.
            if distance < progress + 0.05 or distance > end_s:
                continue
            plan.append((wp, option))
            progress, max_error = distance, max(max_error, error)
    plan.append((goal, RoadOption.LANEFOLLOW))
    return planner, plan, start, goal, allowed, {
        "vehicle_class": vehicle_class,
        "source_length_m": line.length, "spawn_inset_m": start_s, "end_inset_m": 3.0,
        "start_route_s_m": start_s, "goal_route_s_m": end_s,
        "planned_waypoints": len(plan), "via_points": len(anchors),
        "max_plan_offset_m": max_error,
        "allowed_road_ids": sorted(allowed),
        "shared_source_waypoint_aliases": aliases,
        "grp_loose_end_vectors_completed": loose_end_vectors,
        "planner": "NativeSourceSuccessors" if source_successors else type(planner).__name__,
        "compatible_movements": [{"road_id": rid, "source_lane_path": source_path(mapping["roads"][str(rid)])}
                                 for rid in sorted(allowed)
                                 if mapping['roads'][str(rid)].get('kind') != 'lane_group'
                                 and len(source_path(mapping["roads"][str(rid)])) > 1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--roads-dir", type=Path, required=True)
    parser.add_argument("--route-json", type=Path, required=True)
    parser.add_argument("--signal-bindings", type=Path, help="Exported SUMO-to-CARLA signal binding manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default=os.environ.get("MAP_RPC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MAP_RPC_PORT", "23000")))
    parser.add_argument("--carla-root", type=Path, default=Path(os.environ.get(
        "CARLA_ROOT", str(Path(__file__).resolve().parents[3] / "apps/carla-0.9.16"))))
    parser.add_argument("--speed", type=float, default=15.0, help="Target speed in km/h")
    parser.add_argument("--base-min-distance", type=float, default=3.0,
                        help="Official LocalPlanner waypoint removal distance base, in metres")
    parser.add_argument("--distance-ratio", type=float, default=0.5,
                        help="Official LocalPlanner speed-dependent distance coefficient, in seconds")
    parser.add_argument("--max-seconds", type=float, default=240.0, help="Simulated driving time limit")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(args.carla_root.resolve() / "PythonAPI/carla"))
    import carla
    import numpy as np
    from agents.navigation.basic_agent import BasicAgent
    from shapely.geometry import LineString, Point

    output = args.output.resolve()
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    scene = json.loads(args.scene.read_text())
    route = json.loads(args.route_json.read_text())["route"]
    mapping = json.loads((args.roads_dir / "road_lane_ids.json").read_text())
    lanes = {lane["id"]: lane for lane in scene["fusion_map"]["lanes"]}
    line = LineString(route["positions"])
    report = {"passed": False, "termination": "initializing", "scene": str(args.scene.resolve()),
              "roads_dir": str(args.roads_dir.resolve()), "route_json": str(args.route_json.resolve()),
              "host": args.host, "port": args.port, "fixed_delta_seconds": 0.05,
              "camera_sensor_tick_seconds": 0.0,
              "target_speed_kmh": args.speed, "route_lane_ids": route["lane_ids"],
              "scope": "Single passenger vehicle with official BasicAgent on the supplied fine/coarse route; no background traffic or signal-policy validation"}
    world = original_settings = vehicle = camera = collision_sensor = None
    signal_runner = None
    collisions, cleanup_errors, checkpoints, rows = [], [], [], []
    visited_sources, unexpected_roads = set(), Counter()
    last_image = None
    signal_frames = 0
    lights = []
    termination = "exception"
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        world = client.get_world()
        carla_map = world.get_map()
        report.update(map=carla_map.name, client_version=client.get_client_version(),
                      server_version=client.get_server_version())
        planner, plan, start_wp, goal_wp, allowed_roads, route_report = plan_route(
            carla_map, route, lanes, mapping, line)
        report["route_plan"] = route_report
        (output / "plan.json").write_text(json.dumps([
            {"road_id": wp.road_id, "lane_id": wp.lane_id, "s": wp.s,
             "position_enu_m": enu(wp.transform.location), "option": option.name}
            for wp, option in plan], indent=2) + "\n")
        landmarks = route.get("landmarks", [landmark for area in scene["areas"] for landmark in area["landmarks"]])
        for landmark in landmarks:
            point = Point(landmark["position"][:2])
            station = line.project(point)
            checkpoints.append({"id": landmark["id"], "route_s_m": station,
                                "landmark_xy": list(landmark["position"][:2]),
                                "landmark_offset_from_route_m": line.distance(point),
                                "point_xy": list(line.interpolate(station).coords[0][:2]),
                                "minimum_distance_m": math.inf, "passed": False})
        seams = [lanes[lane_id] for lane_id in route["lane_ids"]
                 if is_boundary_seam(lanes[lane_id])]
        if not seams:
            raise ValueError("Expected a fine/coarse seam on this route")
        for seam in seams:
            for end, point in (("entry", seam["centerline"][0]), ("exit", seam["centerline"][-1])):
                station = line.project(Point(point[:2]))
                checkpoints.append({"id": seam["id"] + ":" + end, "route_s_m": station,
                                    "point_xy": list(line.interpolate(station).coords[0][:2]),
                                    "minimum_distance_m": math.inf, "passed": False})
        if any(not route_report["start_route_s_m"] + 3 < p["route_s_m"] < route_report["goal_route_s_m"] - 3 for p in checkpoints):
            raise ValueError("A required checkpoint is too close to a route endpoint margin")

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        blueprints = world.get_blueprint_library()
        bp = blueprints.find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "mapbuild_route_check")
        spawn = start_wp.transform
        spawn.location.z += 0.5
        vehicle = world.spawn_actor(bp, spawn)
        if vehicle.bounding_box.extent.x >= route_report["spawn_inset_m"]:
            raise ValueError("Vehicle extends beyond the planned spawn margin")
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(20):
            world.tick()
        report["settling_seconds_before_sensors"] = 1.0
        report["vehicle_id"] = vehicle.id
        report["vehicle_length_m"] = 2 * vehicle.bounding_box.extent.x
        report["vehicle_width_m"] = 2 * vehicle.bounding_box.extent.y
        report["vehicle_height_m"] = 2 * vehicle.bounding_box.extent.z
        if args.signal_bindings:
            from map_generation.traffic_signals import SignalProgramRunner, check_runtime_signal_bindings

            manifest = json.loads(args.signal_bindings.read_text())
            report["signal_bindings"] = check_runtime_signal_bindings(world, manifest)
            signal_runner = SignalProgramRunner(world, manifest)

        collision_sensor = world.spawn_actor(blueprints.find("sensor.other.collision"),
                                              carla.Transform(), attach_to=vehicle)

        def on_collision(event):
            collisions.append({"frame": event.frame, "other_actor_id": event.other_actor.id,
                               "other_actor_type": event.other_actor.type_id,
                               "normal_impulse": xyz(event.normal_impulse)})

        collision_sensor.listen(on_collision)
        camera_bp = blueprints.find("sensor.camera.rgb")
        # Set TickInterval to zero for one sensor frame per world tick.
        for key, value in {"image_size_x": "640", "image_size_y": "360", "sensor_tick": "0.0"}.items():
            camera_bp.set_attribute(key, value)
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2)), attach_to=vehicle)
        images = queue.Queue()
        camera.listen(images.put)
        controller_options = {"dt": 0.05, "sampling_resolution": 1.0,
                              "base_min_distance": args.base_min_distance, "distance_ratio": args.distance_ratio}
        report["basic_agent_options"] = controller_options
        agent = BasicAgent(vehicle, target_speed=args.speed, map_inst=carla_map, grp_inst=planner,
                           opt_dict=controller_options)
        agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
        lights = list(world.get_actors().filter("traffic.traffic_light*"))
        best_progress, progress_anchor, stationary_since = route_report["start_route_s_m"], route_report["start_route_s_m"], 0.0
        red_wait_since = None
        off_route_steps = 0
        termination = "time_limit"
        with (output / "trajectory.jsonl").open("w") as trajectory:
            for step in range(int(args.max_seconds / 0.05)):
                if signal_runner is not None:
                    signal_runner.update(step * 0.05)
                control = agent.run_step()
                vehicle.apply_control(control)
                frame = world.tick()
                image = synchronous_rgb(images, frame)
                last_image = image
                snapshot = world.get_snapshot()
                location, velocity = vehicle.get_location(), vehicle.get_velocity()
                position = Point(location.x, -location.y)
                progress, offset = line.project(position), line.distance(position)
                elapsed = (step + 1) * 0.05
                best_progress = max(best_progress, progress)
                if best_progress >= progress_anchor + 0.5:
                    progress_anchor, stationary_since = best_progress, elapsed
                wp = carla_map.get_waypoint(location)
                raw_road_id = wp.road_id
                wp, alias = compatible_waypoint(carla_map, wp, mapping, allowed_roads)
                wp, overlap = resolve_overlapping_movement(carla_map, wp, location,
                    vehicle.get_transform().get_forward_vector(), mapping, lanes, allowed_roads)
                meta = mapping["roads"].get(str(wp.road_id), {})
                if meta.get("source"):
                    visited_sources.add(meta["source"])
                unexpected = wp.road_id not in allowed_roads
                if unexpected:
                    unexpected_roads[str(wp.road_id)] += 1
                off_route_steps = off_route_steps + 1 if offset > 3.0 or unexpected else 0
                passed_now = []
                for checkpoint in checkpoints:
                    checkpoint["minimum_distance_m"] = min(checkpoint["minimum_distance_m"], position.distance(Point(checkpoint["point_xy"])))
                    if not checkpoint["passed"] and progress >= checkpoint["route_s_m"] + 3 and checkpoint["minimum_distance_m"] <= 3:
                        checkpoint.update(passed=True, frame=frame)
                        passed_now.append(checkpoint["id"])
                # This is the red signal detected by the official BasicAgent,
                # including before the ego enters the trigger volume itself.
                agent_light = agent._last_traffic_light
                waiting_red = agent_light is not None and agent_light.get_state() == carla.TrafficLightState.Red
                if waiting_red:
                    red_wait_since = elapsed if red_wait_since is None else red_wait_since
                    stationary_since = elapsed
                else:
                    red_wait_since = None
                states = [{"id": light.id, "state": str(light.get_state())} for light in lights]
                trigger_light = vehicle.get_traffic_light() if vehicle.is_at_traffic_light() else None
                if trigger_light is not None or waiting_red:
                    signal_frames += 1
                transform = vehicle.get_transform()
                row = {"frame": frame, "camera_frame": image.frame, "sim_time_s": elapsed, "world_time_s": snapshot.timestamp.elapsed_seconds,
                       "position_enu_m": enu(location), "position_carla_m": xyz(location),
                       "rotation_pitch_yaw_roll_deg": [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
                       "bounding_box_world_enu_m": [enu(vertex) for vertex in vehicle.bounding_box.get_world_vertices(transform)],
                       "speed_kmh": 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
                       "route_s_m": progress, "route_offset_m": offset, "road_id": wp.road_id,
                       "native_query_road_id": raw_road_id, "route_overlap_resolution": overlap,
                       "lane_id": wp.lane_id, "source": meta.get("source"), "native_is_junction": wp.is_junction,
                       "shared_source_waypoint_alias": alias,
                       "control": {"throttle": control.throttle, "steer": control.steer, "brake": control.brake},
                       "lights": states, "ego_trigger_light_id": trigger_light.id if trigger_light else None,
                       "agent_red_light_id": agent_light.id if waiting_red else None,
                       "new_checkpoints": passed_now}
                trajectory.write(json.dumps(row) + "\n")
                rows.append(row)
                if step % 100 == 0 or passed_now:
                    image.save_to_disk(str(output / "rgb" / ("frame_%08d.png" % frame)))
                    trajectory.flush()
                if collisions:
                    termination = "collision"
                    break
                if off_route_steps >= 20:
                    termination = "left_supplied_route"
                    break
                if red_wait_since is not None and elapsed - red_wait_since > 120:
                    termination = "red_signal_wait_timeout"
                    break
                if not waiting_red and elapsed - stationary_since > 30:
                    termination = "stalled"
                    break
                if progress >= route_report["goal_route_s_m"] - 1.5 and location.distance(goal_wp.transform.location) < 2:
                    termination = "reached_goal"
                    break
                if agent.done():
                    termination = "planner_finished_before_goal"
                    break
        report["progress_m"] = best_progress - route_report["start_route_s_m"]
        report["completion_fraction"] = min(1.0, report["progress_m"] / (route_report["goal_route_s_m"] - route_report["start_route_s_m"]))
    except Exception as error:
        report["error"] = "%s: %s" % (type(error).__name__, error)
        termination = "exception"
    finally:
        for actor in (camera, collision_sensor, vehicle):
            if actor is not None:
                try:
                    if actor.type_id.startswith("sensor."):
                        actor.stop()
                    if not actor.destroy():
                        cleanup_errors.append("Actor %s destruction was not acknowledged" % actor.id)
                except Exception as error:
                    cleanup_errors.append(str(error))
        if signal_runner is not None:
            try:
                signal_runner.close()
            except Exception as error:
                cleanup_errors.append("restore traffic lights: " + str(error))
        if original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception as error:
                cleanup_errors.append("restore world settings: " + str(error))
        if last_image is not None:
            last_image.save_to_disk(str(output / "rgb" / "last.png"))
            report["rgb_standard_deviation"] = float(np.frombuffer(last_image.raw_data, dtype=np.uint8).reshape(360, 640, 4)[:, :, :3].std())
        for checkpoint in checkpoints:
            if not math.isfinite(checkpoint["minimum_distance_m"]):
                checkpoint["minimum_distance_m"] = None
        report.update(termination=termination, ticks=len(rows), collisions=collisions,
                      checkpoints=checkpoints, visited_source_types=sorted(visited_sources),
                      unexpected_road_frames=dict(unexpected_roads), cleanup_errors=cleanup_errors,
                      traffic_lights={"actor_count": len(lights), "ego_signal_frames": signal_frames,
                                      "status": "sumo_program_states_recorded" if signal_runner is not None else
                                                "actual_states_recorded" if lights else "unverified_no_signal_actors",
                                      "binding_manifest": str(args.signal_bindings.resolve()) if args.signal_bindings else None,
                                      "states_changed_by_this_script": signal_runner is not None,
                                      "compliance_verified": False})
        report["checks"] = {
            "reached_goal": termination == "reached_goal", "no_collisions": not collisions,
            "loaded_signal_bindings_match": report.get("signal_bindings", {}).get("passed", False) if args.signal_bindings else None,
            "all_route_checkpoints_crossed": bool(checkpoints) and all(p["passed"] for p in checkpoints),
            "fine_and_coarse_driven": {"av2", "osm_sumo"}.issubset(visited_sources),
            "rgb_received": report.get("rgb_standard_deviation", 0) > 1,
            "actors_cleaned_and_settings_restored": not cleanup_errors,
        }
        report["passed"] = all(value for value in report["checks"].values() if value is not None)
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
