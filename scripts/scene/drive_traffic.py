"""Run a three-vehicle interaction check using source routes and BasicAgent.

A lead-vehicle stop probes following behavior under the supplied signal plan.
"""

import argparse
import json
import math
import os
from pathlib import Path
import queue
import sys


def cross_route(scene, main_route):
    """Choose a passenger-legal crossing through the first core landmark."""
    import networkx as nx
    import numpy as np
    from shapely.geometry import LineString, Point

    lanes = {l["id"]: l for l in scene["fusion_map"]["lanes"] if "passenger" in l["allowed_classes"]}
    curves = {key: LineString(lane["centerline"]) for key, lane in lanes.items()}
    graph = nx.DiGraph()
    graph.add_nodes_from(lanes)
    for key, lane in lanes.items():
        graph.add_edges_from((key, other, {"weight": curves[other].length})
                             for other in lane["successors"] if other in lanes)
    landmark = scene["areas"][0]["landmarks"][0]
    center = Point(landmark["position"][:2])
    main = LineString(main_route["positions"])
    station = main.project(center)
    direction = np.asarray(main.interpolate(station + 1).coords[0][:2]) - np.asarray(main.interpolate(station - 1).coords[0][:2])
    direction /= np.linalg.norm(direction)
    choices = []
    for key, curve in curves.items():
        if not (lanes[key].get("is_intersection") or lanes[key].get("internal")) or curve.distance(center) > 20:
            continue
        tangent = np.asarray(curve.coords[-1][:2]) - np.asarray(curve.coords[0][:2])
        if np.linalg.norm(tangent) < 0.1 or abs(float(direction @ (tangent / np.linalg.norm(tangent)))) > 0.65:
            continue
        before = nx.single_source_dijkstra_path(graph.reverse(copy=False), key, cutoff=130, weight="weight")
        after = nx.single_source_dijkstra_path(graph, key, cutoff=130, weight="weight")
        entries = [list(reversed(path)) for node, path in before.items()
                   if 35 < Point(curves[node].coords[0][:2]).distance(center) < 180]
        exits = [path for node, path in after.items()
                 if 35 < Point(curves[node].coords[-1][:2]).distance(center) < 180]
        for entry in entries:
            for exit_path in exits:
                ids = entry + exit_path[1:]
                if len(set(ids)) != len(ids):
                    continue
                start = np.asarray(curves[ids[0]].coords[0][:2]) - np.asarray(center.coords[0])
                end = np.asarray(curves[ids[-1]].coords[-1][:2]) - np.asarray(center.coords[0])
                # Require opposite sides of the junction and avoid a parallel
                # route that merely turns through the candidate internal lane.
                if float(start @ end) >= 0 or abs(float(direction @ start / np.linalg.norm(start))) > 0.75:
                    continue
                length = sum(curves[i].length for i in ids)
                choices.append((length, ids))
    if not choices:
        raise ValueError("No passenger-legal crossing route found through the first core junction")
    _, ids = min(choices)
    return {"lane_ids": ids, "positions": [p for key in ids for p in lanes[key]["centerline"]],
            "selection": "Source-successor passenger path crossing " + landmark["id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scene", "roads-dir", "route-json", "signal-bindings", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--host", default=os.environ.get("MAP_RPC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MAP_RPC_PORT", "23000")))
    parser.add_argument("--carla-root", type=Path, default=Path(os.environ.get(
        "CARLA_ROOT", str(Path(__file__).resolve().parents[3] / "apps/carla-0.9.16"))))
    parser.add_argument("--max-seconds", type=float, default=240)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(args.carla_root.resolve() / "PythonAPI/carla"))
    import carla
    from agents.navigation.basic_agent import BasicAgent
    from shapely.geometry import LineString, Point
    from shapely.ops import substring
    from map_generation.traffic_signals import SignalProgramRunner, check_runtime_signal_bindings
    from scripts.scene.drive_route import compatible_waypoint, enu, plan_route, synchronous_rgb, xyz
    from scripts.scene.traffic_evidence import evaluate_traffic_evidence

    output = args.output.resolve()
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    scene = json.loads(args.scene.read_text())
    primary = json.loads(args.route_json.read_text())["route"]
    mapping = json.loads((args.roads_dir / "road_lane_ids.json").read_text())
    lanes = {lane["id"]: lane for lane in scene["fusion_map"]["lanes"]}
    leader = dict(primary, positions=list(substring(LineString(primary["positions"]), 22, LineString(primary["positions"]).length).coords))
    routes = {"ego": primary, "leader": leader, "crossing": cross_route(scene, primary)}
    (output / "routes.json").write_text(json.dumps(routes, indent=2) + "\n")
    report = {"passed": False, "scene": str(args.scene.resolve()), "route_json": str(args.route_json.resolve()),
              "signal_bindings": str(args.signal_bindings.resolve()), "fixed_delta_seconds": 0.05,
              "camera_sensor_tick_seconds": 0.0,
              "scope": "Three-car map response check with explicit source routes"}
    world = original_settings = camera = runner = None
    vehicles, sensors, states, collisions = {}, {}, {}, []
    cleanup_errors, rows, vehicle_lengths = [], [], {}
    last_image, ticks, termination = None, 0, "exception"
    hold_start = hold_end = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60)
        world = client.get_world()
        carla_map = world.get_map()
        report["map"] = carla_map.name
        plans = {name: plan_route(carla_map, route, lanes, mapping, LineString(route["positions"]))
                 for name, route in routes.items()}
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, 0.05, False
        world.apply_settings(settings)
        blueprints = world.get_blueprint_library()
        for name, route in routes.items():
            planner, plan, start, goal, allowed, detail = plans[name]
            bp = blueprints.find("vehicle.tesla.model3")
            bp.set_attribute("role_name", "mapbuild_traffic_" + name)
            transform = start.transform
            transform.location.z += 0.5
            vehicles[name] = world.spawn_actor(bp, transform)
            vehicle_lengths[name] = 2 * vehicles[name].bounding_box.extent.x
            vehicles[name].apply_control(carla.VehicleControl(brake=1))
            agent = BasicAgent(vehicles[name], target_speed=15, map_inst=carla_map, grp_inst=planner,
                               opt_dict={"dt": 0.05, "sampling_resolution": 1.0})
            agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
            states[name] = {"agent": agent, "line": LineString(route["positions"]), "goal": goal,
                            "allowed": allowed, "plan": detail, "done": False, "off_route": 0,
                            "best_progress": 5.0, "last_progress_time": 0.0, "red_since": None}
        for _ in range(20):
            world.tick()
        for name, vehicle in vehicles.items():
            sensor = world.spawn_actor(blueprints.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)
            sensor.listen(lambda event, name=name: collisions.append({
                "vehicle": name, "frame": event.frame, "other_actor_id": event.other_actor.id,
                "other_actor_type": event.other_actor.type_id, "normal_impulse": xyz(event.normal_impulse)}))
            sensors[name] = sensor
        bp = blueprints.find("sensor.camera.rgb")
        # Match the strict one-RGB-frame-per-world-tick wait below.
        for key, value in {"image_size_x": "640", "image_size_y": "360", "sensor_tick": "0.0"}.items():
            bp.set_attribute(key, value)
        camera = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.5, z=2)), attach_to=vehicles["ego"])
        images = queue.Queue()
        camera.listen(images.put)
        manifest = json.loads(args.signal_bindings.read_text())
        report["runtime_signal_bindings"] = check_runtime_signal_bindings(world, manifest)
        runner = SignalProgramRunner(world, manifest)
        lights = {light.id: light for light in world.get_actors().filter("traffic.traffic_light*")}
        termination = "time_limit"
        with (output / "trajectory.jsonl").open("w") as log:
            for step in range(int(args.max_seconds / 0.05)):
                elapsed = step * 0.05
                runner.update(elapsed)
                if hold_start is None and states["leader"]["best_progress"] >= 20:
                    hold_start, hold_end = elapsed, elapsed + 8.0
                holding = hold_start is not None and elapsed < hold_end
                controls = {}
                for name, vehicle in vehicles.items():
                    state = states[name]
                    control = state["agent"].run_step() if not state["done"] else carla.VehicleControl(brake=1)
                    if name == "leader" and holding:
                        control = carla.VehicleControl(brake=1)
                    controls[name] = control
                    vehicle.apply_control(control)
                frame = world.tick()
                image = synchronous_rgb(images, frame)
                last_image, ticks = image, step + 1
                ego_position = vehicles["ego"].get_location()
                row = {"frame": frame, "camera_frame": image.frame, "sim_time_s": elapsed + 0.05, "leader_forced_stop": holding,
                       "lights": {str(key): str(light.get_state()) for key, light in lights.items()}, "vehicles": {}}
                finished = []
                for name, vehicle in vehicles.items():
                    state = states[name]
                    location, velocity = vehicle.get_location(), vehicle.get_velocity()
                    speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    point = Point(location.x, -location.y)
                    progress, offset = state["line"].project(point), state["line"].distance(point)
                    wp = carla_map.get_waypoint(location)
                    wp, alias = compatible_waypoint(carla_map, wp, mapping, state["allowed"])
                    signal = state["agent"]._last_traffic_light
                    red = signal is not None and signal.get_state() == carla.TrafficLightState.Red
                    if progress > state["best_progress"] + 0.5:
                        state["best_progress"], state["last_progress_time"] = progress, elapsed
                    if red or (name == "leader" and holding):
                        state["last_progress_time"] = elapsed
                    state["red_since"] = (elapsed if state["red_since"] is None else state["red_since"]) if red else None
                    state["off_route"] = state["off_route"] + 1 if offset > 3 or wp.road_id not in state["allowed"] else 0
                    distance = location.distance(ego_position)
                    control = controls[name]
                    row["vehicles"][name] = {"actor_id": vehicle.id, "position_enu_m": enu(location),
                        "speed_kmh": speed, "route_s_m": progress, "route_offset_m": offset,
                        "road_id": wp.road_id, "lane_id": wp.lane_id, "distance_from_ego_m": distance,
                        "shared_source_waypoint_alias": alias,
                        "agent_red_light_id": signal.id if red else None,
                        "control": {"throttle": control.throttle, "steer": control.steer, "brake": control.brake}}
                    if not state["done"] and progress >= state["plan"]["goal_route_s_m"] - 1.5 and location.distance(state["goal"].transform.location) < 2:
                        state.update(done=True, goal_frame=frame)
                        if name != "ego":
                            finished.append(name)
                    if not state["done"] and (state["off_route"] >= 20 or elapsed - state["last_progress_time"] > 40
                            or (state["red_since"] is not None and elapsed - state["red_since"] > 120)):
                        termination = "route_or_progress_failure:" + name
                row["finished_and_removed"] = finished
                log.write(json.dumps(row) + "\n")
                rows.append(row)
                if step % 100 == 0:
                    image.save_to_disk(str(output / "rgb" / ("frame_%08d.png" % frame)))
                    log.flush()
                for name in finished:
                    sensor = sensors.pop(name)
                    sensor.stop()
                    if not sensor.destroy():
                        cleanup_errors.append("Finished-trip sensor destruction not acknowledged: " + name)
                    if not vehicles.pop(name).destroy():
                        cleanup_errors.append("Finished-trip vehicle destruction not acknowledged: " + name)
                if collisions:
                    termination = "collision"
                    break
                if termination.startswith("route_or_progress_failure"):
                    break
                if all(state["done"] for state in states.values()):
                    termination = "all_routes_completed"
                    break
    except Exception as error:
        report["error"] = "%s: %s" % (type(error).__name__, error)
        termination = "exception"
    finally:
        for actor in [camera, *sensors.values(), *vehicles.values()]:
            if actor is not None:
                try:
                    if actor.type_id.startswith("sensor."):
                        actor.stop()
                    if not actor.destroy():
                        cleanup_errors.append("Actor destruction not acknowledged: %s" % actor.id)
                except Exception as error:
                    cleanup_errors.append(str(error))
        if runner is not None:
            try:
                runner.close()
            except Exception as error:
                cleanup_errors.append("Restore signals: " + str(error))
        if original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception as error:
                cleanup_errors.append("Restore world settings: " + str(error))
        if last_image is not None:
            last_image.save_to_disk(str(output / "rgb/last.png"))
        evidence = evaluate_traffic_evidence(rows, vehicle_lengths) if rows else {
            "checks": {"leader_stop_response": False, "red_stop_then_green_motion_observed": False,
                       "vehicles_beyond_50m_update": False}}
        checks = {"all_routes_completed": termination == "all_routes_completed", "no_collisions": not collisions,
                  "loaded_signal_bindings_match": report.get("runtime_signal_bindings", {}).get("passed", False),
                  **evidence["checks"],
                  "owned_actors_and_settings_cleaned": not cleanup_errors}
        report.update(termination=termination, ticks=ticks, collisions=collisions, cleanup_errors=cleanup_errors,
                      routes_completed={name: state["done"] for name, state in states.items()},
                      response_evidence=evidence, vehicle_lengths_m=vehicle_lengths,
                      checks=checks, passed=all(checks.values()), formal_signal_compliance_verified=False)
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
