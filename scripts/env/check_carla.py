"""Check synchronous ticks, vehicle motion, and RGB capture in a dedicated CARLA instance."""

import argparse
import json
import math
import queue
from pathlib import Path

import carla
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    vehicle = camera = None
    try:
        world.apply_settings(settings)
        carla_map = world.get_map()
        blueprints = world.get_blueprint_library()
        vehicle_bp = blueprints.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "mapbuild_environment_check")
        for endpoint in carla_map.get_spawn_points():
            # Generated maps place spawn points at lane endpoints. Move inward
            # so the entire vehicle rests on the road, including on short maps.
            inward = carla_map.get_waypoint(endpoint.location).next(5.0)
            if not inward:
                continue
            spawn = inward[0].transform
            spawn.location.z += 0.5
            vehicle = world.try_spawn_actor(vehicle_bp, spawn)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("No free vehicle spawn point in this map")

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "320")
        camera_bp.set_attribute("image_size_y", "180")
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=2.0)),
            attach_to=vehicle,
        )
        frames = queue.Queue()
        camera.listen(frames.put)
        start = None
        image = None
        for step in range(80):
            if step == 20:
                start = vehicle.get_location()
                vehicle.apply_control(carla.VehicleControl(throttle=0.4))
            tick = world.tick()
            image = frames.get(timeout=30.0)
            while image.frame < tick:
                image = frames.get(timeout=30.0)
            if image.frame != tick:
                raise RuntimeError(f"Camera frame {image.frame} differs from tick {tick}")

        end = vehicle.get_location()
        distance = end.distance(start)
        horizontal_distance = math.hypot(end.x - start.x, end.y - start.y)
        pixels = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(180, 320, 4)
        rgb_std = float(pixels[:, :, :3].std())
        image.save_to_disk(str(args.output / "rgb.png"))
        report = {
            "client_version": client.get_client_version(),
            "server_version": client.get_server_version(),
            "map": carla_map.name,
            "ticks": 80,
            "fixed_delta_seconds": 0.05,
            "camera_frame": image.frame,
            "vehicle_displacement_m": distance,
            "horizontal_displacement_m": horizontal_distance,
            "start_xyz_m": [start.x, start.y, start.z],
            "end_xyz_m": [end.x, end.y, end.z],
            "rgb_standard_deviation": rgb_std,
            "passed": horizontal_distance > 0.5 and rgb_std > 1.0,
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise RuntimeError("Vehicle motion or RGB output check failed")
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
