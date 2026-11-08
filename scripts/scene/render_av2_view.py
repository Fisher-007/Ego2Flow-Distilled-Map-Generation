"""Prepare and render an AV2 camera pose in CARLA.

Source poses and calibration define the camera. Projection maps Unreal
(forward, right, up) coordinates to camera (right, down, forward) coordinates.
"""

import argparse
import json
import math
from pathlib import Path
import queue
import shutil

import carla
import cv2
import numpy as np
import pandas as pd
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.se3 import SE3

from map_generation.coordinates import LocalFrame, local_to_carla, paired_observation_context, register_local_pose, source_log_registration


CAMERA = "ring_front_center"
CV_FROM_CARLA_CAMERA = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)


def camera_transform(city_se3_camera, frame):
    """Convert both world handedness and camera basis, then CARLA Euler angles."""
    rotation = np.diag([1, -1, 1]) @ city_se3_camera.rotation @ CV_FROM_CARLA_CAMERA
    position = local_to_carla(frame.city_to_local(city_se3_camera.translation))
    pitch = math.asin(float(np.clip(rotation[2, 0], -1, 1)))
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    roll = math.atan2(-rotation[2, 1], rotation[2, 2])
    transform = carla.Transform(carla.Location(*position), carla.Rotation(
        pitch=math.degrees(pitch), yaw=math.degrees(yaw), roll=math.degrees(roll)))
    expected = np.eye(4)
    expected[:3, :3], expected[:3, 3] = rotation, position
    return transform, expected


def render_intrinsics(target_k, width, height):
    """A symmetric overscan camera covers every desired AV2 pixel ray."""
    fx, fy, cx, cy = target_k[0, 0], target_k[1, 1], target_k[0, 2], target_k[1, 2]
    focal = max(fx, fy)
    render_width = 2 * math.ceil(max(cx, width - 1 - cx) * focal / fx + 2)
    render_height = 2 * math.ceil(max(cy, height - 1 - cy) * focal / fy + 2)
    fov = math.degrees(2 * math.atan(render_width / (2 * focal)))
    render_k = np.array([[focal, 0, render_width / 2], [0, focal, render_height / 2], [0, 0, 1]])
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    map_x = ((u - cx) * focal / fx + render_width / 2).astype(np.float32)
    map_y = ((v - cy) * focal / fy + render_height / 2).astype(np.float32)
    return render_k, render_width, render_height, fov, map_x, map_y


def project_carla(local_points, transform, intrinsics):
    xyz = local_to_carla(local_points)
    matrix = np.asarray(transform.get_matrix(), dtype=np.float64)
    # Subtract the local origin before rotation to reduce float32 translation error.
    sensor = (xyz-matrix[:3,3]) @ matrix[:3,:3]
    cv_points = sensor @ CV_FROM_CARLA_CAMERA.T
    image = cv_points @ intrinsics.T
    return image[:, :2] / image[:, 2:3], cv_points


def lane_points(scene):
    points, labels, runs = [], [], []
    for lane in scene["lanes"]:
        for side in ("left_boundary", "right_boundary"):
            start = len(points)
            points.extend(lane[side])
            labels.extend(f"{lane['id']}:{side}:{i}" for i in range(len(lane[side])))
            runs.append((start, len(points), side))
    return np.asarray(points), labels, runs


def overlay(image, uv, valid, runs):
    result = image.copy()
    for start, end, side in runs:
        color = (30, 220, 40) if side == "left_boundary" else (240, 180, 30)
        for i in range(start, end - 1):
            if valid[i] and valid[i + 1]:
                cv2.line(result, tuple(np.rint(uv[i]).astype(int)), tuple(np.rint(uv[i + 1]).astype(int)), color, 2)
    return result


def prepare(scene, log_dir, output, timestamp=None, camera_name=CAMERA, log_registration=None):
    if log_registration is None:log_registration=source_log_registration(scene,log_dir.name)
    requested_timestamp = timestamp is not None
    frame = LocalFrame(scene["coordinates"]["city"], scene["coordinates"]["origin_city_xyz"])
    loader = AV2SensorDataLoader(log_dir.parent, log_dir.parent)
    observation_city = loader.get_city_name(log_dir.name)
    if observation_city != frame.city:
        raise ValueError("Observation and SceneIR must use the same AV2 city frame")
    camera = loader.get_log_pinhole_camera(log_dir.name, camera_name)
    paths = loader.get_ordered_log_cam_fpaths(log_dir.name, camera_name)
    poses = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
    # Explicit source frames provide their own registration references.
    landmark = next((item for area in scene["areas"] for item in area.get("landmarks",[])
                     if "AV2" in item["coverage"]), None)
    target = frame.local_to_city(landmark["position"][:2]) if landmark else None
    if timestamp is None:
        if target is None:
            raise ValueError("Supply an exact camera timestamp for a scene without an AV2 core landmark")
        available = poses[poses.timestamp_ns.isin([int(path.stem) for path in paths])].copy()
        available["core_distance_m"] = np.linalg.norm(available[["tx_m", "ty_m"]].to_numpy() - target, axis=1)
        timestamp = int(available.loc[available.core_distance_m.idxmin(), "timestamp_ns"])
    image_path = log_dir / "sensors" / "cameras" / camera_name / f"{timestamp}.jpg"
    real = cv2.imread(str(image_path))
    if real is None or real.shape[:2] != (camera.height_px, camera.width_px):
        raise ValueError("Selected image is absent or its dimensions differ from official calibration")
    city_se3_ego = loader.get_city_SE3_ego(log_dir.name, timestamp)
    observation_context = paired_observation_context(scene, log_dir.name, observation_city, city_se3_ego.translation)
    source_city_se3_ego = city_se3_ego
    if log_registration is not None:
        if log_registration["log_id"] != log_dir.name or log_registration["map_id"] != scene["map_id"]:
            raise ValueError("Log registration belongs to another observation or map")
        position, rotation = register_local_pose(city_se3_ego.translation-frame.origin,
                                                  city_se3_ego.rotation, log_registration)
        city_se3_ego = SE3(rotation=rotation, translation=position+frame.origin)
        observation_context["registered_context"] = paired_observation_context(
            scene, log_dir.name, observation_city, city_se3_ego.translation)
        observation_context["registered_context"]["registration"] = "Explicit log-to-map rigid transform; see log_registration."
    city_se3_camera = city_se3_ego.compose(camera.ego_SE3_cam)
    transform, expected = camera_transform(city_se3_camera, frame)
    render_k, width, height, fov, map_x, map_y = render_intrinsics(camera.intrinsics.K, camera.width_px, camera.height_px)
    local_points, labels, runs = lane_points(scene)
    ego_points = city_se3_ego.inverse().transform_point_cloud(frame.local_to_city(local_points))
    with np.errstate(divide="ignore", invalid="ignore"):
        av2_uv, av2_cam_points, av2_valid = camera.project_ego_to_img(ego_points)
        carla_uv, _ = project_carla(local_points, transform, camera.intrinsics.K)
        raw_uv, _ = project_carla(local_points, transform, render_k)
    valid = av2_valid & (av2_cam_points[:, 2] > 2)
    if int(valid.sum()) < 6:
        raise ValueError("Selected view has fewer than six visible source lane landmarks")
    ray_map = render_k @ np.linalg.inv(camera.intrinsics.K)
    expected_raw = np.column_stack([av2_uv[valid], np.ones(int(valid.sum()))]) @ ray_map.T
    pixel_error = np.linalg.norm(av2_uv[valid] - carla_uv[valid], axis=1)
    raw_error = np.linalg.norm(raw_uv[valid] - expected_raw[:, :2], axis=1)
    actual = np.asarray(transform.get_matrix())
    geometry_checks = {
        "visible_lane_landmarks": int(valid.sum()),
        "rotation_determinant": float(np.linalg.det(expected[:3, :3])),
        "rotation_roundtrip_max_abs": float(np.max(np.abs(actual[:3, :3] - expected[:3, :3]))),
        "translation_roundtrip_max_m": float(np.max(np.abs(actual[:3, 3] - expected[:3, 3]))),
        "av2_vs_carla_projection_max_px": float(pixel_error.max()),
        "av2_vs_overscan_projection_max_px": float(raw_error.max()),
        "remap_all_pixels_inside_render": bool(map_x.min() >= 1 and map_x.max() < width - 1
                                                and map_y.min() >= 1 and map_y.max() < height - 1),
        "translation_tolerance_m": 1e-3,
        "projection_tolerance_px": .1,
    }
    # Use submillimeter and pixel-space bounds for float32 camera registration.
    geometry_checks["passed"] = bool(geometry_checks["rotation_roundtrip_max_abs"] < 1e-6
        and geometry_checks["translation_roundtrip_max_m"] < geometry_checks["translation_tolerance_m"]
        and max(pixel_error.max(), raw_error.max()) < geometry_checks["projection_tolerance_px"]
        and geometry_checks["remap_all_pixels_inside_render"])
    visible = np.flatnonzero(valid)
    probes = visible[np.linspace(0, len(visible) - 1, min(30, len(visible))).astype(int)]
    metadata = {
        "map_id": scene["map_id"], "log_id": log_dir.name, "camera": camera_name, "timestamp_ns": timestamp,
        "observation_context": observation_context,
        "log_registration": log_registration,
        "source_city_SE3_ego": source_city_se3_ego.transform_matrix.tolist(),
        "selection": "Requested exact timestamp" if requested_timestamp else
                     "One observed camera frame; default selects closest recorded ego pose to fine-core landmark",
        "core_landmark": landmark["id"] if landmark else None,
        "ego_distance_to_core_m": float(np.linalg.norm(city_se3_ego.translation[:2] - target)) if target is not None else None,
        "K_av2": camera.intrinsics.K.tolist(), "target_width_height": [camera.width_px, camera.height_px],
        "K_render": render_k.tolist(), "render_width_height": [width, height], "render_horizontal_fov_deg": fov,
        "target_to_render_pixel_homography": ray_map.tolist(),
        "city_SE3_ego": city_se3_ego.transform_matrix.tolist(), "ego_SE3_camera_cv": camera.ego_SE3_cam.transform_matrix.tolist(),
        "city_SE3_camera_cv": city_se3_camera.transform_matrix.tolist(), "origin_city_xyz": frame.origin.tolist(),
        "carla_world_SE3_camera_fru": actual.tolist(),
        "carla_pose": {"xyz_m": actual[:3, 3].tolist(), "pitch_yaw_roll_deg": [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll]},
        "cv_from_carla_camera": CV_FROM_CARLA_CAMERA.tolist(), "geometry_checks": geometry_checks,
        "signal_phase_alignment_verified": False,
        "projection_landmarks": [{"id": labels[i], "local_xyz": local_points[i].tolist(), "av2_uv": av2_uv[i].tolist(),
                                  "carla_uv": carla_uv[i].tolist()} for i in probes],
        "limitations": [
            "Geometry uses the official AV2 pinhole calibration and exact camera timestamp pose; no additional distortion is inferred.",
            "CARLA renders a symmetric overscan K; bilinear remapping reproduces AV2 pixel rays, not identical photometry or subpixel sampling.",
            "Source lane overlays check coordinate/projection consistency, not visibility, scene reconstruction accuracy, or pixel alignment of rendered assets.",
            "Static camera only: lighting, traffic and materials are not reconstructed from the real image. Only map appearance is inspected here.",
            "Real and simulated signal phases are not aligned by this script; a visible lamp needs separate observation-based matching.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_path, output / "real.jpg")
    cv2.imwrite(str(output / "real_projection.png"), overlay(real, av2_uv, valid, runs))
    (output / "camera.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if not geometry_checks["passed"]:
        raise RuntimeError("Camera geometry check failed; inspect camera.json")
    return metadata, transform, (map_x, map_y), (av2_uv, valid, runs)


def render(metadata, transform, remap, projections, output, port, expected_map, warmup_frames,
           capture_gbuffers=False, capture_semantics=False, capture_depth=False,
           runtime=None):
    if runtime is None:
        client = carla.Client("127.0.0.1", port)
        client.set_timeout(60.0)
        world = client.get_world()
        native = world.get_map()
    else:
        # Reuse the client, world, and map objects across a render batch.
        client, world, native = runtime
    if native.name.rsplit("/", 1)[-1] != expected_map:
        raise RuntimeError(f"Expected {expected_map}, got {native.name}")
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, 0.05, False
    sensor = None
    semantic_sensor = None
    depth_sensor = None
    buffers = {}
    try:
        world.apply_settings(settings)
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        attributes = {"image_size_x": str(metadata["render_width_height"][0]),
                      "image_size_y": str(metadata["render_width_height"][1]),
                      "fov": str(metadata["render_horizontal_fov_deg"]), "sensor_tick": "0",
                      "lens_k": "0", "lens_kcube": "0", "enable_postprocess_effects": "True",
                      "motion_blur_intensity": "0", "lens_flare_intensity": "0",
                      "bloom_intensity": "0", "chromatic_aberration_intensity": "0"}
        # Keep lighting postprocessing enabled and disable lens effects individually.
        for key, value in attributes.items():
            blueprint.set_attribute(key, value)
        sensor = world.spawn_actor(blueprint, transform)
        frames = queue.Queue()
        sensor.listen(frames.put)
        semantic_frames = queue.Queue()
        if capture_semantics:
            # Reuse CARLA's manual_control.py semantic camera and palette. Its
            # raw red channel stores the official class ID (BGRA byte index 2).
            semantic_blueprint = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
            for key in ("image_size_x", "image_size_y", "fov", "sensor_tick", "lens_k", "lens_kcube"):
                semantic_blueprint.set_attribute(key, attributes[key])
            semantic_sensor = world.spawn_actor(semantic_blueprint, transform)
            semantic_sensor.listen(semantic_frames.put)
        depth_frames = queue.Queue()
        if capture_depth:
            depth_blueprint = world.get_blueprint_library().find("sensor.camera.depth")
            for key in ("image_size_x", "image_size_y", "fov", "sensor_tick", "lens_k", "lens_kcube"):
                depth_blueprint.set_attribute(key, attributes[key])
            depth_sensor = world.spawn_actor(depth_blueprint, transform)
            depth_sensor.listen(depth_frames.put)
        if capture_gbuffers:
            # Follow CARLA's tutorial_gbuffer.py. These expose actual material
            # inputs and normals when diagnosing lighting, without editing them.
            sensor.enable_gbuffers(True)
            for name in ("GBufferA", "GBufferB", "GBufferC"):
                stream = queue.Queue()
                buffers[name] = stream
                sensor.listen_to_gbuffer(getattr(carla.GBufferTextureID, name), stream.put)
        for _ in range(warmup_frames + 1):
            tick = world.tick()
            image = frames.get(timeout=30)
            while image.frame < tick:
                image = frames.get(timeout=30)
            if image.frame != tick:
                raise RuntimeError("Camera output does not match the requested synchronous frame")
            buffer_images = {}
            for name, stream in buffers.items():
                buffer_image = stream.get(timeout=30)
                while buffer_image.frame < tick:
                    buffer_image = stream.get(timeout=30)
                if buffer_image.frame != tick:
                    raise RuntimeError(name + " does not match the requested synchronous frame")
                buffer_images[name] = buffer_image
            if semantic_sensor is not None:
                semantic_image = semantic_frames.get(timeout=30)
                while semantic_image.frame < tick:
                    semantic_image = semantic_frames.get(timeout=30)
                if semantic_image.frame != tick:
                    raise RuntimeError("Semantic camera does not match the requested synchronous frame")
            if depth_sensor is not None:
                depth_image = depth_frames.get(timeout=30)
                while depth_image.frame < tick:
                    depth_image = depth_frames.get(timeout=30)
                if depth_image.frame != tick:
                    raise RuntimeError("Depth camera does not match the requested synchronous frame")
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
        simulated = cv2.remap(raw, *remap, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(output / "sim_raw.png"), raw)
        cv2.imwrite(str(output / "sim.png"), simulated)
        cv2.imwrite(str(output / "sim_projection.png"), overlay(simulated, *projections))
        real = cv2.imread(str(output / "real.jpg"))
        cv2.imwrite(str(output / "paired.png"), np.concatenate([real, simulated], axis=1))
        for name, buffer_image in buffer_images.items():
            pixels = np.frombuffer(buffer_image.raw_data, dtype=np.uint8).reshape(buffer_image.height, buffer_image.width, 4)[:, :, :3]
            cv2.imwrite(str(output / (name + ".png")), cv2.remap(pixels, *remap, interpolation=cv2.INTER_LINEAR))
        semantic_report = None
        depth_report = None
        if depth_sensor is not None:
            pixels = np.frombuffer(depth_image.raw_data, dtype=np.uint8).reshape(depth_image.height, depth_image.width, 4).astype(np.float64)
            # Decode CARLA's BGRA camera buffer.
            depth = ((pixels[:, :, 2] + 256 * pixels[:, :, 1] + 65536 * pixels[:, :, 0]) / 16777215 * 1000).astype(np.float32)
            depth = cv2.remap(depth, *remap, interpolation=cv2.INTER_NEAREST)
            np.savez_compressed(output / "depth.npz", depth_m=depth)
            depth_report = {"frame": depth_image.frame, "file": "depth.npz",
                            "remap": "Nearest native metric depth; no edge interpolation"}
        if semantic_sensor is not None:
            pixels = np.frombuffer(semantic_image.raw_data, dtype=np.uint8).reshape(semantic_image.height, semantic_image.width, 4)
            labels = cv2.remap(pixels[:, :, 2], *remap, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(output / "semantic_ids.png"), labels)
            semantic_image.convert(carla.ColorConverter.CityScapesPalette)
            colors = np.frombuffer(semantic_image.raw_data, dtype=np.uint8).reshape(semantic_image.height, semantic_image.width, 4)[:, :, :3]
            cv2.imwrite(str(output / "semantic_palette.png"), cv2.remap(colors, *remap, interpolation=cv2.INTER_NEAREST))
            ids, counts = np.unique(labels, return_counts=True)
            semantic_report = {"frame": semantic_image.frame, "class_pixel_counts": dict(zip(map(str, ids.tolist()), counts.tolist())),
                               "ids_file": "semantic_ids.png", "palette_file": "semantic_palette.png",
                               "remap": "Nearest-neighbor class IDs and official CARLA CityScapesPalette; no interpolated labels",
                               "camera_attributes": dict(semantic_sensor.attributes),
                               "scope": "Rendered asset labels at the RGB pose; not a ground-truth segmentation of the real AV2 image"}
        actual_transform = sensor.get_transform()
        actual = np.asarray(actual_transform.get_matrix())
        requested = np.asarray(metadata["carla_world_SE3_camera_fru"])
        pose_error = float(np.max(np.abs(actual - requested)))
        # Check actor translation and rotation separately against camera-registration tolerances.
        landmark_points = np.asarray([p["local_xyz"] for p in metadata["projection_landmarks"]])
        native_uv, _ = project_carla(landmark_points, actual_transform, np.asarray(metadata["K_av2"]))
        source_uv = np.asarray([p["av2_uv"] for p in metadata["projection_landmarks"]])
        # CARLA stores angles/matrices as floats. Measure the relative rotation
        # itself rather than comparing a coordinate-dependent matrix element.
        # The existing source projection budget supplies its angular scale.
        left, _, right = np.linalg.svd(actual[:3, :3] @ requested[:3, :3].T)
        relative_rotation = left @ right
        skew = np.array([relative_rotation[2, 1] - relative_rotation[1, 2],
                         relative_rotation[0, 2] - relative_rotation[2, 0],
                         relative_rotation[1, 0] - relative_rotation[0, 1]]) / 2
        rotation_angle = float(np.arctan2(np.linalg.norm(skew),
                                         (np.trace(relative_rotation) - 1) / 2))
        focal = max(metadata["K_av2"][0][0], metadata["K_av2"][1][1])
        angular_tolerance = float(np.arctan(
            metadata["geometry_checks"]["projection_tolerance_px"] / focal))
        pose_checks = {
            "translation_max_m": float(np.max(np.abs(actual[:3, 3] - requested[:3, 3]))),
            "rotation_max_abs": float(np.max(np.abs(actual[:3, :3] - requested[:3, :3]))),
            "source_landmark_projection_max_px": float(np.linalg.norm(native_uv - source_uv, axis=1).max()),
            "translation_tolerance_m": metadata["geometry_checks"]["translation_tolerance_m"],
            "rotation_tolerance": 1e-6,
            "rotation_tolerance_role": "Matrix-element reference; rotation acceptance uses angular and pixel errors",
            "rotation_angle_rad": rotation_angle,
            "rotation_angle_tolerance_rad": angular_tolerance,
            "rotation_precision_basis": "Existing projection tolerance / maximum AV2 focal length; actual source-landmark projection also checked",
            "projection_tolerance_px": metadata["geometry_checks"]["projection_tolerance_px"],
            "landmark_count": len(landmark_points),
        }
        pose_checks["passed"] = (pose_checks["translation_max_m"] < pose_checks["translation_tolerance_m"]
            and pose_checks["rotation_angle_rad"] < pose_checks["rotation_angle_tolerance_rad"]
            and pose_checks["source_landmark_projection_max_px"] < pose_checks["projection_tolerance_px"])
        report = {"rendered": True, "server_version": client.get_server_version(), "port": port,
                  "map": native.name, "frame": image.frame, "simulation_timestamp_s": image.timestamp,
                  "camera_attributes": dict(sensor.attributes), "requested_camera_attributes": attributes,
                  "world_weather": str(world.get_weather()),
                  # Read light state on the captured RGB tick and store world-space display bounds.
                  "traffic_lights": [{"opendrive_id": str(a.get_opendrive_id()),
                      "state": str(a.get_state()), "frozen": a.is_frozen(),
                      "lamp_boxes_world_xyz_m": [[[v.x, v.y, v.z] for v in b.get_world_vertices(carla.Transform())]
                          for b in a.get_light_boxes()]}
                      for a in world.get_actors().filter("traffic.traffic_light*")],
                  "camera_pose_matrix_max_abs_error": pose_error,
                  "actual_carla_world_SE3_camera_fru": actual.tolist(),
                  "native_camera_checks": pose_checks, "rgb_std": float(simulated.std()),
                  "source_timestamp_ns": metadata["timestamp_ns"],
                  "semantics": semantic_report,
                  "depth": depth_report,
                  "gbuffers": {name: {"frame": value.frame, "file": name + ".png"} for name, value in buffer_images.items()},
                  "gbuffer_interpretation": "A: encoded world normal; B RGB: metallic/specular/roughness; C: base color. Images include CARLA's format/gamma conversion; PNG values are not raw linear material parameters.",
                  "passed": pose_checks["passed"] and float(simulated.std()) > 1}
        (output / "render_result.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise RuntimeError("Static render or spawned camera pose check failed")
        return report
    finally:
        if depth_sensor is not None:
            depth_sensor.stop()
            depth_sensor.destroy()
        if semantic_sensor is not None:
            semantic_sensor.stop()
            semantic_sensor.destroy()
        if sensor is not None:
            for name in buffers:
                sensor.stop_gbuffer(getattr(carla.GBufferTextureID, name))
            if buffers:
                sensor.enable_gbuffers(False)
            sensor.stop()
            sensor.destroy()
        world.apply_settings(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--timestamp", type=int, help="Exact ring_front_center image timestamp in nanoseconds")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare geometry and projection checks on CPU")
    parser.add_argument("--expected-map", required=True)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--gbuffers", action="store_true", help="Record synchronized normals and material buffers for appearance diagnosis")
    parser.add_argument("--semantics", action="store_true", help="Record synchronized official semantic class IDs and palette")
    args = parser.parse_args()
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be nonnegative")
    metadata, transform, remap, projections = prepare(json.loads(args.scene.read_text()), args.log_dir, args.output, args.timestamp)
    result = {"timestamp_ns": metadata["timestamp_ns"], "geometry_checks": metadata["geometry_checks"]}
    if not args.prepare_only:
        result["render"] = render(metadata, transform, remap, projections, args.output, args.port, args.expected_map,
                                  args.warmup_frames, args.gbuffers, args.semantics)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
