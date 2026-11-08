"""Render AV2 actor snapshots with fixed source poses and calibrated cameras.

Native assets use kinematic placement; the report records shape and category
differences from the source cuboids.
"""

import argparse
import json
import math
from pathlib import Path

import carla
import numpy as np

from scripts.scene.render_av2_view import prepare, render


def transform(matrix):
    r = matrix[:3, :3]
    return carla.Transform(carla.Location(*matrix[:3, 3]), carla.Rotation(
        pitch=math.degrees(math.asin(float(np.clip(r[2, 0], -1, 1)))),
        yaw=math.degrees(math.atan2(r[1, 0], r[0, 0])),
        roll=math.degrees(math.atan2(-r[2, 1], r[2, 2]))))


def actor_pose(box_pose, bounding_box):
    """T_world_actor = T_world_source_box @ inverse(T_actor_blueprint_box)."""
    source = np.eye(4)
    source[:3, :3] = box_pose["rotation_matrix"]
    source[:3, 3] = box_pose["center_xyz_m"]
    relative = np.asarray(carla.Transform(bounding_box.location, bounding_box.rotation).get_matrix())
    return transform(source @ np.linalg.inv(relative))


PREFERRED = {
    "BUS": ["vehicle.mitsubishi.fusorosa"],
    # Class-compatible existing bus; exact school-bus geometry/color remains unmatched.
    "SCHOOL_BUS": ["vehicle.mitsubishi.fusorosa"],
    "REGULAR_VEHICLE": ["vehicle.lincoln.mkz_2020", "vehicle.audi.tt"],
    "BOX_TRUCK": ["vehicle.mercedes.sprinter"],
    "TRUCK": ["vehicle.carlamotors.european_hgv", "vehicle.mercedes.sprinter"],
    "LARGE_VEHICLE": ["vehicle.mercedes.sprinter"],
    "PEDESTRIAN": ["walker.pedestrian.0001"],
    "BICYCLIST": ["vehicle.diamondback.century"],
    "MOTORCYCLIST": ["vehicle.yamaha.yzf"],
}


def spawn_snapshot(world, snapshot, front_only=True, replay_assets=None, native_blueprints=None):
    """Optionally substitute named tracks with dimension-checked rigid assets.

    These assets retain source box poses and are for kinematic replay only.
    A positive spawn mass makes CARLA's generic mesh movable; physics is then
    disabled before placement, just as for the original vehicle blueprints.
    """
    library = world.get_blueprint_library()
    available = {bp.id: bp for bp in library}
    spawned, records, skipped = [], [], []
    selected = [row for row in snapshot["actors"] if not front_only or row.get("front_camera_box")]
    # Query ground before placing actors and retain missing or mismatched support results.
    ground = {}
    for row in selected:
        center = row["carla_box_pose"]["center_xyz_m"]
        hit = world.project_point(carla.Location(*center), carla.Vector3D(z=-1), 20.)
        ground[row["track_uuid"]] = None if hit is None else {"xyz_m": [hit.location.x, hit.location.y, hit.location.z], "label": str(hit.label)}
    try:
        for row in selected:
            override = (replay_assets or {}).get(row["track_uuid"])
            if override:
                blueprint = library.find("static.prop.mesh")
                blueprint.set_attribute("mesh_path", override["mesh_path"])
                blueprint.set_attribute("mass", "1")
                if override.get("sensor_trace", False):
                    blueprint.set_attribute("sensor_trace", "true")
                blueprint.set_attribute("scale", str(override.get("uniform_scale", 1.0)))
            elif row['track_uuid'] in (native_blueprints or {}):
                blueprint=library.find(native_blueprints[row['track_uuid']])
            elif row.get('native_blueprint'):
                blueprint=library.find(row['native_blueprint'])
            else:
                blueprint = next((available[name] for name in PREFERRED.get(row["category"], []) if name in available), None)
            if blueprint is None:
                skipped.append({"track_uuid": row["track_uuid"], "category": row["category"], "reason": "No configured class-compatible blueprint"})
                continue
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "av2_replay_" + row["track_uuid"])
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "true")
            if blueprint.has_attribute('color') and row.get('appearance',{}).get('color_rgb'):
                blueprint.set_attribute('color',','.join(map(str,row['appearance']['color_rgb'])))
            pose = row["carla_box_pose"]
            # Spawn outside collisions, disable physics, then apply the source pose.
            initial = carla.Transform(carla.Location(pose["center_xyz_m"][0], pose["center_xyz_m"][1], pose["center_xyz_m"][2] + 1000.))
            actor = world.try_spawn_actor(blueprint, initial)
            if actor is None:
                skipped.append({"track_uuid": row["track_uuid"], "category": row["category"], "reason": "CARLA refused spawn at the elevated source position", "blueprint": blueprint.id})
                continue
            spawned.append(actor)
            actor.set_simulate_physics(False)
            actor.set_enable_gravity(False)
            if override:
                actual_size = np.array([actor.bounding_box.extent.x, actor.bounding_box.extent.y, actor.bounding_box.extent.z]) * 2
                if not np.allclose(actual_size, override["expected_dimensions_lwh_m"], atol=.01, rtol=0):
                    raise ValueError(f"Replay mesh bounds differ from its build report: {actual_size}")
            target = actor_pose(pose, actor.bounding_box)
            anchored_box = None
            if override and 'source_anchor_local_xyz_m' in override:
                # Transform the source cuboid-center anchor using the mesh's uniform scale.
                anchor = np.asarray(override['source_anchor_local_xyz_m'], dtype=float) * float(override.get('uniform_scale', 1.0))
                local_center = np.array([actor.bounding_box.location.x, actor.bounding_box.location.y, actor.bounding_box.location.z])
                shift = np.asarray(pose['rotation_matrix']) @ (local_center - anchor)
                target.location.x += float(shift[0])
                target.location.y += float(shift[1])
                target.location.z += float(shift[2])
                anchored_box = dict(pose, center_xyz_m=(np.asarray(pose['center_xyz_m']) + shift).tolist())
            actor.set_transform(target)
            extent = actor.bounding_box.extent
            dimensions = np.array([extent.x, extent.y, extent.z]) * 2
            source_dimensions = np.asarray(row["dimensions_lwh_m"])
            hit = ground[row["track_uuid"]]
            halfspan = np.abs(np.asarray(pose["rotation_matrix"])[2])
            contact = {"surface_hit_below_source_center": hit}
            if hit is not None:
                contact.update(source_box_bottom_minus_hit_z_m=float(pose["center_xyz_m"][2] - halfspan @ source_dimensions / 2 - hit["xyz_m"][2]),
                               blueprint_box_bottom_minus_hit_z_m=float(pose["center_xyz_m"][2] - halfspan @ dimensions / 2 - hit["xyz_m"][2]))
            records.append({"track_uuid": row["track_uuid"], "category": row["category"], "actor_id": actor.id,
                "blueprint": blueprint.id, "source_dimensions_lwh_m": source_dimensions.tolist(),
                "blueprint_dimensions_lwh_m": dimensions.tolist(),
                "dimension_difference_lwh_m": (dimensions - source_dimensions).tolist(),
                "dimension_relative_abs_error_lwh": (np.abs(dimensions - source_dimensions) / source_dimensions).tolist(),
                "source_box_pose": pose, "actor_transform_requested": np.asarray(target.get_matrix()).tolist(),
                "source_observation_age_s": row["observation_age_s"], "alignment_method": row["alignment_method"],
                "source_center_surface_probe": contact})
            if row.get('appearance'):records[-1]['appearance']=row['appearance']
            if override:
                records[-1]["replay_asset"] = override
                records[-1]["actual_mesh_path"] = actor.attributes.get("mesh_path")
                if anchored_box is not None:
                    records[-1]['render_box_pose'] = anchored_box
                    records[-1]['source_anchor_local_xyz_m'] = override['source_anchor_local_xyz_m']
        return spawned, records, skipped
    except Exception:
        for actor in reversed(spawned):
            actor.destroy()
        raise


def placement_records(actors, records):
    by_id = {actor.id: actor for actor in actors}
    for row in records:
        actor = by_id[row["actor_id"]]
        actual = np.asarray(actor.get_transform().get_matrix())
        relative = np.asarray(carla.Transform(actor.bounding_box.location, actor.bounding_box.rotation).get_matrix())
        box = actual @ relative
        row["actual_box_center_xyz_m"] = box[:3, 3].tolist()
        row["box_center_error_m"] = float(np.linalg.norm(box[:3, 3] - row["source_box_pose"]["center_xyz_m"]))
        row["box_rotation_matrix_max_abs_error"] = float(np.max(np.abs(box[:3, :3] - row["source_box_pose"]["rotation_matrix"])))
        target=row.get('render_box_pose',row['source_box_pose'])
        row['target_box_center_error_m']=float(np.linalg.norm(box[:3,3]-target['center_xyz_m']))
        row['target_box_rotation_error']=float(np.max(abs(box[:3,:3]-target['rotation_matrix'])))
        if 'source_anchor_local_xyz_m' in row:
            anchor = np.asarray(row['source_anchor_local_xyz_m'], dtype=float) * float(row['replay_asset'].get('uniform_scale', 1.0))
            actual_anchor = (actual @ np.r_[anchor, 1.])[:3]
            row['actual_source_anchor_xyz_m'] = actual_anchor.tolist()
            row['actual_source_anchor_error_m'] = float(np.linalg.norm(actual_anchor - row['source_box_pose']['center_xyz_m']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--expected-map", default="PitMarkedSurface")
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--timestamp", type=int, help="If supplied, must match the prepared snapshot")
    parser.add_argument("--gbuffers", action="store_true")
    parser.add_argument("--semantics", action="store_true", help="Accepted for wrapper compatibility; this actor check always records semantics")
    parser.add_argument("--all-objects", action="store_true", help="Also place objects outside the front camera envelope.")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    from map_generation.coordinates import source_log_registration
    from scripts.scene.prepare_dynamic_replay import register_replay
    scene = json.loads(args.scene.read_text())
    snapshot = register_replay(snapshot, source_log_registration(scene, args.log_dir.name))
    if "Causal" not in snapshot["scope"]:
        raise ValueError("Use anchor_snapshot.json from the causal preparer, not an offline future file")
    if args.timestamp is not None and args.timestamp != snapshot["timestamp_ns"]:
        raise ValueError("Requested camera time differs from the prepared actor snapshot")
    metadata, camera_transform, remap, projections = prepare(scene, args.log_dir, args.output, snapshot["timestamp_ns"])
    metadata["actor_snapshot"] = {"file": str(args.snapshot), "timestamp_ns": snapshot["timestamp_ns"],
                                  "alignment": "Past-only constant velocity/yaw estimates; no future annotations"}
    metadata["limitations"] = [text for text in metadata["limitations"] if not text.startswith("Static camera only:")]
    metadata["limitations"].append("Actors use a causal source snapshot with approximate blueprint geometry; lighting and materials remain unmatched. No policy is evaluated.")
    (args.output / "camera.json").write_text(json.dumps(metadata, indent=2) + "\n")
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(60.)
    world = client.get_world()
    if world.get_map().name.rsplit("/", 1)[-1] != args.expected_map:
        raise ValueError("Connected CARLA map differs from the requested map")
    actors, records, skipped = spawn_snapshot(world, snapshot, not args.all_objects)
    try:
        result = render(metadata, camera_transform, remap, projections, args.output, args.port,
                        args.expected_map, args.warmup_frames, capture_gbuffers=args.gbuffers, capture_semantics=True)
        placement_records(actors, records)
        report = {"timestamp_ns": snapshot["timestamp_ns"], "snapshot_file": str(args.snapshot),
            "scope": "One causal camera-time actor snapshot; fixed-pose reconstruction, not closed-loop motion or planner validation.",
            "source_objects_total": len(snapshot["actors"]),
            "selection": "All source objects" if args.all_objects else "Objects whose projected cuboid envelope intersects the front camera",
            "selected_objects_total": sum(args.all_objects or row.get("front_camera_box") is not None for row in snapshot["actors"]),
            "vehicle_blueprints": [{"id": bp.id, "base_type": bp.get_attribute("base_type").as_str() if bp.has_attribute("base_type") else None}
                                   for bp in world.get_blueprint_library().filter("vehicle.*")],
            "spawned": len(actors), "actors": records, "skipped": skipped,
            "placement_passed": bool(records) and all(r["box_center_error_m"] < .001 and r["box_rotation_matrix_max_abs_error"] < 1e-5 for r in records),
            "render": result,
            "limitations": ["Blueprint geometry is not rescaled; source/blueprint dimensions and unmatched categories remain explicit.",
                            "Source box centers are not snapped to the ground; center-ray and box-bottom probes are not wheel-contact validation.",
                            "Front-frustum selection is only for this view; off-camera shadows, reflections, full-world interactions and 360-degree reconstruction are not validated.",
                            "No vehicle brake/indicator or traffic-light phase is inferred from cuboid motion.",
                            "Actors remain at the estimated source-time pose during rendering; their CARLA velocity is not the observed velocity."]}
        (args.output / "dynamic_replay_result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"spawned": report["spawned"], "skipped": len(skipped), "placement_passed": report["placement_passed"], "frame": result["frame"]}))
    finally:
        for actor in reversed(actors):
            actor.destroy()


if __name__ == "__main__":
    main()
