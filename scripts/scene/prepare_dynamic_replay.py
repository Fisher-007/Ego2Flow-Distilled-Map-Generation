"""Prepare camera-time AV2 actor snapshots from causal history.

Two preceding annotation frames supplement the supplied history when available.
Optional future annotations are written separately for alignment analysis.
"""

import argparse
from copy import deepcopy
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np


FLIP = np.diag([1., -1., 1.])


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def yaw(rotation):
    return math.atan2(rotation[1][0], rotation[0][0])


def wrapped(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def carla_pose(position, rotation):
    """Box center and body orientation: flip world AND object handedness."""
    r = FLIP @ np.asarray(rotation) @ FLIP
    return {"center_xyz_m": (FLIP @ np.asarray(position)).tolist(),
            "rotation_matrix": r.tolist(),
            "pitch_yaw_roll_deg": [math.degrees(math.asin(float(np.clip(r[2, 0], -1, 1)))),
                                   math.degrees(math.atan2(r[1, 0], r[0, 0])),
                                   math.degrees(math.atan2(-r[2, 1], r[2, 2]))]}


def register_state(row, registration):
    """Transform a completed source-frame estimate, including its velocity."""
    from map_generation.coordinates import register_local_pose
    row = deepcopy(row)
    position, rotation = register_local_pose(row['position_local_xyz_m'], row['city_rotation'], registration)
    basis = np.asarray(registration['rotation_matrix'])
    row.update(position_local_xyz_m=position.tolist(), city_rotation=rotation.tolist())
    if 'observed_position_local_xyz_m' in row:
        row['observed_position_local_xyz_m'] = (basis @ row['observed_position_local_xyz_m'] + registration['translation_local_m']).tolist()
    if 'velocity_local_xyz_mps' in row:
        row['velocity_local_xyz_mps'] = (basis @ row['velocity_local_xyz_mps']).tolist()
    if 'yaw_rate_rad_s' in row:
        row['yaw_rate_axis_local_xyz'] = basis[:, 2].tolist()
    if 'carla_box_pose' in row:
        row['carla_box_pose'] = carla_pose(position, rotation)
    return row


def register_replay(data, registration):
    """Register a snapshot/history once, keeping timestamps and sensor extrinsics."""
    if registration is None or data.get('log_registration') == registration:
        return data
    if data.get('log_registration') is not None:
        raise ValueError('Replay already uses a different source registration')
    result = deepcopy(data)
    frames = result.get('frames', [result])
    for frame in frames:
        if 'ego' in frame:
            frame['ego'] = register_state(frame['ego'], registration)
        frame['actors'] = [register_state(row, registration) for row in frame['actors']]
    result['log_registration'] = registration
    return result


def vertices(position, rotation, dimensions):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return signs * np.asarray(dimensions)[None] / 2 @ np.asarray(rotation).T + position


def image_box(position, rotation, dimensions, camera, origin):
    points = vertices(position, rotation, dimensions) + origin
    t = np.asarray(camera["city_SE3_camera_cv"])
    cv = (points - t[:3, 3]) @ t[:3, :3]
    visible = cv[:, 2] > .1
    if not visible.any():
        return None
    pixels = cv[visible] @ np.asarray(camera["K_av2"]).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    width, height = camera["target_width_height"]
    lo = np.maximum(pixels.min(axis=0), [0, 0])
    hi = np.minimum(pixels.max(axis=0), [width - 1, height - 1])
    if (hi <= lo).any():
        return None
    return {"xyxy_px": [*lo.tolist(), *hi.tolist()],
            "clipped_box_area_px": float(np.prod(hi - lo)),
            "scope": "Projected 3D box envelope; image occlusion is not tested."}


def snapshot(rows, timestamp, max_velocity_gap_s=.25):
    """Use actors in the latest observed frame, with past-only CV + yaw rate."""
    preceding = [row for row in rows if row["timestamp_ns"] <= timestamp]
    if not preceding:
        return {"timestamp_ns": timestamp, "actors": [], "annotation_timestamp_ns": None}
    latest = max(row["timestamp_ns"] for row in preceding)
    tracks = defaultdict(list)
    for row in preceding:
        tracks[row["track_uuid"]].append(row)
    actors = []
    for track, records in sorted(tracks.items()):
        records.sort(key=lambda r: r["timestamp_ns"])
        current = records[-1]
        if current["timestamp_ns"] != latest:
            continue  # The latest observation defines active tracks.
        velocity = np.zeros(3)
        yaw_rate = 0.
        velocity_times = [latest]
        if len(records) > 1:
            previous = records[-2]
            dt = (latest - previous["timestamp_ns"]) / 1e9
            if 0 < dt <= max_velocity_gap_s:
                velocity = (np.asarray(current["position_local_xyz_m"]) - previous["position_local_xyz_m"]) / dt
                yaw_rate = wrapped(yaw(current["city_rotation"]) - yaw(previous["city_rotation"])) / dt
                velocity_times.insert(0, previous["timestamp_ns"])
        age = (timestamp - latest) / 1e9
        position = np.asarray(current["position_local_xyz_m"]) + age * velocity
        angle = age * yaw_rate
        r = np.array([[math.cos(angle), -math.sin(angle), 0],
                      [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]) @ np.asarray(current["city_rotation"])
        actors.append({"track_uuid": track, "category": current["category"],
            "source_timestamp_ns": latest, "target_timestamp_ns": timestamp,
            "observation_age_s": age, "motion_source_timestamps_ns": velocity_times,
            "alignment_method": "past_two_observation_constant_velocity_and_yaw_rate" if len(velocity_times) == 2 else "last_observation_hold",
            "dimensions_lwh_m": current["dimensions_lwh_m"],
            "source_num_interior_pts": current["num_interior_pts"],
            "observed_position_local_xyz_m": current["position_local_xyz_m"],
            "position_local_xyz_m": position.tolist(), "city_rotation": r.tolist(),
            "velocity_local_xyz_mps": velocity.tolist(), "yaw_rate_rad_s": yaw_rate,
            "causal_alignment_displacement_m": float(np.linalg.norm(age * velocity)),
            "carla_box_pose": carla_pose(position, r)})
    return {"timestamp_ns": timestamp, "annotation_timestamp_ns": latest,
            "annotation_age_s": (timestamp - latest) / 1e9, "actors": actors}


def preceding_context(log_dir, history_start, origin):
    """Initialize the first history image from observations strictly before it."""
    import pandas as pd
    from av2.structures.cuboid import CuboidList
    from av2.utils.io import read_city_SE3_ego
    table = pd.read_feather(log_dir / "annotations.feather")
    times = sorted(int(t) for t in table.timestamp_ns.unique() if t < history_start)[-2:]
    selected = table[table.timestamp_ns.isin(times)].sort_values(["timestamp_ns", "track_uuid"])
    poses = read_city_SE3_ego(log_dir)
    rows = []
    for source, cuboid in zip(selected.itertuples(index=False), CuboidList.from_dataframe(selected).cuboids):
        timestamp = int(cuboid.timestamp_ns)
        box = cuboid.transform(poses[timestamp])
        rows.append({"timestamp_ns": timestamp, "track_uuid": str(source.track_uuid),
            "category": str(source.category), "num_interior_pts": int(source.num_interior_pts),
            "dimensions_lwh_m": [float(source.length_m), float(source.width_m), float(source.height_m)],
            "position_local_xyz_m": (box.dst_SE3_object.translation - origin).tolist(),
            "city_rotation": box.dst_SE3_object.rotation.tolist()})
    return rows, times


def build(reference, output, log_dir=None):
    output.mkdir(parents=True, exist_ok=True)
    history = json.loads((reference / "history.json").read_text())
    manifest = json.loads((reference / "manifest.json").read_text())
    camera = json.loads((reference / "view/camera.json").read_text())
    timestamp = history["timestamp_ns"]
    if any(row["timestamp_ns"] > timestamp for row in history["actors"] + history["ego"]):
        raise ValueError("Input history contains future observations")
    origin = np.asarray(manifest["coordinates"]["origin_city_xyz"])
    prefix, prefix_times = [], []
    if log_dir is not None:
        if log_dir.name != manifest["log_id"]:
            raise ValueError("Initialization log differs from the paired reference")
        prefix, prefix_times = preceding_context(log_dir, history["ego"][0]["timestamp_ns"], origin)
    input_actors = prefix + history["actors"]
    initialization = {"scope": "Past-only actor context preceding the requested history; not extra model image frames.",
        "source_log": str(log_dir) if log_dir is not None else None,
        "source_timestamps_ns": prefix_times, "actors": prefix}
    save(output / "history_initialization.json", initialization)
    anchor = snapshot(input_actors, timestamp)
    ego = np.asarray(camera["city_SE3_ego"])
    for actor in anchor["actors"]:
        local = np.asarray(actor["position_local_xyz_m"])
        actor["position_anchor_ego_xyz_m"] = ((local + origin - ego[:3, 3]) @ ego[:3, :3]).tolist()
        actor["front_camera_box"] = image_box(local, actor["city_rotation"], actor["dimensions_lwh_m"], camera, origin)
    save(output / "anchor_snapshot.json", {"scope": "Causal camera-time estimates from preceding annotations; not exact observed camera-time truth.",
        "map_id": manifest["map_id"], "log_id": manifest["log_id"], **anchor})

    if log_dir is not None:
        available_images = {int(p.stem) for p in (log_dir / "sensors/cameras/ring_front_center").glob("*.jpg")}
        camera_times = sorted({row["timestamp_ns"] for row in history["ego"]} & available_images)
    else:
        camera_times = [timestamp]
    frames = [snapshot(input_actors, t) for t in camera_times]
    poses = {row["timestamp_ns"]: row for row in history["ego"]}
    save(output / "history_replay.json", {"scope": "Only observed input history; snapshots use observations at or before each camera frame.",
        "anchor_timestamp_ns": timestamp, "frames": [dict(f, ego=poses[f["timestamp_ns"]]) for f in frames],
        "initialization_source_timestamps_ns": prefix_times,
        "ego_SE3_camera_cv": camera["ego_SE3_camera_cv"], "K_av2": camera["K_av2"],
        "origin_city_xyz": origin.tolist(), "target_width_height": camera["target_width_height"]})

    # Separate diagnostic: these later labels do not alter either file above.
    future = json.loads((reference / "reference_future.json").read_text())
    offline_rows = []
    next_by_track = {}
    for row in sorted(future["actors"], key=lambda r: r["timestamp_ns"]):
        next_by_track.setdefault(row["track_uuid"], row)
        offline_rows.append(dict(row, carla_box_pose=carla_pose(row["position_local_xyz_m"], row["city_rotation"])))
    save(output / "offline_future_tracks.json", {"scope": "Recorded future for offline scoring/world replay only; never a planner input.",
         "anchor_timestamp_ns": timestamp, "ego": future["ego"], "actors": offline_rows})
    diagnostics = []
    for actor in anchor["actors"]:
        later = next_by_track.get(actor["track_uuid"])
        if later is None or (later["timestamp_ns"] - actor["source_timestamp_ns"]) / 1e9 > .25:
            continue
        alpha = (timestamp - actor["source_timestamp_ns"]) / (later["timestamp_ns"] - actor["source_timestamp_ns"])
        held = np.asarray(actor["observed_position_local_xyz_m"])
        interpolated = held * (1 - alpha) + np.asarray(later["position_local_xyz_m"]) * alpha
        diagnostics.append({"track_uuid": actor["track_uuid"], "category": actor["category"],
            "bracketing_timestamps_ns": [actor["source_timestamp_ns"], later["timestamp_ns"]],
            "hold_vs_offline_linear_interpolation_m": float(np.linalg.norm(held - interpolated)),
            "causal_vs_offline_linear_interpolation_m": float(np.linalg.norm(np.asarray(actor["position_local_xyz_m"]) - interpolated))})
    save(output / "alignment_diagnostic.json", {"scope": "Offline bracketing-label interpolation is a synchronization diagnostic, not observed camera-time ground truth or planner validation.", "actors": diagnostics})

    visible = sorted([a for a in anchor["actors"] if a["front_camera_box"]], key=lambda a: -a["front_camera_box"]["clipped_box_area_px"])
    motions = [a["causal_alignment_displacement_m"] for a in anchor["actors"]]
    checks = {"history_snapshots_use_only_past": all(t <= f["timestamp_ns"] <= timestamp for f in frames for a in f["actors"] for t in a["motion_source_timestamps_ns"]),
              "initialization_precedes_history": all(t < camera_times[0] for t in prefix_times),
              "future_kept_separate": all(row["timestamp_ns"] > timestamp for row in offline_rows),
              "carla_rotation_and_position_roundtrip": all(np.max(np.abs(FLIP @ np.array(a["carla_box_pose"]["rotation_matrix"]) @ FLIP - a["city_rotation"])) < 1e-10 and
                    np.max(np.abs(FLIP @ a["carla_box_pose"]["center_xyz_m"] - a["position_local_xyz_m"])) < 1e-10 for a in anchor["actors"])}
    report = {"source_reference": str(reference), "map_id": manifest["map_id"], "log_id": manifest["log_id"],
        "timestamp_ns": timestamp, "history_camera_frames": len(frames), "history_snapshot_objects": sum(len(f["actors"]) for f in frames),
        "initialization_source_timestamps_ns": prefix_times,
        "first_history_frame_objects": len(frames[0]["actors"]),
        "anchor_annotation_age_s": anchor["annotation_age_s"], "anchor_objects": len(anchor["actors"]),
        "anchor_categories": dict(Counter(a["category"] for a in anchor["actors"])),
        "front_camera_box_categories": dict(Counter(a["category"] for a in visible)),
        "anchor_alignment_displacement_m": {"mean": float(np.mean(motions)), "max": float(max(motions))},
        "offline_future_objects": len(offline_rows), "offline_alignment_track_count": len(diagnostics),
        "most_prominent_front_boxes": [{k: a[k] for k in ("track_uuid", "category", "dimensions_lwh_m", "position_anchor_ego_xyz_m", "velocity_local_xyz_mps", "causal_alignment_displacement_m", "front_camera_box")} for a in visible[:12]],
        "traffic_lights": {"source_state": "unknown", "matched_to_carla": False, "reason": "Existing record has no verified ego-facing lamp phase; cuboid annotations do not provide it."},
        "checks": checks, "passed": all(checks.values()),
        "limitations": ["Constant-velocity/yaw extrapolation spans the observed annotation-to-camera delay; annotation noise and acceleration remain.",
                        "Projected cuboid visibility excludes neither occlusion nor truncation; source dimensions are preserved separately from CARLA blueprint dimensions.",
                        "CPU preparation does not validate actor placement, road contact, CARLA rendering, or visual planning.",
                        "The recorded future is not a model input; reference ego replay is not a planner prediction."]}
    save(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, help="Read history camera frames from the local source log")
    args = parser.parse_args()
    print(json.dumps(build(args.reference, args.output, args.log_dir), indent=2))


if __name__ == "__main__":
    main()
