"""Separate stationary context and traffic using AV2 tracks and lane geometry."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from av2.structures.cuboid import CuboidList
from av2.utils.io import read_city_SE3_ego
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.prepared import prep

from scripts.scene.prepare_dynamic_replay import snapshot, register_state, register_replay
from map_generation.coordinates import source_log_registration, register_local_pose


VEHICLES = {"REGULAR_VEHICLE", "BUS", "SCHOOL_BUS", "BOX_TRUCK", "LARGE_VEHICLE", "MOTORCYCLIST"}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def source_tracks(log_dir, origin):
    table = pd.read_feather(log_dir/"annotations.feather").sort_values(["timestamp_ns", "track_uuid"])
    poses = read_city_SE3_ego(log_dir)
    tracks = defaultdict(list)
    for row, cuboid in zip(table.itertuples(index=False), CuboidList.from_dataframe(table).cuboids):
        timestamp = int(row.timestamp_ns)
        box = cuboid.transform(poses[timestamp])
        tracks[str(row.track_uuid)].append({"timestamp_ns": timestamp, "track_uuid": str(row.track_uuid),
            "category": str(row.category), "num_interior_pts": int(row.num_interior_pts),
            "dimensions_lwh_m": [float(row.length_m), float(row.width_m), float(row.height_m)],
            "position_local_xyz_m": (box.dst_SE3_object.translation-origin).tolist(),
            "city_rotation": box.dst_SE3_object.rotation.tolist()})
    return tracks, poses


def classify_tracks(tracks, scene_lanes):
    lanes = [lane for lane in scene_lanes if lane.get("lane_type") == "VEHICLE"]
    polygons = [Polygon(np.r_[np.asarray(lane["left_boundary"])[:,:2],
                               np.asarray(lane["right_boundary"])[::-1,:2]]).buffer(0) for lane in lanes]
    road = unary_union(polygons)
    prepared_road = prep(road)
    lines = [LineString(np.asarray(lane["centerline"])[:,:2]) for lane in lanes]
    features, all_rows = [], []
    for track, rows in tracks.items():
        all_rows.extend(rows)
        times = np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64)
        xyz = np.asarray([row["position_local_xyz_m"] for row in rows])
        center = np.median(xyz, axis=0)
        span = (int(times[-1])-int(times[0]))/1e9
        displacement = float(np.linalg.norm(np.quantile(xyz[:,:2], .95, axis=0)-np.quantile(xyz[:,:2], .05, axis=0)))
        speeds = []
        for i in range(len(rows)):
            j = int(np.searchsorted(times, int(times[i])-500_000_000))
            dt = (int(times[i])-int(times[j]))/1e9
            if .35 <= dt <= .65:
                speeds.append(float(np.linalg.norm(xyz[i,:2]-xyz[j,:2])/dt))
        p90_speed = float(np.quantile(speeds, .9)) if speeds else None
        on_road = float(np.mean([prepared_road.covers(Point(p[:2])) for p in xyz]))
        stationary = span >= 2 and displacement < .75 and p90_speed is not None and p90_speed < .5
        category = rows[0]["category"]
        if category == "STOP_SIGN":
            role = "static_control_evidence"
        elif category == "BICYCLE" and stationary:
            role = "stationary_nonvehicle_context"
        elif category in VEHICLES and stationary and on_road < .05 and road.distance(Point(center[:2])) > .5:
            role = "observed_stationary_background"
        elif category in VEHICLES:
            role = "traffic_participant" if span >= .5 and displacement >= 2 and p90_speed is not None and p90_speed >= .75 else "traffic_unresolved"
        else:
            role = "other_participant"
        nearest = sorted(range(len(lines)), key=lambda i: lines[i].distance(Point(center[:2])))[:3]
        features.append({"track_uuid":track,"category":category,"role":role,
            "observation_count":len(rows),"observed_duration_s":span,"first_timestamp_ns":int(times[0]),
            "last_timestamp_ns":int(times[-1]),"robust_motion_extent_m":displacement,"p90_half_second_speed_mps":p90_speed,
            "center_in_driving_polygon_fraction":on_road,"center_clearance_to_driving_polygon_m":float(road.distance(Point(center[:2]))),
            "median_position_local_xyz_m":center.tolist(),"median_dimensions_lwh_m":np.median([r["dimensions_lwh_m"] for r in rows],axis=0).tolist(),
            "nearest_lanes":[{"lane_id":lanes[i]["id"],"distance_m":float(lines[i].distance(Point(center[:2])))} for i in nearest]})
    return features, all_rows, lanes, road, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--anchor-ns", type=int, required=True)
    parser.add_argument("--native-replay", type=Path, help="Existing same-log anchor actor placement report supplies native blueprint transforms")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-information-graph", action="store_true",
                        help="Export measured actor layers without constructing the optional evidence graph")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = json.loads(args.scene.read_text())
    origin = np.asarray(scene["coordinates"]["origin_city_xyz"])
    tracks, poses = source_tracks(args.log_dir, origin)
    registration = source_log_registration(scene, args.log_dir.name)
    source_rows = [row for rows in tracks.values() for row in rows]
    if registration:
        tracks = {key:[register_state(row,registration) for row in rows] for key,rows in tracks.items()}
    features, all_rows, lanes, road, lines = classify_tracks(tracks, scene["lanes"])
    all_rows = source_rows
    anchor = register_replay(snapshot(all_rows, args.anchor_ns), registration)
    by_id = {row["track_uuid"]:row for row in features}
    for row in anchor["actors"]:
        row["role"] = by_id[row["track_uuid"]]["role"]
    counts = dict(Counter(row["role"] for row in features))
    report = {"log_id":args.log_dir.name,"scene":str(args.scene.resolve()),"anchor_timestamp_ns":args.anchor_ns,
        "log_registration":registration,
        "classification_scope":"Offline construction using the whole observed log; not a model input or a claim of permanent parking.",
        "criteria":{"stationary_min_duration_s":2,"robust_motion_extent_max_m":.75,"p90_speed_max_mps":.5,
                    "driving_polygon_fraction_max":.05,"driving_polygon_clearance_min_m":.5},
        "rules_scope":"Measurements propose actor roles. Ambiguous/brief/stopped-in-lane vehicles remain removable controllable participants, never permanent map meshes.",
        "counts":counts,"tracks":features}
    save(args.output/"actor_layers.json", report)
    save(args.output/"anchor_snapshot.json", anchor)
    # The exported actor context is optional run-time content, separate from the map.
    native = {r['track_uuid']:r for r in json.loads(args.native_replay.read_text())['actors']} if args.native_replay else {}
    for actor in anchor['actors']:
        if actor['track_uuid'] in native:
            row=native[actor['track_uuid']]
            actor['native_asset']={'blueprint':row['blueprint'],'actor_transform_carla':row['actor_transform_requested'],
                                   'dimensions_lwh_m':row['blueprint_dimensions_lwh_m']}
    save(args.output/"runtime_context.json", {"source_log":args.log_dir.name,"source_timestamp_ns":args.anchor_ns,
        "log_registration":registration,
        "scope":"Optional observed stationary background; source-time moving actors are not baked into the map.",
        "parked_actors":[row for row in anchor["actors"] if row["role"] == "observed_stationary_background"],
        "traffic_initial_states":[row for row in anchor["actors"] if row["role"] in ("traffic_participant","traffic_unresolved")],
        "unresolved_static_controls":[row for row in anchor["actors"] if row["role"] == "static_control_evidence"]})
    # Sample spaced observation times within the supplied spatial support.
    samples = []
    for timestamp in sorted({r["timestamp_ns"] for r in all_rows})[10::10]:
        pose = poses[timestamp]
        ego = pose.translation-origin
        if registration:ego,_=register_local_pose(ego,pose.rotation,registration)
        window = Point(ego[:2]).buffer(50.)
        length = sum(line.intersection(window).length for line in lines)
        state = register_replay(snapshot(all_rows, timestamp),registration)
        participants = []
        for row in state["actors"]:
            pos = Point(row["position_local_xyz_m"][:2])
            if by_id[row["track_uuid"]]["role"] not in ("traffic_participant","traffic_unresolved") or not window.covers(pos) or not road.covers(pos):
                continue
            nearest = min(range(len(lines)), key=lambda i:lines[i].distance(pos))
            participants.append({"track_uuid":row["track_uuid"],"source_lane_id":lanes[nearest]["id"],
                "role":by_id[row["track_uuid"]]["role"],
                "position_local_xyz_m":row["position_local_xyz_m"],"velocity_local_xyz_mps":row["velocity_local_xyz_mps"],
                "speed_mps":float(np.linalg.norm(row["velocity_local_xyz_mps"][:2])) if len(row['motion_source_timestamps_ns'])==2 else None,"dimensions_lwh_m":row["dimensions_lwh_m"]})
        moving=[p for p in participants if p['role']=='traffic_participant']
        valid_speeds=[p['speed_mps'] for p in moving if p['speed_mps'] is not None]
        samples.append({"timestamp_ns":timestamp,"ego_local_xyz_m":ego.tolist(),"lane_length_m":length,
            "candidate_traffic_count":len(participants),"observed_moving_count":len(moving),"unresolved_count":len(participants)-len(moving),
            "moving_density_veh_per_lane_km":len(moving)*1000/length if length else None,
            "candidate_density_upper_veh_per_lane_km":len(participants)*1000/length if length else None,
            "moving_mean_speed_mps":float(np.mean(valid_speeds)) if valid_speeds else None,"actors":participants})
    save(args.output/"macro_samples.json", {"source_log":args.log_dir.name,"radius_m":50,"sample_period_s":1,
        "log_registration":registration,
        "scope":"Observed local traffic within a moving geometric 50m disk and AV2 vehicle lanes; annotated objects only. Not full visibility, long-term demand, or a city density estimate. Sample joint count/speed/state records; do not independently mix incompatible marginals.","samples":samples})
    # Link actor roles, source lanes, and observations in the graph.
    if not args.no_information_graph:
        from map_generation.information_graph import enrich
        nodes, edges = [], []
        for lane in lanes:
            nodes.append({"id":lane["id"],"kind":"source_lane","source":"AV2","immutable_fields":["topology","permissions","measured_boundaries"]})
        for row in features:
            key="object:"+row["track_uuid"]
            nodes.append({"id":key,"kind":"observed_object",**row,"realization":"pending" if row["role"]=="static_control_evidence" else "runtime_layer",
                          "unresolved_question":"Which approach does this sign regulate? Verify image face, geometry and lane relation before assigning a rule." if row["role"]=="static_control_evidence" else None})
            evidence="evidence:"+row["track_uuid"]
            nodes.append({"id":evidence,"kind":"source_track","source":str(args.log_dir/"annotations.feather"),"track_uuid":row["track_uuid"]})
            edges.append({"source":key,"relation":"supported_by","target":evidence,"status":"observed"})
            for near in row["nearest_lanes"]:
                edges.append({"source":key,"relation":"near_lane","target":near["lane_id"],"distance_m":near["distance_m"],"status":"measured_proximity_not_control_relation"})
        graph = enrich({"nodes":nodes,"edges":edges,"decisions":[]},scene,args.scene.resolve())
        save(args.output/"evidence_graph.json", graph)
    print(json.dumps({"counts":counts,"anchor_counts":dict(Counter(r["role"] for r in anchor["actors"])),"macro_samples":len(samples),"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
