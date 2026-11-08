"""Replay causal AV2 history with persistent kinematic actors and calibrated cameras."""
import argparse
from collections import Counter
import json
from pathlib import Path
import queue

import carla
import cv2
import numpy as np

from scripts.scene.render_av2_view import prepare, CV_FROM_CARLA_CAMERA
from scripts.scene.render_dynamic_av2_view import actor_pose, placement_records, spawn_snapshot, transform


def camera_pose(history, frame):
    ego = np.eye(4)
    ego[:3, :3] = frame['ego']['city_rotation']
    ego[:3, 3] = frame['ego']['position_local_xyz_m']
    camera = ego @ np.asarray(history['ego_SE3_camera_cv'])
    flip = np.diag([1., -1., 1.])
    expected = np.eye(4)
    expected[:3, :3] = flip @ camera[:3, :3] @ CV_FROM_CARLA_CAMERA
    expected[:3, 3] = flip @ camera[:3, 3]
    return transform(expected), expected


def receive(stream, tick):
    image = stream.get(timeout=30)
    while image.frame < tick:
        image = stream.get(timeout=30)
    if image.frame != tick:
        raise RuntimeError('Sensor image differs from the requested synchronous frame')
    return image


def replay(args):
    history = json.loads(args.history.read_text())
    from map_generation.coordinates import source_log_registration
    from scripts.scene.prepare_dynamic_replay import register_replay
    scene = json.loads(args.scene.read_text())
    history = register_replay(history, source_log_registration(scene, args.log_dir.name))
    frames = history['frames'][getattr(args,'frame_start',0):]
    appearance=json.loads(args.actor_appearance.read_text())['tracks'] if getattr(args,'actor_appearance',None) else {}
    if getattr(args,'conditions',None):
        from map_generation.native_conditions import selected_appearance
        # Apply supplied graph-selection fields over observed tracks.
        for track, selected in selected_appearance(json.loads(args.conditions.read_text())).items():
            appearance[track] = dict(appearance.get(track, {}), **selected)
    surface=None
    if getattr(args,'asset_surface',None):
        from scripts.scene.replay_asset_fit import Surface
        surface=Surface(args.asset_surface)
    times = [f['timestamp_ns'] for f in frames]
    if any(t > f['timestamp_ns'] or f['timestamp_ns'] > history['anchor_timestamp_ns']
           for f in frames for a in f['actors'] for t in a['motion_source_timestamps_ns']):
        raise ValueError('History has a future observation')
    if not frames[0]['actors']:
        raise ValueError('First history frame has no actors; check past initialization before replay')
    dt = float(np.median(np.diff(np.asarray(times, dtype=np.int64)))) / 1e9
    offsets = (np.asarray(times, dtype=np.int64)-times[0])/1e9
    if np.max(np.abs(offsets-np.arange(len(times))*dt)) > 1e-4:
        raise ValueError('This short replay requires camera times on one regular synchronous grid')
    args.output.mkdir(parents=True, exist_ok=True)
    calibration_time=history.get('calibration_timestamp_ns',history['anchor_timestamp_ns'])
    calibration_frame=next(f for f in history['frames'] if f['timestamp_ns']==calibration_time)
    # Verify calibration with an in-window pose containing source-lane landmarks.
    metadata, _, remap, _ = prepare(scene, args.log_dir,
                                  args.output/'anchor_calibration', calibration_time,
                                  camera_name=history.get('camera', 'ring_front_center'))
    _, anchor_matrix = camera_pose(history, calibration_frame)
    source_matrix = np.asarray(metadata['carla_world_SE3_camera_fru'])
    if (np.max(np.abs(anchor_matrix[:3,:3]-source_matrix[:3,:3])) > 1e-6 or
            np.max(np.abs(anchor_matrix[:3,3]-source_matrix[:3,3])) > metadata['geometry_checks']['translation_tolerance_m']):
        raise ValueError('History camera differs from official reference calibration')
    client = carla.Client('127.0.0.1', args.port)
    client.set_timeout(getattr(args,'native_map_timeout_s',60.))
    world = client.get_world()
    native = world.get_map()
    if native.name.rsplit('/',1)[-1] != args.expected_map:
        raise ValueError('Unexpected replay map')
    if surface:
        from scripts.scene.replay_asset_fit import catalog,choose,DIMENSION_MATCHED_CATEGORIES
        assets=catalog(world)
        (args.output/'asset_catalog.json').write_text(json.dumps(assets,indent=2)+'\n')
        dimensions={row['track_uuid']:(row['dimensions_lwh_m'],row['category']) for f in frames for row in f['actors'] if row['category'] in DIMENSION_MATCHED_CATEGORIES}
        choices={key:choose(size,assets,appearance.get(key,{}),category) for key,(size,category) in dimensions.items()}
    else:choices={}
    for frame in frames:
        for actor in frame['actors']:
            if actor['track_uuid'] in appearance:actor['appearance']=appearance[actor['track_uuid']]
            if actor['track_uuid'] in choices:actor['native_blueprint']=choices[actor['track_uuid']]
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    settings.no_rendering_mode = False
    pool, provenance, skipped, sensors = {}, [], [], []
    captures = []
    initial_simulation_time = None
    signal_replay = None
    if getattr(args, 'signal_replay', None):
        from map_generation.signal_observations import ObservedSignalReplay
        signal_replay = ObservedSignalReplay(world, json.loads(args.signals.read_text()),
            json.loads(args.signal_replay.read_text()), args.log_dir.name,
            history.get('camera', 'ring_front_center'))
    try:
        world.apply_settings(settings)
        streams = {}
        pose, _ = camera_pose(history, frames[0])
        for name, kind in [('rgb','sensor.camera.rgb'),('semantic','sensor.camera.semantic_segmentation')]:
            bp = world.get_blueprint_library().find(kind)
            attrs = {'image_size_x':str(metadata['render_width_height'][0]),
                'image_size_y':str(metadata['render_width_height'][1]),
                'fov':str(metadata['render_horizontal_fov_deg']), 'sensor_tick':'0', 'lens_k':'0','lens_kcube':'0'}
            if name == 'rgb':
                attrs.update(enable_postprocess_effects='True', motion_blur_intensity='0',
                    lens_flare_intensity='0', bloom_intensity='0', chromatic_aberration_intensity='0')
            for key,value in attrs.items():bp.set_attribute(key,value)
            sensor=world.spawn_actor(bp,pose);stream=queue.Queue()
            sensor.listen(stream.put);sensors.append(sensor);streams[name]=stream
        (args.output/'rgb').mkdir(exist_ok=True)
        (args.output/'semantic').mkdir(exist_ok=True)
        with (args.output/'frames.jsonl').open('w') as logfile:
            for index, frame in enumerate(frames):
                source = {a['track_uuid']:a for a in frame['actors']}
                for track in list(pool):
                    if track not in source:
                        pool.pop(track).destroy()
                newcomers = [a for track,a in source.items() if track not in pool]
                actors, records, omissions = spawn_snapshot(world, {'actors':newcomers}, front_only=False)
                for actor, record in zip(actors, records):
                    pool[record['track_uuid']]=actor
                    record['first_history_index']=index
                    provenance.append(record)
                skipped.extend(dict(r,history_index=index) for r in omissions)
                pose, expected = camera_pose(history,frame)
                for sensor in sensors:sensor.set_transform(pose)
                active_records=[]
                for track,actor in pool.items():
                    row=source[track]
                    fitted=None;target_box=row['carla_box_pose']
                    if surface and actor.type_id.startswith('vehicle.') and row['category'] in ('REGULAR_VEHICLE','LARGE_VEHICLE','BOX_TRUCK','BUS','SCHOOL_BUS'):
                        target_pose,target_box,fitted=surface.fit(row,actor)
                    else:target_pose=actor_pose(row['carla_box_pose'],actor.bounding_box)
                    actor.set_transform(target_pose)
                    active_records.append({'track_uuid':track,'actor_id':actor.id,'category':row['category'],
                        'source_box_pose':row['carla_box_pose'],'source_dimensions_lwh_m':row['dimensions_lwh_m'],
                        'render_box_pose':target_box,'asset_surface_fit':fitted,
                        'asset_dimensions_lwh_m':[2*actor.bounding_box.extent.x,2*actor.bounding_box.extent.y,2*actor.bounding_box.extent.z],
                        'source_velocity_local_xyz_mps':row['velocity_local_xyz_mps'],
                        'source_observation_age_s':row['observation_age_s'],
                        'motion_source_timestamps_ns':row['motion_source_timestamps_ns']})
                signal_selection = signal_replay.apply(frame['timestamp_ns']) if signal_replay else None
                # Warm the first pose once; subsequent source frames advance by
                # exactly one world tick, with no per-image warmup or camera reset.
                for _ in range(args.warmup_frames+1 if index == 0 else 1):
                    tick=world.tick()
                    rgb=receive(streams['rgb'],tick)
                    semantic=receive(streams['semantic'],tick)
                placement_records(list(pool.values()),active_records)
                raw=np.frombuffer(rgb.raw_data,dtype=np.uint8).reshape(rgb.height,rgb.width,4)
                image=cv2.remap(raw[:,:,:3],*remap,interpolation=cv2.INTER_LINEAR)
                raw_sem=np.frombuffer(semantic.raw_data,dtype=np.uint8).reshape(semantic.height,semantic.width,4)
                labels=cv2.remap(raw_sem[:,:,2],*remap,interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(args.output/'rgb'/f'{index:03d}.png'),image)
                cv2.imwrite(str(args.output/'semantic'/f'{index:03d}.png'),labels)
                if initial_simulation_time is None:initial_simulation_time=rgb.timestamp
                actual=np.asarray(rgb.transform.get_matrix())
                capture={'history_index':index,'source_timestamp_ns':frame['timestamp_ns'],
                    'source_relative_time_s':(frame['timestamp_ns']-history['anchor_timestamp_ns'])/1e9,
                    'world_frame':tick,'rgb_frame':rgb.frame,'semantic_frame':semantic.frame,
                    'simulation_timestamp_s':rgb.timestamp,
                    'time_mapping_error_s':float(rgb.timestamp-initial_simulation_time-offsets[index]),
                    'source_objects':len(source),'replayed_objects':len(pool),
                    'camera_matrix_max_error':float(np.max(np.abs(actual-expected))),
                    'camera_translation_max_error_m':float(np.max(np.abs(actual[:3,3]-expected[:3,3]))),
                    'camera_rotation_max_error':float(np.max(np.abs(actual[:3,:3]-expected[:3,:3]))),
                    'actors':active_records,
                    'native_lights':{light.get_opendrive_id():str(light.get_state()) for light in world.get_actors().filter('traffic.traffic_light*')}}
                if signal_replay:
                    capture['source_signal_selection'] = signal_selection
                    capture['source_signal_check'] = signal_replay.check()
                logfile.write(json.dumps(capture,allow_nan=False)+'\n');logfile.flush()
                captures.append(capture)
        report={'scope':__doc__,'map':native.name,'history_file':str(args.history),
            'log_registration':history.get('log_registration'),
            'calibration_timestamp_ns':calibration_time,
            'frames':len(captures),'history_duration_s':float(offsets[-1]), 'fixed_delta_seconds':dt,
            'total_replayed_object_frames':sum(c['replayed_objects'] for c in captures),
            'unique_source_tracks':len({a['track_uuid'] for f in frames for a in f['actors']}),
            'spawn_events':len(provenance),'skipped_by_category':dict(Counter(s['category'] for s in skipped)),
            'skipped':skipped,'spawn_records':provenance,
            'max_box_center_error_m':max(a['box_center_error_m'] for c in captures for a in c['actors']),
            'max_box_rotation_error':max(a['box_rotation_matrix_max_abs_error'] for c in captures for a in c['actors']),
            'asset_surface_file':str(args.asset_surface) if surface else None,
            'actor_appearance_file':str(args.actor_appearance) if getattr(args,'actor_appearance',None) else None,
            'graph_conditions_file':str(args.conditions) if getattr(args,'conditions',None) else None,
            'max_target_box_center_error_m':max(a['target_box_center_error_m'] for c in captures for a in c['actors']),
            'max_target_box_rotation_error':max(a['target_box_rotation_error'] for c in captures for a in c['actors']),
            'max_camera_matrix_error':max(c['camera_matrix_max_error'] for c in captures),
            'max_camera_translation_error_m':max(c['camera_translation_max_error_m'] for c in captures),
            'max_camera_rotation_error':max(c['camera_rotation_max_error'] for c in captures),
            'max_time_mapping_error_s':max(abs(c['time_mapping_error_s']) for c in captures),
            'checks':{'synchronized_rgb_semantics':all(c['world_frame']==c['rgb_frame']==c['semantic_frame'] for c in captures),
                'contiguous_world_frames':all(b['world_frame']==a['world_frame']+1 for a,b in zip(captures,captures[1:])),
                'source_observations_not_from_future':all(t<=c['source_timestamp_ns'] for c in captures for a in c['actors'] for t in a['motion_source_timestamps_ns'])},
            'limitations':['No policy or prediction; camera/actors follow causal recorded history with physics disabled.',
                'Asset dimensions, articulation and surface contact remain different from source boxes.',
                'Native light states are recorded but not matched to historical AV2 states.',
                'All class-supported observed objects are placed; skipped categories remain explicit.',
                'Source velocities are past estimates; CARLA kinematic actor velocity is not a recovered dynamic state.']}
        if signal_replay:
            selected = [c for c in captures if c['source_signal_selection']['applied']]
            report['source_signal_replay'] = dict(
                conditions=str(args.signal_replay), bindings=str(args.signals),
                applied_frames=len(selected),
                directly_observed_frames=sum(c['source_signal_selection']['evidence_kind']=='reviewed_sample' for c in selected),
                inferred_frames=sum(c['source_signal_selection']['evidence_kind']=='inferred_interval' for c in selected),
                uncovered_frames=len(captures)-len(selected),
                matched_frames=sum(bool(c['source_signal_check']['matched']) for c in selected))
            report['checks']['source_signal_constraints'] = all(c['source_signal_check']['matched'] for c in selected)
            report['limitations'][2] = 'Selected source-light intervals distinguish directly reviewed frames from inferred holds. Uncovered lights and historical transition timing remain unverified.'
        # Check translation and rotation with separate calibration tolerances.
        report['camera_matrix_tolerance']=metadata['geometry_checks']['translation_tolerance_m']
        if surface:
            fits=[a['asset_surface_fit'] for c in captures for a in c['actors'] if a['asset_surface_fit'] and a['asset_surface_fit']['applied']]
            gaps=[abs(g) for f in fits for g in f['support_gaps_m'] if np.isfinite(g)]
            report['asset_contact_summary']={'fitted_object_frames':len(fits),'abs_support_gap_m_quantiles':np.quantile(gaps,[.5,.95,1]).tolist(),
                'source_z_adjustment_abs_m_quantiles':np.quantile([abs(f['source_center_delta_z_m']) for f in fits],[.5,.95,1]).tolist()}
            report['limitations'].append('Vehicle render height/tilt adapts to the native surface and selected asset. Source z/rotation errors are retained; target-placement success is not source-geometry fidelity.')
        report['checks'].update(box_centers=report['max_target_box_center_error_m']<.001,
            box_rotations=report['max_target_box_rotation_error']<1e-5,
            camera_geometry=report['max_camera_translation_error_m']<report['camera_matrix_tolerance']
                and report['max_camera_rotation_error']<1e-6,
            clock_mapping=report['max_time_mapping_error_s']<1e-4)
        report['passed_replay_checks']=all(report['checks'].values())
        (args.output/'replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if not report['passed_replay_checks']:raise RuntimeError('Replay checks failed; inspect replay.json')
        print(json.dumps({k:v for k,v in report.items() if k not in ['spawn_records','skipped']},indent=2))
    finally:
        if signal_replay:signal_replay.close()
        for sensor in reversed(sensors):sensor.stop();sensor.destroy()
        for actor in reversed(list(pool.values())):actor.destroy()
        world.apply_settings(original)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--history',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--expected-map',default='PitDrivableHoles')
    parser.add_argument('--warmup-frames',type=int,default=20)
    parser.add_argument('--conditions',type=Path)
    parser.add_argument('--signals',type=Path,help='Native source control bindings for optional replay light intervals')
    parser.add_argument('--signal-replay',type=Path,help='Reviewed samples and explicit inferred finite source-state intervals')
    parser.add_argument('--actor-appearance',type=Path)
    parser.add_argument('--asset-surface',type=Path,help='Explicit native road/terrain mesh directory for rendering fit; source states remain separate')
    parser.add_argument('--native-map-timeout-s',type=float,default=60.)
    args=parser.parse_args();replay(args)


if __name__=='__main__':main()
