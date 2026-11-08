"""Render source-timestamped area views with calibrated cameras and actor snapshots."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','views','histories','appearance','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--map',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--areas',nargs='+');p.add_argument('--cameras',nargs='+')
    p.add_argument('--gbuffers',action='store_true',help='Save source-view material and geometry buffers for visibility feedback')
    p.add_argument('--weather',type=Path)
    p.add_argument('--signals',type=Path,help='Native source-control bindings for reviewed snapshot light states')
    p.add_argument('--signal-evidence',type=Path,help='Reviewed source light states at exact image timestamps')
    p.add_argument('--replay-assets',type=Path,help='Explicit per-track rigid mesh overrides')
    p.add_argument('--period-assets',type=Path,help='Optional fixed source-epoch native meshes at explicit poses, keyed by exact log/camera/time; removed after capture')
    p.add_argument('--actor-interactions',type=Path,help='Optional exact-view native person roles and coarse occupancy poses, separate from original AV2 rows')
    p.add_argument('--runtime-context',type=Path,help='Optional source-observed context in registered city coordinates, keyed by exact log/camera/time; kept separate from AV2 snapshots')
    p.add_argument('--surfaces',type=Path,help='Fit native vehicle shapes to the source ground layer, retaining original cuboids')
    p.add_argument('--prepare-only',action='store_true',help='Check real camera geometry and snapshots without connecting to CARLA')
    p.add_argument('--rgb-only',action='store_true',help='Review calibrated RGB views without additional depth and semantic sensors')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import carla
    from map_generation.coordinates import source_log_registration
    from scripts.scene.prepare_dynamic_replay import register_replay
    from scripts.scene.render_av2_view import prepare,render
    from scripts.scene.render_dynamic_av2_view import spawn_snapshot,placement_records
    from scripts.scene.replay_asset_fit import Surface,catalog,choose,DIMENSION_MATCHED_CATEGORIES
    replay_assets=json.loads(a.replay_assets.read_text())['tracks'] if a.replay_assets else {}
    scene=json.loads(a.scene.read_text());areas=json.loads(a.views.read_text())['areas']
    period_views=json.loads(a.period_assets.read_text())['views'] if a.period_assets else []
    context_views=json.loads(a.runtime_context.read_text())['views'] if a.runtime_context else []
    interaction_views=json.loads(a.actor_interactions.read_text())['views'] if a.actor_interactions else []
    if not a.prepare_only:
        client=carla.Client('127.0.0.1',a.port);client.set_timeout(600.)
        world=client.get_world();native=world.get_map()
        if native.name.removeprefix('/Game/')!=a.map.removeprefix('/Game/'):
            raise ValueError('Unexpected city map: '+native.name)
        client.set_timeout(60.)
        print('Loaded native map; preparing source camera and actor views',flush=True)
    if a.weather and not a.prepare_only:
        weather=world.get_weather()
        for key,value in json.loads(a.weather.read_text()).items():setattr(weather,key,float(value))
        world.set_weather(weather)
    assets=catalog(world) if not a.prepare_only else [];results=[]
    surface=Surface(a.surfaces) if a.surfaces and not a.prepare_only else None
    (a.output/'vehicle_catalog.json').write_text(json.dumps(assets,indent=2)+'\n')
    signal_snapshot=None
    if a.signal_evidence and not a.prepare_only:
        from map_generation.signal_observations import ObservedSignalSnapshot
        signal_snapshot=ObservedSignalSnapshot(world,json.loads(a.signals.read_text()),json.loads(a.signal_evidence.read_text()))
    for area in areas:
        if a.areas and area['area_id'] not in a.areas:continue
        log_id=area['log_id'];appearance=json.loads((a.appearance/log_id/'appearance.json').read_text())
        for view in area['views']:
            camera=view['camera']
            if a.cameras and camera not in a.cameras:continue
            output=a.output/area['area_id']/camera;output.mkdir(parents=True,exist_ok=True)
            print('Preparing',area['area_id'],camera,flush=True)
            row=dict(area_id=area['area_id'],log_id=log_id,camera=camera,timestamp_ns=view['timestamp_ns'],passed=False)
            actors=[]
            try:
                history=json.loads((a.histories/log_id/camera/'history.json').read_text())
                snapshot=next(frame for frame in history['frames'] if frame['timestamp_ns']==view['timestamp_ns'])
                snapshot=register_replay(snapshot,source_log_registration(scene,log_id))
                for actor in snapshot['actors']:
                    color=appearance['tracks'].get(actor['track_uuid'],{})
                    if color:actor['appearance']=color
                    if actor['category'] in DIMENSION_MATCHED_CATEGORIES and assets:
                        actor['native_blueprint']=choose(actor['dimensions_lwh_m'],assets,color,actor['category'])
                context_rows=[actor.copy() for context in context_views
                    if context['log_id']==log_id and context['camera']==camera and context['timestamp_ns']==view['timestamp_ns']
                    for actor in context['actors']]
                for actor in context_rows:
                    if actor['category'] in DIMENSION_MATCHED_CATEGORIES and assets:
                        actor['native_blueprint']=choose(actor['dimensions_lwh_m'],assets,actor.get('appearance',{}),actor['category'])
                interaction_conditions=[actor for interaction in interaction_views
                    if interaction['log_id']==log_id and interaction['camera']==camera and interaction['timestamp_ns']==view['timestamp_ns']
                    for actor in interaction['actors']]
                native_blueprints={x['track_uuid']:x['native_blueprint']for x in interaction_conditions}
                period_pieces=[piece for period in period_views
                    if period['log_id']==log_id and period['camera']==camera and period['timestamp_ns']==view['timestamp_ns']
                    for piece in period['pieces']]
                log_dir=Path(view['file']).parents[3]
                metadata,pose,remap,projections=prepare(scene,log_dir,output,view['timestamp_ns'],camera_name=camera)
                if a.prepare_only:
                    (output/'snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
                    (output/'period_assets.json').write_text(json.dumps(period_pieces,indent=2)+'\n')
                    (output/'runtime_context.json').write_text(json.dumps(context_rows,indent=2)+'\n')
                    row.update(passed=metadata['geometry_checks']['passed'],camera_checks=metadata['geometry_checks'],
                        source_view=view,actors=len(snapshot['actors']),output=str(output),native_rendered=False)
                    results.append(row)
                    (a.output/'review.json').write_text(json.dumps(dict(map=a.map,views=results,prepare_only=True,scope=__doc__),indent=2)+'\n')
                    print(row,flush=True)
                    continue
                from scripts.scene.source_period_assets import spawn_fixed_period_meshes,readback_period_meshes
                period_actors,period_records=spawn_fixed_period_meshes(world,period_pieces)
                actors.extend(period_actors)
                snapshot_actors,records,skipped=spawn_snapshot(world,snapshot,front_only=False,replay_assets=replay_assets,native_blueprints=native_blueprints)
                actors.extend(snapshot_actors)
                context_actors,context_records,context_skipped=spawn_snapshot(world,dict(actors=context_rows),front_only=False,replay_assets=replay_assets,native_blueprints=native_blueprints) if context_rows else ([],[],[])
                actors.extend(context_actors)
                for record in context_records:record['source_origin']='observed_runtime_context_not_av2_track'
                if surface:
                    by_id={row['track_uuid']:row for row in snapshot['actors']+context_rows}
                    by_actor={actor.id:actor for actor in actors}
                    for record in records+context_records:
                        actor=by_actor[record['actor_id']]
                        if 'source_anchor_local_xyz_m' in record:
                            record['asset_support_fit'] = dict(applied=False, reason='Explicit source-anchored observed mesh retains its source pose; generic body centering/ground fit is not applicable.')
                            continue
                        pose_fit,box,fit=surface.fit(by_id[record['track_uuid']],actor)
                        record.update(render_box_pose=box,asset_support_fit=fit)
                        actor.set_transform(pose_fit)
                interaction_results=[]
                if interaction_conditions:
                    from scripts.scene.source_actor_interactions import apply_interactions
                    interaction_results=apply_interactions(world,actors,records+context_records,interaction_conditions)
                detail=dict(actors=records,skipped=skipped,source_snapshot=snapshot,
                    source_period=dict(manifest=str(a.period_assets) if a.period_assets else None,
                        pieces=period_records,scope='Stationary source-epoch objects, mass zero at declared poses. Removed after capture; no permanent navigation permission change.'),
                    actor_interactions_manifest=str(a.actor_interactions) if a.actor_interactions else None,
                    interaction_results=interaction_results,
                    runtime_context=dict(manifest=str(a.runtime_context) if a.runtime_context else None,
                        source_objects=context_rows,actors=context_records,skipped=context_skipped,
                        scope='Optional separately identified context; physics disabled and removed after this capture. Not AV2 tracks or trajectories.'),
                    replay_assets_manifest=str(a.replay_assets) if a.replay_assets else None,
                    unresolved_appearance=appearance.get('unresolved_tracks',{}),
                    scope=('Native shapes fitted to the source ground layer; original source cuboids, asset dimensions and support changes retained.' if surface else 'Native assets at source cuboid centers; dimensions and ground gaps reported.')+' Physics disabled for source snapshot review; separate controlled traffic runs measure driving.')
                # Save placement inputs before reporting a camera-check failure.
                (output/'actors.json').write_text(json.dumps(detail,indent=2)+'\n')
                if signal_snapshot:
                    row['signal_observation']=signal_snapshot.apply(log_id,camera,view['timestamp_ns'])
                    (output/'signal_state.json').write_text(json.dumps(row['signal_observation'],indent=2)+'\n')
                rendered=render(metadata,pose,remap,projections,output,a.port,a.map.rsplit('/',1)[-1],20,
                    capture_semantics=not a.rgb_only,capture_depth=not a.rgb_only,
                    capture_gbuffers=a.gbuffers,runtime=(client,world,native))
                if signal_snapshot and row['signal_observation']['observed_state_applied']:
                    row['signal_observation']['native_after_render']=signal_snapshot.check()
                    (output/'signal_state.json').write_text(json.dumps(row['signal_observation'],indent=2)+'\n')
                    if not row['signal_observation']['native_after_render']['matched']:
                        raise ValueError('Source snapshot light state changed during native capture')
                readback_period_meshes(period_actors,period_records)
                placement_records(actors,records)
                placement_records(context_actors,context_records)
                if interaction_conditions:
                    from scripts.scene.source_actor_interactions import readback_interactions
                    readback_interactions(actors,records+context_records)
                (output/'actors.json').write_text(json.dumps(detail,indent=2)+'\n')
                row.update(passed=rendered['passed'],actors=len(records),runtime_context_actors=len(context_records),skipped=skipped,
                    source_period_pieces=len(period_records),
                    runtime_context_skipped=context_skipped,
                    camera_checks=metadata['geometry_checks'],source_view=view,output=str(output))
            except Exception as error:
                row.update(error_type=type(error).__name__,error=str(error))
            finally:
                if signal_snapshot:signal_snapshot.close()
                for actor in reversed(actors):actor.destroy()
            results.append(row)
            (a.output/'review.json').write_text(json.dumps(dict(map=a.map,views=results,prepare_only=a.prepare_only,scope=__doc__),indent=2)+'\n')
            print(row,flush=True)


if __name__=='__main__':main()
