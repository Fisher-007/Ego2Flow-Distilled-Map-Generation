"""Test multi-lane routes with CARLA GlobalRoutePlanner and BasicAgent."""
import argparse,json,os,queue,socket,subprocess,sys,time
from pathlib import Path
from scripts.env.check_carla_launch import wait_for_world,stop_server
from scripts.scene.drive_route import synchronous_rgb


def drive(world,case,output,seconds=70,native=None,planner=None,signal_manifest=None,signal_waypoints=None):
    import carla
    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    native=native or world.get_map();settings=world.get_settings();sync=world.get_settings()
    render=not case.get('no_rendering',False)
    sync.synchronous_mode=True;sync.fixed_delta_seconds=.05;sync.no_rendering_mode=not render;world.apply_settings(sync)
    bp=world.get_blueprint_library();actors=[];collisions=[];images=queue.Queue();rows=[]
    output.mkdir(parents=True,exist_ok=True);(output/'rgb').mkdir(exist_ok=True)
    result=dict(passed=False,case=case,map=native.name,collisions=collisions,
                rendering_enabled=render,controller_input='Native route, vehicle and traffic state; BasicAgent uses no images.')
    encoder=None;video_frames=0;signal_runner=None
    try:
        planner=planner or GlobalRoutePlanner(native,1.)
        waypoint_at=getattr(planner,'waypoint_at',native.get_waypoint_xodr)
        start=waypoint_at(*case['start']);goal=waypoint_at(*case['goal'])
        if 'native_plan' in case:
            from agents.navigation.local_planner import RoadOption
            plan=[(waypoint_at(r['road_id'],r['lane_id'],r['s_m']),RoadOption[r['road_option']])
                  for r in case['native_plan']]
            from shapely.geometry import LineString,Point
            import numpy as np
            plan_xy=np.array([[w.transform.location.x,w.transform.location.y]for w,_ in plan])
            plan_segments=plan_xy[1:]-plan_xy[:-1]
            plan_segment_lengths_sq=np.sum(plan_segments**2,axis=1)
            source_line=LineString(case['source_route']['positions'])
            goal_s=source_line.project(Point(goal.transform.location.x,-goal.transform.location.y))
        else:
            plan=planner.trace_route(start.transform.location,goal.transform.location)
        (output/'plan.json').write_text(json.dumps([dict(road_id=w.road_id,lane_id=w.lane_id,section_id=w.section_id,s=w.s,option=o.name,lane_type=str(w.lane_type)) for w,o in plan],indent=2)+'\n')
        result['planned_options']=sorted({o.name for _,o in plan})
        result['lane_log_semantics']=dict(road_id='Nearest waypoint from the configured lane_query_type (default Driving); contextual lookup, not proof of occupied lane or permissions.',
            explicit_plan_context='Closest XY segment of the supplied native plan; geometric route context, not a live lane-type or legality query.')
        result['planned_lane_types']=sorted({str(w.lane_type)for w,_ in plan})
        spawn=start.transform;spawn.location.z+=.5
        vehicle=world.spawn_actor(bp.find(case.get('vehicle_blueprint','vehicle.tesla.model3')),spawn);actors.append(vehicle)
        box=vehicle.bounding_box
        result['vehicle_geometry']=dict(blueprint=vehicle.type_id,
            bounding_box_center_m=[box.location.x,box.location.y,box.location.z],
            bounding_box_extent_m=[box.extent.x,box.extent.y,box.extent.z])
        gnss_queue=None;gnss_frame=None
        if case.get('gnss_frame'):
            from map_generation.native_gnss import NativeGnssFrame
            gnss_frame=NativeGnssFrame.from_config(case['gnss_frame'])
            gnss_queue=queue.Queue();gbp=bp.find('sensor.other.gnss')
            for key in ['noise_lat_stddev','noise_lon_stddev','noise_alt_stddev','noise_lat_bias','noise_lon_bias','noise_alt_bias']:
                gbp.set_attribute(key,'0.0')
            gbp.set_attribute('sensor_tick','0.0');gbp.set_attribute('noise_seed','0')
            gnss=world.spawn_actor(gbp,carla.Transform(carla.Location(z=1.0)),attach_to=vehicle)
            actors.append(gnss);gnss.listen(gnss_queue.put)
            result['gnss_condition']=dict(frame_config=case['gnss_frame'],mount_m=[0,0,1],noise='all declared biases and standard deviations zero',input='Measured GNSS only; sensor capture transform separately retained as validation reference.')
        vehicle.apply_control(carla.VehicleControl(brake=1.))
        for _ in range(20):world.tick()
        collision=world.spawn_actor(bp.find('sensor.other.collision'),carla.Transform(),attach_to=vehicle);actors.append(collision)
        def record_collision(event):
            impulse=event.normal_impulse;pose=vehicle.get_transform()
            collisions.append(dict(frame=event.frame,other_type=event.other_actor.type_id,
                other_actor_id=event.other_actor.id,normal_impulse=[impulse.x,impulse.y,impulse.z],
                vehicle_position_carla_m=[pose.location.x,pose.location.y,pose.location.z],
                vehicle_rotation_deg=[pose.rotation.pitch,pose.rotation.yaw,pose.rotation.roll]))
        collision.listen(record_collision)
        if render:
            cbp=bp.find('sensor.camera.rgb')
            width,height=(1280,720) if case.get('video_ffmpeg') else (800,450)
            for k,v in dict(image_size_x=str(width),image_size_y=str(height),sensor_tick='0.0').items():cbp.set_attribute(k,v)
            camera=world.spawn_actor(cbp,carla.Transform(carla.Location(x=1.5,z=2)),attach_to=vehicle);actors.append(camera);camera.listen(images.put)
        if render and case.get('video_ffmpeg'):
            encoder=subprocess.Popen([case['video_ffmpeg'],'-hide_banner','-loglevel','error','-y',
                '-f','rawvideo','-pixel_format','bgra','-video_size',f'{width}x{height}','-framerate','10','-i','-',
                '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output/'front.mp4')],stdin=subprocess.PIPE)
        control_options=dict(dt=.05,sampling_resolution=1.,base_min_distance=case.get('base_min_distance',2.),distance_ratio=case.get('distance_ratio',.25))
        if 'max_steering' in case:control_options['max_steering']=case['max_steering']
        agent=BasicAgent(vehicle,target_speed=case.get('speed_kmh',15),map_inst=native,grp_inst=planner,opt_dict=control_options)
        result['controller_options']=control_options
        # Extend terminal control waypoints along the exit while retaining the original goal.
        control_plan=list(plan)
        if case.get('directed_station_conditions'):
            from map_generation.directed_route import coalesce_directed_indices
            positions=[[w.transform.location.x,-w.transform.location.y,w.transform.location.z]for w,_ in plan]
            indices,review=coalesce_directed_indices(positions,[w.road_id for w,_ in plan],case['directed_station_conditions'])
            control_plan=[plan[i]for i in indices]
            result['directed_station_coalescing']=review
        if 'native_plan' in case:
            waypoint=plan[-1][0];tail=[]
            for _ in range(6):
                candidates=[candidate for candidate in waypoint.next(1.)
                            if candidate.lane_type == carla.LaneType.Driving]
                if len(candidates)!=1:break
                waypoint=candidates[0];tail.append((waypoint,RoadOption.LANEFOLLOW))
            control_plan.extend(tail)
            result['terminal_control_continuation']=[dict(road_id=w.road_id,lane_id=w.lane_id,s_m=w.s)for w,_ in tail]
        agent.set_global_plan(control_plan,stop_waypoint_creation=True,clean_queue=True)
        if signal_manifest is not None:
            from map_generation.traffic_signals import SignalProgramRunner
            from map_generation.signal_detection import install_route_signal_detection
            signal_runner=SignalProgramRunner(world,signal_manifest)
            result['route_signal_references']=install_route_signal_detection(agent,plan,affected_waypoints=signal_waypoints,route_lane_keys=case.get('signal_route_lane_keys'))
            result['signal_control']=('Generated source-bound phase timing with native affected-lane references; not observed AV2 phase history.'
                if signal_manifest['bindings'] else 'Existing native free-running lights with native affected-lane route adapter; no supplied phase program applied.')
        previous=None;changed=[];reached=False
        with (output/'trajectory.jsonl').open('w') as f:
            for step in range(int(seconds/.05)):
                if signal_runner is not None:signal_runner.update(step*.05)
                control=agent.run_step();vehicle.apply_control(control);frame=world.tick()
                image=synchronous_rgb(images,frame) if render else None
                if encoder and step%2==0:
                    encoder.stdin.write(image.raw_data);video_frames+=1
                transform=vehicle.get_transform();pos=transform.location;wp=native.get_waypoint(pos,lane_type=getattr(carla.LaneType,case.get('lane_query_type','Driving')))
                current=(wp.road_id,wp.section_id,wp.lane_id)
                if current!=previous:
                    changed.append(dict(frame=frame,elapsed_s=(step+1)*.05,native_lane=list(current)))
                    if image is not None:image.save_to_disk(str(output/'rgb'/('transition_%08d.png'%frame)))
                    previous=current
                speed=vehicle.get_velocity().length();distance=pos.distance(goal.transform.location)
                detected_light=agent._last_traffic_light
                waiting_red=detected_light is not None and detected_light.get_state()==carla.TrafficLightState.Red
                trigger_light=vehicle.get_traffic_light() if vehicle.is_at_traffic_light() else None
                row=dict(frame=frame,t=(step+1)*.05,position_carla_m=[pos.x,pos.y,pos.z],road_id=wp.road_id,lane_id=wp.lane_id,section_id=wp.section_id,s_m=wp.s,nearest_waypoint_query_lane_type=case.get('lane_query_type','Driving'),nearest_waypoint_lane_type=str(wp.lane_type),speed_mps=speed,goal_distance_m=distance,
                    pitch_deg=transform.rotation.pitch,roll_deg=transform.rotation.roll,yaw_deg=transform.rotation.yaw,vertical_velocity_mps=vehicle.get_velocity().z,
                    detected_red_light_id=detected_light.id if waiting_red else None,
                    native_trigger_light_id=trigger_light.id if trigger_light else None,
                    native_trigger_light_state=str(trigger_light.get_state()) if trigger_light else None,
                    control=dict(throttle=control.throttle,brake=control.brake,steer=control.steer))
                row['lane_query_type']=case.get('lane_query_type','Driving');row['native_lane_type']=str(wp.lane_type)
                if gnss_queue is not None:
                    sample=synchronous_rgb(gnss_queue,frame)
                    row['gnss']=gnss_frame.convert(sample.latitude,sample.longitude,sample.altitude)
                    loc=sample.transform.location
                    row['gnss_capture_reference']=dict(frame=sample.frame,timestamp=sample.timestamp,position_carla_m=[loc.x,loc.y,loc.z],scope='Validation truth only; not passed to GNSS conversion.')
                if 'native_plan' in case:
                    point=Point(pos.x,-pos.y)
                    row.update(route_s_m=source_line.project(point),route_offset_m=source_line.distance(point))
                    delta=np.array([pos.x,pos.y])-plan_xy[:-1]
                    fractions=np.clip(np.divide(np.sum(delta*plan_segments,axis=1),plan_segment_lengths_sq,
                        out=np.zeros(len(plan_segments)),where=plan_segment_lengths_sq>0),0.,1.)
                    distances_sq=np.sum((delta-fractions[:,None]*plan_segments)**2,axis=1)
                    index=int(np.argmin(distances_sq));planned=plan[index][0]
                    row['explicit_plan_context']=dict(segment_index=index,segment_fraction=float(fractions[index]),
                        xy_offset_m=float(np.sqrt(distances_sq[index])),road_id=planned.road_id,
                        lane_id=planned.lane_id,section_id=planned.section_id,lane_type=str(planned.lane_type))
                f.write(json.dumps(row)+'\n');rows.append(row)
                if image is not None and step%100==0:image.save_to_disk(str(output/'rgb'/('frame_%08d.png'%frame)))
                reached=distance<2.5 and wp.road_id==goal.road_id and wp.lane_id==goal.lane_id
                if 'native_plan' in case:
                    heading=transform.get_forward_vector();expected=goal.transform.get_forward_vector()
                    reached=distance<2.5 and row['route_s_m']>=goal_s-1. and heading.x*expected.x+heading.y*expected.y>.8
                if reached or collisions:break
            if image is not None:image.save_to_disk(str(output/'rgb'/'last.png'))
        result.update(reached_goal=reached,ticks=len(rows),elapsed_s=len(rows)*.05,lane_transitions=changed,
            passed=reached and not collisions,final_goal_distance_m=rows[-1]['goal_distance_m'],
            red_wait_s=.05*sum(r['detected_red_light_id'] is not None and r['speed_mps']<.1 for r in rows),
            max_abs_vertical_velocity_mps=max(abs(r['vertical_velocity_mps']) for r in rows),
            scope='Official GlobalRoutePlanner and BasicAgent with unchanged source-backed markings. Lane histories and collision events recorded; no background traffic.')
        if 'native_plan' in case:
            result.update(max_route_offset_m=max(r['route_offset_m'] for r in rows),
                scope='Source-selected Area route on complete physical map, adapted CARLA native-identity planner and official BasicAgent. Exit position, heading, continuous route offset and collisions checked. No background traffic or AV2 replay.')
            result['passed'] &= result['max_route_offset_m']<2.
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if signal_runner is not None:signal_runner.close()
        for actor in reversed(actors):
            if actor.type_id.startswith('sensor.'):actor.stop()
            actor.destroy()
        world.apply_settings(settings)
        if encoder:
            try:encoder.stdin.close()
            except BrokenPipeError:pass
            result['video']=dict(path=str(output/'front.mp4'),frames=video_frames,fps=10,
                width=width,height=height,first_sample_s=.05,sample_interval_s=.1,encoder_exit=encoder.wait(),
                scope='Actual synchronous driving frames, every second 20 Hz tick; source data are not replayed.')
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roads',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=Path,help='Native start/goal triples for source-backed road/lane/station routes')
    p.add_argument('--map',required=True)
    p.add_argument('--imported',action='store_true',help='Use the imported physical map instead of the procedural road diagnostic')
    a=p.parse_args();port=int(os.environ['MAP_RPC_PORT']);a.output.mkdir(parents=True,exist_ok=True)
    for value in [port,port+1,port+2]:
        with socket.socket() as sock:sock.bind(('127.0.0.1',value))
    sys.path.insert(0,str(Path(os.environ['CARLA_ROOT'])/'PythonAPI/carla'))
    import carla
    section=json.loads((a.roads/'section.json').read_text());length=section['length_m']
    last=section.get('native_sections',[dict(road_id=1,length_m=length)])[-1]
    cases=json.loads(a.cases.read_text()) if a.cases else [dict(name='left',start=[1,-2,10.],goal=[last['road_id'],-1,last['length_m']-10.]),dict(name='right',start=[1,-2,10.],goal=[last['road_id'],-3,last['length_m']-10.])]
    with (a.output/'server.log').open('w') as log:
        server=subprocess.Popen(['bash','scripts/env/run_carla.sh',a.map,'-quality-level=Epic'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            wait_for_world(server,port,3600,a.output/'server.log');print('Owned server ready',flush=True)
            client=carla.Client('127.0.0.1',port);client.set_timeout(300);world=client.get_world()
            if not a.imported:
                world=client.generate_opendrive_world((a.roads/'roads.xodr').read_text(),carla.OpendriveGenerationParameters(vertex_distance=.5,max_road_length=50,wall_height=0,additional_width=0,smooth_junctions=False))
            results=[]
            for case in cases:
                try:
                    result=drive(world,case,a.output/case['name']);results.append(result)
                except Exception:
                    results.append(json.loads((a.output/case['name']/'result.json').read_text()))
                    raise
                finally:
                    (a.output/'results.json').write_text(json.dumps(dict(results=results,procedural=not a.imported),indent=2)+'\n')
                print({k:result[k] for k in ['case','passed','ticks','lane_transitions','final_goal_distance_m']},flush=True)
            (a.output/'results.json').write_text(json.dumps(dict(results=results,procedural=not a.imported),indent=2)+'\n')
        finally:stop_server(server)


if __name__=='__main__':main()
