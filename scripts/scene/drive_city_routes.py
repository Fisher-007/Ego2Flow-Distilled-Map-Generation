"""Drive all prepared Area traversals on one already running complete map."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def driving_plans(directory, route_names=None):
    """Select native plans without depending on a particular city's filename."""
    rows = []
    for path in sorted(directory.glob('*.json')):
        if route_names and path.stem not in route_names:
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or 'source' not in data or 'native_plan' not in data:
            continue
        if not data['native_plan'] or not data['source'].get('route', {}).get('lane_ids'):
            raise ValueError('Empty native driving plan: ' + str(path))
        rows.append((path, data))
    if not rows:
        raise ValueError('No native driving plans matched in ' + str(directory))
    missing = set(route_names or []) - {path.stem for path, _ in rows}
    if missing:
        raise ValueError('Requested driving plans were not found: ' + ', '.join(sorted(missing)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--map',required=True)
    p.add_argument('--port',type=int,default=int(os.environ.get('MAP_RPC_PORT','23000')))
    p.add_argument('--seconds',type=float,default=120.)
    p.add_argument('--route-names',nargs='+',help='Run only these prepared route file stems')
    p.add_argument('--speed-kmh',type=float,default=10.)
    p.add_argument('--video-ffmpeg',help='Record the actual drive at 1280x720, 10 fps using this FFmpeg binary')
    p.add_argument('--no-rendering',action='store_true',help='Run BasicAgent driving checks with RGB rendering disabled')
    p.add_argument('--base-min-distance-m',type=float,default=.8,
                   help='BasicAgent waypoint lookahead base; recorded with each route')
    p.add_argument('--distance-ratio',type=float,default=.1,
                   help='BasicAgent speed-dependent waypoint lookahead')
    p.add_argument('--signals',type=Path,help='Check native light groups and source lane bindings before driving')
    p.add_argument('--launch',action='store_true',help='Launch and stop an isolated assigned CARLA process')
    a=p.parse_args()
    driving_plans(a.plans, a.route_names)
    sys.path.insert(0,str(Path(os.environ['CARLA_ROOT'])/'PythonAPI/carla'))
    import carla
    from map_generation.native_route_planner import NativeRoutePlanner
    from scripts.scene.check_multilane_runtime import drive
    a.output.mkdir(parents=True,exist_ok=True)
    if a.launch:
        from scripts.env.check_carla_launch import stop_server
        for port in (a.port,a.port+1,a.port+2):
            with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        with (a.output/'server.log').open('w') as log:
            server=subprocess.Popen(['bash','scripts/env/run_carla.sh',a.map,'-quality-level=Epic'],
                env=dict(os.environ,MAP_RPC_PORT=str(a.port)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+3600
                while time.monotonic()<deadline:
                    if server.poll() is not None:raise RuntimeError('Owned CARLA exited; inspect server.log')
                    client=carla.Client('127.0.0.1',a.port);client.set_timeout(5.)
                    try:
                        client.get_world()
                        break
                    except RuntimeError:time.sleep(1.)
                else:raise TimeoutError('Owned world did not start within 3600 seconds')
                # Retrieve OpenDRIVE once for the loaded map.
                execute(a,carla)
            finally:stop_server(server)
    else:execute(a,carla)


def prepare_runtime(a,carla):
    from map_generation.native_route_planner import NativeRoutePlanner
    client=carla.Client('127.0.0.1',a.port);client.set_timeout(600.)
    world=client.get_world();native=world.get_map()
    if native.name.removeprefix('/Game/')!=a.map.removeprefix('/Game/'):
        raise ValueError('Unexpected map: '+native.name)
    signal_manifest=None;signal_review=None
    if a.signals:
        from map_generation.traffic_signals import check_runtime_signal_bindings
        if world.get_settings().synchronous_mode:
            world.tick(60.)
        else:
            world.wait_for_tick(60.)
        signal_manifest=json.loads(a.signals.read_text())
        signal_review=check_runtime_signal_bindings(world,signal_manifest)
        print({k:v for k,v in signal_review.items() if k!='signals'},flush=True)
        if not signal_review['passed']:
            a.output.mkdir(parents=True,exist_ok=True)
            (a.output/'signals.json').write_text(json.dumps(signal_review,indent=2)+'\n')
            raise RuntimeError('Resolve native signal groups and bindings before driving')
    planner=NativeRoutePlanner(native,1.)
    print('Loaded complete native map; preparing source Area routes',flush=True)
    return dict(world=world,native=native,planner=planner,signal_manifest=signal_manifest,
                signal_review=signal_review,signal_waypoints={})


def execute(a,carla,runtime=None):
    """Run a route group, optionally reusing the same loaded-world queries."""
    plans=driving_plans(a.plans,a.route_names)
    from scripts.scene.check_multilane_runtime import drive
    runtime=prepare_runtime(a,carla)if runtime is None else runtime
    world,native,planner=(runtime[k]for k in ('world','native','planner'))
    signal_manifest=runtime['signal_manifest']
    a.output.mkdir(parents=True,exist_ok=True)
    if runtime['signal_review'] is not None:
        (a.output/'signals.json').write_text(json.dumps(runtime['signal_review'],indent=2)+'\n')
    # Native light geometry is fixed throughout this loaded world. Reuse its
    # actual affected waypoints across routes; signal states remain live.
    results=[];signal_waypoints=runtime['signal_waypoints']
    for path,data in plans:
        points=data['native_plan']
        triple=lambda r:[r['road_id'],r['lane_id'],r['s_m']]
        case=dict(name=data['source']['area_id'],start=triple(points[0]),goal=triple(points[-1]),
            native_plan=points,source_route=data['source']['route'],
            speed_kmh=a.speed_kmh,base_min_distance=a.base_min_distance_m,distance_ratio=a.distance_ratio,
            vehicle_blueprint=data['source'].get('vehicle_blueprint','vehicle.tesla.model3'))
        if data.get('directed_station_conditions'):case['directed_station_conditions']=data['directed_station_conditions']
        if a.video_ffmpeg:case['video_ffmpeg']=a.video_ffmpeg
        if getattr(a,'no_rendering',False):case['no_rendering']=True
        output=a.output/case['name']
        try:
            result=drive(world,case,output,a.seconds,native=native,planner=planner,signal_manifest=signal_manifest,signal_waypoints=signal_waypoints)
        except Exception:
            if not (output/'result.json').exists():
                raise
            result=json.loads((output/'result.json').read_text())
        summary={k:v for k,v in result.items() if k not in ('case','lane_transitions')}
        summary['area_id']=case['name'];results.append(summary)
        (a.output/'review.json').write_text(json.dumps(dict(map=native.name,results=results,
            source_plans=str(a.plans),scope=__doc__),indent=2)+'\n')
        print(summary,flush=True)
    return runtime


if __name__=='__main__':main()
