"""Prepare a real AV2 video window and matching causal actor/camera replay inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from scripts.data.av2_source import listing, S3
from scripts.assets.prepare_actor_layers import source_tracks, save
from scripts.scene.prepare_dynamic_replay import snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--split', choices=['train','val','test'], help='Defaults to the split directory containing log-dir')
    p.add_argument('--start-ns',type=int,required=True);p.add_argument('--duration-s',type=float,default=6)
    p.add_argument('--camera',default='ring_front_center');p.add_argument('--stride',type=int,default=2)
    p.add_argument('--calibration-ns',type=int,
                   help='Source time inside the window with visible lane landmarks for camera-geometry verification')
    p.add_argument('--layers',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--local-images',action='store_true',help='Use all existing source camera files without a remote listing')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    split = a.split or a.log_dir.parent.name
    if split not in {'train','val','test'}:
        p.error('Set --split when log-dir is outside a standard AV2 split directory')
    prefix = f'datasets/av2/sensor/{split}/'
    if a.local_images:
        rows=[{'key':prefix+a.log_dir.name+'/sensors/cameras/'+a.camera+'/'+f.name,'size':f.stat().st_size}
              for f in (a.log_dir/'sensors/cameras'/a.camera).glob('*.jpg')]
    else:
        rows,_=listing(prefix+a.log_dir.name+'/sensors/cameras/'+a.camera+'/')
    images=sorted([r for r in rows if r['key'].endswith('.jpg')],key=lambda r:r['key'])
    start=min(range(len(images)),key=lambda i:abs(int(Path(images[i]['key']).stem)-a.start_ns))
    chosen=[r for r in images[start::a.stride] if int(Path(r['key']).stem)<=a.start_ns+int(a.duration_s*1e9)]
    def download(r):
        path=a.log_dir/'sensors/cameras'/a.camera/Path(r['key']).name
        if not path.exists() or path.stat().st_size!=r['size']:
            with urlopen(S3+r['key'],timeout=120) as response:content=response.read()
            if len(content)!=r['size']:raise RuntimeError('Incomplete frame '+str(path))
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        return {'timestamp_ns':int(path.stem),'source_image':str(path),'source_url':S3+r['key']}
    with ThreadPoolExecutor(max_workers=4) as pool:images=list(pool.map(download,chosen))
    scene=json.loads(a.scene.read_text());origin=np.asarray(scene['coordinates']['origin_city_xyz'])
    tracks,_=source_tracks(a.log_dir,origin);rows=[r for v in tracks.values() for r in v]
    roles={r['track_uuid']:r['role'] for r in json.loads(a.layers.read_text())['tracks']}
    loader=AV2SensorDataLoader(a.log_dir.parent,a.log_dir.parent)
    camera=loader.get_log_pinhole_camera(a.log_dir.name,a.camera)
    frames=[]
    for image in images:
        t=image['timestamp_ns'];state=snapshot(rows,t);ego=loader.get_city_SE3_ego(a.log_dir.name,t)
        for actor in state['actors']:actor['role']=roles[actor['track_uuid']]
        frames.append({**state,'ego':{'timestamp_ns':t,'position_local_xyz_m':(ego.translation-origin).tolist(),
            'city_rotation':ego.rotation.tolist()},'source_image':image['source_image']})
    calibration=min(frames,key=lambda f:abs(f['timestamp_ns']-a.calibration_ns))['timestamp_ns'] if a.calibration_ns else images[-1]['timestamp_ns']
    save(a.output/'history.json',{'scope':'Observed source camera window, using past-only actor snapshots. Offline role labels never supply future trajectories to a policy.',
        'camera':a.camera,'anchor_timestamp_ns':images[-1]['timestamp_ns'],'frames':frames,
        'calibration_timestamp_ns':calibration,
        'ego_SE3_camera_cv':camera.ego_SE3_cam.transform_matrix.tolist(),'K_av2':camera.intrinsics.K.tolist(),
        'origin_city_xyz':origin.tolist(),'target_width_height':[camera.width_px,camera.height_px]})
    save(a.output/'source_images.json',{'camera':a.camera,'images':images})
    print({'camera':a.camera,'frames':len(frames),'duration_s':(images[-1]['timestamp_ns']-images[0]['timestamp_ns'])/1e9},flush=True)


if __name__=='__main__':main()
