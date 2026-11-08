"""Reuse calibrated AV2 actor appearance extraction on each city's source log."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

from scripts.assets.prepare_actor_layers import source_tracks
from scripts.scene.prepare_dynamic_replay import snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','config','data-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cameras',nargs='+',default=['ring_front_center','ring_front_left','ring_front_right'])
    p.add_argument('--logs',nargs='+',help='Limit new extraction to selected source logs')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    scene=json.loads(a.scene.read_text());config=json.loads(a.config.read_text())
    origin=np.array(scene['coordinates']['origin_city_xyz']);reviews=[]
    for relative in config['sources']['av2_logs']:
        log=a.data_root/relative;root=a.output/log.name;root.mkdir(exist_ok=True)
        if a.logs and log.name not in a.logs:continue
        tracks,poses=source_tracks(log,origin);by_time=defaultdict(list)
        for rows in tracks.values():
            for row in rows:by_time[row['timestamp_ns']].append(row)
        loader=AV2SensorDataLoader(log.parent,log.parent);combined={};views=[]
        for name in a.cameras:
            out=root/name;out.mkdir(exist_ok=True);camera=loader.get_log_pinhole_camera(log.name,name)
            history=dict(origin_city_xyz=origin.tolist(),K_av2=camera.intrinsics.K.tolist(),
                ego_SE3_camera_cv=camera.ego_SE3_cam.transform_matrix.tolist(),frames=[],
                coordinates='Raw source city frame minus common origin; source registration is applied only by subsequent native placement.')
            for image in sorted((log/'sensors/cameras'/name).glob('*.jpg')):
                timestamp=int(image.stem)
                preceding=[r for t,rows in by_time.items() if timestamp-250_000_000<=t<=timestamp for r in rows]
                state=snapshot(preceding,timestamp);pose=poses[timestamp]
                history['frames'].append(dict(state,source_image=str(image),ego=dict(
                    position_local_xyz_m=(pose.translation-origin).tolist(),city_rotation=pose.rotation.tolist())))
            path=out/'history.json';path.write_text(json.dumps(history,separators=(',',':'))+'\n')
            subprocess.run([sys.executable,'-m','scripts.assets.prepare_actor_appearance','--history',str(path),
                '--log-dir',str(log),'--output',str(out),'--frame-stride','1'],check=True)
            result=json.loads((out/'appearance.json').read_text());views.append(dict(camera=name,
                frames=len(history['frames']),tracks=len(result['tracks']),skipped_sweeps=len(result['skipped_unpaired_sweeps'])))
            for key,row in result['tracks'].items():
                if key not in combined or row['support_points']>combined[key]['support_points']:
                    combined[key]=dict(row,selected_camera=name,source_appearance=str(out/'appearance.json'))
        (root/'appearance.json').write_text(json.dumps(dict(tracks=combined,views=views,
            scope='Best supported calibrated view per observed vehicle identity. Source colors stay estimates; unobserved tracks remain unassigned.'),indent=2)+'\n')
        reviews.append(dict(log_id=log.name,tracks_with_color=len(combined),views=views,appearance=str(root/'appearance.json')))
        (a.output/'review.json').write_text(json.dumps(dict(logs=reviews,total_tracks=sum(r['tracks_with_color'] for r in reviews)),indent=2)+'\n')
        print(reviews[-1],flush=True)


if __name__=='__main__':main()
