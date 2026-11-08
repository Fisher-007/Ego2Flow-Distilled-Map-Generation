"""Reuse source LiDAR canopy extraction for the declared city observations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import shapely
from av2.utils.io import read_city_SE3_ego
from map_generation.coordinates import source_log_registration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','visual-review','data-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--declared-areas-only',action='store_true',help='Extract the declared Area sensor logs only; retain map-only source geometry without requiring unrelated camera downloads')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    scene=json.loads(a.scene.read_text());views=json.loads(a.visual_review.read_text())
    logs={}
    for area in views['areas']:
        row=logs.setdefault(area['log_id'],dict(split=area['split'],views=[]))
        row['views'].extend(dict(v,id=area['area_id']+'_'+v['camera']) for v in area['views'])
    for lane in scene['lanes']:
        for observation in lane.get('source_observations', []):
            source=Path(observation['source']);log_id=source.parent.parent.name
            if log_id not in logs and not a.declared_areas_only:
                split=source.parent.parent.parent.name
                log=a.data_root/'av2/sensor'/split/log_id
                selected=[]
                for camera in ('ring_front_left','ring_front_center','ring_front_right'):
                    files=sorted((log/'sensors/cameras'/camera).glob('*.jpg'))
                    image=files[len(files)//2]
                    selected.append(dict(id='source_'+camera,camera=camera,timestamp_ns=int(image.stem),file=str(image)))
                logs[log_id]=dict(split=split,views=selected)
    buildings=scene['buildings'];lanes=scene['fusion_map']['lanes'];fine=scene['lanes']
    building_shapes=[shapely.Polygon(b['exterior'],b.get('holes',[])) for b in buildings]
    lane_shapes=[shapely.LineString(l['centerline']) for l in lanes]
    fine_shapes=[shapely.LineString(l['centerline']) for l in fine]
    indexes=[shapely.STRtree(g) for g in (building_shapes,lane_shapes,fine_shapes)]
    records=[]
    for log_id,config in sorted(logs.items()):
        folder=a.output/log_id;folder.mkdir(exist_ok=True)
        evidence=folder/'extracted/evidence.json'
        if not evidence.exists():
            log=a.data_root/'av2/sensor'/config['split']/log_id
            poses=read_city_SE3_ego(log)
            xy=np.array([pose.translation for pose in poses.values()])-scene['coordinates']['origin_city_xyz']
            registration=source_log_registration(scene,log_id)
            if registration:
                xy=xy@np.asarray(registration['rotation_matrix']).T+registration['translation_local_m']
            bounds=shapely.MultiPoint(xy[:,:2]).envelope.buffer(180.)
            subsets=[[rows[int(i)] for i in tree.query(bounds,predicate='intersects')]
                     for tree,rows in zip(indexes,(buildings,lanes,fine))]
            crop=dict(map_id=scene['map_id'],coordinates=scene['coordinates'],areas=scene['areas'],
                source_info=dict(assembly_review=dict(log_registrations=scene.get('source_info',{}).get('assembly_review',{}).get('log_registrations',[]))),
                buildings=subsets[0],fusion_map=dict(lanes=subsets[1]),lanes=subsets[2])
            # Retain the exact source-registration declaration used by cameras,
            # actors and road assembly; the crop only limits clearance queries.
            (folder/'scene.json').write_text(json.dumps(crop,separators=(',',':'))+'\n')
            (folder/'views.json').write_text(json.dumps(dict(views=config['views']))+'\n')
            command=[sys.executable,'-u','-m','scripts.assets.prepare_av2_context','--scene',str(folder/'scene.json'),
                '--log-dir',str(log),'--views',str(folder/'views.json'),'--output',str(folder/'extracted')]
            with (folder/'extract.log').open('w') as handle:
                subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        data=json.loads(evidence.read_text())
        records.append(dict(log_id=log_id,canopy_candidates=len(data['canopy_candidates']),
            source_registration=data.get('source_log_registration'),static_voxels=data['static_voxels'],
            evidence=str(evidence)))
        (a.output/'review.json').write_text(json.dumps(dict(logs=records,
            source_log_policy='declared Area sensor logs only; other map-only observations retained without source-context extraction' if a.declared_areas_only else 'declared Areas plus other source lane logs',
            scope='Source-supported canopy candidates with annotation removal and reviewed source registration. Camera review and native realization remain separate.'),indent=2)+'\n')
        print(records[-1],flush=True)


if __name__=='__main__':main()
