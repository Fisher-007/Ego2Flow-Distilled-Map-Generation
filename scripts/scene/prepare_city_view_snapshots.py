"""Align selected city views with available past object observations.

Keep the Area and original view as evidence. When a camera lies outside the
annotation time span, retrieve the nearest camera frame with preceding source
observations and rebuild its snapshot with the existing causal replay method.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from scripts.data.av2_source import S3
from scripts.assets.prepare_actor_layers import source_tracks
from scripts.scene.prepare_dynamic_replay import snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('views','histories','data-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    read=lambda path:json.loads(path.read_text())
    spec=read(a.views);histories={};sources={};outputs={};reports=[]
    for area in spec['areas']:
        log_id=area['log_id']
        log=Path(area['views'][0]['file']).parents[3]
        times=np.unique(pd.read_feather(log/'annotations.feather',columns=['timestamp_ns'])['timestamp_ns'])
        for view in area['views']:
            key=(log_id,view['camera']);original=deepcopy(view);t=view['timestamp_ns']
            relative=view.get('history_relative_path', str(Path(log_id)/view['camera']/'history.json'))
            history_path=a.histories/relative
            if history_path not in histories:histories[history_path]=read(history_path)
            history=histories[history_path];frame=next(r for r in history['frames'] if r['timestamp_ns']==t)
            changed=t<int(times[0]) or t>int(times[-1])
            if changed:
                objects=read(a.data_root/'av2/metadata'/(log_id+'_objects.json'))
                prefix=f"datasets/av2/sensor/{area['split']}/{log_id}/sensors/cameras/{view['camera']}/"
                candidates=[]
                for obj in objects:
                    if not obj['key'].startswith(prefix) or not obj['key'].endswith('.jpg'):continue
                    stamp=int(Path(obj['key']).stem)
                    # Estimate velocity from two preceding source observations.
                    before=times[(times<=stamp)&(times>=stamp-250_000_000)]
                    if len(before)>=2:candidates.append((abs(stamp-t),stamp,obj))
                _,stamp,obj=min(candidates,key=lambda row:row[:2])
                target=log/'sensors/cameras'/view['camera']/Path(obj['key']).name
                if not target.exists():
                    with urlopen(S3+obj['key'],timeout=90) as response:payload=response.read()
                    if len(payload)!=obj['size']:raise IOError('Incomplete source image: '+str(target))
                    target.write_bytes(payload)
                if log_id not in sources:sources[log_id]=source_tracks(log,np.asarray(history['origin_city_xyz']))
                tracks,poses=sources[log_id]
                rows=[row for records in tracks.values() for row in records if stamp-250_000_000<=row['timestamp_ns']<=stamp]
                frame=snapshot(rows,stamp);pose=poses[stamp]
                frame.update(source_image=str(target),ego=dict(position_local_xyz_m=(pose.translation-np.asarray(history['origin_city_xyz'])).tolist(),city_rotation=pose.rotation.tolist()))
                view.update(file=str(target),timestamp_ns=stamp,area_anchor_difference_s=view['area_anchor_difference_s']+(stamp-t)/1e9,
                    previous_view=original,selection_feedback='Nearest frame with existing past annotations; original view precedes or follows the annotated span.')
            row=dict(area_id=area['area_id'],camera=view['camera'],original_timestamp_ns=t,timestamp_ns=view['timestamp_ns'],
                changed=changed,time_shift_s=(view['timestamp_ns']-t)/1e9,actors=len(frame['actors']),annotation_timestamp_ns=frame['annotation_timestamp_ns'],
                annotation_span_ns=[int(times[0]),int(times[-1])],source_image=view['file'])
            reports.append(row)
            if key not in outputs:outputs[key]=dict(history,frames=[])
            if not any(r['timestamp_ns']==frame['timestamp_ns'] for r in outputs[key]['frames']):outputs[key]['frames'].append(frame)
            # The renderer's existing log/camera interface holds every selected
            # timestamp; only the independent preparation jobs need Area paths.
            view['history_relative_path']=str(Path(log_id)/view['camera']/'history.json')
    for (log_id,camera),history in outputs.items():
        history['frames'].sort(key=lambda frame:frame['timestamp_ns'])
        out=a.output/'histories'/log_id/camera;out.mkdir(parents=True,exist_ok=True)
        (out/'history.json').write_text(json.dumps(history,separators=(',',':'))+'\n')
    spec['paired_view_preparation']=dict(original_views=str(a.views),report=str(a.output/'review.json'),scope=__doc__)
    (a.output/'views.json').write_text(json.dumps(spec,indent=2)+'\n')
    (a.output/'review.json').write_text(json.dumps(dict(views=reports,changed_views=sum(r['changed'] for r in reports),
        source_areas_unchanged=True,scope='Paired snapshots with source views, annotation-span alignment, and recorded time shifts.'),indent=2)+'\n')
    print(dict(views=len(reports),changed=sum(r['changed'] for r in reports),actors=sum(r['actors'] for r in reports)),flush=True)


if __name__=='__main__':main()
