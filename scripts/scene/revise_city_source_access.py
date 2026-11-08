"""Revise explicitly reviewed local source accesses while preserving other observations.

Reuse the normal source-path construction and original-interval restoration.
Every new seam remains a candidate for constrained geometry and native review.
"""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
from map_generation.access_fusion import restore_access,section
from scripts.scene.restore_city_source_continuations import restore
from scripts.scene.realize_city_source_components import observed_source_ids


def revise(scene,conditions,reviewed):
    before={l['id']:l for l in scene['fusion_map']['lanes']}
    observed=observed_source_ids(scene);retired=set(conditions['retire_seams'])
    if retired&observed:raise ValueError('A local access revision cannot retire observed lanes')
    for path in conditions.get('observed_paths',[]):
        if any(k not in observed for k in path):raise ValueError('Reviewed fine path is not observed')
        for a,b in zip(path[:-1],path[1:]):
            if b not in before[a]['successors']:raise ValueError('Reviewed observed path is absent')
        for row in reviewed['connections']:
            if path[0]==row['fine_lane_id']and not set(row['permissions'])<=set.intersection(*(set(before[k]['allowed_classes'])for k in path)):
                raise ValueError('Observed route loses a required source mode')
    out=deepcopy(scene);fused=out['fusion_map'];old_access=deepcopy(fused['source_access_review'])
    old_continuation=deepcopy(fused.get('source_continuation_review'))
    fused['lanes']=[l for l in fused['lanes']if l['id']not in retired]
    for lane in fused['lanes']:
        if any(lane.get(f)in retired for f in ['left_neighbor','right_neighbor']):
            raise ValueError('A retired seam has a lateral owner')
        lane['successors']=[k for k in lane['successors']if k not in retired]
    target_ids={r['fine_lane_id']for r in reviewed['connections']}
    if not target_ids<=observed:raise ValueError('Local receiving stations must be observed')
    out=restore_access(out,conditions.get('options',{}),deepcopy(reviewed))
    new_access=deepcopy(out['fusion_map']['source_access_review'])
    previous_connections=old_access.get('connections',[])
    retired_connections=[r for r in previous_connections if r['seam_id']in retired]
    old_access['connections']=[r for r in previous_connections if r['seam_id']not in retired]+new_access['connections']
    old_access.setdefault('retired_connections',[]).extend(retired_connections)
    old_access.setdefault('fine_station_splits',{}).update(new_access['fine_station_splits'])
    old_access.setdefault('unresolved',[]).extend(new_access['unresolved'])
    old_access.setdefault('geometry_rejected_candidates',[]).extend(new_access['geometry_rejected_candidates'])
    out['fusion_map']['source_access_review']=old_access
    continuation=None
    if conditions.get('coarse_continuations'):
        out,continuation=restore(out,dict(connections=conditions['coarse_continuations'],reason=conditions['reason']))
        if old_continuation:continuation['previous_review']=old_continuation
    after={l['id']:l for l in out['fusion_map']['lanes']}
    if len(after)!=len(out['fusion_map']['lanes']):raise ValueError('Local reconstruction produced duplicate IDs')
    for key,lane in before.items():
        if key in retired or key in target_ids:continue
        a=deepcopy(lane);b=deepcopy(after[key]);a.pop('successors');b.pop('successors')
        if a!=b:raise ValueError('An unrelated source observation changed: '+key)
        if not set(lane['successors'])-retired<=set(after[key]['successors']):
            raise ValueError('An unrelated existing successor was lost')
    for key,parts in new_access['fine_station_splits'].items():
        original=before[key]
        for part in parts:
            lane=after[part['id']];expected=section(original['centerline'],*part['interval_m'])
            if not np.allclose(lane['centerline'],expected,atol=1e-8,rtol=0):raise ValueError('A fine split changed original geometry')
            for field in ['allowed_classes','width_m','speed_mps']:
                if lane[field]!=original[field]:raise ValueError('A fine split changed source permissions or dimensions')
        if not set(original['successors'])<=set(after[parts[-1]['id']]['successors']):
            raise ValueError('A split dropped its original downstream branches')
    report=dict(conditions=conditions,retired_source_observations=[before[k]for k in sorted(retired)],
        retired_connections=retired_connections,new_access_review=new_access,source_continuation_review=continuation,
        split_original_observations=[before[k]for k in sorted(target_ids)],
        lane_count_before=len(before),lane_count_after=len(after),
        new_seam_ids=[r['seam_id']for r in new_access['connections']],
        restored_source_intervals=continuation['inserted_intervals']if continuation else 0,
        existing_geometry_changes=0,existing_permission_changes=0,
        status='Explicit local source correction; inferred geometry requires constrained generation and native review.')
    out['fusion_map'].setdefault('local_source_access_revisions',[]).append(report)
    return out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','conditions','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();conditions=json.loads(a.conditions.read_text())
    reviewed=json.loads(Path(conditions['reviewed_connections']).read_text())
    out,report=revise(json.loads(a.scene.read_text()),conditions,reviewed)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n')
    (a.output/'local_access_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:report[k]for k in ['lane_count_before','lane_count_after','new_seam_ids','restored_source_intervals','existing_geometry_changes','existing_permission_changes','status']},flush=True)

if __name__=='__main__':main()
