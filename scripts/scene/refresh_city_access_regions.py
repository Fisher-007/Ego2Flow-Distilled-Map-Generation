"""Refresh coupled generation regions after explicit source corrections.

Retain surviving reviewed seeds, add explicitly declared new interfaces, regroup
actual current topology, and preserve prior region IDs where evidence overlaps.
"""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
from map_generation.access_fusion import section
from scripts.scene.group_access_repair_regions import group_regions


def refresh(scene,previous,conditions):
    lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    old_seeds={s for r in previous['regions']for s in r['seed_seams']}
    retired=sorted(old_seeds-set(lanes));seeds=sorted((old_seeds&set(lanes))|set(conditions['additional_seed_ids']))
    if any(lanes[k]['source']=='av2'for k in seeds):raise ValueError('A fixed observed lane cannot become a generation seed')
    groups=group_regions(scene,[dict(seam_id=k,needs_review=True)for k in seeds],previous['margin_m'])
    old={r['id']:r for r in previous['regions']};used=set();records=[];next_id=max(int(k.rsplit('_',1)[1])for k in old)+1
    for region in groups:
        available=[]
        keys={v['lane_id']for v in region['intervals']}
        for key,row in old.items():
            if key in used:continue
            shared=len(set(region['seed_seams'])&set(row['seed_seams']))
            common=len(keys&{v['lane_id']for v in row['intervals']})
            if shared or common:available.append((-shared,-common,key))
        prior=min(available)[2]if available else None
        ident=prior or f'access_region_{next_id:03d}'
        if not prior:next_id+=1
        used.add(ident);region['id']=ident
        records.append(dict(region=ident,prior_region=prior,seed_count=len(region['seed_seams'])))
    for scope in conditions.get('scope_decisions',[]):
        decision=json.loads(Path(scope).read_text());before=decision['before'];after=decision['after']
        matches=[r for r in groups if set(before['seed_seams'])<=set(r['seed_seams'])]
        if len(matches)!=1:raise ValueError('Scope decision does not identify one current region')
        region=matches[0]
        # Match disjoint source-lane intervals by their reviewed bounds.
        for old_value,value in zip(before['intervals'],after['intervals'],strict=True):
            if value==old_value:continue
            candidates=[v for v in region['intervals']if v==old_value]
            if len(candidates)!=1:raise ValueError('Reviewed scope interval differs from current region')
            candidates[0].update(value)
        region['scope_feedback']=after['scope_feedback']
    merge_records=[]
    for decision in conditions.get('coupled_scope_merges',[]):
        members=[r for r in groups if r['id']in decision['regions']]
        if len(members)!=len(decision['regions']):raise ValueError('Reviewed coupled scope region missing')
        merged=deepcopy(members[0]);merged['id']=decision['id']
        merged['seed_seams']=sorted({k for r in members for k in r['seed_seams']})
        merged['fixed_av2_ports']=[list(v)for v in sorted({tuple(p)for r in members for p in r['fixed_av2_ports']})]
        intervals={}
        for r in members:
            for value in r['intervals']:intervals.setdefault(value['lane_id'],[]).append(deepcopy(value))
        combined=[]
        for key,values in sorted(intervals.items()):
            joined=[]
            for value in sorted(values,key=lambda v:(v['start_m'],v['end_m'])):
                if joined and value['start_m']<=joined[-1]['end_m']+1e-6:
                    joined[-1]['end_m']=max(joined[-1]['end_m'],value['end_m'])
                else:joined.append(value)
            combined.extend(joined)
        merged['intervals']=combined;merged['mutable_lane_count']=len(intervals)
        merged['scope_feedback']={'coupled_merge':decision,'member_feedback':[r.get('scope_feedback')for r in members]}
        groups=[r for r in groups if r['id']not in decision['regions']]+[merged]
        merge_records.append(decision)
        for record in records:
            if record['region']in decision['regions']:record['merged_into']=decision['id']
    for region in groups:
        points=np.vstack([section(lanes[v['lane_id']]['centerline'],v['start_m'],v['end_m'])for v in region['intervals']])[:,:2]
        region['bbox_local_xy']=list(map(float,[*points.min(axis=0),*points.max(axis=0)]))
        region['extent_xy_m']=(points.max(axis=0)-points.min(axis=0)).tolist()
        region['mutable_length_m']=sum(v['end_m']-v['start_m']for v in region['intervals'])
    ownership={}
    for region in groups:
        for value in region['intervals']:
            key=value['lane_id']
            for other,low,high in ownership.get(key,[]):
                if min(high,value['end_m'])-max(low,value['start_m'])>.001:
                    raise ValueError('Revised inference regions overlap: '+other+' / '+region['id'])
            ownership.setdefault(key,[]).append((region['id'],value['start_m'],value['end_m']))
    result=deepcopy(previous);result.update(source_scene=conditions['source_scene'],regions=sorted(groups,key=lambda r:r['id']),
        status='source_corrected_regions_pending_model_preparation',refresh_conditions=conditions,
        region_refresh=dict(previous_regions=len(old),current_regions=len(groups),surviving_seeds=len(old_seeds)-len(retired),
            retired_seed_ids=retired,added_seed_ids=conditions['additional_seed_ids'],region_identity=records,
            previous_regions_without_current_owner=sorted(set(old)-{r['id']for r in groups}),coupled_scope_merges=merge_records))
    result.pop('reuse_reason',None)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','previous','conditions','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();result=refresh(json.loads(a.scene.read_text()),json.loads(a.previous.read_text()),json.loads(a.conditions.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    review=result['region_refresh']
    print(dict(previous_regions=review['previous_regions'],current_regions=review['current_regions'],
        retired_seeds=len(review['retired_seed_ids']),added_seeds=len(review['added_seed_ids']),
        previous_regions_without_current_owner=review['previous_regions_without_current_owner'],
        maximum_extent_m=max(max(r['extent_xy_m'])for r in result['regions'])),flush=True)

if __name__=='__main__':main()
