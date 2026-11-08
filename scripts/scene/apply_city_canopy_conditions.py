"""Apply explicit Agent crown ownership/placement conditions without source edits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('evidence','conditions','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();e=json.loads(a.evidence.read_text());c=json.loads(a.conditions.read_text())
    rows=deepcopy(e['canopy_candidates']);held=[];merges=[]
    def match(row,area,ids):return row['source_area']==area and sorted(row['source_candidate_ids'])==sorted(ids)
    for rule in c.get('hold_placements',[]):
        selected=[r for r in rows if match(r,rule['area'],rule['source_candidate_ids'])]
        if len(selected)!=1:raise ValueError('Placement condition must identify one exact source envelope')
        row=selected[0];rows.remove(row);held.append(dict(observation=row,condition=rule))
    for rule in c.get('merge_crowns',[]):
        members=[]
        for ids in rule['members']:
            selected=[r for r in rows if match(r,rule['area'],ids)]
            if len(selected)!=1:raise ValueError('Crown ownership must identify one exact source envelope per member')
            members.append(selected[0])
        center=np.mean([r['position_enu_m']for r in members],axis=0)
        top=max(r['observed_top_z_m']for r in members)
        row=deepcopy(members[0]);row.update(position_enu_m=center.tolist(),observed_top_z_m=top,
            target_height_m=float(top-center[2]),
            crown_radius_m=max(float(np.linalg.norm(np.array(r['position_enu_m'][:2])-center[:2]))+r['crown_radius_m']for r in members),
            source_candidate_ids=sorted(k for r in members for k in r['source_candidate_ids']),
            merged_source_observations=members,ownership_condition=rule,
            support_points=sum(r['support_points']for r in members),
            clearance_scope='Merged approximate anchor; actual native stem/crown clearance must be remeasured')
        for r in members:rows.remove(r)
        rows.append(row);merges.append(dict(source_area=rule['area'],source_candidate_ids=row['source_candidate_ids']))
    for row in rows:
        row['id']='canopy:'+row['source_area']+':'+','.join(map(str,row['source_candidate_ids']))
        row['appearance_condition']=c['appearance_by_area'][row['source_area']]
        row['native_realization_status']='pending_verified_ground_asset_and_clearance'
    result=dict(e,canopy_candidates=rows,source_evidence=str(a.evidence),agent_conditions=str(a.conditions),
        held_source_placements=held,crown_ownership_merges=merges,
        selected_native_candidate_count=len(rows),native_acceptance=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(dict(source_refined_candidates=len(e['canopy_candidates']),native_candidates=len(rows),
        held_placements=len(held),merged_crowns=len(merges),native_acceptance=False),flush=True)


if __name__=='__main__':main()
