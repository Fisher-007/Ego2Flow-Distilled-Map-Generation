"""Realize explicit Agent choices among observed lane identities and segmentations."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from map_generation.city_registration import register_sources
from map_generation.city_sources import combine_exact_observations
from scripts.data.propose_city_source_alignment import boundary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data-root','whole-proposal','chain-proposal','neighbor-records','conditions','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();condition=json.loads(a.conditions.read_text())
    whole=json.loads(a.whole_proposal.read_text());chains=json.loads(a.chain_proposal.read_text())['proposals']
    neighbors=json.loads(a.neighbor_records.read_text());config=json.loads(a.config.read_text())
    sources=[]
    for rel in config['sources']['av2_logs']:
        path=next((a.data_root/rel/'map').glob('log_map_archive*.json'))
        sources.append((str(path.relative_to(a.data_root)),json.loads(path.read_text())))
    sources,_=register_sources(sources,dict(registrations=whole['proposed_registrations']))
    lanes={int(k):v for _,data in sources for k,v in data['lane_segments'].items()}
    parent={k:k for k in lanes}
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    def select(keys,owner):
        targets={root(k)for k in keys};selected=root(owner)
        for target in targets:
            if lanes[target]['lane_type']!=lanes[selected]['lane_type']:
                raise ValueError('Source ownership crosses travel modes')
            parent[target]=selected
    if condition['accept_whole_lane_components']:
        for k,v in whole['proposed_geometry_aliases'].items():select([int(k),v],v)
    decisions=[]
    for i in condition['accepted_chain_indices']:
        row=chains[i];select([row['long_lane_id'],*row['source_path']],row['long_lane_id'])
        decisions.append(dict(kind='complete_observed_chain',proposal_index=i,**row))
    for i in condition['accepted_neighbor_identity_indices']:
        row=neighbors[i];select(row['candidates'],row['owner_observed_neighbor'])
        decisions.append(dict(kind='reviewed_adjacent_source_identity',**row))
    aliases={str(k):root(k)for k in lanes if root(k)!=k}
    ownership=dict(source_aliases=aliases,reason=condition['reason'],evidence=condition['evidence'],
                   source_decisions=decisions,neighbor_choices={},
                   source_id_choices=condition.get('source_id_choices',{}))
    preview=combine_exact_observations(sources,ownership_review=ownership)
    if condition.get('conflicting_neighbor_policy')=='selected_geometry_source':
        for row in preview['neighbor_conflicts']:
            key=row['canonical_lane_id'];field=row['field']
            original=lanes[key][field]
            canonical=preview['aliases'].get(str(original),original) if original is not None else None
            if original is not None and canonical not in row['candidate_ids']:continue
            ownership['neighbor_choices'].setdefault(str(key),{})[field]=dict(source_neighbor_id=original,
                reason='Use the adjacency recorded with the exact selected geometry source; alternate epoch/segmentation observations remain in the source review.')
    result=combine_exact_observations(sources,ownership_review=ownership)
    positions={lane['id']:np.stack([boundary(lane,'left_lane_boundary')[[0,-1]],boundary(lane,'right_lane_boundary')[[0,-1]]]).mean(0)for lane in result['lanes']}
    joins=[]
    for lane in result['lanes']:
        for nxt in lane['successors']:
            if nxt in positions:
                delta=positions[lane['id']][1]-positions[nxt][0]
                joins.append(dict(source=lane['id'],target=nxt,xy_m=float(np.linalg.norm(delta[:2])),z_m=float(abs(delta[2]))))
    report=dict(registrations=whole['proposed_registrations'],ownership_review=ownership,
                conditions=str(a.conditions),scope=condition['scope'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    a.output.with_suffix('.preview.json').write_text(json.dumps(dict(
        **{k:v for k,v in result.items()if k!='lanes'},lane_count=len(result['lanes']),
        source_counts=dict(Counter(Path(o['source']).parts[-3]for lane in result['lanes']for o in lane['source_observations'])),
        source_join_measurements=joins),indent=2)+'\n')
    print(json.dumps(dict(lanes=len(result['lanes']),aliases=len(aliases),
         same_id_conflicts=result['source_id_geometry_conflicts'],neighbor_conflicts=result['neighbor_conflicts'],
         recorded_neighbor_choices=len(result['source_ownership_neighbor_resolutions']),
         source_join_count=len(joins),source_join_xy_p95=float(np.percentile([r['xy_m']for r in joins],95)),
         source_join_xy_max=max(r['xy_m']for r in joins),source_joins_over_two_m=sum(r['xy_m']>2 for r in joins)),indent=2))


if __name__=='__main__':
    main()
