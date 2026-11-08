"""Extend an existing reviewed AV2 ownership with explicit whole-lane/chain choices."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from map_generation.city_registration import register_sources
from map_generation.city_sources import combine_exact_observations
from scripts.data.propose_city_source_alignment import boundary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data-root','base-review','conditions','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text());base=json.loads(a.base_review.read_text());condition=json.loads(a.conditions.read_text())
    sources=[]
    for rel in config['sources']['av2_logs']:
        path=next((a.data_root/rel/'map').glob('log_map_archive*.json'))
        sources.append((str(path.relative_to(a.data_root)),json.loads(path.read_text())))
    sources,_=register_sources(sources,base)
    maps={Path(path).parts[-3]:data['lane_segments'] for path,data in sources}
    lanes={int(k):v for data in maps.values()for k,v in data.items()}
    logs=defaultdict(set)
    for log,data in maps.items():
        for k in data:logs[int(k)].add(log)
    parent={k:k for k in lanes}
    def root(k):
        if parent[k]!=k:parent[k]=root(parent[k])
        return parent[k]
    for k,v in base['ownership_review']['source_aliases'].items():parent[int(k)]=int(v)
    decisions=list(base['ownership_review'].get('source_decisions',[]));rejected=[]
    def select(keys,owner):
        selected=root(owner);targets={root(k)for k in keys}
        if any(lanes[k]['lane_type']!=lanes[selected]['lane_type']for k in targets):
            raise ValueError('Reviewed identity crosses travel modes')
        for k in targets:parent[k]=selected
    for row in condition['complete_chains']:
        data=maps[row['path_log']];path=row['source_path']
        for left,right in zip(path[:-1],path[1:]):
            if [x for x in data[str(left)]['successors']if str(x)in data]!=[right] or [x for x in data[str(right)]['predecessors']if str(x)in data]!=[left]:
                raise ValueError('Chain contains an observed intermediate branch')
        select([row['long_lane_id'],*path],row['long_lane_id'])
        decisions.append(dict(kind='reviewed_complete_chain_extension',**row))
    for row in condition['whole_identities']:
        keys=row['members'];roots={root(k)for k in keys}
        if len(roots)>1:
            component_logs=[set().union(*(logs[k]for k in lanes if root(k)==target))for target in roots]
            if any(x&y for i,x in enumerate(component_logs)for y in component_logs[i+1:]):
                rejected.append(dict(choice=row,reason='Distinct current source components co-occur in a log; separate path evidence required.'));continue
            select(keys,row['owner'])
        decisions.append(dict(kind='reviewed_whole_identity_extension',**row))
    aliases={str(k):root(k)for k in lanes if root(k)!=k}
    ownership=dict(source_aliases=aliases,reason=condition['reason'],evidence=condition['evidence'],source_decisions=decisions,neighbor_choices={})
    preview=combine_exact_observations(sources,ownership_review=ownership)
    for row in preview['neighbor_conflicts']:
        key=row['canonical_lane_id'];field=row['field'];original=lanes[key][field]
        canonical=preview['aliases'].get(str(original),original)if original is not None else None
        if original is not None and canonical not in row['candidate_ids']:continue
        ownership['neighbor_choices'].setdefault(str(key),{})[field]=dict(source_neighbor_id=original,reason='Keep adjacency recorded by the exact selected geometry owner; all alternative observations remain recorded.')
    result=combine_exact_observations(sources,ownership_review=ownership)
    positions={lane['id']:np.stack([boundary(lane,'left_lane_boundary')[[0,-1]],boundary(lane,'right_lane_boundary')[[0,-1]]]).mean(0)for lane in result['lanes']}
    joins=[]
    for lane in result['lanes']:
        for nxt in lane['successors']:
            if nxt in positions:
                delta=positions[lane['id']][1]-positions[nxt][0]
                joins.append(dict(source=lane['id'],target=nxt,xy_m=float(np.linalg.norm(delta[:2])),z_m=float(abs(delta[2]))))
    report=dict(base,ownership_review=ownership,conditions=str(a.conditions),base_review=str(a.base_review),scope=condition['scope'])
    review=dict(**{k:v for k,v in result.items()if k!='lanes'},rejected_extensions=rejected,lane_count=len(result['lanes']),source_join_measurements=joins)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    a.output.with_suffix('.preview.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(dict(lanes=len(result['lanes']),aliases=len(aliases),rejected_extensions=len(rejected),same_id_conflicts=result['source_id_geometry_conflicts'],neighbor_conflicts=result['neighbor_conflicts'],source_joins=len(joins),xy_p95=float(np.percentile([r['xy_m']for r in joins],95)),xy_max=max(r['xy_m']for r in joins),z_max=max(r['z_m']for r in joins),joins_over_two_m=sum(r['xy_m']>2 for r in joins)),indent=2))


if __name__=='__main__':main()
