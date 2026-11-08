"""Expand selected coarse-road fans into source-connected movement paths."""
import argparse,json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np


def expand(scene,conditions):
    result=deepcopy(scene);lanes={l['id']:l for l in result['fusion_map']['lanes']}
    records=[]
    for decision in conditions['expansions']:
        before=deepcopy(lanes);paths=decision['paths'];duplicates=decision.get('realized_duplicates',[])
        retired={k for row in paths for k in row['source_path']}|{row['lane_id']for row in duplicates}
        if any(before[k]['source']=='av2'for k in retired):raise ValueError('Observed AV2 cannot be expanded')
        predecessors=defaultdict(list)
        for lane in before.values():
            for target in lane['successors']:predecessors[target].append(lane['id'])
        starts=defaultdict(list);created={}
        for row in paths:
            chain=row['source_path'];source=[before[k]for k in chain]
            if any(b not in before[a]['successors']for a,b in zip(chain,chain[1:])):
                raise ValueError('Declared movement is not an existing ordered source path')
            if any(p in retired for p in predecessors[chain[0]]) or any(t in retired for t in source[-1]['successors']):
                raise ValueError('Movement must reach both outside boundaries of the expanded fan')
            permissions=set(source[0]['allowed_classes'])
            for lane in source[1:]:permissions.intersection_update(lane['allowed_classes'])
            if not permissions:raise ValueError('Movement has no shared source permission')
            points=[]
            for lane in source:
                p=np.asarray(lane['centerline'])
                if points and np.linalg.norm(points[-1][-1]-p[0])>.01:raise ValueError('Source path has an unresolved geometric join')
                points.append(p if not points else p[1:])
            first_width=source[0].get('end_widths_m',[source[0]['width_m']]*2)[0]
            last_width=source[-1].get('end_widths_m',[source[-1]['width_m']]*2)[1]
            key=row['id']
            if key in before or key in created:raise ValueError('Movement ID already exists')
            created[key]=dict(id=key,source='inferred_source_movement',internal=True,is_intersection=True,
                edge_id=decision['id'],width_m=max(first_width,last_width),end_widths_m=[first_width,last_width],
                speed_mps=min(l['speed_mps']for l in source),allowed_classes=sorted(permissions),
                successors=list(source[-1]['successors']),centerline=np.vstack(points).tolist(),
                height_source=source[0].get('height_source','retained source profile'),
                inferred_movement_source_path=chain,source_generation_status='Complete source movement hypothesis; geometry generation and native review pending.')
            starts[chain[0]].append(key)
        # Compare the expanded movement set with the original directed paths and permissions.
        covered={(tuple(row['source_path']))for row in paths}
        actual=set()
        def walk(chain):
            nexts=[k for k in before[chain[-1]]['successors']if k in retired and k not in {d['lane_id']for d in duplicates}]
            if not nexts:actual.add(tuple(chain));return
            for key in nexts:
                if key in chain:raise ValueError('A cyclic inferred fan requires separate source review')
                walk(chain+[key])
        for first in starts:walk([first])
        if actual!=covered:raise ValueError('Declared expansion differs from complete original movement set')
        duplicate_records=[]
        for row in duplicates:
            old=before[row['lane_id']];route=row['realized_path']
            candidates={**before,**created}
            if any(b not in candidates[a]['successors']for a,b in zip(route,route[1:])):
                raise ValueError('Duplicate realization is not a complete path')
            if route[-1]not in old['successors']:raise ValueError('Duplicate realization has another destination')
            first=created[route[0]];origin=next(p['source_path'][0]for p in paths if p['id']==route[0])
            if set(predecessors[row['lane_id']])!=set(predecessors[origin]):raise ValueError('Duplicate has another approach')
            modes=set(first['allowed_classes'])
            for key in route[1:]:modes.intersection_update(candidates[key]['allowed_classes'])
            if not set(old['allowed_classes'])<=modes:raise ValueError('Duplicate realization loses a permission')
            starts[row['lane_id']].append(route[0]);duplicate_records.append(row)
        for key in retired:
            for pred in predecessors[key]:
                if pred not in retired and key not in starts:raise ValueError('An outside branch would be lost')
        for key,lane in list(lanes.items()):
            if key in retired:del lanes[key];continue
            lane['successors']=list(dict.fromkeys(v for target in lane['successors']for v in(starts[target]if target in retired else[target])))
        lanes.update(created)
        if any(t not in lanes for l in lanes.values()for t in l['successors']):raise ValueError('Expanded graph has a missing successor')
        changed_observed=[k for k,l in before.items()if l['source']=='av2'and lanes[k]!=l]
        if changed_observed:raise ValueError('Observed AV2 changed')
        record=dict(decision=decision,original_inferred_lanes=[before[k]for k in sorted(retired)],
            created_movement_ids=sorted(created),complete_original_movements=len(actual),complete_new_movements=len(created),
            exact_local_movement_set_preserved=True,changed_av2=[],realized_duplicates=duplicate_records,
            accepted_geometry=False,reason='Source topology refinement only; every complete original route and permission retained. Independent generated movement geometry remains required.')
        records.append(record)
    result['fusion_map']['lanes']=list(lanes.values())
    result['fusion_map'].setdefault('inferred_movement_expansions',[]).extend(records)
    review=dict(expansions=[{k:v for k,v in r.items()if k not in ['original_inferred_lanes','decision']}for r in records],
        original_lanes=len(scene['fusion_map']['lanes']),current_lanes=len(lanes),
        source_records_preserved=True,source_geometry_accepted=False)
    return result,review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','conditions','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();scene,review=expand(json.loads(a.scene.read_text()),json.loads(a.conditions.read_text()))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(scene,separators=(',',':'))+'\n')
    (a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(review,indent=2))

if __name__=='__main__':main()
