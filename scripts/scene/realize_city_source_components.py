"""Apply selected inferred-corridor ownership to existing AV2 paths."""
import argparse,json
from collections import deque
from copy import deepcopy
from pathlib import Path
import numpy as np
from map_generation.access_fusion import section
from shapely.geometry import LineString


def observed_source_ids(scene):
    originals={l['id']:l for l in scene['fusion_map'].get('source_conflict_original_lanes',[])}
    result=set()
    for lane in scene['fusion_map']['lanes']:
        if lane['source']=='av2':result.add(lane['id']);continue
        if lane['source']!='source_conflict_transition' or lane.get('original_source')!='av2':continue
        original=originals[lane['source_conflict_parent_lane_id']]
        points=section(original['centerline'],*lane['source_conflict_parent_interval_m'])
        actual=np.asarray(lane['centerline'])
        # Uncut observations retain their original sampling instead of adding
        # section endpoints; compare the original array in that case as well.
        candidates=[points,np.asarray(original['centerline'])]
        unchanged=any(p.shape==actual.shape and np.allclose(p,actual,atol=1e-8,rtol=0)for p in candidates)
        if unchanged and lane['allowed_classes']==original['allowed_classes']:result.add(lane['id'])
    return result

def corridor_modes(lanes,body,start,target):
    queue=deque([(start,set(lanes[start]['allowed_classes']))]);seen=set();modes=set()
    while queue:
        key,allowed=queue.popleft();state=(key,tuple(sorted(allowed)))
        if state in seen:continue
        seen.add(state)
        if key==target:modes.update(allowed);continue
        for other in lanes[key]['successors']:
            if other not in body and other!=target:continue
            common=allowed&set(lanes[other]['allowed_classes'])
            if common:queue.append((other,common))
    return modes

def realize(scene,conditions,evidence):
    before={l['id']:l for l in scene['fusion_map']['lanes']}
    observed=observed_source_ids(scene)
    components={r['region']:r for r in evidence}
    retired=set();records=[];alias_owners={};alias_observations=[];neighbor_changes=[]
    for decision in conditions['decisions']:
        component=components[decision['region']];body=set(component['body'])
        if body&retired:raise ValueError('Reviewed components overlap')
        if any(k in observed for k in body):raise ValueError('Cannot retire an observed lane')
        inputs=sorted(k for k,l in before.items()if k not in body and any(t in body for t in l['successors']))
        outputs=sorted({t for k in body for t in before[k]['successors']if t not in body})
        if inputs!=component['inputs']or outputs!=component['outputs']:
            raise ValueError('Source component evidence differs from the current scene')
        if decision['decision']=='canonicalize_covered_observed_prefixes':
            aliases=decision['aliases']
            if set(a['alias']for a in aliases)!=set(inputs):
                raise ValueError('Every duplicate source prefix must be explicitly accounted for')
            observations=[]
            for a in aliases:
                key=a['alias'];owner=a['canonical']
                if key not in observed or owner not in observed:
                    raise ValueError('Both prefix and canonical owner must be original observations')
                lane=before[key];canonical=before[owner]
                if set(lane['allowed_classes'])!=set(canonical['allowed_classes']):
                    raise ValueError('A source interval alias cannot change permissions')
                incoming=[k for k,l in before.items()if key in l['successors']]
                if not incoming or any(owner not in before[k]['successors']for k in incoming):
                    raise ValueError('Canonical incoming paths must already exist')
                if any(t not in body for t in lane['successors']):
                    raise ValueError('The source prefix has an independent observed continuation')
                fragment=section(canonical['centerline'],*a['interval_m'])
                points=np.asarray(lane['centerline'])
                difference=LineString(points[:,:2]).hausdorff_distance(LineString(fragment[:,:2]))
                if difference>a['maximum_xy_difference_m']:
                    raise ValueError('Declared canonical interval does not cover the observed prefix')
                if np.dot(points[-1,:2]-points[0,:2],fragment[-1,:2]-fragment[0,:2])<=0:
                    raise ValueError('Source interval directions disagree')
                observation=dict(alias=key,canonical=owner,canonical_interval_m=a['interval_m'],
                    maximum_xy_difference_m=float(difference),existing_predecessors=incoming,
                    original_observation=deepcopy(lane))
                observations.append(observation);alias_observations.append(observation)
                alias_owners[key]=owner;retired.add(key)
            records.append(dict(region=decision['region'],decision=decision,inputs=inputs,
                original_outputs=outputs,observed_interval_aliases=observations,
                source_observations=[deepcopy(before[k])for k in sorted(body)]))
            retired.update(body)
            continue
        if decision['decision']=='realize_inferred_component_through_existing_observed_paths':
            paths=[p['path']for p in component['paths']]
            if not all(paths):raise ValueError('An inferred destination lacks an observed replacement')
            corrected=False
        elif decision['decision']=='correct_contradicted_inferred_receiver':
            paths=decision['replacement_paths'];corrected=True
            if sorted(decision['contradicted_receivers'])!=outputs:
                raise ValueError('Every former receiver must be explicitly reviewed')
        else:raise ValueError('Unknown ownership decision')
        if set(p[0]for p in paths)!=set(inputs):raise ValueError('An original approach is unaccounted for')
        routes=[]
        for path in paths:
            if any(k not in observed for k in path):
                raise ValueError('Replacement must be an existing observed source path')
            for a,b in zip(path[:-1],path[1:]):
                if b not in before[a]['successors']:raise ValueError('Missing observed source continuation')
            required=set()
            for target in outputs if corrected else [path[-1]]:
                required.update(corridor_modes(before,body,path[0],target))
            available=set.intersection(*(set(before[k]['allowed_classes'])for k in path))
            if not required<=available:raise ValueError('Observed replacement loses a permitted source mode')
            routes.append(dict(path=path,original_permissions=sorted(required),observed_permissions=sorted(available)))
        records.append(dict(region=decision['region'],decision=decision,
            inputs=inputs,original_outputs=outputs,observed_routes=routes,
            source_observations=[deepcopy(before[k])for k in sorted(body)]))
        retired.update(body)
    out=deepcopy(scene);kept=[]
    for lane in out['fusion_map']['lanes']:
        if lane['id']in retired:continue
        for field in ['left_neighbor','right_neighbor']:
            if lane.get(field)in alias_owners:
                old=lane[field];lane[field]=alias_owners[old]
                neighbor_changes.append(dict(lane_id=lane['id'],field=field,alias=old,canonical=lane[field]))
            elif lane.get(field)in retired:raise ValueError('A retained lateral relation needs explicit ownership review')
        lane['successors']=[k for k in lane['successors']if k not in retired];kept.append(lane)
    out['fusion_map']['lanes']=kept
    access=out['fusion_map'].get('source_access_review',{})
    retired_access=[r for r in access.get('connections',[])if r['seam_id']in retired]
    access['connections']=[r for r in access.get('connections',[])if r['seam_id']not in retired]
    access.setdefault('retired_connections',[]).extend(retired_access)
    matches=out['fusion_map'].get('boundary_matches',[])
    retired_matches=[r for r in matches if r['seam_id']in retired]
    out['fusion_map']['boundary_matches']=[r for r in matches if r['seam_id']not in retired]
    for lane in kept:
        a=deepcopy(before[lane['id']]);b=deepcopy(lane);a.pop('successors');b.pop('successors')
        for change in neighbor_changes:
            if change['lane_id']==lane['id']:a[change['field']]=change['canonical']
        if a!=b:raise ValueError('Retained source properties changed')
    report=dict(conditions=conditions,records=records,retired_lane_ids=sorted(retired),
        retired_access_connections=retired_access,retired_boundary_matches=retired_matches,
        observed_interval_aliases=alias_observations,observed_alias_count=len(alias_observations),
        retired_inferred_count=sum(k not in observed for k in retired),neighbor_ownership_changes=neighbor_changes,
        lane_count_before=len(before),lane_count_after=len(kept),
        observed_lane_count=sum(l['id']in observed for l in kept),
        geometry_changes=0,permission_changes=0,added_physical_connections=0,
        source_component_count=len(records),
        interpretation='Physical inferred duplicates and source-proven wrong attachments are represented by existing observed paths. Original coarse and inferred observations remain here; a corrected receiver is not a newly asserted traffic prohibition.')
    out['fusion_map'].setdefault('observed_source_interval_aliases',[]).extend(alias_observations)
    out['fusion_map'].setdefault('source_component_ownership_reviews',[]).append(report)
    return out,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['scene','conditions','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();conditions=json.loads(a.conditions.read_text())
    evidence=json.loads(Path(conditions['component_evidence']).read_text())
    scene,report=realize(json.loads(a.scene.read_text()),conditions,evidence)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(scene,separators=(',',':'),allow_nan=False)+'\n')
    (a.output/'ownership_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items()if k not in ['conditions','records','retired_access_connections','retired_boundary_matches']},flush=True)

if __name__=='__main__':main()
