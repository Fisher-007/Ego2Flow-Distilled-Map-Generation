"""Expose selected AV2 interface collars as generation inputs."""
import argparse,json,heapq
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from map_generation.access_fusion import section


def merge_ranges(parts):
    result=[]
    for lo,hi in sorted(parts):
        if hi-lo<1e-6:continue
        if result and lo<=result[-1][1]+1e-6:result[-1][1]=max(result[-1][1],hi)
        else:result.append([float(lo),float(hi)])
    return result


def expose(scene,conditions,distance):
    original={l['id']:l for l in scene['fusion_map']['lanes']}
    lengths={k:LineString(np.asarray(l['centerline'])[:,:2]).length for k,l in original.items()}
    links=defaultdict(set)
    for k,lane in original.items():
        for target in lane['successors']:
            if target in original:
                links[k,1].add((target,0));links[target,0].add((k,1))
    intervals=defaultdict(list);dependencies=defaultdict(set);expansions=[]
    for decision in conditions['decisions']:
        if decision['decision']!='infer_local_source_conflict_interface':continue
        queue=[(0.,p['lane_id'],p['end'])for p in decision['ports']];heapq.heapify(queue);visited={};rows=[]
        while queue:
            used,key,end=heapq.heappop(queue)
            if used>=visited.get((key,end),float('inf'))or used>distance:continue
            visited[key,end]=used
            if original[key]['source']!='av2':continue
            length=lengths[key];remaining=distance-used
            lo,hi=(max(0.,length-remaining),length)if end else (0.,min(length,remaining))
            if hi-lo>1e-6:
                intervals[key].append((lo,hi));dependencies[key].add(decision['component_id'])
                rows.append(dict(lane_id=key,start_m=lo,end_m=hi))
            if remaining>=length:heapq.heappush(queue,(used+length,key,1-end))
            for other,side in links[key,end]:heapq.heappush(queue,(used,other,side))
        expansions.append(dict(component_id=decision['component_id'],source_intervals=rows))
    pieces={};report=[];changed=[]
    archived={lane['id']:deepcopy(lane)for lane in scene['fusion_map'].get('source_conflict_original_lanes',[])}
    occupied=set(original)
    snapshots={}
    for key in sorted(intervals):
        snapshot=key
        if snapshot in archived and archived[snapshot]!=original[key]:
            serial=1
            while key+':source_conflict_snapshot:'+str(serial)in archived:serial+=1
            snapshot=key+':source_conflict_snapshot:'+str(serial)
        record=deepcopy(original[key])
        if snapshot!=key:
            record['id']=snapshot
            record['source_conflict_snapshot_lane_id']=key
        archived[snapshot]=record;snapshots[key]=snapshot
    for key,parts in intervals.items():
        ranges=merge_ranges(parts);length=lengths[key];cuts=sorted({0.,length,*[v for pair in ranges for v in pair]})
        # Cut endpoints at observed source precision, without introducing tiny
        # residual pieces at a nearly coincident source port.
        cuts=[0.]+[v for v in cuts[1:-1]if v>.001 and length-v>.001]+[length]
        ids=[key];serial=1
        while len(ids)<len(cuts)-1:
            candidate=key+':source_conflict:'+str(serial);serial+=1
            if candidate in occupied:continue
            ids.append(candidate);occupied.add(candidate)
        entries=[];lane=original[key]
        for n,(lo,hi)in enumerate(zip(cuts[:-1],cuts[1:])):
            mutable=any(a-1e-6<=(lo+hi)/2<=b+1e-6 for a,b in ranges)
            part=deepcopy(lane);part['id']=ids[n]
            if lo>1e-8 or hi<length-1e-8:
                part['centerline']=section(lane['centerline'],lo,hi).tolist()
                for field in ['left_boundary','right_boundary']:
                    bl=LineString(np.asarray(lane[field])[:,:2]).length
                    part[field]=section(lane[field],lo/length*bl,hi/length*bl).tolist()
                part.pop('parametric_xy_segments',None);part.pop('boundary_shape',None)
            part['source_conflict_parent_lane_id']=snapshots[key];part['source_conflict_parent_interval_m']=[lo,hi]
            part['successors']=ids[n+1:n+2]or list(lane['successors'])
            if mutable:
                part['source']='source_conflict_transition';part['original_source']='av2'
                part['source_conflict_components']=sorted(dependencies[key]);part['generation_eligible']=True
                part['source_geometry_status']='Observed conflicting interface retained as model input; inferred geometry has not yet been sampled or applied.'
                changed.append(ids[n])
            entries.append(part);report.append(dict(id=ids[n],parent=snapshots[key],input_lane_id=key,start_m=lo,end_m=hi,source=part['source'],inferred=mutable))
        pieces[key]=entries
    output=[]
    for key,lane in original.items():output.extend(pieces.get(key,[deepcopy(lane)]))
    for lane in output:lane['successors']=[pieces[t][0]['id']if t in pieces else t for t in lane['successors']]
    # Internal serial piece edges already use their exact new identifiers.
    out=deepcopy(scene);out['fusion_map']['lanes']=output
    out['fusion_map']['source_conflict_original_lanes']=[archived[k]for k in sorted(archived)]
    result=dict(collar_distance_m=distance,reviewed_conflict_components=len(expansions),affected_original_lanes=len(pieces),
        inferred_pieces=len(changed),retained_source_pieces=sum(not r['inferred']for r in report),original_fused_lanes=len(original),
        resulting_fused_lanes=len(output),inferred_lane_ids=changed,pieces=report,expansions=expansions,
        source_conditions=conditions,scope='Exact source interval exposure only. Original source observations, source topology, vehicle permissions and unselected intervals are preserved. Actual model sampling and constrained projection still required.')
    history=deepcopy(scene['fusion_map'].get('source_conflict_exposure_history',[]))
    if not history and scene['fusion_map'].get('source_conflict_exposure'):history.append(deepcopy(scene['fusion_map']['source_conflict_exposure']))
    history.append(result);out['fusion_map']['source_conflict_exposure_history']=history
    out['fusion_map']['source_conflict_exposure']=result
    return out,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['scene','conditions','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--distance',type=float,default=8.);a=p.parse_args()
    s=json.loads(a.scene.read_text());c=json.loads(a.conditions.read_text());s,r=expose(s,c,a.distance)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(s,separators=(',',':'),allow_nan=False)+'\n')
    (a.output/'exposure.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items()if k not in ['inferred_lane_ids','pieces','expansions','source_conditions']},indent=2),flush=True)
if __name__=='__main__':main()
