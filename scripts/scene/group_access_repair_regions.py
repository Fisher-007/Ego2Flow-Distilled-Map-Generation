"""Group inferred-road intervals connected by shared ports for joint refinement."""
import argparse
from collections import defaultdict
import heapq
import json
from pathlib import Path
import numpy as np
from shapely.geometry import LineString


def group_regions(scene,findings,margin):
    lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    mutable={k:l for k,l in lanes.items() if l['source']!='av2'}
    lengths={k:LineString(np.asarray(l['centerline'])[:,:2]).length for k,l in mutable.items()}
    links=defaultdict(set)
    for k,lane in lanes.items():
        for target in lane['successors']:
            if target not in lanes:continue
            links[k,1].add((target,0));links[target,0].add((k,1))
    seeds=[r['seam_id'] for r in findings if r['needs_review']]
    expanded=[];interval_index=defaultdict(list)
    for i,seed in enumerate(seeds):
        intervals=defaultdict(list);intervals[seed].append((0.,lengths[seed]))
        queue=[(0.,seed,0),(0.,seed,1)];visited={};fixed=set()
        while queue:
            distance,key,end=heapq.heappop(queue);port=(key,end)
            if distance>=visited.get(port,float('inf')):continue
            visited[port]=distance
            if key not in mutable:fixed.add(port);continue
            remaining=max(0.,margin-distance);length=lengths[key]
            interval=(max(0.,length-remaining),length) if end else (0.,min(length,remaining))
            if interval[1]-interval[0]>.001:intervals[key].append(interval)
            if remaining>=length:heapq.heappush(queue,(distance+length,key,1-end))
            for other,side in links[port]:heapq.heappush(queue,(distance,other,side))
        merged={}
        for key,parts in intervals.items():
            result=[]
            for low,high in sorted(parts):
                if result and low<=result[-1][1]+.001:result[-1][1]=max(result[-1][1],high)
                else:result.append([low,high])
            merged[key]=result
            for low,high in result:interval_index[key].append((low,high,i))
        expanded.append(dict(seed=seed,intervals=merged,fixed_av2_ports=sorted(fixed)))
    parent=list(range(len(seeds)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for key,rows in interval_index.items():
        rows=sorted(rows);active=[]
        for low,high,i in rows:
            active=[r for r in active if r[1]>=low-.001]
            for _,_,j in active:parent[root(i)]=root(j)
            active.append((low,high,i))
    groups=defaultdict(list)
    for i,row in enumerate(expanded):groups[root(i)].append(row)
    output=[]
    from map_generation.access_fusion import section
    for index,rows in enumerate(sorted(groups.values(),key=lambda rs:rs[0]['seed'])):
        intervals=defaultdict(list);ports=set()
        for row in rows:
            ports.update(tuple(p) for p in row['fixed_av2_ports'])
            for k,values in row['intervals'].items():intervals[k].extend(values)
        merged=[];points=[]
        for key,parts in sorted(intervals.items()):
            result=[]
            for low,high in sorted(parts):
                if result and low<=result[-1][1]+.001:result[-1][1]=max(result[-1][1],high)
                else:result.append([low,high])
            for low,high in result:
                points.extend(section(lanes[key]['centerline'],low,high))
                merged.append(dict(lane_id=key,start_m=low,end_m=high,length_m=lengths[key]))
        xy=np.asarray(points)[:,:2];box=[*xy.min(axis=0),*xy.max(axis=0)]
        output.append(dict(id=f'access_region_{index:03d}',seed_seams=[r['seed'] for r in rows],
            intervals=merged,fixed_av2_ports=sorted(ports),bbox_local_xy=list(map(float,box)),
            extent_xy_m=(xy.max(axis=0)-xy.min(axis=0)).tolist(),
            mutable_lane_count=len(intervals),mutable_length_m=sum(r['end_m']-r['start_m'] for r in merged)))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('scene','geometry-review','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--margin',type=float,default=20.)
    a=p.parse_args();scene=json.loads(a.scene.read_text());findings=json.loads(a.geometry_review.read_text())['findings']
    groups=group_regions(scene,findings,a.margin)
    result=dict(source_scene=str(a.scene),geometry_review=str(a.geometry_review),margin_m=a.margin,regions=groups,
        status='agent_region_review_pending',
        scope='Coupled inference regions; source topology, vehicle permissions and AV2 geometry remain hard constraints. No source lane is removed by this proposal.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(dict(regions=len(groups),seed_seams=sum(len(r['seed_seams']) for r in groups),
        maximum_mutable_lanes=max(r['mutable_lane_count'] for r in groups),
        maximum_extent_m=max(max(r['extent_xy_m']) for r in groups),
        large_regions=[dict(id=r['id'],lanes=r['mutable_lane_count'],extent=r['extent_xy_m']) for r in groups if max(r['extent_xy_m'])>90]),flush=True)


if __name__=='__main__':main()
