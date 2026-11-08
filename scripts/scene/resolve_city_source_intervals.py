"""Resolve Agent-confirmed partial AV2 identities as exact source intervals."""
import argparse,json
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import numpy as np
from shapely.geometry import LineString,Point
from map_generation.access_fusion import section
from map_generation.fusion import station_xyz,fuse_scene

def realize(scene,condition):
    scene=deepcopy(scene)
    original=scene['lanes']+scene.get('av2_context',[])
    lanes={x['id']:x for x in original}
    lines={k:LineString(np.asarray(x['centerline'])[:,:2])for k,x in lanes.items()}
    lengths={k:x.length for k,x in lines.items()}
    rank={k:(lengths[k],k)for k in lanes}
    transfers=defaultdict(list);rejected=[];seen=set()
    lateral=float(condition['maximum_centerline_difference_m'])
    for row in condition['pairs']:
        first,second=row['first'],row['second']
        if first==second:continue
        if first not in lanes or second not in lanes:
            rejected.append(dict(pair=row,reason='Source lane absent from declared fine extent'));continue
        if tuple(sorted([first,second]))in seen:continue
        seen.add(tuple(sorted([first,second])))
        if lanes[first]['lane_type']!=lanes[second]['lane_type']:
            raise ValueError('Partial ownership cannot cross travel modes')
        owner,loser=sorted([first,second],key=lambda k:rank[k],reverse=True)
        endpoints=np.asarray(lanes[owner]['centerline'])[[0,-1],:2]
        lo,hi=[float(lines[loser].project(Point(p)))for p in endpoints]
        if hi-lo<.001:
            rejected.append(dict(pair=row,reason='No positive directed common station interval'));continue
        samples=np.linspace(lo,hi,max(3,int((hi-lo)/2)+1))
        points=np.array([station_xyz(lanes[loser]['centerline'],s)for s in samples])
        stations=np.array([lines[owner].project(Point(p[:2]))for p in points])
        distances=np.array([lines[owner].distance(Point(p[:2]))for p in points])
        if np.min(np.diff(stations))<-.001 or max(distances)>lateral or stations[-1]-stations[0]<.001:
            rejected.append(dict(pair=row,reason='Reviewed interval correspondence needs more geometry evidence',maximum_centerline_difference_m=float(max(distances)),owner_station_progress_m=float(stations[-1]-stations[0])));continue
        transfers[loser].append(dict(owner=owner,interval=[lo,hi],maximum_centerline_difference_m=float(max(distances)),source_pair=row))
    policies={}
    for key in lanes:
        cuts=sorted({0.,lengths[key],*(s for r in transfers[key]for s in r['interval'])})
        intervals=[]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            if hi-lo<1e-7:continue
            mid=(lo+hi)/2;active=[r for r in transfers[key]if r['interval'][0]-1e-7<=mid<=r['interval'][1]+1e-7]
            owner=max(active,key=lambda r:rank[r['owner']])['owner']if active else key
            intervals.append((lo,hi,owner))
        policies[key]=intervals
    def project(key,owner,station):
        point=station_xyz(lanes[key]['centerline'],station)
        return float(lines[owner].project(Point(point[:2])))
    def resolve(key,lo,hi):
        out=[]
        for a,b,owner in policies[key]:
            start,end=max(a,lo),min(b,hi)
            if end-start<1e-7:continue
            if owner==key:out.append((key,start,end))
            else:
                if rank[owner]<=rank[key]:raise ValueError('Source ownership must be acyclic')
                sa,sb=project(key,owner,start),project(key,owner,end)
                if sb-sa<1e-7:raise ValueError('Partial identity collapsed a positive source interval')
                out.extend(resolve(owner,sa,sb))
        return out
    mappings={k:resolve(k,0.,lengths[k])for k in lanes}
    cuts=defaultdict(list)
    for rows in mappings.values():
        for key,a,b in rows:cuts[key].extend([a,b])
    canonical={};segments=defaultdict(list)
    for key,values in cuts.items():
        stations=[]
        for s in sorted(values):
            if not stations or s-stations[-1]>1e-6:stations.append(s)
        for a,b in zip(stations[:-1],stations[1:]):
            if b-a<1e-6:continue
            mid=(a+b)/2
            if not any(lo-1e-6<=mid<=hi+1e-6 and owner==key for lo,hi,owner in policies[key]):continue
            n=len(segments[key]);ident=key if n==0 else key+':source_interval:'+str(n)
            lane=deepcopy(lanes[key]);lane['id']=ident;lane['centerline']=section(lane['centerline'],a,b).tolist()
            for field in ('left_boundary','right_boundary'):
                bl=LineString(np.asarray(lane[field])[:,:2]).length
                lane[field]=section(lane[field],a/lengths[key]*bl,b/lengths[key]*bl).tolist()
            lane['source_parent_lane_id']=key;lane['source_parent_interval_m']=[a,b]
            lane['source_interval_owner']=dict(source_lane_id=lanes[key].get('source_lane_id'),original_source_id=key,interval_m=[a,b],reason=condition['reason'])
            lane['successors']=[];lane['source_predecessors']=[];lane['internal_predecessors']=[]
            canonical[ident]=lane;segments[key].append((a,b,ident))
    paths={}
    for key,rows in mappings.items():
        path=[]
        for owner,a,b in rows:
            for lo,hi,ident in segments[owner]:
                if lo>=a-2e-6 and hi<=b+2e-6:
                    if not path or path[-1]!=ident:path.append(ident)
        if not path:raise ValueError('Original observed road has no retained source interval: '+key)
        paths[key]=path
    edges=set();source_edges=0;contracted_edges=[]
    for key,path in paths.items():
        edges.update(zip(path[:-1],path[1:]))
        for target in lanes[key]['successors']:
            if target not in paths:continue
            source_edges+=1
            first,last=path[-1],paths[target][0]
            if first in paths[target]:contracted_edges.append(dict(source=key,target=target,shared_interval=first,reason='Source successor continues through an already shared directed interval; its forward remainder is retained in the target path'))
            else:edges.add((first,last))
    compactions=[]
    minimum=float(condition.get('minimum_serial_interval_m',.1))
    port_tolerance=float(condition.get('coincident_port_tolerance_m',0.))
    while True:
        incoming=defaultdict(set);outgoing=defaultdict(set)
        for a,b in edges:outgoing[a].add(b);incoming[b].add(a)
        found=None
        for key,lane in canonical.items():
            lo,hi=lane['source_parent_interval_m']
            if hi-lo>=minimum:continue
            alternatives=[(a,key)for a in incoming[key]]+[(key,b)for b in outgoing[key]]
            for a,b in alternatives:
                la,lb=canonical[a],canonical[b]
                if (la['source_parent_lane_id']==lb['source_parent_lane_id'] and
                    abs(la['source_parent_interval_m'][1]-lb['source_parent_interval_m'][0])<2e-6 and
                    ((outgoing[a]=={b} and incoming[b]=={a}) or
                     (hi-lo<=port_tolerance and ((key==b and incoming[b]=={a}) or (key==a and outgoing[a]=={b}))))):
                    found=(a,b);break
            if found:break
        if not found:break
        a,b=found;la,lb=canonical[a],canonical[b];parent=la['source_parent_lane_id']
        tiny_length=canonical[key]['source_parent_interval_m'][1]-canonical[key]['source_parent_interval_m'][0]
        lo,hi=la['source_parent_interval_m'][0],lb['source_parent_interval_m'][1]
        la['source_parent_interval_m']=[lo,hi];la['source_interval_owner']['interval_m']=[lo,hi]
        for field in ('centerline','left_boundary','right_boundary'):
            total=LineString(np.asarray(lanes[parent][field])[:,:2]).length
            la[field]=section(lanes[parent][field],lo/lengths[parent]*total,hi/lengths[parent]*total).tolist()
        moved_port=tiny_length if (outgoing[a]!={b} or incoming[b]!={a}) else 0.
        edges={(a if x==b else x,a if y==b else y)for x,y in edges if (x,y)!=(a,b)}
        for key,path in paths.items():
            remapped=[a if x==b else x for x in path]
            paths[key]=[x for i,x in enumerate(remapped)if i==0 or x!=remapped[i-1]]
        del canonical[b];compactions.append(dict(first=a,second=b,parent=parent,retained_interval_m=[lo,hi],maximum_port_station_reinterpretation_m=moved_port,reason='Adjacent slices retain the complete exact source boundary interval; serial joins are exact and nearly coincident branch ports may share a station only within the declared source-precision tolerance'))
    segments=defaultdict(list)
    for key,lane in canonical.items():
        lo,hi=lane['source_parent_interval_m'];segments[lane['source_parent_lane_id']].append((lo,hi,key))
    for rows in segments.values():rows.sort()
    for first,last in sorted(edges):
        canonical[first]['successors'].append(last);canonical[last]['source_predecessors'].append(first);canonical[last]['internal_predecessors'].append(first)
    def at_station(key,s):
        for a,b,owner in policies[key]:
            if a-1e-6<=s<=b+1e-6:
                if owner!=key:return at_station(owner,project(key,owner,s))
                rows=segments[key]
                return min(rows,key=lambda row:abs((row[0]+row[1])/2-s))[2]
        return paths[key][-1]
    neighbor_changes=[]
    for ident,lane in canonical.items():
        key=lane['source_parent_lane_id'];point=np.asarray(lane['centerline'])[len(lane['centerline'])//2,:2]
        for field in ('left_neighbor','right_neighbor'):
            neighbor=lanes[key].get(field)
            if neighbor in paths:
                new=at_station(neighbor,lines[neighbor].project(Point(point)))
                lane[field]=None if new==ident else new
                if new!=neighbor:neighbor_changes.append(dict(lane=ident,field=field,source_neighbor=neighbor,selected_interval=lane[field]))
        lane['source_interval_observations']=[]
    for source,path in paths.items():
        for ident in path:
            canonical[ident]['source_interval_observations'].append(dict(source_lane=source,source_lane_id=lanes[source].get('source_lane_id'),source_observations=lanes[source].get('source_observations',[])))
    original_ids=set(lanes)
    preserved=[dict(source=k,length_m=lengths[k],path=paths[k],resolved_intervals=[dict(owner=o,interval_m=[a,b])for o,a,b in mappings[k]])for k in lanes if len(paths[k])!=1 or paths[k][0]!=k or transfers[k]]
    join_rows=[]
    for a,b in edges:
        delta=np.asarray(canonical[a]['centerline'])[-1]-np.asarray(canonical[b]['centerline'])[0]
        join_rows.append(dict(source=a,target=b,xy_m=float(np.linalg.norm(delta[:2])),z_m=float(abs(delta[2]))))
    report=dict(serial_interval_compactions=compactions,original_lanes=len(lanes),canonical_intervals=len(canonical),reviewed_pairs=len(condition['pairs']),accepted_transfers=sum(map(len,transfers.values())),rejected_pairs=rejected,
        affected_source_paths=preserved,source_successor_edges=source_edges,retained_interval_edges=len(edges),contracted_contiguous_source_edges=contracted_edges,neighbor_interval_changes=neighbor_changes,
        source_join_measurements=join_rows,minimum_interval_length_m=min(LineString(np.asarray(x['centerline'])[:,:2]).length for x in canonical.values()),
        scope='Agent-confirmed common road intervals use exact slices of one original observed boundary pair. Every original source trace and successor is mapped; exterior portions and intermediate source ports are retained. Different-epoch handoff gaps remain measured construction dependencies.')
    scene['source_interval_original_lanes']=original
    scene['lanes']=list(canonical.values());scene.pop('av2_context',None);scene['source_interval_ownership']=dict(conditions=condition,review=report)
    scene.pop('fusion_map',None)
    return scene,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('scene','conditions','config','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();s,r=realize(json.loads(a.scene.read_text()),json.loads(a.conditions.read_text()))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'source_scene.json').write_text(json.dumps(s,separators=(',',':'),allow_nan=False)+'\n')
    (a.output/'interval_review.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items()if k not in ['affected_source_paths','neighbor_interval_changes','source_join_measurements','contracted_contiguous_source_edges','serial_interval_compactions']},indent=2),flush=True)
    fused=fuse_scene(s,json.loads(a.config.read_text()))
    (a.output/'scene.json').write_text(json.dumps(fused,separators=(',',':'),allow_nan=False)+'\n')
    print(dict(fused_lanes=len(fused['fusion_map']['lanes']),matched_boundaries=len(fused['fusion_map']['boundary_matches'])),flush=True)
if __name__=='__main__':main()
