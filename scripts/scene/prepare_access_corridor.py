"""Extract a source-supported transition and approach as a local generation input."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from shapely.geometry import LineString
from map_generation.access_fusion import section
from map_generation.information_graph import add_node,add_edge


def source_region(lanes,pred,request,margin,reason):
    seam_id=request['seam_id']
    incoming=request['incoming'];chain=[];remaining=margin
    key=request['coarse_lane_id']
    while remaining>.01:
        lane=lanes[key]
        if lane['source']!='osm_sumo':break
        adjacent=lane['successors'] if incoming else pred[key]
        if len(adjacent)!=1:break
        length=LineString(np.asarray(lane['centerline'])[:,:2]).length
        # Leave a real outer boundary segment when the approach is long enough.
        if length>remaining+1.:
            start,end=(length-remaining,length) if incoming else (0.,remaining)
            chain.append(dict(lane_id=key,start_m=start,end_m=end,length_m=length));remaining=0.;break
        chain.append(dict(lane_id=key,start_m=0.,end_m=length,length_m=length));remaining-=length
        beyond=pred[key] if incoming else lane['successors']
        if len(beyond)!=1:break
        key=beyond[0]
    if not chain or remaining>.01:
        raise ValueError('No single source-supported approach contains the requested margin; review the junction as a group')
    ordered=list(reversed(chain)) if incoming else chain
    seam=lanes[seam_id];seam_length=LineString(np.asarray(seam['centerline'])[:,:2]).length
    entry=dict(lane_id=seam_id,start_m=0.,end_m=seam_length,length_m=seam_length)
    segments=ordered+[entry] if incoming else [entry]+ordered
    outer=chain[-1];outer_lane=deepcopy(lanes[outer['lane_id']])
    keep=(0.,outer['start_m']) if incoming else (outer['end_m'],outer['length_m'])
    return dict(seam_id=seam_id,incoming=incoming,segments=segments,outer_lane_id=outer_lane['id'],outer_retained_interval_m=keep,
        requested_margin_m=margin,reason=reason,original_successors={r['lane_id']:lanes[r['lane_id']]['successors'] for r in segments},
        scope='Temporary local generator conditioning; distribute accepted geometry back onto these original route IDs before map compilation.')


def prepare(scene,graph,seam_id,margin,reason):
    lanes={l['id']:l for l in scene['fusion_map']['lanes']};pred=defaultdict(list)
    for k,l in lanes.items():
        for t in l['successors']:pred[t].append(k)
    request=next(r for r in scene['fusion_map']['source_access_review']['connections'] if r['seam_id']==seam_id)
    manifest=source_region(lanes,pred,request,margin,reason)
    segments=manifest['segments'];incoming=manifest['incoming']
    seam=lanes[seam_id];outer_lane=deepcopy(lanes[manifest['outer_lane_id']])
    keep=manifest['outer_retained_interval_m']
    outer_lane['centerline']=section(outer_lane['centerline'],*keep).tolist()
    outer_lane.pop('boundary_shape',None)
    source=np.vstack([section(lanes[r['lane_id']]['centerline'],r['start_m'],r['end_m']) for r in segments])
    source=source[np.r_[True,np.linalg.norm(np.diff(source[:,:2],axis=0),axis=1)>1e-7]]
    proposal=deepcopy(seam);proposal['centerline']=source.tolist();proposal.pop('boundary_shape',None)
    if incoming:outer_lane['successors']=[seam_id]
    else:proposal['successors']=[outer_lane['id']]
    first,last=lanes[segments[0]['lane_id']],lanes[segments[-1]['lane_id']]
    proposal['end_widths_m']=[first['width_m'],last['width_m']]
    mutable={r['lane_id'] for r in segments if r['lane_id']!=seam_id}
    updated=dict(scene);updated['fusion_map']=dict(scene['fusion_map'])
    updated['fusion_map']['lanes']=[outer_lane if l['id']==outer_lane['id'] else proposal if l['id']==seam_id else l
        for l in scene['fusion_map']['lanes'] if l['id'] not in mutable or l['id']==outer_lane['id']]
    graph=deepcopy(graph);task='task:city_corridor_generation';node='corridor:'+seam_id
    removed=mutable-{outer_lane['id']}
    graph['nodes']=[n for n in graph['nodes'] if n['id'] not in removed]
    graph['edges']=[e for e in graph['edges'] if e['source'] not in removed and e['target'] not in removed]
    graph['decisions']=[d for d in graph['decisions'] if d['source_id'] not in removed]
    graph['edges']=[e for e in graph['edges'] if not(e['relation']=='successor' and e['source'] in mutable|{seam_id})]
    for l in (outer_lane,proposal):
        for t in l['successors']:add_edge(graph,l['id'],'successor',t,status='temporary_corridor',evidence=[node])
    add_node(graph,dict(id=task,kind='simulation_task'))
    add_node(graph,dict(id=node,kind='reviewed_geometry_region',content=manifest))
    add_edge(graph,seam_id,'depends_on',node)
    graph['decisions'].append(dict(source_id=seam_id,task=task,choice='enhance',agent='construction_agent',reason=reason,
        required_properties=['source_route_identity','observed_fine_geometry','continuous_boundaries','drivable_curvature'],evidence_ids=[node]))
    return updated,graph,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','graph','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seam',required=True);p.add_argument('--margin',type=float,default=20.);p.add_argument('--reason',required=True)
    a=p.parse_args();s,g,m=prepare(json.loads(a.scene.read_text()),json.loads(a.graph.read_text()),a.seam,a.margin,a.reason)
    a.output.mkdir(parents=True,exist_ok=True);m['source_scene']=str(a.scene.resolve());g['source_scene']=str((a.output/'scene.json').resolve())
    for name,value in [('scene',s),('graph',g),('corridor',m)]:
        with (a.output/(name+'.json')).open('w') as f:json.dump(value,f,separators=(',',':'))
    decision=dict(source_scene=g['source_scene'],road=dict(seam_id=a.seam,seed=0,start_t=9))
    (a.output/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(dict(seam=a.seam,mutable_source_lanes=len(m['segments'])-1,margin_m=a.margin),flush=True)


if __name__=='__main__':main()
