"""Propagate explicit inferred road XY/width revisions through source transitions."""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.sparse import coo_matrix,diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from shapely.geometry import LineString,Point


def propagate(previous,proposal):
    out=deepcopy(proposal);before={l['id']:l for l in previous['fusion_map']['lanes']};lanes={l['id']:l for l in out['fusion_map']['lanes']}
    if set(before)!=set(lanes):raise ValueError('Source identities changed')
    revised={k for k in lanes if lanes[k].get('source_corridor_revision')!=before[k].get('source_corridor_revision')}
    transitions=sorted(k for k,l in lanes.items()if k not in revised and l['source']!='av2'and(l['source']!='osm_sumo'or l.get('internal')))
    index={k:i for i,k in enumerate(transitions)};predecessors={k:[]for k in lanes}
    for k,l in lanes.items():
        for t in l['successors']:
            if t in predecessors:predecessors[t].append(k)
    parent=np.arange(2*len(index))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    for k,i in index.items():
        for t in lanes[k]['successors']:
            if t in index:parent[find(2*i+1)]=find(2*index[t])
    roots=[find(i)for i in range(len(parent))];unique={r:i for i,r in enumerate(sorted(set(roots)))};vertices=np.asarray([unique[r]for r in roots])
    delta={}
    for k,l in lanes.items():
        if k in index:continue
        old=before[k];xy=np.asarray(l['centerline'])[[0,-1],:2]-np.asarray(old['centerline'])[[0,-1],:2]
        widths=np.asarray(l.get('end_widths_m',[l['width_m']]*2))-old.get('end_widths_m',[old['width_m']]*2)
        delta[k]=np.c_[xy,widths]
    anchors={};rr=[];cc=[];vv=[]
    for k,i in index.items():
        for e,neighbors,side in [(0,predecessors[k],0),(1,lanes[k]['successors'],1)]:
            node=int(vertices[2*i+e])
            for other in neighbors:
                if other in delta:anchors.setdefault(node,[]).append(dict(lane_id=other,correction=delta[other][1 if e==0 else 0].tolist()))
        left,right=vertices[2*i:2*i+2];p=np.asarray(lanes[k]['centerline']);length=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).sum()
        if left!=right:rr.extend([left,right]);cc.extend([right,left]);vv.extend([1/max(length,1e-6)]*2)
    n=len(unique);graph=coo_matrix((vv,(rr,cc)),shape=(n,n)).tocsr();lap=diags(np.asarray(graph.sum(axis=1)).ravel())-graph
    fixed=np.zeros(n,dtype=bool);correction=np.zeros((n,3))
    for node,rows in anchors.items():fixed[node]=True;correction[node]=np.mean([r['correction']for r in rows],axis=0)
    _,components=connected_components(graph,directed=False)
    for component in np.unique(components):
        members=np.flatnonzero(components==component)
        if not fixed[members].any():fixed[members[0]]=True
    free=np.flatnonzero(~fixed)
    if len(free):correction[free]=spsolve(lap[free][:,free],-lap[free][:,fixed]@correction[fixed])
    updated=[]
    for k,i in index.items():
        ends=correction[vertices[2*i:2*i+2]]
        if abs(ends).max()<1e-10:continue
        lane=lanes[k];old=before[k];p=np.asarray(old['centerline']);ss=np.r_[0.,np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).cumsum()]
        field=CubicHermiteSpline([0.,ss[-1]],ends,np.zeros((2,3)))(ss)
        q=p.copy();q[:,:2]+=field[:,:2];lane['centerline']=q.tolist()
        for name in ['left_boundary','right_boundary']:
            if name not in old:continue
            line=LineString(p[:,:2]);points=np.asarray(old[name]);at=np.array([line.project(Point(x[:2]))for x in points]);values=CubicHermiteSpline([0.,ss[-1]],ends,np.zeros((2,3)))(at)
            points[:,:2]+=values[:,:2];lane[name]=points.tolist()
        widths=np.asarray(old.get('end_widths_m',[old['width_m']]*2))+ends[:,2]
        lane['end_widths_m']=widths.tolist();lane['width_m']=float(np.mean(widths))
        for name in ['boundary_shape','parametric_xy_segments','parametric_elevation_rows']:lane.pop(name,None)
        lane['source_corridor_transition_revision']=dict(maximum_xy_change_m=float(np.linalg.norm(field[:,:2],axis=1).max()),endpoint_width_changes_m=ends[:,2].tolist(),requires_current_generation=True)
        updated.append(dict(lane_id=k,**lane['source_corridor_transition_revision']))
    conflicts=[dict(port=node,maximum_correction_spread=float(np.ptp([r['correction']for r in rows],axis=0).max()),sources=rows)for node,rows in anchors.items()if np.ptp([r['correction']for r in rows],axis=0).max()>1e-6]
    gaps=[]
    for k,l in lanes.items():
        for t in l['successors']:
            if t not in lanes:continue
            old_gap=np.linalg.norm(np.asarray(before[k]['centerline'][-1])-before[t]['centerline'][0]);new_gap=np.linalg.norm(np.asarray(l['centerline'][-1])-lanes[t]['centerline'][0])
            if new_gap>old_gap+1e-6:gaps.append(dict(before=k,after=t,old_gap_m=float(old_gap),new_gap_m=float(new_gap)))
    report=dict(source_lanes=len(revised),transition_updates=updated,anchor_conflicts=conflicts,increased_port_gaps=gaps,
        changed_av2=[k for k,l in lanes.items()if l['source']=='av2'and l!=before[k]],
        topology_and_permissions_unchanged=all(l['successors']==before[k]['successors']and l['allowed_classes']==before[k]['allowed_classes']for k,l in lanes.items()),
        accepted_map=False,method='Joint endpoint-displacement graph anchored by revised inferred corridors and unchanged observed/other road ends; zero-slope cubic XY correction and endpoint width transfer. Original source evidence remains in the previous scene. Fresh generation and complete assembly checks remain required.')
    out['fusion_map']['source_corridor_port_revision']=report;return out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['previous','proposal','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();scene,review=propagate(json.loads(a.previous.read_text()),json.loads(a.proposal.read_text()));a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(scene,separators=(',',':'))+'\n');(a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    print({k:v for k,v in review.items()if k not in ['transition_updates','anchor_conflicts','increased_port_gaps']},flush=True)
    print(dict(transition_updates=len(review['transition_updates']),anchor_conflicts=len(review['anchor_conflicts']),increased_port_gaps=review['increased_port_gaps'][:30]),flush=True)
if __name__=='__main__':main()
