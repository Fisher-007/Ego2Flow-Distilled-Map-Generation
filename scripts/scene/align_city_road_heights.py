"""Fit coarse road elevations to AV2 along the connected road graph.

The bare-earth DEM supplies unobserved road profiles. Measured AV2 elevations
anchor matching road stations; graph endpoints enforce continuous road joins.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree
from map_generation.fusion import station_xyz


def share_carriageway_profiles(lanes):
    """Use one inferred longitudinal residual for each directed source edge.

    Source XY polylines are retained exactly as piecewise-linear geometry.
    Original stations are included in the common normalized station set.
    """
    from collections import defaultdict
    groups=defaultdict(list)
    for i,lane in enumerate(lanes):
        if not lane.get('internal') and 'passenger' in lane['allowed_classes']:
            groups[lane['edge_id']].append(i)
    shared=[]
    for edge,indices in groups.items():
        if len(indices)<2:continue
        arrays=[np.asarray(lanes[i]['centerline'],dtype=float)for i in indices]
        stations=[np.r_[0.,np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1).cumsum()]for a in arrays]
        if any(s[-1]<.1 for s in stations):continue
        fractions=np.unique(np.concatenate([s/s[-1]for s in stations]))
        fractions=fractions[np.r_[True,np.diff(fractions)>1e-9]]
        fractions[0]=0.;fractions[-1]=1.
        profiles=np.asarray([np.interp(fractions,s/s[-1],a[:,2])-((1-fractions)*a[0,2]+fractions*a[-1,2])
                             for a,s in zip(arrays,stations)])
        residual=np.median(profiles,axis=0)
        for i,a,s in zip(indices,arrays,stations):
            points=np.column_stack([np.interp(fractions,s/s[-1],a[:,c])for c in range(3)])
            points[:,2]=(1-fractions)*a[0,2]+fractions*a[-1,2]+residual
            lanes[i]['centerline']=points.tolist()
        shared.append(dict(edge_id=edge,lane_indices=indices,normalized_station_count=len(fractions),
                           removed_residual_spread_m=float(np.ptp(profiles,axis=0).max())))
    return shared


def align(scene, additional_anchors=(), carriageway_weight=0., lateral_sample_step=8., shared_carriageway_profile=False):
    lanes=scene['coarse_map']['lanes'];lookup={l['id']:i for i,l in enumerate(lanes)}
    shared=share_carriageway_profiles(lanes) if shared_carriageway_profile else []
    parent=np.arange(2*len(lanes))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    def union(a,b):parent[find(b)]=find(a)
    for i,l in enumerate(lanes):
        for target in l['successors']:
            j=lookup[target]
            if np.linalg.norm(np.asarray(l['centerline'][-1])[:2]-np.asarray(lanes[j]['centerline'][0])[:2])<2:
                union(2*i+1,2*j)
    for group in shared:
        first=group['lane_indices'][0]
        for i in group['lane_indices'][1:]:
            union(2*first,2*i);union(2*first+1,2*i+1)
    roots=[find(i) for i in range(len(parent))];ids={r:n for n,r in enumerate(sorted(set(roots)))}
    vertices=np.array([ids[r] for r in roots]);n=len(ids)
    original_ends=np.asarray([z for l in lanes for z in (l['centerline'][0][2],l['centerline'][-1][2])])
    base=np.bincount(vertices,weights=original_ends)/np.bincount(vertices)
    shapes=[LineString(np.asarray(l['centerline'])[:,:2]) for l in lanes];tree=STRtree(shapes)
    rows=[];cols=[];values=[];rhs=[];anchors=[]
    def equation(coefficients,target,weight):
        row=len(rhs);w=weight**.5
        for col,value in coefficients:rows.append(row);cols.append(col);values.append(value*w)
        rhs.append(target*w)
    for i,l in enumerate(lanes):
        a,b=vertices[2*i:2*i+2];length=max(shapes[i].length,.1)
        equation([(a,-1),(b,1)],original_ends[2*i+1]-original_ends[2*i],1/length)
    for i,z in enumerate(base):equation([(i,1)],z,.001)
    for fine in scene['lanes']:
        points=fine['centerline'];line=LineString(np.asarray(points)[:,:2])
        if line.length<.1:continue
        for fs in np.linspace(0,line.length,max(2,int(line.length/8)+1)):
            point=station_xyz(points,fs)
            direction=station_xyz(points,min(line.length,fs+1))-station_xyz(points,max(0,fs-1))
            direction=direction[:2]/max(np.linalg.norm(direction[:2]),1e-9);candidates=[]
            for i in tree.query(box(point[0]-3,point[1]-3,point[0]+3,point[1]+3)):
                lane=lanes[i]
                if fine['lane_type']=='BIKE' and 'passenger' in lane['allowed_classes']:continue
                if fine['lane_type']!='BIKE' and not set(lane['allowed_classes'])&{'passenger','bus'}:continue
                length=shapes[i].length
                if length<.1:continue
                station=shapes[i].project(Point(point[:2]));p=station_xyz(lane['centerline'],station)
                distance=np.linalg.norm(p[:2]-point[:2])
                d=station_xyz(lane['centerline'],min(length,station+1))-station_xyz(lane['centerline'],max(0,station-1))
                cosine=float(direction@d[:2]/max(np.linalg.norm(d[:2]),1e-9))
                if distance<3 and cosine>.9:candidates.append((distance+3*(1-cosine),int(i),station,p[2]))
            if not candidates:continue
            _,i,station,z=min(candidates);u=station/shapes[i].length;a,b=vertices[2*i:2*i+2]
            profile=z-((1-u)*original_ends[2*i]+u*original_ends[2*i+1])
            target=point[2]-profile
            equation([(a,1-u),(b,u)],target,20.)
            anchors.append(dict(fine_lane_id=fine['id'],coarse_lane_id=lanes[i]['id'],station_m=station,
                before_z_m=float(z),target_z_m=float(point[2]),variables=[int(a),int(b)],fraction=float(u),profile_z_m=float(profile)))
    # Reviewed road correspondence can supply observed heights where a narrow
    # centerline-distance match misses the same physical road.
    for observation in additional_anchors:
        i=lookup[observation['coarse_lane_id']];station=observation['station_m']
        if not 0<=station<=shapes[i].length+1e-6:
            raise ValueError('Height observation lies outside its source road')
        z=station_xyz(lanes[i]['centerline'],station)[2]
        u=station/shapes[i].length;a,b=vertices[2*i:2*i+2]
        profile=z-((1-u)*original_ends[2*i]+u*original_ends[2*i+1])
        equation([(a,1-u),(b,u)],observation['target_z_m']-profile,20.)
        anchors.append(dict(observation,before_z_m=float(z),variables=[int(a),int(b)],
                            fraction=float(u),profile_z_m=float(profile)))
    lateral = []
    if carriageway_weight:
        from collections import defaultdict
        groups = defaultdict(list)
        for i, lane in enumerate(lanes):
            if not lane.get('internal') and 'passenger' in lane['allowed_classes']:
                groups[lane['edge_id']].append(i)
        for edge_id, indices in groups.items():
            indices.sort(key=lambda i: int(lanes[i]['id'].rsplit('_', 1)[1]))
            for i, j in zip(indices, indices[1:]):
                if shapes[i].length < .1 or shapes[j].length < .1:
                    continue
                # Same SUMO edge identifies one directed source carriageway.
                # Its transverse level is inferred; AV2 itself stays untouched.
                count=max(3, int(max(shapes[i].length,shapes[j].length)/lateral_sample_step)+1)
                for u in np.linspace(0,1,count):
                    zs=[float(station_xyz(lanes[k]['centerline'],u*shapes[k].length)[2]) for k in (i,j)]
                    profiles=[zs[c]-((1-u)*original_ends[2*k]+u*original_ends[2*k+1]) for c,k in enumerate((i,j))]
                    ai,bi=vertices[2*i:2*i+2];aj,bj=vertices[2*j:2*j+2]
                    coefficients=[(ai,1-u),(bi,u),(aj,-(1-u)),(bj,-u)]
                    target=profiles[1]-profiles[0]
                    equation(coefficients,target,carriageway_weight)
                    lateral.append(dict(edge_id=edge_id,lane_ids=[lanes[i]['id'],lanes[j]['id']],
                        fraction=float(u),coefficients=[(int(a),float(b))for a,b in coefficients],
                        target=float(target),before_delta_m=zs[0]-zs[1]))
    matrix=coo_matrix((values,(rows,cols)),shape=(len(rhs),n)).tocsr()
    normal=(matrix.T@matrix).tocsr();target=matrix.T@(np.asarray(rhs)-matrix@base)
    iterations=[0]
    def count_iteration(_):iterations[0]+=1
    delta,stop=cg(normal,target,M=diags(1/normal.diagonal()),rtol=1e-8,atol=1e-10,
                  maxiter=5000,callback=count_iteration)
    if stop!=0:raise RuntimeError(f'Road height solve did not converge: {stop}')
    height=base+delta;changes=[]
    for i,l in enumerate(lanes):
        p=np.asarray(l['centerline'],dtype=float);stations=np.r_[0.,np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).cumsum()]
        fraction=stations/max(stations[-1],1e-8);a,b=vertices[2*i:2*i+2]
        delta=(1-fraction)*(height[a]-p[0,2])+fraction*(height[b]-p[-1,2])
        p[:,2]+=delta;l['centerline']=p.tolist();l['height_source']='AV2-anchored connected road profile with DEM residual shape'
        changes.append(float(np.max(abs(delta))))
    before=[];after=[]
    for a in anchors:
        i,j=a['variables'];u=a['fraction'];z=(1-u)*height[i]+u*height[j]+a['profile_z_m'];a['after_z_m']=float(z)
        before.append(a['before_z_m']-a['target_z_m']);after.append(z-a['target_z_m'])
    def stats(v):return dict(rmse_m=float(np.sqrt(np.mean(np.square(v)))),absolute_p95_m=float(np.percentile(abs(np.asarray(v)),95)),max_m=float(np.max(abs(np.asarray(v)))))
    report=dict(anchors=len(anchors),graph_variables=n,solver_stop=int(stop),iterations=iterations[0],
        reviewed_additional_anchors=len(additional_anchors),
        before=stats(before),after=stats(after),max_correction_m=max(changes),anchor_observations=anchors,
        method=__doc__,scope='Elevation construction from observed road stations; bridge-layer assignment and native driving still require review.')
    if shared:
        report['shared_carriageway_profiles']=dict(group_count=len(shared),groups=shared,
            method='Common normalized source stations and median longitudinal DEM residual; shared start/end elevation variables. Original source XY, lane topology and separate edge identities retained; AV2 observations remain unchanged.',
            crossfall='Level transverse surface is an explicit inference where road banking is not observed.')
    if lateral:
        for item in lateral:
            item['after_delta_m']=float(sum(height[a]*b for a,b in item['coefficients'])-item['target'])
        report['carriageway_coupling']=dict(weight=carriageway_weight,sample_step_m=lateral_sample_step,
            observations=len(lateral),before=stats([x['before_delta_m']for x in lateral]),
            after=stats([x['after_delta_m']for x in lateral]),
            scope='Same directed source edge only; inferred level transverse profile. Retain AV2 observations, source XY, topology, permissions and separate source edge/level identities.',
            worst=sorted(lateral,key=lambda x:abs(x['after_delta_m']),reverse=True)[:30])
    scene['coarse_map']['height_alignment']={k:v for k,v in report.items() if k!='anchor_observations'}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fine-scene',type=Path,help='Reuse the assembled fine observations after a coarse-only revision')
    p.add_argument('--additional-anchors',type=Path,help='Observed or explicitly inferred support selected through reviewed road correspondence')
    p.add_argument('--carriageway-weight',type=float,default=0.,help='Transverse level constraint weight for a directed source edge; zero disables this constraint')
    p.add_argument('--lateral-sample-step',type=float,default=8.)
    p.add_argument('--shared-carriageway-profile',action='store_true',help='Use a common inferred longitudinal residual and endpoint elevations for each directed source carriageway')
    a=p.parse_args();scene=json.loads(a.scene.read_text())
    if a.fine_scene:
        fine=json.loads(a.fine_scene.read_text())
        if scene['coordinates']!=fine['coordinates']:raise ValueError('Fine and coarse observations use different frames')
        scene['lanes']=fine['lanes'];del fine
    additional=json.loads(a.additional_anchors.read_text())['anchor_observations'] if a.additional_anchors else []
    report=align(scene,additional,a.carriageway_weight,a.lateral_sample_step,a.shared_carriageway_profile);a.output.mkdir(parents=True,exist_ok=True)
    # A fresh fusion will apply these elevations to all retained source pieces.
    scene.pop('fusion_map',None)
    with (a.output/'scene.json').open('w') as stream:json.dump(scene,stream,separators=(',',':'),allow_nan=False)
    (a.output/'height_review.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ('anchor_observations','method','scope')})

if __name__=='__main__':main()
