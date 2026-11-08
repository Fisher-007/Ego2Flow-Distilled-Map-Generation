"""Fit buildable coarse-road profiles to AV2 anchors and terrain-derived heights.

AV2 stays unchanged. Shared source endpoints and bounded longitudinal grades
replace bare-earth microstructure with an inferred continuous road profile.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
import osqp
from map_generation.fusion import station_xyz


def editable_coarse_sources(scene, review, distance):
    import networkx as nx
    lanes={l['id']:l for l in scene['coarse_map']['lanes']}
    lengths={k:float(np.linalg.norm(np.diff(np.asarray(l['centerline'])[:,:2],axis=0),axis=1).sum())for k,l in lanes.items()}
    graph=nx.Graph();graph.add_nodes_from(lanes)
    for k,l in lanes.items():
        graph.add_weighted_edges_from((k,t,(lengths[k]+lengths[t])/2)for t in l['successors'])
    seeds={k for c in review['corridors']for k in c['coarse_lane_ids']}
    return set(nx.multi_source_dijkstra_path_length(graph,seeds,cutoff=distance))


def smooth(scene, anchors, spacing=5., max_grade=.12, editable=None, reference=None, shared_carriageway_profile=False, carriageway_groups=(), height_bounds=(), include_solve_knots=False, source_prior_weights=None, source_prior_profiles=None, station_height_links=()):
    lanes=scene['coarse_map']['lanes'];lookup={l['id']:i for i,l in enumerate(lanes)}
    shapes=[np.asarray(l['centerline'],float) for l in lanes]
    stations=[np.r_[0.,np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).cumsum()] for p in shapes]
    knots=[np.linspace(0,s[-1],max(2,int(np.ceil(s[-1]/spacing))+1)) for s in stations]
    groups=[]
    if shared_carriageway_profile:
        from collections import defaultdict
        by_edge=defaultdict(list)
        for i,lane in enumerate(lanes):
            if not lane.get('internal') and 'passenger' in lane['allowed_classes']:
                by_edge[lane['edge_id']].append(i)
        groups=[ids for ids in by_edge.values()if len(ids)>1]
        for ids in groups:
            count=max(len(knots[i])for i in ids)
            for i in ids:knots[i]=np.linspace(0,stations[i][-1],count)
    reversed_lanes=set()
    explicit_groups=[]
    for selection in carriageway_groups:
        members=[lookup[row['lane_id']] for row in selection['lanes']]
        explicit_groups.append(members)
        reversed_lanes.update(lookup[row['lane_id']] for row in selection['lanes'] if row['reversed'])
    explicit_members={i for group in explicit_groups for i in group}
    groups=[ids for ids in groups if not set(ids)&explicit_members]+explicit_groups
    output_stations={}
    grouped_lanes={i for group in groups for i in group}
    for ids in groups:
        # Include every source XY knot and every solve knot in common physical
        # orientation. Returning the full shared knot set preserves both the
        # exact source polyline and the common piecewise-linear height field.
        fractions=[]
        for i in ids:
            values=np.r_[stations[i],knots[i]]/stations[i][-1]
            fractions.append(1-values if i in reversed_lanes else values)
        common=np.unique(np.round(np.concatenate(fractions),12))
        common[0]=0.;common[-1]=1.
        count=max(len(knots[i])for i in ids)
        for i in ids:
            forward=1-common[::-1] if i in reversed_lanes else common
            output_stations[i]=forward*stations[i][-1]
            knots[i]=np.linspace(0,stations[i][-1],count)
    offsets=np.r_[0,np.cumsum([len(s) for s in knots])];parent=np.arange(offsets[-1])
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    for i,lane in enumerate(lanes):
        for key in lane['successors']:
            j=lookup[key]
            if np.linalg.norm(shapes[i][-1,:2]-shapes[j][0,:2])<.02:
                parent[find(offsets[j])]=find(offsets[i+1]-1)
    for ids in groups:
        first=ids[0]
        first_vars=list(range(offsets[first],offsets[first+1]))
        if first in reversed_lanes:first_vars.reverse()
        for i in ids[1:]:
            other_vars=list(range(offsets[i],offsets[i+1]))
            if i in reversed_lanes:other_vars.reverse()
            for a,b in zip(first_vars,other_vars,strict=True):
                parent[find(b)]=find(a)
    roots=[find(i) for i in range(len(parent))];unique={r:i for i,r in enumerate(sorted(set(roots)))}
    variables=np.asarray([unique[r] for r in roots]);size=len(unique)
    rr=[];cc=[];vv=[];target=[]
    ar=[];ac=[];av=[];upper=[]
    def objective(pairs,z,weight):
        r=len(target);scale=weight**.5
        for c,v in pairs:rr.append(r);cc.append(c);vv.append(scale*v)
        target.append(z*scale)
    lanevars=[]
    for i,lane in enumerate(lanes):
        ids=variables[offsets[i]:offsets[i+1]];lanevars.append(ids);length=stations[i][-1]
        for index,(s,v) in enumerate(zip(knots[i],ids)):
            prior=(source_prior_profiles or {}).get(lane['id'])
            z=float(np.interp(s,prior['station_m'],prior['z_m'])) if prior else float(np.interp(s,stations[i],shapes[i][:,2]))
            objective([(int(v),1)],z,.01*max(length/(len(ids)-1),.1)*(source_prior_weights or {}).get(lane['id'],1.))
        for a,b,ds in zip(ids[:-1],ids[1:],np.diff(knots[i])):
            if editable is not None and lane['id'] not in editable:continue
            r=len(upper);ar.extend([r,r]);ac.extend([int(a),int(b)]);av.extend([-1.,1.]);upper.append(max_grade*ds)
        for a,b,c,ds1,ds2 in zip(ids[:-2],ids[1:-1],ids[2:],np.diff(knots[i])[:-1],np.diff(knots[i])[1:]):
            if min(ds1,ds2)>1e-6:objective([(int(a),1/ds1),(int(b),-1/ds1-1/ds2),(int(c),1/ds2)],0.,.1)
    for a in anchors:
        i=lookup[a['coarse_lane_id']];s=a['station_m'];ss=knots[i];j=int(np.clip(np.searchsorted(ss,s)-1,0,len(ss)-2))
        u=float(np.clip((s-ss[j])/max(ss[j+1]-ss[j],1e-8),0,1));v=lanevars[i]
        objective([(int(v[j]),1-u),(int(v[j+1]),u)],a['target_z_m'],20.)
    design=coo_matrix((vv,(rr,cc)),shape=(len(target),size)).tocsc()
    constraint=coo_matrix((av,(ar,ac)),shape=(len(upper),size)).tocsc()
    upper=np.asarray(upper);lower=-upper
    if editable is not None:
        from scipy.sparse import vstack
        fixed={}
        reference_by_id={l['id']:l for l in reference['coarse_map']['lanes']}
        for i,lane in enumerate(lanes):
            if lane['id']in editable:continue
            original=np.asarray(reference_by_id[lane['id']]['centerline'])
            values=np.interp(knots[i],stations[i],original[:,2])
            for variable,z in zip(lanevars[i],values):
                if variable in fixed and abs(fixed[variable]-z)>1e-5:
                    raise ValueError('Unchanged source heights disagree at a shared endpoint')
                fixed[int(variable)]=float(z)
        ordered=sorted(fixed);values=np.array([fixed[k]for k in ordered])
        locks=coo_matrix((np.ones(len(ordered)),(np.arange(len(ordered)),ordered)),shape=(len(ordered),size)).tocsc()
        constraint=vstack([constraint,locks],format='csc');lower=np.r_[lower,values];upper=np.r_[upper,values]
    if height_bounds:
        from scipy.sparse import vstack
        br=[];bc=[];bv=[];blo=[];bhi=[]
        for bound in height_bounds:
            i=lookup[bound['lane_id']];ss=knots[i];station=bound['station_m']
            j=int(np.clip(np.searchsorted(ss,station)-1,0,len(ss)-2))
            u=float(np.clip((station-ss[j])/max(ss[j+1]-ss[j],1e-8),0,1))
            row=len(blo);br.extend([row,row]);bc.extend([int(lanevars[i][j]),int(lanevars[i][j+1])]);bv.extend([1-u,u])
            blo.append(bound.get('minimum_z_m',-np.inf));bhi.append(bound.get('maximum_z_m',np.inf))
        bounded=coo_matrix((bv,(br,bc)),shape=(len(blo),size)).tocsc()
        constraint=vstack([constraint,bounded],format='csc')
        lower=np.r_[lower,blo];upper=np.r_[upper,bhi]
    if station_height_links:
        from scipy.sparse import vstack
        lr=[];lc=[];lv=[];limits=[]
        for link in station_height_links:
            row=len(limits)
            for prefix,sign in [('from',1.),('to',-1.)]:
                i=lookup[link[prefix+'_lane_id']];ss=knots[i];station=link[prefix+'_station_m']
                j=int(np.clip(np.searchsorted(ss,station)-1,0,len(ss)-2))
                u=float(np.clip((station-ss[j])/max(ss[j+1]-ss[j],1e-8),0,1))
                lr.extend([row,row]);lc.extend([int(lanevars[i][j]),int(lanevars[i][j+1])]);lv.extend([sign*(1-u),sign*u])
            limits.append(link['maximum_difference_m'])
        links=coo_matrix((lv,(lr,lc)),shape=(len(limits),size)).tocsc()
        constraint=vstack([constraint,links],format='csc')
        lower=np.r_[lower,-np.asarray(limits)];upper=np.r_[upper,limits]
    solver=osqp.OSQP()
    solver.setup(P=(design.T@design).tocsc(),q=-design.T@np.asarray(target),A=constraint,
                 l=lower,u=upper,eps_abs=1e-5,eps_rel=1e-5,max_iter=30000,verbose=False,polishing=True)
    result=solver.solve()
    if result.info.status_val!=1:raise RuntimeError('Road profile solve: '+result.info.status)
    before=[];after=[];movement=[]
    for i,lane in enumerate(lanes):
        p=shapes[i].copy();ds=np.diff(stations[i]);valid=ds>1e-7
        before.extend((abs(np.diff(p[:,2]))[valid]/ds[valid]).tolist())
        if i in grouped_lanes:
            at=output_stations[i]
            p=np.column_stack([np.interp(at,stations[i],shapes[i][:,c])for c in range(3)])
            p[:,2]=np.interp(at,knots[i],result.x[lanevars[i]])
        else:
            # Keep the original repeated endpoints of zero-length SUMO links;
            # uniquing station zero would produce an invalid one-point line.
            if include_solve_knots and stations[i][-1] > 0 and (editable is None or lane['id'] in editable):
                at=np.unique(np.r_[stations[i],knots[i]])
                p=np.column_stack([np.interp(at,stations[i],shapes[i][:,c])for c in range(3)])
                p[:,2]=np.interp(at,knots[i],result.x[lanevars[i]])
            else:
                p[:,2]=np.interp(stations[i],knots[i],result.x[lanevars[i]])
        if editable is not None and lane['id']not in editable:
            p=np.asarray(reference_by_id[lane['id']]['centerline']).copy()
        current_stations=np.r_[0.,np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).cumsum()]
        ds=np.diff(current_stations);valid=ds>1e-7
        after.extend((abs(np.diff(p[:,2]))[valid]/ds[valid]).tolist())
        movement.extend(abs(p[:,2]-np.interp(current_stations,stations[i],shapes[i][:,2])).tolist())
        lane['centerline']=p.tolist();lane['height_source']='AV2-constrained continuous road profile; terrain supplies unobserved height prior'
    errors=[]
    for a in anchors:
        i=lookup[a['coarse_lane_id']]
        errors.append(float(station_xyz(lanes[i]['centerline'],a['station_m'])[2]-a['target_z_m']))
    report=dict(method=__doc__,solver=result.info.status,variables=size,anchors=len(anchors),
                knot_spacing_m=spacing,maximum_inferred_grade=max_grade,
                before_segments_above_15pct=int(sum(v>.15 for v in before)),after_segments_above_15pct=int(sum(v>.15 for v in after)),
                before_max_grade=max(before),after_max_grade=max(after),maximum_height_change_m=max(movement),
                anchor_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),anchor_absolute_p95_m=float(np.percentile(abs(np.asarray(errors)),95)),
                scope='Construction fit, not independent elevation accuracy. Bridge clearance and native surface response remain to be checked.')
    report['explicit_oriented_carriageway_groups']=len(explicit_groups)
    report['explicit_station_height_bounds']=len(height_bounds)
    report['explicit_station_height_links']=len(station_height_links)
    report['source_prior_weights']=source_prior_weights or {}
    report['source_prior_profiles']=source_prior_profiles or {}
    report['reversed_source_lanes']=len(reversed_lanes)
    if groups:
        report['shared_carriageway_groups']=len(groups)
        report['carriageway_method']='Common source station variables across directed source edges and explicitly selected orientation-aware counterpart groups; all source XY knots, bounded grades and original AV2 anchors retained.'
    scene['coarse_map']['profile_smoothing']=report
    if editable is not None:report.update(editable_coarse_lanes=len(editable),outside_profiles_preserved=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','height-review','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--preserve-outside-scene',type=Path)
    p.add_argument('--support-review',type=Path)
    p.add_argument('--propagation-distance-m',type=float,default=300.)
    p.add_argument('--shared-carriageway-profile',action='store_true')
    p.add_argument('--carriageway-groups',type=Path,help='Explicit reviewed source-lane orientation groups sharing one physical inferred profile')
    a=p.parse_args();s=json.loads(a.scene.read_text());anchors=json.loads(a.height_review.read_text())['anchor_observations']
    editable=reference=None
    if a.preserve_outside_scene:
        reference=json.loads(a.preserve_outside_scene.read_text());review=json.loads(a.support_review.read_text())
        editable=editable_coarse_sources(s,review,a.propagation_distance_m)
    report=smooth(s,anchors,editable=editable,reference=reference,shared_carriageway_profile=a.shared_carriageway_profile,carriageway_groups=json.loads(a.carriageway_groups.read_text())['groups'] if a.carriageway_groups else ());a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'profile_review.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.output/'scene.json').open('w') as stream:json.dump(s,stream,separators=(',',':'),allow_nan=False)
    print(report,flush=True)


if __name__=='__main__':main()
