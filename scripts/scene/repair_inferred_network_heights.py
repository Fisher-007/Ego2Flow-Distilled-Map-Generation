"""Fit selected inferred-road elevations with fixed source and outer contacts.

OSQP solves shared endpoint heights and interior profiles under grade bounds;
the compiled profile is measured after fitting.
"""
import argparse, json
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import numpy as np
import osqp
from scipy.sparse import coo_matrix, vstack, eye, diags
from map_generation.road_builder import fit_lane

def repair(scene, condition):
    result=deepcopy(scene)
    before={l['id']:l for l in scene['fusion_map']['lanes']}
    lanes={l['id']:l for l in result['fusion_map']['lanes']}
    keys=condition['lane_ids'];selected=set(keys)
    coarse=set(condition['coarse_lane_ids'])
    points={k:np.asarray(before[k]['centerline'],float) for k in keys}
    if any(before[k]['source']=='av2' or 'parametric_xy_segments' in before[k] or 'boundary_shape' in before[k] for k in keys):
        raise ValueError('Selected roads must be plain inferred profiles, not observed or accepted parametric roads')
    ports=[(k,i) for k in keys for i in range(len(points[k]))]
    parent={p:p for p in ports}
    def root(p):
        while parent[p]!=p:
            parent[p]=parent[parent[p]];p=parent[p]
        return p
    def union(a,b):parent[root(a)]=root(b)
    pred=defaultdict(list)
    for k,l in before.items():
        for target in l['successors']:
            if target in before:pred[target].append(k)
            if k in selected and target in selected:
                union((k,len(points[k])-1),(target,0))
    stations={}
    for k,p in points.items():
        ds=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)
        stations[k]=np.r_[0.,ds.cumsum()]
        for i,d in enumerate(ds):
            if d<=1e-7:union((k,i),(k,i+1))
    groups=defaultdict(list)
    for p in ports:groups[root(p)].append(p)
    index={n:i for i,n in enumerate(groups)}
    get=lambda k,i:index[root((k,i))]
    fixed=defaultdict(list)
    moving_ends={}
    for k,p in points.items():
        neighbors=[pred[k],before[k]['successors']]
        moving=[]
        for end,nbrs in enumerate(neighbors):
            at=0 if end==0 else len(p)-1
            outside=[n for n in nbrs if n not in selected]
            if outside:
                fixed[get(k,at)].extend(float(before[n]['centerline'][-1 if end==0 else 0][2]) for n in outside)
            if not nbrs:fixed[get(k,at)].append(float(p[at,2]))
            moving.append(bool(nbrs) and not outside)
        moving_ends[k]=moving
        if k in coarse:
            arc=stations[k];d=condition['coarse_transition_distance_m']
            for i,s in enumerate(arc):
                if not ((moving[0] and s<d) or (moving[1] and arc[-1]-s<d)):
                    fixed[get(k,i)].append(float(p[i,2]))
    conflicts={str(i):values for i,values in fixed.items() if np.ptp(values)>1e-6}
    if conflicts:return None,dict(passed=False,status='incompatible_fixed_heights',fixed_conflicts=conflicts)
    n=len(index);prior=np.zeros(n);weight=np.zeros(n)
    for k,p in points.items():
        arc=stations[k];w=np.maximum(.05,np.r_[np.diff(arc)[0]/2,(arc[2:]-arc[:-2])/2,np.diff(arc)[-1]/2])
        for i in range(len(p)):
            j=get(k,i);prior[j]+=w[i]*p[i,2];weight[j]+=w[i]
    prior/=weight
    rows=[];cols=[];values=[];limits=[]
    smooth_rows=[];smooth_cols=[];smooth_values=[];nr=0
    for k,p in points.items():
        arc=stations[k]
        for i,d in enumerate(np.diff(arc)):
            if d<=1e-7:continue
            row=len(limits);rows.extend([row,row]);cols.extend([get(k,i),get(k,i+1)]);values.extend([-1.,1.])
            limits.append(condition['maximum_polyline_grade']*d)
        for i in range(1,len(p)-1):
            a,b=arc[i]-arc[i-1],arc[i+1]-arc[i]
            if min(a,b)<=1e-7:continue
            smooth_rows.extend([nr]*3);smooth_cols.extend([get(k,i-1),get(k,i),get(k,i+1)])
            smooth_values.extend([1/a,-1/a-1/b,1/b]);nr+=1
    D=coo_matrix((values,(rows,cols)),shape=(len(limits),n)).tocsc()
    S=coo_matrix((smooth_values,(smooth_rows,smooth_cols)),shape=(nr,n)).tocsc()
    P=diags(weight,format='csc')+condition['slope_smoothing_weight']*(S.T@S)
    low=prior-condition['maximum_height_change_m'];high=prior+condition['maximum_height_change_m']
    for i,values in fixed.items():low[i]=high[i]=float(np.mean(values))
    # Couple selected directional profiles through source cross-section height constraints.
    cross=condition.get('cross_section_height_relations',[])
    cr=[];cc=[];cv=[];cl=[];cu=[]
    for row,relation in enumerate(cross):
        for sign,name in ((1.,'a'),(-1.,'b')):
            endpoint=relation[name];k=endpoint['lane_id'];s=endpoint['station_m'];arc=stations[k]
            j=min(int(np.searchsorted(arc,s,side='right'))-1,len(arc)-2)
            t=(s-arc[j])/(arc[j+1]-arc[j])
            cr.extend([row,row]);cc.extend([get(k,j),get(k,j+1)]);cv.extend([sign*(1-t),sign*t])
        target=relation['height_difference_m'];tolerance=relation['tolerance_m']
        cl.append(target-tolerance);cu.append(target+tolerance)
    C=coo_matrix((cv,(cr,cc)),shape=(len(cross),n)).tocsc()
    solver_settings=dict(max_iter=condition.get('solver_max_iterations',30000),
        eps_abs=condition.get('solver_eps_abs',1e-8),eps_rel=condition.get('solver_eps_rel',1e-8),
        polishing=True,verbose=False)
    solver=osqp.OSQP();solver.setup(P=P,q=-weight*prior,A=vstack([D,C,eye(n)],format='csc'),
        l=np.r_[-np.array(limits),cl,low],u=np.r_[limits,cu,high],**solver_settings)
    solution=solver.solve()
    report=dict(condition=condition,status=solution.info.status,selected_roads=len(keys),
        solver_settings=solver_settings,solver_version=osqp.__version__,
        solver_info=dict(iter=int(solution.info.iter),prim_res=float(solution.info.prim_res),
                         dual_res=float(solution.info.dual_res),status_val=int(solution.info.status_val)),
        variables=n,fixed_height_nodes=len(fixed),method=__doc__)
    if solution.info.status_val not in (1,2):
        import networkx as nx
        graph=nx.Graph();graph.add_nodes_from(range(n))
        rowmat=D.tocsr()
        for row,limit in enumerate(limits):
            ids=rowmat[row].indices
            if len(ids)==2:graph.add_edge(int(ids[0]),int(ids[1]),weight=float(limit/condition['maximum_polyline_grade']))
        inverse={i:groups[node] for node,i in index.items()};violations=[]
        for i,values in fixed.items():
            distances=nx.single_source_dijkstra_path_length(graph,i)
            for j,other in fixed.items():
                if j<=i or j not in distances or distances[j]<=1e-7:continue
                grade=abs(float(np.mean(values)-np.mean(other)))/distances[j]
                if grade>condition['maximum_polyline_grade']+1e-6:
                    violations.append(dict(first=inverse[i],second=inverse[j],distance_m=distances[j],required_mean_grade=grade,heights=[float(np.mean(values)),float(np.mean(other))]))
        report['fixed_cut_grade_conflicts']=sorted(violations,key=lambda r:-r['required_mean_grade'])[:30]
        return None,dict(report,passed=False)

    z=solution.x.copy()
    for i,values in fixed.items():z[i]=float(np.mean(values))
    cross_values=np.asarray(C@z).ravel()
    report['cross_section_height_relations']=[dict(relation,
        actual_height_difference_m=float(value),
        target_error_m=float(abs(value-relation['height_difference_m'])))
        for relation,value in zip(cross,cross_values)]
    report['cross_section_constraints_satisfied']=all(
        r['target_error_m']<=r['tolerance_m']+1e-6 for r in report['cross_section_height_relations'])
    records=[]
    for k,p in points.items():
        updated=p.copy();updated[:,2]=[z[get(k,i)] for i in range(len(p))]
        lanes[k]['centerline']=updated.tolist()
        geom,_=fit_lane(lanes[k],1.)
        at=np.linspace(0,geom.length,max(17,int(np.ceil(geom.length/.1))+1))
        xyz=np.array([geom.xyz(s) for s in at])
        grade=np.abs(np.diff(xyz[:,2]))/np.maximum(1e-12,np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1))
        oldgeom,_=fit_lane(before[k],1.);oldxyz=np.array([oldgeom.xyz(s) for s in at])
        records.append(dict(id=k,source=before[k]['source'],old_fitted_maximum_grade=float(np.max(np.abs(np.diff(oldxyz[:,2]))/np.maximum(1e-12,np.linalg.norm(np.diff(oldxyz[:,:2],axis=0),axis=1)))),
            fitted_maximum_grade=float(grade.max()),maximum_z_motion_m=float(np.max(abs(updated[:,2]-p[:,2]))),
            xy_exactly_preserved=bool(np.array_equal(updated[:,:2],p[:,:2])),moving_ends=moving_ends[k]))
    gaps=[]
    for k,l in lanes.items():
        for t in l['successors']:
            if t in lanes and (k in selected or t in selected):
                gap=abs(l['centerline'][-1][2]-lanes[t]['centerline'][0][2])
                if gap>1e-6:gaps.append(dict(source=k,target=t,height_gap_m=gap))
    preserved=all(lanes[k]==before[k] for k in before if k not in selected)
    topology=all(all(lanes[k].get(f)==before[k].get(f) for f in ('id','source','successors','allowed_classes','width_m','end_widths_m')) for k in before)
    report.update(records=records,endpoint_height_gaps=gaps,unselected_lanes_unchanged=preserved,
        topology_permissions_preserved=topology,maximum_fitted_grade=max(r['fitted_maximum_grade'] for r in records),
        maximum_z_motion_m=max(r['maximum_z_motion_m'] for r in records))
    report['passed']=bool(report['cross_section_constraints_satisfied'] and preserved and topology and not gaps and all(r['xy_exactly_preserved'] and r['fitted_maximum_grade']<=condition['maximum_fitted_grade']+1e-6 for r in records))
    result['fusion_map']['inferred_network_height_feedback']=dict(condition_id=condition['id'],review_scope='Selected inferred height profiles only; not native simulation acceptance')
    return result,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','conditions','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result,review=repair(json.loads(a.scene.read_text()),json.loads(a.conditions.read_text()))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'review.json').write_text(json.dumps(review,indent=2))
    if result is not None:(a.output/'scene.json').write_text(json.dumps(result,separators=(',',':')))
    print({k:v for k,v in review.items() if k not in ('records','condition','method')},flush=True)
    if not review['passed']:raise SystemExit(1)
if __name__=='__main__':main()
