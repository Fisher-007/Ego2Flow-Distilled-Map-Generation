"""Project generated road paths onto fixed measured ports and shared inferred ports."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,least_squares,OptimizeResult
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix
from scipy.signal import find_peaks
from shapely.geometry import LineString,Polygon
from shapely.ops import unary_union
from map_generation.boundary_shape import BoundaryShape
from map_generation.road_builder import fit_lane
from scripts.scene.prepare_access_clearance import obstacle_shapes,margins


def unit(v):return v/max(np.linalg.norm(v),1e-10)


def interpolate(points,u):
    p=np.asarray(points);arc=np.r_[0.,np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).cumsum()];arc/=arc[-1]
    return np.column_stack([np.interp(u,arc,p[:,i]) for i in range(p.shape[1])])


def project(condition,raw,limit=.2,movement_limit=3.,restore_feasibility=False,variable_tangents=False,constraint_samples=129,width_aware_curvature=False,source_initialization=False,initial_paths=None,max_iterations=250,port_direction_initialization="source",solver_ftol=1e-9,checkpoint_path=None,sparse_feasibility=False,compact_solver_constraints=False,adaptive_constraint_generation=False,sparse_penalty_projection=False,penalty_weights=None,pathwise_feasibility_polish=False):
    lanes={l['id']:l for l in condition['lanes']};keys=condition['generated_path_ids'];ports=[(k,e) for k in keys for e in (0,1)]
    parent={p:p for p in ports}
    def root(p):
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    for row in condition['boundaries']:
        if not row['fixed']:parent[root((row['lane_id'],row['end']))]=root((row['neighbor_id'],row['neighbor_end']))
    collapsed=[key for key in keys if root((key,0))==root((key,1))]
    if collapsed:raise ValueError('Positive source paths have identical shared start/end ports; review source relation realization: '+', '.join(collapsed))
    groups=defaultdict(list)
    for p in ports:groups[root(p)].append(p)
    fixed={};cache={}
    for row in condition['boundaries']:
        if not row['fixed']:continue
        key=row['neighbor_id'];end=row['neighbor_end']
        if key not in cache:cache[key]=fit_lane(lanes[key],1.)[0]
        g=cache[key];point=g.xyz(0 if end==0 else g.length)
        direction=g.curves[0].tangent(0) if end==0 else g.curves[-1].tangent(1)
        ident=root((row['lane_id'],row['end']))
        if ident in fixed and np.linalg.norm(fixed[ident][0]-point)>.01:raise ValueError('Conflicting measured boundary positions')
        fixed[ident]=(point,direction)
    # Unconnected outer source ports are fixed as well.
    joined={(r['lane_id'],r['end']) for r in condition['boundaries']}
    for port in ports:
        if port not in joined:
            key,end=port;p=np.asarray(lanes[key]['centerline']);fixed[root(port)]=(p[0 if end==0 else -1],unit(np.diff(p[:2 if end==0 else len(p)],axis=0)[0 if end==0 else -1,:2]))
    free=[];nodes={};parameters=[];bounds=[]
    for ident,members in groups.items():
        if ident in fixed:nodes[ident]=dict(fixed=True,point=fixed[ident][0],direction=fixed[ident][1]);continue
        positions=[];directions=[]
        for key,end in members:
            p=np.asarray(lanes[key]['centerline']);positions.append(p[0 if end==0 else -1])
            vector=(p[-1]-p[0])[:2] if port_direction_initialization=='chord' else (p[1]-p[0])[:2] if end==0 else (p[-1]-p[-2])[:2]
            directions.append(unit(vector))
        point=np.mean(positions,axis=0);direction=unit(np.sum(directions,axis=0));angle=np.arctan2(direction[1],direction[0]);offset=len(parameters)
        nodes[ident]=dict(fixed=False,offset=offset,point=point,direction=direction);free.append(ident)
        parameters.extend([point[0],point[1],angle]);bounds.extend([(point[0]-movement_limit,point[0]+movement_limit),(point[1]-movement_limit,point[1]+movement_limit),(angle-np.pi,angle+np.pi)])
    def endpoints(x,key):
        points=[];directions=[]
        for end in (0,1):
            node=nodes[root((key,end))]
            if node['fixed']:points.append(node['point'][:2]);directions.append(node['direction'])
            else:
                offset=node['offset'];points.append(x[offset:offset+2]);angle=x[offset+2];directions.append([np.cos(angle),np.sin(angle)])
        return np.asarray(points),np.asarray(directions)
    knots=np.linspace(0,1,5);u=np.linspace(0,1,constraint_samples);targets={};source_targets={};offsets={};raw={r['id']:r for r in raw['paths']}
    for key in keys:
        ends,directions=endpoints(np.asarray(parameters),key);p=np.asarray(raw[key]['centerline_xy_enu_m']);reference=interpolate(p,u)
        reference+=(1-u[:,None])*(ends[0]-p[0])+u[:,None]*(ends[1]-p[-1]);targets[key]=reference
        source_targets[key]=interpolate(lanes[key]['centerline'],u)[:,:2]
        base=BoundaryShape(ends,directions,knots,np.zeros((3,2)))
        start_reference=reference
        if source_initialization:
            start_reference=source_targets[key].copy()
            start_reference+=(1-u[:,None])*(ends[0]-start_reference[0])+u[:,None]*(ends[1]-start_reference[-1])
        interior=np.column_stack([np.interp(knots[1:-1],u,start_reference[:,axis]) for axis in (0,1)])-base.base(knots[1:-1])
        offsets[key]=len(parameters);parameters.extend(interior.ravel())
        extent=np.linalg.norm(np.asarray(condition['region']['bbox_local_xy'][2:])-condition['region']['bbox_local_xy'][:2])+2*movement_limit
        bounds.extend([(-extent,extent)]*6 if restore_feasibility else [(None,None)]*6)
    scale_offsets={}
    if variable_tangents:
        for key in keys:
            scale_offsets[key]=len(parameters);parameters.extend([1.,1.]);bounds.extend([(.25,3.)]*2)
    initial=np.asarray(parameters)
    if initial_paths is not None:
        previous={r['id']:r['boundary_shape']for r in initial_paths['paths']}
        for ident in free:
            key,end=groups[ident][0];shape=previous[key];offset=nodes[ident]['offset']
            initial[offset:offset+2]=shape['endpoints_xy_m'][end]
            direction=shape['tangents_xy_unit'][end];angle=np.arctan2(direction[1],direction[0])
            initial[offset+2]+=np.arctan2(np.sin(angle-initial[offset+2]),np.cos(angle-initial[offset+2]))
        for key in keys:
            shape=previous[key];initial[offsets[key]:offsets[key]+6]=np.asarray(shape['interior_residual_xy_m']).ravel()
            if variable_tangents:initial[scale_offsets[key]:scale_offsets[key]+2]=shape['endpoint_tangent_scales']
    def shapes(x):
        result=[]
        for key in keys:
            ends,directions=endpoints(x,key);start=offsets[key]
            scales=x[scale_offsets[key]:scale_offsets[key]+2] if variable_tangents else None
            result.append(BoundaryShape(ends,directions,knots,x[start:start+6].reshape(3,2),scales))
        return result
    hermite=np.array([
        [2*u**3-3*u*u+1,-2*u**3+3*u*u,u**3-2*u*u+u,u**3-u*u],
        [6*u*u-6*u,-6*u*u+6*u,3*u*u-4*u+1,3*u*u-2*u],
        [12*u-6,-12*u+6,6*u-4,6*u-2]]).transpose(0,2,1)
    residual_spline=CubicSpline(knots,np.vstack([np.zeros(3),np.eye(3),np.zeros(3)]),
        bc_type=((1,np.zeros(3)),(1,np.zeros(3))))
    residual_basis=np.array([residual_spline(u,n) for n in range(3)])
    cache_x=None;cache_values=None;cache_lengths=None
    def evaluate(x):
        nonlocal cache_x,cache_values,cache_lengths
        if cache_x is not None and np.array_equal(cache_x,x):return cache_values,cache_lengths
        controls=[];residuals=[];lengths=[]
        for key in keys:
            ends,directions=endpoints(x,key);length=np.linalg.norm(ends[1]-ends[0]);lengths.append(length)
            scales=x[scale_offsets[key]:scale_offsets[key]+2] if variable_tangents else np.ones(2)
            controls.append(np.vstack([ends,length*directions*scales[:,None]]))
            residuals.append(x[offsets[key]:offsets[key]+6].reshape(3,2))
        cache_values=(np.einsum('dtb,nbc->dntc',hermite,np.asarray(controls))+
                      np.einsum('dtk,nkc->dntc',residual_basis,np.asarray(residuals)))
        cache_x=x.copy();cache_lengths=np.asarray(lengths)
        return cache_values,cache_lengths
    def objective(x):
        xy=evaluate(x)[0][0]
        return float(np.mean((xy-np.asarray([targets[key] for key in keys]))**2)+
                     .1*np.mean((xy-np.asarray([source_targets[key] for key in keys]))**2))
    obstacles=obstacle_shapes(condition);has_clearance=bool(obstacles)
    clearance=float(condition.get('physical_clearance',{}).get('clearance_m',0.))
    clearance_start=(6 if restore_feasibility else 2)*len(keys)*len(u)
    # Apply selected forward-progress constraints to inferred paths.
    directed_progress=condition.get('directed_progress',{})
    progress_axes={key:unit(np.asarray(row['axis_xy'],dtype=float)) for key,row in directed_progress.items()}
    progress_start=clearance_start+(len(keys)*len(u) if has_clearance else 0)
    def progress_margins(derivatives,key):
        if key not in directed_progress:return np.ones(len(derivatives))
        cosine=derivatives@progress_axes[key]/np.maximum(np.linalg.norm(derivatives,axis=1),1e-9)
        return cosine-float(directed_progress[key].get('minimum_cosine',0.))
    def path_constraint_rows(index):
        count=len(keys);samples=len(u)
        rows=[np.arange(index*samples,(index+1)*samples),np.arange((count+index)*samples,(count+index+1)*samples)]
        if restore_feasibility:
            rows.extend([np.arange((2*count+2*index)*samples,(2*count+2*index+2)*samples),np.arange((4*count+2*index)*samples,(4*count+2*index+2)*samples)])
        if has_clearance:rows.append(clearance_start+np.arange(index*samples,(index+1)*samples))
        if directed_progress:rows.append(progress_start+np.arange(index*samples,(index+1)*samples))
        return np.concatenate(rows)
    def constraints(x):
        (xy,d,dd),lengths=evaluate(x);speed=np.linalg.norm(d,axis=2)
        k=abs(d[:,:,0]*dd[:,:,1]-d[:,:,1]*dd[:,:,0])/np.maximum(speed,1e-9)**3
        threshold=np.asarray([limit if set(lanes[key]['allowed_classes']) & {'passenger','bus','delivery','truck','taxi','emergency'} else .5 for key in keys])
        curvature_limit=threshold[:,None]
        if width_aware_curvature:
            widths=np.asarray([np.interp(u,[0,1],lanes[key]['end_widths_m'])for key in keys])
            # Bound inner-offset curvature while retaining source widths.
            curvature_limit=np.minimum(curvature_limit,1.9/widths)
        values=[(curvature_limit-k).ravel(),(speed/np.maximum(lengths[:,None],1e-6)-.2).ravel()]
        if restore_feasibility:
            box=condition['region']['bbox_local_xy']
            values.extend([(xy-np.asarray(box[:2])+movement_limit).ravel(),
                           (np.asarray(box[2:])+movement_limit-xy).ravel()])
        if has_clearance:
            values.append(np.concatenate([margins(xy[i],np.interp(u,[0,1],lanes[key]['end_widths_m']),obstacles[key],clearance)for i,key in enumerate(keys)]))
        if directed_progress:values.append(np.concatenate([progress_margins(d[i],key)for i,key in enumerate(keys)]))
        return np.concatenate(values)
    def solver_constraints(x):
        values=constraints(x)
        if not compact_solver_constraints:return values
        count=len(keys);samples=len(u)
        # Use per-path minima to represent all sampled inequalities with a smaller Jacobian.
        compact=[values[:count*samples].reshape(count,samples).min(axis=1),
                 values[count*samples:2*count*samples].reshape(count,samples).min(axis=1)]
        if restore_feasibility:
            compact.extend([values[2*count*samples:4*count*samples].reshape(count,samples,2).min(axis=1).ravel(),
                            values[4*count*samples:6*count*samples].reshape(count,samples,2).min(axis=1).ravel()])
        if has_clearance:compact.append(values[clearance_start:clearance_start+count*samples].reshape(count,samples).min(axis=1))
        if directed_progress:compact.append(values[progress_start:].reshape(count,samples).min(axis=1))
        return np.concatenate(compact)
    feasibility=None
    if restore_feasibility and initial_paths is None:
        low=np.array([b[0] for b in bounds]);high=np.array([b[1] for b in bounds])
        initial=np.clip(initial,low+1e-8,high-1e-8)
        # Recover a bounded feasible starting shape after a failed constrained
        # solve; final acceptance still uses the original hard constraints.
        reference=initial.copy()
        def residual(x):
            return np.r_[np.minimum(constraints(x),0)*100.,(x-reference)*.001]
        sparse_options={}
        if sparse_feasibility:
            # Use the path constraints' sparse dependency on shared ports and interior variables.
            count=len(keys);samples=len(u);constraint_count=len(constraints(initial))
            row_indices=[];column_indices=[]
            for index,key in enumerate(keys):
                columns=set(range(offsets[key],offsets[key]+6))
                if variable_tangents:columns.update(range(scale_offsets[key],scale_offsets[key]+2))
                for end in (0,1):
                    node=nodes[root((key,end))]
                    if not node['fixed']:columns.update(range(node['offset'],node['offset']+3))
                columns=np.array(sorted(columns),dtype=int)
                rows=path_constraint_rows(index)
                row_indices.append(np.repeat(rows,len(columns)));column_indices.append(np.tile(columns,len(rows)))
            row_indices.append(constraint_count+np.arange(len(initial)));column_indices.append(np.arange(len(initial)))
            ri=np.concatenate(row_indices);ci=np.concatenate(column_indices)
            sparsity=coo_matrix((np.ones(len(ri)),(ri,ci)),shape=(constraint_count+len(initial),len(initial))).tocsr()
            sparse_options['jac_sparsity']=sparsity
            print(dict(region=condition['region']['id'],feasibility='sparse_local_path_jacobian',
                variables=len(initial),residuals=constraint_count+len(initial),nonzero_dependencies=sparsity.nnz),flush=True)
        recovered=least_squares(residual,initial,bounds=(low,high),max_nfev=150,
            ftol=1e-8,xtol=1e-8,gtol=1e-8,**sparse_options)
        initial=recovered.x
        feasibility=dict(evaluations=int(recovered.nfev),minimum_margin=float(constraints(initial).min()),
                         method='Bounded feasibility recovery before the same hard-constrained geometric projection',
                         sparse_local_path_jacobian=sparse_feasibility)
    history=[]
    def callback(x):
        history.append(dict(iteration=len(history)+1,objective=objective(x),minimum_margin=float(constraints(x).min())))
        if len(history)%25==0:
            print(dict(region=condition['region']['id'],**history[-1]),flush=True)
            if checkpoint_path:
                target=Path(checkpoint_path)
                state=dict(paths=[dict(id=k,boundary_shape=shape.as_dict())for k,shape in zip(keys,shapes(x))],
                    region=condition['region']['id'],iteration=len(history),objective=history[-1]['objective'],
                    minimum_constraint_margin=history[-1]['minimum_margin'],
                    status='Intermediate optimizer state; not an accepted or completed geometry result.')
                temporary=target.with_suffix('.tmp')
                temporary.write_text(json.dumps(state,separators=(',',':'),allow_nan=False)+'\n')
                temporary.replace(target)
    constraint_rounds=[]
    penalty_rounds=[]
    pathwise_rounds=[]
    if pathwise_feasibility_polish:
        # Refine individual path interiors with shared ports fixed.
        if initial_paths is None:raise ValueError('Pathwise polish requires an actual saved curve state')
        current=initial.copy();count=len(keys);samples=len(u);all_success=True;iterations=0
        for index,key in enumerate(keys):
            selected=path_constraint_rows(index);before=float(constraints(current)[selected].min())
            if before>=-1e-7:continue
            columns=list(range(offsets[key],offsets[key]+6))
            if variable_tangents:columns.extend(range(scale_offsets[key],scale_offsets[key]+2))
            columns=np.asarray(columns,dtype=int);center=current.copy()
            def expand(local):
                value=center.copy();value[columns]=local;return value
            local=minimize(lambda value:objective(expand(value)),current[columns],method='SLSQP',
                bounds=[bounds[j]for j in columns],
                constraints=[dict(type='ineq',fun=lambda value:constraints(expand(value))[selected]-1e-5)],
                options=dict(maxiter=max_iterations,ftol=solver_ftol))
            current=expand(local.x);iterations+=int(local.nit);all_success=all_success and bool(local.success)
            row=dict(path=key,variables=len(columns),iterations=int(local.nit),solver_success=bool(local.success),
                before_margin=before,after_margin=float(constraints(current)[selected].min()),message=str(local.message))
            pathwise_rounds.append(row);print(dict(region=condition['region']['id'],**row),flush=True)
        margin=float(constraints(current).min())
        solution=OptimizeResult(x=current,success=bool(all_success and margin>=-1e-7),
            message='Fixed shared-port path-local constrained correction; full sampled feasibility '+str(margin),
            nit=iterations,fun=objective(current))
    elif sparse_penalty_projection:
        # Find feasible starts with bounded sparse least squares and increasing inequality penalties.
        count=len(keys);samples=len(u);nc=len(constraints(initial))
        ri=[];ci=[]
        for index,key in enumerate(keys):
            columns=set(range(offsets[key],offsets[key]+6))
            if variable_tangents:columns.update(range(scale_offsets[key],scale_offsets[key]+2))
            for end in (0,1):
                node=nodes[root((key,end))]
                if not node['fixed']:columns.update(range(node['offset'],node['offset']+3))
            columns=np.asarray(sorted(columns),dtype=int)
            rows=[path_constraint_rows(index)]
            rows.extend([nc+np.arange(2*index*samples,2*(index+1)*samples),
                         nc+2*count*samples+np.arange(2*index*samples,2*(index+1)*samples)])
            rows=np.concatenate(rows);ri.append(np.repeat(rows,len(columns)));ci.append(np.tile(columns,len(rows)))
        ri=np.concatenate(ri);ci=np.concatenate(ci)
        pattern=coo_matrix((np.ones(len(ri)),(ri,ci)),shape=(nc+4*count*samples,len(initial))).tocsr()
        target_array=np.asarray([targets[key]for key in keys]);source_array=np.asarray([source_targets[key]for key in keys])
        norm=np.sqrt(target_array.size);current=initial;total=0
        low=np.asarray([-np.inf if b[0] is None else b[0]for b in bounds])
        high=np.asarray([np.inf if b[1] is None else b[1]for b in bounds])
        for weight in (penalty_weights or (10.,100.,1000.,10000.,100000.)):
            def penalty_residual(x):
                xy=evaluate(x)[0][0]
                return np.r_[np.minimum(constraints(x)-1e-5,0)*weight,
                    ((xy-target_array)/norm).ravel(),(np.sqrt(.1)*(xy-source_array)/norm).ravel()]
            # Optimize local displacements to decouple solver tolerances from city-coordinate magnitude.
            center=current.copy()
            answer=least_squares(lambda delta:penalty_residual(center+delta),np.zeros_like(center),
                bounds=(low-center,high-center),jac_sparsity=pattern,x_scale='jac',
                max_nfev=max_iterations,ftol=1e-10,xtol=1e-10,gtol=1e-8)
            current=center+answer.x;total+=int(answer.nfev);margin=float(constraints(current).min())
            record=dict(penalty_weight=weight,evaluations=int(answer.nfev),optimizer_success=bool(answer.success),
                objective=objective(current),minimum_constraint_margin=margin)
            penalty_rounds.append(record);print(dict(region=condition['region']['id'],**record),flush=True)
            if checkpoint_path:
                target=Path(checkpoint_path)
                target.write_text(json.dumps(dict(paths=[dict(id=k,boundary_shape=shape.as_dict())for k,shape in zip(keys,shapes(current))],
                    region=condition['region']['id'],penalty_round=record,status='Unaccepted sparse penalty state; final dense review required.'),separators=(',',':'))+'\n')
            if answer.success and margin>=-1e-7:break
        solution=OptimizeResult(x=current,success=bool(answer.success and margin>=-1e-7),
            message=str(answer.message)+('; full sampled inequalities feasible'if margin>=-1e-7 else '; full sampled inequalities require feedback'),
            nit=total,fun=objective(current))
    elif adaptive_constraint_generation:
        # Keep active samples fixed within each solve, then add violated dense extrema.
        count=len(keys);samples=len(u);indices=np.arange(len(constraints(initial)))
        series=[indices[:count*samples].reshape(count,samples),
                indices[count*samples:2*count*samples].reshape(count,samples)]
        if restore_feasibility:
            for start in [2*count*samples,4*count*samples]:
                block=indices[start:start+2*count*samples].reshape(count,samples,2)
                series.extend([block[:,:,axis]for axis in range(2)])
        if has_clearance:series.append(indices[clearance_start:clearance_start+count*samples].reshape(count,samples))
        if directed_progress:series.append(indices[progress_start:].reshape(count,samples))
        uniform=np.unique(np.linspace(0,samples-1,9).astype(int))
        active=set(int(i)for block in series for line in block for i in line[uniform])
        tolerance=max(solver_ftol,1e-6)
        def add_violated_extrema(x):
            values=constraints(x)
            for block in series:
                for line in block:
                    v=values[line];extrema=np.r_[0,find_peaks(-v)[0],samples-1,int(v.argmin())]
                    active.update(int(line[j])for j in extrema if v[j]<-tolerance)
            return values
        current=initial;iterations=0
        for outer in range(8):
            add_violated_extrema(current);selected=np.array(sorted(active),dtype=int)
            def selected_constraints(x):return constraints(x)[selected]
            solution=minimize(objective,current,method='SLSQP',bounds=bounds,
                constraints=[dict(type='ineq',fun=selected_constraints)],callback=callback,
                options=dict(maxiter=max_iterations,ftol=solver_ftol))
            current=solution.x;iterations+=int(solution.nit);full=constraints(current)
            record=dict(round=outer+1,active_constraints=len(selected),full_constraints=len(full),
                solver_success=bool(solution.success),iterations=int(solution.nit),minimum_full_margin=float(full.min()),
                objective=float(solution.fun))
            constraint_rounds.append(record);print(dict(region=condition['region']['id'],**record),flush=True)
            if solution.success and full.min()>=-tolerance:break
            before_count=len(active);add_violated_extrema(current)
            if len(active)==before_count and not solution.success:break
        solution.nit=iterations
        if constraints(solution.x).min()<-tolerance:
            solution.success=False
            solution.message=str(solution.message)+'; dense sampled inequalities still require feedback'
    else:
        solution=minimize(objective,initial,method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=solver_constraints)],
            callback=callback,options=dict(maxiter=max_iterations,ftol=solver_ftol))
    output=[]
    for key,shape in zip(keys,shapes(solution.x)):
        dense=np.linspace(0,1,1025);xy=shape(dense);d,dd=shape(dense,1),shape(dense,2)
        k=abs(d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/np.maximum(np.linalg.norm(d,axis=1),1e-9)**3
        normal=np.c_[-d[:,1],d[:,0]];normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        widths=np.interp(dense,[0,1],lanes[key]['end_widths_m']);envelope=Polygon(np.r_[xy+normal*widths[:,None]/2,(xy-normal*widths[:,None]/2)[::-1]])
        clearance_metrics={}
        if has_clearance:
            obstacle=obstacles[key];margin=margins(xy,widths,obstacle,clearance)
            clearance_metrics=dict(minimum_building_clearance_margin_m=float(margin.min()),building_envelope_overlap_m2=float(envelope.intersection(obstacle).area)if envelope.is_valid else None)
        if key in directed_progress:
            clearance_metrics['minimum_directed_progress_margin']=float(progress_margins(d,key).min())
        original=interpolate(lanes[key]['centerline'],dense);z=original[:,2].copy()
        for end in (0,1):
            delta=nodes[root((key,end))]['point'][2]-original[0 if end==0 else -1,2]
            z+=((1-dense) if end==0 else dense)*delta
        arc=np.r_[0.,np.linalg.norm(np.diff(xy,axis=0),axis=1).cumsum()]
        output.append(dict(id=key,centerline=np.c_[xy,z].tolist(),boundary_shape=shape.as_dict(),
            simple_xy=bool(LineString(xy).is_simple),valid_width_envelope=bool(envelope.is_valid),
            maximum_curvature_per_m=float(k.max()),maximum_grade=float(np.max(abs(np.diff(z))/np.maximum(np.diff(arc),1e-8))),
            source_curve_deviation_m=float(LineString(xy).hausdorff_distance(LineString(original[:,:2]))),**clearance_metrics))
    review=dict(solver_success=bool(solution.success),message=str(solution.message),iterations=int(solution.nit),
        objective=float(solution.fun),minimum_constraint_margin=float(constraints(solution.x).min()),
        free_shared_ports=len(free),fixed_boundary_ports=len(fixed),history=history,
        path_metrics=[{k:v for k,v in row.items() if k not in ('centerline','boundary_shape')} for row in output],
        method='Joint SLSQP geometric projection of generated paths, shared inferred port position/tangent, exact fixed AV2 or coarse outer boundaries. Source paths retained. Separate from learned sampling.')
    if feasibility is not None:review['feasibility_recovery']=feasibility
    review['variable_endpoint_tangent_magnitude']=variable_tangents
    review['constraint_samples']=constraint_samples
    review['width_aware_curvature']=width_aware_curvature
    review['initialization']=('previous_exact_curve_state'if initial_paths is not None else'source_feasibility_start_with_unchanged_sampled_objective'if source_initialization else'sampled')
    review['maximum_solver_iterations']=max_iterations
    review['sparse_penalty_projection']=sparse_penalty_projection
    review['penalty_projection_rounds']=penalty_rounds
    if sparse_penalty_projection:
        review['penalty_interior_constraint_margin']=1e-5
        review['penalty_solver_coordinates']='Displacements from current state with exact translated bounds and Jacobian column scaling'
    if sparse_penalty_projection:review['method']='Bounded sparse trust-region least-squares projection of the unchanged learned/source objective with increasing full sampled hard-inequality penalties; convergence and original dense physical acceptance required.'
    review['pathwise_feasibility_polish']=pathwise_feasibility_polish
    review['pathwise_polish_rounds']=pathwise_rounds
    if pathwise_feasibility_polish:review['pathwise_numerical_feasibility_tolerance']=1e-7
    if pathwise_feasibility_polish:review['method']='Fixed shared-port path-local SLSQP correction of actual residual failures; unchanged learned/source objective and full physical acceptance.'
    review['adaptive_constraint_generation']=adaptive_constraint_generation
    review['constraint_generation_rounds']=constraint_rounds
    review['compact_solver_constraints']=compact_solver_constraints
    review['solver_constraint_count']=len(solver_constraints(solution.x))
    review['full_constraint_count']=len(constraints(solution.x))
    review['solver_ftol']=solver_ftol
    review['port_direction_initialization']=port_direction_initialization
    review['directed_progress']=dict(paths=directed_progress,dense_verification_samples=1025)
    review['physical_building_clearance']=dict(active=has_clearance,clearance_m=clearance,source_scene=condition.get('physical_clearance',{}).get('source_scene'),dense_verification_samples=1025)
    review['passed']=bool(solution.success and all(r['simple_xy'] and r['valid_width_envelope'] and r['maximum_grade']<=.15 and
        r.get('minimum_directed_progress_margin',0.)>=-1e-4 and
        r['maximum_curvature_per_m']<=(limit if set(lanes[r['id']]['allowed_classes']) & {'passenger','bus','delivery','truck','taxi','emergency'} else .5)+.001 and
        (not has_clearance or (r['minimum_building_clearance_margin_m']>=-1e-4 and r['building_envelope_overlap_m2'] is not None and r['building_envelope_overlap_m2']<.01))for r in output))
    return output,review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('condition','raw','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--maximum-curvature',type=float,default=.2);p.add_argument('--movement-limit',type=float,default=3.)
    p.add_argument('--restore-feasibility',action='store_true',help='Recover a bounded starting point after an observed failed projection')
    p.add_argument('--variable-tangents',action='store_true',help='Optimize endpoint parameter speed while retaining exact endpoint positions and heading directions')
    p.add_argument('--constraint-samples',type=int,default=129,help='Increase for an observed curvature peak between the original constraint samples')
    p.add_argument('--width-aware-curvature',action='store_true',help='Tighten curvature where an observed wide road envelope folds')
    p.add_argument('--source-initialization',action='store_true',help='Initialize from supplied source shape after observed raw-initialized solve failure; sampled objective remains unchanged')
    p.add_argument('--initial-paths',type=Path,help='Continue an actual iteration-limited solve from its exact saved curve state')
    p.add_argument('--max-iterations',type=int,default=250)
    p.add_argument('--sparse-feasibility',action='store_true',help='Use exact path-local Jacobian sparsity for large bounded feasibility initialization')
    p.add_argument('--pathwise-feasibility-polish',action='store_true',help='Correct actual saved local residual failures without moving any shared port or other path')
    p.add_argument('--penalty-weights',type=float,nargs='+',help='Explicit continuation penalty schedule for an actual saved near-feasible state')
    p.add_argument('--sparse-penalty-projection',action='store_true',help='Project the unchanged learned objective using bounded sparse penalty continuation and full physical acceptance')
    p.add_argument('--adaptive-constraint-generation',action='store_true',help='Add dense violated extrema between fixed smooth constrained solves; require final full feasibility')
    p.add_argument('--compact-solver-constraints',action='store_true',help='Equivalent per-path minimum sampled inequalities for large SLSQP groups')
    p.add_argument('--solver-ftol',type=float,default=1e-9,help='Explicit SLSQP objective/convergence accuracy; physical acceptance checks are unchanged')
    p.add_argument('--checkpoint-path',type=Path,help='Save an unaccepted exact curve state every25 iterations for continued feedback')
    p.add_argument('--port-direction-initialization',choices=['source','chord'],default='source',help='Alternative initialization for an observed free-port heading basin caused by folded inferred source stubs; fixed headings remain exact')
    p.add_argument('--reason',required=True)
    a=p.parse_args();condition=json.loads(a.condition.read_text());raw=json.loads(a.raw.read_text())
    paths,review=project(condition,raw,a.maximum_curvature,a.movement_limit,a.restore_feasibility,a.variable_tangents,a.constraint_samples,a.width_aware_curvature,a.source_initialization,json.loads(a.initial_paths.read_text())if a.initial_paths else None,a.max_iterations,a.port_direction_initialization,a.solver_ftol,a.checkpoint_path,a.sparse_feasibility,a.compact_solver_constraints,a.adaptive_constraint_generation,a.sparse_penalty_projection,a.penalty_weights,a.pathwise_feasibility_polish);review.update(agent='decision_agent',reason=a.reason)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    (a.output/'paths.json').write_text(json.dumps(dict(paths=paths,condition=str(a.condition),raw=str(a.raw),passed=review['passed']),separators=(',',':'))+'\n')
    print({k:v for k,v in review.items() if k not in ('history','method')},flush=True)
    if not review['passed']:raise SystemExit(1)


if __name__=='__main__':main()
