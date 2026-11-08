"""Fit a generated corridor's interior shape under steering and endpoint constraints."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import LineString,Polygon
from shapely.ops import unary_union
from map_generation.boundary_shape import BoundaryShape


def constrain(shape,limit):
    u=np.unique(np.r_[shape.knots,np.linspace(0,1,257)])
    original=shape(u);parameters=shape.interior_residual.ravel();size=len(parameters)
    zero=BoundaryShape(shape.endpoints,shape.tangents,shape.knots,np.zeros_like(shape.interior_residual),shape.endpoint_tangent_scales)
    base=[zero(u,n) for n in range(3)];basis=[[],[],[]]
    for j in range(size):
        unit=np.zeros(size);unit[j]=1
        trial=BoundaryShape(shape.endpoints,shape.tangents,shape.knots,unit.reshape(-1,2),shape.endpoint_tangent_scales)
        for n in range(3):basis[n].append(trial(u,n)-base[n])
    basis=[np.asarray(b) for b in basis]
    def values(x,n):return base[n]+np.einsum('i,ijk->jk',x,basis[n])
    chord=np.linalg.norm(shape.endpoints[1]-shape.endpoints[0])
    def objective(x):return float(np.mean((values(x,0)-original)**2))
    def constraints(x):
        d,dd=values(x,1),values(x,2);speed=np.linalg.norm(d,axis=1)
        k=abs(d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/np.maximum(speed,1e-9)**3
        return np.r_[limit-k,speed/chord-.2]
    result=minimize(objective,parameters,method='SLSQP',constraints=[dict(type='ineq',fun=constraints)],
        options=dict(maxiter=250,ftol=1e-10))
    repaired=BoundaryShape(shape.endpoints,shape.tangents,shape.knots,result.x.reshape(-1,2),shape.endpoint_tangent_scales)
    report=dict(solver_success=bool(result.success),message=str(result.message),iterations=int(result.nit),
        target_maximum_curvature_per_m=limit,maximum_interior_displacement_m=float(np.linalg.norm(repaired(u)-original,axis=1).max()),
        objective_mean_squared_displacement_m2=float(result.fun),minimum_sampled_margin=float(constraints(result.x).min()),
        method='SLSQP geometric projection of generated interior residuals, with fixed endpoints and headings; reported separately from learned generation.')
    return repaired,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','candidate','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--maximum-curvature',type=float,required=True);p.add_argument('--reason',required=True)
    a=p.parse_args();source=json.loads(a.candidate.read_text());candidate=deepcopy(source);lane=candidate['lane']
    old=BoundaryShape.from_dict(lane['boundary_shape']);shape,review=constrain(old,a.maximum_curvature)
    u=np.unique(np.r_[shape.knots,np.linspace(0,1,1025)]);xy=shape(u);d=shape(u,1);dd=shape(u,2)
    curvature=abs(d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/np.linalg.norm(d,axis=1)**3
    normals=np.c_[-d[:,1],d[:,0]];normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    widths=np.interp(u,[0,1],lane.get('end_widths_m',[lane['width_m']]*2))
    envelope=Polygon(np.r_[xy+normals*widths[:,None]/2,(xy-normals*widths[:,None]/2)[::-1]])
    scene=json.loads(a.scene.read_text());buildings=unary_union([Polygon(b['exterior'],b['holes']) for b in scene['buildings']])
    review.update(maximum_curvature_per_m=float(curvature.max()),simple_xy=bool(LineString(xy).is_simple),
        valid_width_envelope=bool(envelope.is_valid),inside_building_m=float(LineString(xy).intersection(buildings).length),
        agent='construction_agent',reason=a.reason,source_candidate=str(a.candidate))
    review['passed']=bool(review['solver_success'] and review['simple_xy'] and review['valid_width_envelope']
        and review['inside_building_m']<.01 and review['maximum_curvature_per_m']<=a.maximum_curvature+1e-4)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'constraint_review.json').write_text(json.dumps(review,indent=2)+'\n')
    print(review,flush=True)
    if not review['passed']:raise SystemExit(1)
    previous=np.asarray(lane['centerline']);old_arc=np.r_[0.,np.linalg.norm(np.diff(previous[:,:2],axis=0),axis=1).cumsum()]
    arc=np.r_[0.,np.linalg.norm(np.diff(xy,axis=0),axis=1).cumsum()]
    lane['centerline']=np.c_[xy,np.interp(arc/arc[-1],old_arc/old_arc[-1],previous[:,2])].tolist()
    lane['boundary_shape']=shape.as_dict();lane['generation_provenance']['geometric_constraint_projection']=str(a.output/'constraint_review.json')
    candidate['geometric_constraint_projection']=review
    candidate['pre_projection_static_geometry']=candidate.get('static_geometry')
    candidate['static_geometry']={key:review[key] for key in (
        'maximum_curvature_per_m','simple_xy','valid_width_envelope','inside_building_m')}
    (a.output/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')


if __name__=='__main__':main()
