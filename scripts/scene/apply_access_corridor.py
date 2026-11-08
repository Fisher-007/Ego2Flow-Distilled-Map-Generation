"""Distribute an accepted corridor shape back to its original road identities."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from map_generation.access_fusion import section
from map_generation.road_builder import fit_lane,Curve


def restrict_coefficients(coefficients,start,end):
    c0,c1,c2,c3=coefficients;delta=end-start
    return np.array([c0+start*(c1+start*(c2+start*c3)),
        (c1+start*(2*c2+3*start*c3))*delta,
        (c2+3*start*c3)*delta**2,c3*delta**3])


def slice_geometry(geometry,start,end):
    curves=[]
    for i,curve in enumerate(geometry.curves):
        low,high=max(start,geometry.knots[i]),min(end,geometry.knots[i+1])
        if high-low<1e-8:continue
        p0,p1=curve.parameter(low-geometry.knots[i]),curve.parameter(high-geometry.knots[i])
        curves.append(restrict_coefficients(curve.coefficients,p0,p1).tolist())
    stations=np.linspace(start,end,max(3,int((end-start)/.25)+1))
    return curves,np.array([geometry.xyz(t) for t in stations])


def apply(source,manifest,candidate):
    before={l['id']:l for l in source['fusion_map']['lanes']}
    mutable={r['lane_id'] for r in manifest['segments']}
    updated=dict(source);updated['fusion_map']=dict(source['fusion_map'])
    updated['fusion_map']['lanes']=[deepcopy(l) if l['id'] in mutable else l for l in source['fusion_map']['lanes']]
    after={l['id']:l for l in updated['fusion_map']['lanes']}
    geometry,_=fit_lane(candidate,1.)
    portions=manifest['segments'];lengths=np.array([r['end_m']-r['start_m'] for r in portions]);cuts=np.r_[0.,lengths.cumsum()]/lengths.sum()*geometry.length
    applied=[]
    for i,row in enumerate(portions):
        key=row['lane_id'];lane=after[key];coefficients,points=slice_geometry(geometry,cuts[i],cuts[i+1])
        if row['start_m']>.01:
            prefix=section(before[key]['centerline'],0,row['start_m'])
            g,_=fit_lane(dict(before[key],centerline=prefix.tolist()),1.)
            coefficients=[c.coefficients.tolist() for c in g.curves]+coefficients
            points=np.vstack([prefix[:-1],points])
        if row['end_m']<row['length_m']-.01:
            suffix=section(before[key]['centerline'],row['end_m'],row['length_m'])
            g,_=fit_lane(dict(before[key],centerline=suffix.tolist()),1.)
            coefficients=coefficients+[c.coefficients.tolist() for c in g.curves]
            points=np.vstack([points,suffix[1:]])
        lane['centerline']=points.tolist();lane.pop('boundary_shape',None)
        lane['parametric_xy_segments']=coefficients
        lane['corridor_geometry_revision']=dict(seam_id=manifest['seam_id'],original_source_interval_m=[row['start_m'],row['end_m']],
            generated_corridor_interval_m=[float(cuts[i]),float(cuts[i+1])],generation=candidate.get('generation_provenance'),
            method='Arc-length subdivision of the accepted generated corridor; source route IDs, permissions and widths retained.')
        applied.append(key)
    changed_outside=[k for k in before if k not in applied and before[k]!=after[k]]
    changed_topology=[k for k in before if any(before[k].get(f)!=after[k].get(f) for f in ('successors','allowed_classes','lane_type','width_m','end_widths_m','speed_mps'))]
    changed_av2=[k for k in before if before[k]['source']=='av2' and before[k]!=after[k]]
    gaps=[float(np.linalg.norm(np.asarray(after[a['lane_id']]['centerline'][-1])-after[b['lane_id']]['centerline'][0])) for a,b in zip(portions,portions[1:])]
    outer=manifest['outer_lane_id'];side=0 if manifest['incoming'] else -1
    outer_error=float(np.linalg.norm(np.asarray(after[outer]['centerline'][side])-before[outer]['centerline'][side]))
    seam=manifest['seam_id'];fine_side=-1 if manifest['incoming'] else 0
    fine_error=float(np.linalg.norm(np.asarray(after[seam]['centerline'][fine_side])-before[seam]['centerline'][fine_side]))
    report=dict(applied_lanes=applied,changed_outside=changed_outside,changed_topology=changed_topology,changed_av2=changed_av2,
        internal_boundary_gaps_m=gaps,outer_boundary_error_m=outer_error,fine_boundary_error_m=fine_error,
        source_scene=manifest['source_scene'],method='Shared polynomial subdivision; no lane identity or traffic-rule changes.')
    report['passed']=not(changed_outside or changed_topology or changed_av2) and max(gaps+[outer_error,fine_error])<1e-6
    if not report['passed']:raise ValueError(str(report))
    updated['fusion_map']['corridor_revisions']=[*source['fusion_map'].get('corridor_revisions',[]),dict(manifest=manifest,review=report)]
    return updated,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','candidate','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();m=json.loads(a.manifest.read_text());source=json.loads(Path(m['source_scene']).read_text());c=json.loads(a.candidate.read_text())['lane']
    s,r=apply(source,m,c);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'scene.json').open('w') as f:json.dump(s,f,separators=(',',':'))
    (a.output/'application.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)


if __name__=='__main__':main()
