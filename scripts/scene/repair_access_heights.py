"""Repair inferred group elevations after an actual grade failure.

Generated XY stays unchanged. Shared inferred ports have one height; observed
and outer ports stay fixed. A constrained height fit replaces a source-height
parameterization that became too steep after changing XY.
"""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import osqp
from scipy.sparse import diags,lil_matrix,vstack,eye


def repair(condition,paths,projection,maximum_grade=.12,maximum_change=.5):
    outputs={r['id']:dict(r) for r in paths['paths']};keys=condition['generated_path_ids']
    ports=[(key,end) for key in keys for end in (0,1)];parent={p:p for p in ports}
    def root(p):
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    for r in condition['boundaries']:
        if not r['fixed']:parent[root((r['lane_id'],r['end']))]=root((r['neighbor_id'],r['neighbor_end']))
    groups=defaultdict(list)
    for port in ports:groups[root(port)].append(port)
    nodes=list(groups);index={k:i for i,k in enumerate(nodes)}
    before=np.array([np.mean([outputs[k]['centerline'][0 if end==0 else -1][2] for k,end in groups[node]]) for node in nodes])
    lengths={key:float(np.linalg.norm(np.diff(np.asarray(outputs[key]['centerline'])[:,:2],axis=0),axis=1).sum()) for key in keys}
    joined={(r['lane_id'],r['end']) for r in condition['boundaries']}
    fixed={root((r['lane_id'],r['end'])) for r in condition['boundaries'] if r['fixed']}
    fixed.update(root(p) for p in ports if p not in joined)
    matrix=lil_matrix((len(keys),len(nodes)))
    for i,key in enumerate(keys):
        matrix[i,index[root((key,1))]]+=1.;matrix[i,index[root((key,0))]]-=1.
    limit=maximum_grade*np.array([lengths[k] for k in keys]);low=before-maximum_change;high=before+maximum_change
    for node in fixed:low[index[node]]=high[index[node]]=before[index[node]]
    solver=osqp.OSQP();solver.setup(P=diags(np.ones(len(nodes)),format='csc'),q=-before,
        A=vstack([matrix.tocsc(),eye(len(nodes),format='csc')],format='csc'),
        l=np.r_[-limit,low],u=np.r_[limit,high],eps_abs=1e-9,eps_rel=1e-9,max_iter=20000,polishing=True,verbose=False)
    solution=solver.solve();report=dict(status=solution.info.status,method=__doc__,maximum_grade=maximum_grade,
        maximum_allowed_port_height_change_m=maximum_change,fixed_ports=len(fixed),source_XY_changed=False)
    if solution.info.status_val not in (1,2):return None,dict(report,passed=False)
    heights=solution.x.copy()
    for node in fixed:heights[index[node]]=before[index[node]]
    changes=[]
    for key in keys:
        row=outputs[key];old=np.asarray(row['centerline']);arc=np.r_[0.,np.linalg.norm(np.diff(old[:,:2],axis=0),axis=1).cumsum()]
        z=np.interp(arc,[0,arc[-1]],[heights[index[root((key,0))]],heights[index[root((key,1))]]])
        changes.append(float(np.max(abs(z-old[:,2]))));row['centerline']=np.c_[old[:,:2],z].tolist()
        row['maximum_grade']=float(np.max(abs(np.diff(z))/np.maximum(np.diff(arc),1e-8)))
    lanes={l['id']:l for l in condition['lanes']}
    good=projection.get('geometry_passed',projection['solver_success']) and all(r['simple_xy'] and r['valid_width_envelope'] and r['maximum_grade']<=.15 and
        r['maximum_curvature_per_m']<= (.2 if 'passenger' in lanes[r['id']]['allowed_classes'] else .5)+.001 for r in outputs.values())
    if condition.get('physical_clearance'):
        good=good and all(r.get('minimum_building_clearance_margin_m',-1)>=-1e-4 and r.get('building_envelope_overlap_m2') is not None and r['building_envelope_overlap_m2']<.01 for r in outputs.values())
    report.update(passed=bool(good),maximum_port_height_change_m=float(np.max(abs(heights-before))),
        maximum_profile_height_change_m=max(changes),path_metrics=[{k:v for k,v in r.items() if k not in ('centerline','boundary_shape')} for r in outputs.values()])
    return dict(paths,paths=list(outputs.values()),passed=bool(good),height_repair=report),report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('condition','paths','projection-review','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reason',required=True)
    a=p.parse_args();result,review=repair(json.loads(a.condition.read_text()),json.loads(a.paths.read_text()),json.loads(a.projection_review.read_text()))
    review.update(agent='construction_agent',reason=a.reason,source_paths=str(a.paths))
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    if result is not None:(a.output/'paths.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print({k:v for k,v in review.items() if k not in ('path_metrics','method')},flush=True)
    if not review['passed']:raise SystemExit(1)


if __name__=='__main__':main()
