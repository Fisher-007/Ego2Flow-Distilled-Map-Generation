"""Diagnose one coherent rigid frame per reviewed group of AV2 source logs."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from av2.map.map_api import ArgoverseStaticMap
from scripts.data.source_lane_geometry import source_centerline
from scripts.data.diagnose_city_registration import distribution


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','data-root','diagnostic','conditions','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    config=json.loads(a.config.read_text());conditions=json.loads(a.conditions.read_text())
    diagnostic=json.loads(a.diagnostic.read_text())['cases']
    maps={};raw={};by_id=defaultdict(list)
    for rel in config['sources']['av2_logs']:
        path=next((a.data_root/rel/'map').glob('log_map_archive_*.json'));log=Path(rel).name
        raw[log]=json.loads(path.read_text());maps[log]=ArgoverseStaticMap.from_json(path)
        for ident in raw[log]['lane_segments']:by_id[ident].append(log)
    parent={k:k for k in maps}
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    def merge(logs):
        for k in logs[1:]:parent[root(k)]=root(logs[0])
    for logs in by_id.values():merge(logs)
    frame_join_evidence=[]
    if conditions.get('preserve_exact_referenced_joins',False):
        def endpoint(lane,end):
            return np.mean([[lane[side][-1 if end else 0][axis]for axis in ('x','y','z')]
                            for side in ('left_lane_boundary','right_lane_boundary')],axis=0)
        for log,data in raw.items():
            for ident,lane in data['lane_segments'].items():
                for nxt in lane['successors']:
                    for other in by_id.get(str(nxt),[]):
                        if other==log:continue
                        target=raw[other]['lane_segments'][str(nxt)]
                        first,last=endpoint(lane,True),endpoint(target,False)
                        if np.isfinite(first).all() and np.isfinite(last).all() and np.linalg.norm(first-last)<1e-6:
                            frame_join_evidence.append(dict(source_log=log,target_log=other,
                                source_lane_id=int(ident),target_lane_id=int(nxt),
                                raw_join_xyz_m=first.tolist(),raw_gap_m=float(np.linalg.norm(first-last))))
                            merge([log,other])
    for logs in conditions.get('coincident_frame_groups',[]):merge(logs)
    groups=defaultdict(list)
    for k in maps:groups[root(k)].append(k)
    fixed={root(k)for k in conditions['fixed_logs']}
    origin=np.asarray(config['coordinates']['origin_city_xyz'])
    records=[];linked=set();adjacent=defaultdict(set)
    selected=[(number,diagnostic[number-1]) for number in conditions['pair_indices']]
    selected.extend((case['id'],case) for case in conditions.get('additional_reviewed_pairs',[]))
    for number,c in selected:
        ga,gb=root(c['first_log']),root(c['reference_log'])
        pa,pb=[],[]
        for x,y in c['source_lane_pairs']:
            pa.append(source_centerline(maps[c['first_log']],maps[c['first_log']].vector_lane_segments[x])[::5]-origin)
            pb.append(source_centerline(maps[c['reference_log']],maps[c['reference_log']].vector_lane_segments[y])[::5]-origin)
        pa,pb=np.asarray(pa),np.asarray(pb)
        records.append(dict(index=number,first=c['first_log'],reference=c['reference_log'],ga=ga,gb=gb,a=pa,b=pb,lane_pairs=c['source_lane_pairs']))
        linked.update([ga,gb]);adjacent[ga].add(gb);adjacent[gb].add(ga)
    components=[];unseen=set(groups)
    while unseen:
        queue=[min(unseen)];component=set()
        while queue:
            k=queue.pop()
            if k in component:continue
            component.add(k);queue.extend(adjacent[k]-component)
        unseen-=component;components.append(component)
        if not component&linked:fixed.update(component)
        elif not component&fixed:raise ValueError('Agent must name a fixed source in each observed frame component')
    variable=sorted(linked-fixed);indices={k:i for i,k in enumerate(variable)}
    def transform(points,key,x):
        if key not in indices:return points
        v=x[6*indices[key]:6*indices[key]+6]
        return points@Rotation.from_rotvec(v[:3]).as_matrix().T+v[3:]
    def residual(x):
        chunks=[]
        for r in records:
            if r['ga']==r['gb']:continue
            chunks.append((transform(r['a'][::2].reshape(-1,3),r['ga'],x)-
                           transform(r['b'][::2].reshape(-1,3),r['gb'],x)).ravel())
        return np.concatenate(chunks)
    initial=np.zeros(6*len(variable))
    fit=least_squares(residual,initial,loss='soft_l1',f_scale=.15,
                      x_scale=np.tile([.005,.005,.005,2.,2.,2.],len(variable)),
                      max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-9)
    reviews=[]
    for r in records:
        pa,pb=r['a'][1::2].reshape(-1,3),r['b'][1::2].reshape(-1,3)
        before=pa-pb;after=transform(pa,r['ga'],fit.x)-transform(pb,r['gb'],fit.x)
        reviews.append(dict(index=r['index'],first_log=r['first'],reference_log=r['reference'],
                            source_lane_pairs=r['lane_pairs'],fit_lane_count=len(r['a'][::2]),
                            check_lane_count=len(r['a'][1::2]),
                            check_before_xyz_m=distribution(np.linalg.norm(before,axis=1)),
                            check_after_xyz_m=distribution(np.linalg.norm(after,axis=1)),
                            check_after_xy_m=distribution(np.linalg.norm(after[:,:2],axis=1)),
                            check_after_abs_z_m=distribution(abs(after[:,2]))))
    registrations={}
    for group in variable:
        v=fit.x[6*indices[group]:6*indices[group]+6];basis=Rotation.from_rotvec(v[:3]).as_matrix()
        translation=origin+v[3:]-basis@origin
        component=next(c for c in components if group in c)
        anchor=next(k for k in conditions['fixed_logs'] if root(k)in component)
        for log in groups[group]:
            points=np.array([[p[k]for k in ('x','y','z')]for lane in raw[log]['lane_segments'].values()
                             for side in ('left_lane_boundary','right_lane_boundary')for p in lane[side]],float)
            known=points[np.isfinite(points).all(1)]
            motion=(known-origin)@basis.T+v[3:]-(known-origin)
            registrations[log]=dict(reference_log=anchor,rotation_matrix=basis.tolist(),
                translation_city_m=translation.tolist(),rotation_xyz_deg=Rotation.from_matrix(basis).as_euler('xyz',degrees=True).tolist(),
                motion_xyz_m=distribution(np.linalg.norm(motion,axis=1)),
                frame_group_logs=groups[group],
                reason='Joint rigid source-frame proposal using Agent-reviewed physical corridor pairs. Shared-ID, reviewed exact referenced-join and explicitly coincident source frames move together; no individual measured lane is deformed.',
                applied=False)
    report=dict(proposed_registrations=registrations,pair_reviews=reviews,
                groups=list(groups.values()),exact_referenced_join_evidence=frame_join_evidence,fixed_group_logs=[groups[k]for k in sorted(fixed)],
                optimizer=dict(success=bool(fit.success),message=fit.message,nfev=fit.nfev,cost=fit.cost),
                conditions=str(a.conditions),source_diagnostic=str(a.diagnostic),applied=False,
                scope='Unapplied joint source-frame diagnosis. Alternate whole lanes are held out for checking. Physical identity, epoch ownership and retained interfaces remain Agent decisions; exact source geometry and corresponding camera/object frames must use the same eventual accepted rigid transform.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(optimizer=report['optimizer'],groups=report['groups'],fixed=report['fixed_group_logs'],
                         proposed_logs=len(registrations),pairs=[{k:v for k,v in x.items()if k!='source_lane_pairs'}for x in reviews]),indent=2))


if __name__=='__main__':
    main()
