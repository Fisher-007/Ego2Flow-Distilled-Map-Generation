"""Propose inferred road cross sections fitted to measured walls."""
import argparse,json,re
from copy import deepcopy
from pathlib import Path
import numpy as np
import osqp
from scipy import sparse
from shapely.geometry import Point,LineString
from map_generation.building_passages import BuildingVolumes


def fit(scene,measurements,conditions):
    out=deepcopy(scene);lanes={l['id']:l for l in out['fusion_map']['lanes']}
    before={l['id']:l for l in scene['fusion_map']['lanes']};reviews={r['osm_way_id']:r for r in measurements['records']}
    volume=BuildingVolumes(scene['buildings']);records=[];updates={}
    for choice in conditions['corridors']:
        key=str(choice['osm_way_id']);source=reviews[key];line=LineString(source['original_way_xy'])
        rows=source['stations'];station=np.asarray([r['station_m']for r in rows]);n=len(rows)
        width=float(choice['lane_width_m']);margin=float(choice.get('wall_margin_m',.2));maximum=float(choice.get('maximum_center_shift_m',2.))
        lower=np.asarray([-maximum if r['right_m']is None else max(-maximum,width+margin-r['right_m'])for r in rows])
        upper=np.asarray([maximum if r['left_m']is None else min(maximum,r['left_m']-width-margin)for r in rows])
        if np.any(lower>upper):raise ValueError('Requested two-way cross section does not fit measured corridor '+key)
        d1=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n))
        d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))
        h=sparse.eye(n)+.4*(d1.T@d1)+4*(d2.T@d2)
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(2*h),q=np.zeros(n),A=sparse.eye(n,format='csc'),l=lower,u=upper,verbose=False,eps_abs=1e-8,eps_rel=1e-8,polishing=True)
        result=solver.solve()
        if result.info.status!='solved':raise ValueError('Corridor offset solve failed: '+result.info.status)
        shift=result.x
        chosen=[l for l in lanes.values()if l['source']=='osm_sumo' and not l.get('internal')and re.sub(r'^[+-]','',l.get('edge_id','')).split('#')[0]==key]
        lane_reviews=[]
        for lane in chosen:
            old=deepcopy(lane);p=np.asarray(old['centerline']);q=p.copy();changes=[]
            for i,point in enumerate(p):
                s=line.project(Point(point[:2]));center=np.asarray(line.interpolate(s).coords[0])
                a=np.asarray(line.interpolate(max(0.,s-.05)).coords[0]);b=np.asarray(line.interpolate(min(line.length,s+.05)).coords[0])
                direction=b-a;direction/=np.linalg.norm(direction);normal=np.array([-direction[1],direction[0]])
                previous=float((point[:2]-center)@normal);delta=float(np.interp(s,station,shift))+previous*(width/old['width_m']-1)
                q[i,:2]+=delta*normal;changes.append(float(np.linalg.norm(q[i,:2]-point[:2])))
            lane['centerline']=q.tolist();lane['width_m']=width;lane['end_widths_m']=[width,width]
            for field in ['boundary_shape','parametric_xy_segments','parametric_elevation_rows']:lane.pop(field,None)
            lane['source_corridor_revision']=dict(osm_way_id=key,reason=choice['reason'],source_width_m=old['width_m'],inferred_width_m=width,maximum_xy_change_m=max(changes),
                source_way_and_buildings_retained=True,permissions_retained=True,geometry_status='proposal_requires_transition_and_generation_refresh')
            obstacles=volume.obstacles(q,width);prior_obstacles=volume.obstacles(p,old['width_m'])
            before_overlap=LineString(p[:,:2]).buffer(old['width_m']/2,cap_style='flat').intersection(prior_obstacles).area
            after_overlap=LineString(q[:,:2]).buffer(width/2,cap_style='flat').intersection(obstacles).area
            review=dict(lane_id=lane['id'],maximum_xy_change_m=max(changes),before_building_overlap_m2=float(before_overlap),after_building_overlap_m2=float(after_overlap),
                endpoint_displacements_xy_m=(q[[0,-1],:2]-p[[0,-1],:2]).tolist())
            lane_reviews.append(review);updates[lane['id']]=review
        records.append(dict(osm_way_id=key,inferred_lane_width_m=width,wall_margin_m=margin,source_measurement=source['bounded_width_min_m'],maximum_center_shift_m=float(abs(shift).max()),
            original_station_m=station.tolist(),center_shift_m=shift.tolist(),solver_status=result.info.status,lanes=lane_reviews,reason=choice['reason']))
    gaps=[]
    for lane in out['fusion_map']['lanes']:
        for target in lane['successors']:
            if target not in lanes or not({lane['id'],target}&set(updates)):continue
            old_gap=float(np.linalg.norm(np.asarray(before[lane['id']]['centerline'][-1])-before[target]['centerline'][0]))
            gap=float(np.linalg.norm(np.asarray(lane['centerline'][-1])-lanes[target]['centerline'][0]))
            if gap>old_gap+1e-6:gaps.append(dict(before=lane['id'],after=target,old_gap_m=old_gap,proposal_gap_m=gap))
    av2=all(before[l['id']]==l for l in out['fusion_map']['lanes']if l['source']=='av2')
    report=dict(records=records,updated_lanes=len(updates),changed_observed_av2=not av2,topology_and_permissions_unchanged=all(before[l['id']]['successors']==l['successors']and before[l['id']]['allowed_classes']==l['allowed_classes']for l in out['fusion_map']['lanes']),
        increased_source_port_gaps=gaps,remaining_overlap_lanes=[x for r in records for x in r['lanes']if x['after_building_overlap_m2']>=.01],
        accepted_map=False,scope='Explicit inferred transverse widths and bounded center offsets from source-way/wall evidence. Raw OSM, buildings, observed AV2, topology and source permissions retained. Port propagation and current-input structure generation are required before acceptance.')
    out['fusion_map']['source_corridor_proposal']=report;return out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','measurements','conditions','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(p.read_text())
    scene,report=fit(read(a.scene),read(a.measurements),read(a.conditions));a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(scene,separators=(',',':'))+'\n');(a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items()if k not in ['records','increased_source_port_gaps','remaining_overlap_lanes']},flush=True)
    print(dict(increased_port_gaps=len(report['increased_source_port_gaps']),remaining_overlap_lanes=report['remaining_overlap_lanes']),flush=True)
if __name__=='__main__':main()
