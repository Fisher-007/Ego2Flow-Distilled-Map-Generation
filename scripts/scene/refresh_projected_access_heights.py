"""Refresh inferred road heights while reusing verified unchanged XY objectives."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from shapely.geometry import LineString,Polygon
from map_generation.road_builder import fit_lane
from scripts.scene.constrain_access_group import interpolate
from scripts.scene.constrain_access_batch import projection_inputs
from scripts.scene.repair_access_heights import repair


def rebase(condition,old_condition,raw,old_raw,previous,previous_review,tolerance=1e-9):
    if raw['paths']!=old_raw['paths']:return None,dict(passed=False,status='new_raw_objective_requires_projection')
    if not previous['passed']or not previous_review['passed']:return None,dict(passed=False,status='previous_result_not_accepted')
    if projection_inputs(condition)==projection_inputs(old_condition):
        return dict(previous),dict(previous_review,status='all_projection_dependencies_unchanged',previous_geometry_reused=True,actual_raw_arrays_identical=True,source_xy_arrays_unchanged=True,exact_xyz_reuse=True,method='All actual raw arrays and consumed projection/application dependencies are identical; preserve the complete prior accepted XYZ result and its measured review.')
    if condition.get('physical_clearance')!=old_condition.get('physical_clearance'):return None,dict(passed=False,status='changed_physical_clearance_requires_projection')
    for key in ['generated_path_ids','boundaries','paths']:
        if condition[key]!=old_condition[key]:return None,dict(passed=False,status='changed_'+key)
    if condition['region']['bbox_local_xy']!=old_condition['region']['bbox_local_xy']:return None,dict(passed=False,status='changed_region_bounds')
    lanes={l['id']:l for l in condition['lanes']};old={l['id']:l for l in old_condition['lanes']}
    keys=condition['generated_path_ids'];u=np.linspace(0,1,1025);difference=0.
    for key in keys:
        for field in ['allowed_classes','end_widths_m']:
            if lanes[key][field]!=old[key][field]:return None,dict(passed=False,status='changed_'+field)
        difference=max(difference,float(abs(interpolate(lanes[key]['centerline'],u)[:,:2]-interpolate(old[key]['centerline'],u)[:,:2]).max()))
    ports=[(k,e)for k in keys for e in [0,1]];parent={p:p for p in ports}
    def root(p):
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    for row in condition['boundaries']:
        if not row['fixed']:parent[root((row['lane_id'],row['end']))]=root((row['neighbor_id'],row['neighbor_end']))
    groups=defaultdict(list)
    for port in ports:groups[root(port)].append(port)
    heights={key:float(np.mean([lanes[k]['centerline'][0 if e==0 else -1][2]for k,e in members]))for key,members in groups.items()}
    fixed_points={};fixed_directions={};cache={}
    for row in condition['boundaries']:
        if not row['fixed']:continue
        key,end=row['neighbor_id'],row['neighbor_end'];port=root((row['lane_id'],row['end']))
        if key not in cache:cache[key]=(fit_lane(lanes[key],1.)[0],fit_lane(old[key],1.)[0])
        current,prior=cache[key]
        point=current.xyz(0. if end==0 else current.length);old_point=prior.xyz(0. if end==0 else prior.length)
        direction=current.curves[0].tangent(0.)if end==0 else current.curves[-1].tangent(1.)
        old_direction=prior.curves[0].tangent(0.)if end==0 else prior.curves[-1].tangent(1.)
        difference=max(difference,float(abs(point[:2]-old_point[:2]).max()),float(abs(direction-old_direction).max()))
        heights[port]=float(point[2]);fixed_points[port]=point[:2];fixed_directions[port]=direction
    if difference>tolerance:return None,dict(passed=False,status='changed_xy_objective_or_fixed_port',maximum_xy_dependency_difference=difference)
    joined={(row['lane_id'],row['end'])for row in condition['boundaries']}
    for k,e in ports:
        if(k,e)not in joined:
            port=root((k,e));point=np.asarray(lanes[k]['centerline'][0 if e==0 else -1])
            fixed_points[port]=point[:2];heights[port]=float(point[2])
    outputs=[];fixed_error=0.
    for row in previous['paths']:
        q=dict(row);key=q['id'];points=np.asarray(q['centerline']);sample=np.linspace(0,1,len(points))
        source=interpolate(lanes[key]['centerline'],sample);z=source[:,2].copy()
        for end in [0,1]:
            port=root((key,end));index=0 if end==0 else -1
            z+=((1-sample)if end==0 else sample)*(heights[port]-source[index,2])
            if port in fixed_points:fixed_error=max(fixed_error,float(np.linalg.norm(points[index,:2]-fixed_points[port])))
        q['centerline']=np.c_[points[:,:2],z].tolist();outputs.append(q)
    if fixed_error>1e-6:return None,dict(passed=False,status='previous_curve_fixed_port_mismatch',maximum_fixed_xy_error_m=fixed_error)
    # XY controls, sampled points and their measured envelope/curvature remain
    # exactly the prior accepted arrays. Only Z is reconstructed then solved.
    projection=dict(solver_success=True,geometry_passed=True)
    result,report=repair(condition,dict(previous,paths=outputs,passed=False),projection)
    report.update(status='height_refresh_passed'if report['passed']else'height_refresh_requires_feedback',
        previous_geometry_reused=True,actual_raw_arrays_identical=True,maximum_xy_dependency_difference=difference,
        numerical_xy_equivalence_tolerance=tolerance,maximum_fixed_xy_error_m=fixed_error,
        method='Previous strict XY solution reused only after identical actual learned arrays, identical path/port identities and permissions/widths, and numerical equivalence of all source XY objectives and fixed fitted XY/tangents. Reconstruct current source/shared/fixed heights and solve the ordinary unchanged grade constraints.',
        source_xy_arrays_unchanged=True)
    if result is not None:
        for old_row,new_row in zip(previous['paths'],result['paths'],strict=True):
            if np.asarray(old_row['centerline'])[:,:2].tolist()!=np.asarray(new_row['centerline'])[:,:2].tolist():raise ValueError('XY changed during height refresh')
    return result,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['batch-root','previous-batch','previous-selection','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());selection=read(a.previous_selection)['regions'];records=[];accepted={}
    for row in read(a.batch_root/'manifest.json')['records']:
        name=row['region'];directory=a.batch_root/name;old=a.previous_batch/name;chosen=selection[name]
        result,review=rebase(read(directory/'condition.json'),read(old/'condition.json'),read(directory/'generation/raw_paths.json'),read(old/'generation/raw_paths.json'),read(chosen['paths']),read(chosen['review']))
        review.update(previous_selection=chosen,current_condition=str(directory/'condition.json'),current_raw=str(directory/'generation/raw_paths.json'))
        out=directory/'source_height_refresh_v1';out.mkdir(exist_ok=True);(out/'review.json').write_text(json.dumps(review,indent=2)+'\n')
        if result is not None:
            result.update(condition=str(directory/'condition.json'),raw=str(directory/'generation/raw_paths.json'))
            (out/'paths.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
        item=dict(region=name,passed=review['passed'],status=review['status']);records.append(item)
        if review['passed']:accepted[name]=dict(paths=str((out/'paths.json').resolve()),review=str((out/'review.json').resolve()),reason=review['method'])
        print(item,flush=True)
    a.output.mkdir(parents=True,exist_ok=True);report=dict(configured_regions=len(records),passed=len(accepted),requires_current_projection=len(records)-len(accepted),regions=accepted,records=records)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k not in ['regions','records']},flush=True)
if __name__=='__main__':main()
