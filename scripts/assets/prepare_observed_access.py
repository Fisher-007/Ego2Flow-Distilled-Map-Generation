"""Construct selected access surfaces from calibrated rays, LiDAR, and roads."""
import argparse
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np
import shapely
from scipy.spatial import cKDTree

from scripts.scene.build_assets import prepare_terrain, surface_heights, write_json
from map_generation.native_conditions import observed_accesses
from scripts.scene.frontage_coverage import coverage, candidates


def camera(history, index):
    row=history['frames'][index];ego=np.eye(4)
    ego[:3,:3]=row['ego']['city_rotation'];ego[:3,3]=row['ego']['position_local_xyz_m']
    return ego@np.asarray(history['ego_SE3_camera_cv'])


def ray(history,index,uv):
    pose=camera(history,index)
    direction=pose[:3,:3]@np.linalg.solve(history['K_av2'],[*uv,1.])
    return pose[:3,3],direction/np.linalg.norm(direction)


def project(history,index,xyz):
    pose=camera(history,index);points=(np.asarray(xyz)-pose[:3,3])@pose[:3,:3]
    uv=points@np.asarray(history['K_av2']).T
    return uv[:,:2]/uv[:,2:]


def triangulate(rays):
    planes=[np.eye(3)-np.outer(d,d) for c,d in rays]
    point=np.linalg.solve(sum(planes),sum(p@r[0] for p,r in zip(planes,rays)))
    return point,[float(np.linalg.norm(np.cross(point-c,d))) for c,d in rays]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--measure-only',action='store_true');p.add_argument('--conditions',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(a.selection.read_text());history=json.loads(Path(spec['history']).read_text())
    scene=json.loads(Path(spec['scene']).read_text());roads=json.loads(Path(spec['road_meshes']).read_text())
    points=np.load(spec['static_points'])['xyz'];tree=cKDTree(points[:,:2])
    road_vertices=np.concatenate([np.asarray(m['vertices']) for m in roads['meshes']])
    road_vertex_tree=cKDTree(road_vertices[:,:2])
    def ground_z(xy):
        nominal=road_vertices[road_vertex_tree.query(xy)[1],2]
        near=tree.query_ball_point(xy,1.)
        heights=points[near,2];heights=heights[abs(heights-nominal)<1.0]
        return float(np.quantile(heights,.25)) if len(heights)>=10 else float(nominal)
    sign=spec['sign'];corners=[];residuals=[]
    for k in range(4):
        xyz,errors=triangulate([ray(history,o['history_index'],o['corners_uv'][k]) for o in sign['observations']])
        corners.append(xyz);residuals.extend(errors)
    corners=np.array(corners);center=corners.mean(0)
    tangent=(corners[1]+corners[2]-corners[0]-corners[3])/2
    width=float(np.linalg.norm(tangent[:2]));tangent=np.r_[tangent[:2]/width,0.]
    height=float((corners[:2,2].mean()-corners[2:,2].mean()))
    normal=np.cross(tangent,[0,0,1.])
    if np.dot(normal,camera(history,sign['observations'][-1]['history_index'])[:3,3]-center)<0:normal=-normal;tangent=-tangent
    sign.update(center_local_xyz_m=center.tolist(),width_m=width,height_m=height,
                tangent_xyz=tangent.tolist(),front_normal_xyz=normal.tolist(),ground_z_m=ground_z(center[:2]),
                triangulation_max_ray_distance_m=max(residuals),corner_xyz_m=corners.tolist())
    triangles=np.concatenate([np.asarray(m['vertices'])[m['faces']] for m in roads['meshes']])
    polys=shapely.polygons(triangles[:,:,:2]);road_tree=shapely.STRtree(polys);footprint=shapely.union_all(polys)
    for access in observed_accesses(spec):
        mouth=[]
        for uv in access['mouth_uv']:
            c,d=ray(history,access['history_index'],uv);z=0.
            for _ in range(8):
                xyz=c+d*(z-c[2])/d[2];z=ground_z(xyz[:2])
            mouth.append(xyz[:2])
        mouth=np.array(mouth);width=float(np.linalg.norm(mouth[1]-mouth[0]));mid=mouth.mean(0)
        source_road=next(r for r in scene['roads'] if r['id']==access['source_road'])
        line=np.array([v[:2] for v in source_road['reference_line']])
        if 'interior_distance_m' in access:
            # Restore the observed service-road mouth.
            nearest=int(np.argmin(np.linalg.norm(line-mid,axis=1)))
            other=nearest+1 if nearest<len(line)-1 else nearest-1
            forward=line[other]-line[nearest];forward/=np.linalg.norm(forward)
            if np.dot(forward,line.mean(0)-mid)<0:forward=-forward
            target=mid+forward*access['interior_distance_m']
        else:
            target=line[-1];forward=target-mid;forward/=np.linalg.norm(forward)
        right=np.array([forward[1],-forward[0]])
        polygon=shapely.Polygon([mid-forward*2-right*width/2,mid-forward*2+right*width/2,
                               target+forward*3+right*width/2,target+forward*3-right*width/2])
        missing=shapely.difference(polygon,footprint)
        access.update(mouth_local_xy_m=mouth.tolist(),interior_local_xy_m=target.tolist(),width_m=width,
                      polygon_local_xy_m=list(map(list,polygon.exterior.coords)),
                      added_surface_area_m2=float(missing.area),native_navigation_binding='not_added_in_this_visual_and_physical_repair')
    from scripts.scene.access_ground import fit, apply
    for access in observed_accesses(spec):
        if access.get('fit_lidar_ground'):
            access['ground_fit']=fit(access,points)
    if 'coverage_margin_m' in spec['streetside']:
        spec['coverage']=coverage(history,spec['streetside'])
        spec['streetside']['tangent_range_m']=spec['coverage']['tangent_range_m']
        spec['access_candidates']=candidates(scene,history,spec['streetside'],observed_accesses(spec))
    if a.measure_only:
        write_json(out/'observed_features.json',spec)
        print(out/'observed_features.json');return
    if a.conditions:
        from map_generation.native_conditions import selected_features
        spec=selected_features(spec,json.loads(a.conditions.read_text()))
        sign=spec['sign']
    ground_reports=[apply(scene,roads,access) for access in observed_accesses(spec) if access.get('ground_fit')]
    write_json(out/'access_ground.json',ground_reports)
    triangles=np.concatenate([np.asarray(m['vertices'])[m['faces']] for m in roads['meshes']])
    polys=shapely.polygons(triangles[:,:,:2]);road_tree=shapely.STRtree(polys)
    # Consume each compiled graph selection. Known road faces remain unchanged.
    for access in observed_accesses(spec):
        polygon=shapely.Polygon(access['polygon_local_xy_m'])
        missing=shapely.difference(polygon,footprint)
        tris=shapely.get_parts(shapely.constrained_delaunay_triangles(missing))
        xy=shapely.get_coordinates(tris).reshape(-1,4,2)[:,:3].reshape(-1,2)
        near=shapely.get_coordinates(shapely.shortest_line(shapely.points(xy),footprint.boundary)).reshape(-1,2,2)[:,1]
        z=surface_heights(triangles,road_tree,near,'access endpoints')
        if access.get('ground_fit'):
            model=access['ground_fit']
            measured=np.c_[xy-np.array(model['center_xy_m']),np.ones(len(xy))]@np.array(model['coefficients'])
            # Use road heights at contacts and measured ground heights inside the patch.
            weight=np.clip(np.linalg.norm(xy-near,axis=1)/.75,0,1)
            z=z*(1-weight)+measured*weight
        vertices=np.c_[xy,z];faces=np.arange(len(vertices)).reshape(-1,3)
        wrong=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])[:,2]<0
        faces[wrong]=faces[wrong,::-1]
        roads['meshes'].append({'id':access['id'],'vertices':vertices.tolist(),'faces':faces.tolist(),
            'provenance':{'OSM':access['source_road'],'source_pixels':access['mouth_uv'],
                          'note':'Observed physical access; no invented OpenDRIVE routing relation.'}})
    from scripts.scene.repair_support_surface import repair
    support_report=repair(scene,roads,points,spec['support_region'])
    write_json(out/'support_surface.json',support_report)
    write_json(out/'road_meshes.json',roads);write_json(out/'scene.json',scene)
    with (out/'terrain_build.log').open('w') as log, contextlib.redirect_stdout(log):
        prepare_terrain(scene,roads,out)
    # Fit road-relative curb profiles along the open access mouth.
    terrain=json.loads((out/'terrain_mesh.json').read_text());cfg=spec['streetside']
    tris=np.concatenate([np.asarray(m['vertices'])[m['faces']] for m in roads['meshes']])
    road_polys=shapely.polygons(tris[:,:,:2]);road_tree=shapely.STRtree(road_polys);union=shapely.union_all(road_polys)
    n=np.array(cfg['normal_xy']);t=np.array([-n[1],n[0]])
    openings=shapely.union_all([shapely.Polygon(a['polygon_local_xy_m']) for a in observed_accesses(spec)])
    def profile(vertices):
        v=np.array(vertices);q=v[:,:2]@n;s=v[:,:2]@t
        candidates=(q>=cfg['normal_range_m'][0])&(q<=cfg['normal_range_m'][1]+3)&(s>=cfg['tangent_range_m'][0])&(s<=cfg['tangent_range_m'][1])
        ids=np.flatnonzero(candidates)
        near=shapely.get_coordinates(shapely.shortest_line(shapely.points(v[ids,:2]),union.boundary)).reshape(-1,2,2)[:,1]
        distance=np.linalg.norm(v[ids,:2]-near,axis=1);edgeq=near@n
        active=(edgeq>=cfg['normal_range_m'][0])&(edgeq<=cfg['normal_range_m'][1])&(distance<3.2)
        ids=ids[active];near=near[active];distance=distance[active]
        roadz=surface_heights(tris,road_tree,near,'curb')
        ramp=np.clip(shapely.distance(shapely.points(near),openings)/1.5,0,1)
        target=roadz+cfg['curb_height_m']*ramp
        weight=np.clip((3.2-distance)/(3.2-cfg['sidewalk_width_m']),0,1)
        delta=(target-v[ids,2])*weight;v[ids,2]+=delta
        return v,ids,delta
    v,ids,delta=profile(terrain['vertices']);terrain['vertices']=v.tolist()
    from scripts.scene.repair_support_surface import close_surface_edges
    close_surface_edges(terrain,roads)
    terrain['report']['observed_curb_profile']={'vertices_changed':len(ids),'height_m':cfg['curb_height_m'],
        'max_abs_height_change_m':float(np.max(abs(delta))),'access_ramp_m':1.5,'scope':cfg['height_scope'],
        'tangent_range_m':cfg['tangent_range_m'],'access_ids':[a['id'] for a in observed_accesses(spec)]}
    write_json(out/'terrain_mesh.json',terrain)
    write_json(out/'observed_features.json',spec)
    for index in sorted({*[a['history_index'] for a in observed_accesses(spec)],
                         *[o['history_index'] for o in sign['observations']]}):
        image=cv2.imread(history['frames'][index]['source_image'])
        uv=project(history,index,corners).round().astype(int);cv2.polylines(image,[uv],True,(0,220,255),2)
        for access in observed_accesses(spec):
            if access['history_index']!=index:continue
            polygon=shapely.Polygon(access['polygon_local_xy_m'])
            accessxyz=np.array([[*p,ground_z(p)] for p in polygon.exterior.coords])
            uv=project(history,index,accessxyz).round().astype(int);cv2.polylines(image,[uv],True,(0,220,0),3)
            for uv in access['mouth_uv']:cv2.circle(image,tuple(uv),8,(255,0,255),3)
        cv2.imwrite(str(out/f'evidence_{index}.jpg'),image)
    print(json.dumps({'accesses':observed_accesses(spec),'curb':terrain['report']['observed_curb_profile']},indent=2))


if __name__=='__main__':main()
