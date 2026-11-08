"""Prepare source-based corrections to inferred trunks, paint, and land cover."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import shapely


def inferred_lane_conflicts(canopy, roads, clearance_m=4.5):
    """Review inferred trunks against the final lane layer and vehicle headroom."""
    from map_generation.av2_surfaces import plane_z
    rows=[dict(item,candidate=i) for i,item in enumerate(canopy['canopy_candidates'])
          if item.get('context_kind')=='inferred_outer']
    if not rows:return []
    points=np.asarray([x['position_enu_m'] for x in rows])
    heights=np.asarray([x['target_height_m'] for x in rows])
    tree=shapely.STRtree(shapely.points(points[:,:2]));conflicts={}
    for mesh in roads['meshes']:
        # Shared paving has no lane ownership and is not itself a traffic lane.
        if not mesh.get('source_lane_id') or not mesh['faces']:continue
        tri=np.asarray(mesh['vertices'])[mesh['faces']]
        pairs=tree.query(shapely.polygons(tri[:,:,:2]),predicate='intersects')
        if not pairs.shape[1]:continue
        z=plane_z(tri[pairs[0]],points[pairs[1],:2])
        inside=(z+clearance_m>=points[pairs[1],2]) & (z<=points[pairs[1],2]+heights[pairs[1]])
        for index,height in zip(pairs[1,inside],z[inside]):
            conflicts.setdefault(int(index),[]).append(dict(source_lane_id=mesh['source_lane_id'],
                mesh_id=mesh['id'],surface_z_m=float(height)))
    rejected=[]
    for index,contacts in conflicts.items():
        row=rows[index]
        rejected.append(dict(candidate=row['candidate'],actor_label='MAP_InferredCanopy_'+str(row['candidate']),
            position_enu_m=row['position_enu_m'],target_height_m=row['target_height_m'],
            context_kind='inferred_outer',contacts=contacts,
            reason='Inferred trunk intersects a final lane and its vehicle clearance. Retain source observations and road geometry; review the native instance height before removal.'))
    return rejected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','source-states','canopy','landcover','selection','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    read=lambda path:json.loads(path.read_text())
    spec=read(a.selection);scene=read(a.scene)
    positions=[];bodies=[];observations=[]
    for path in sorted(a.source_states.glob('*/initial_states.json')):
        data=read(path)
        for state in data['samples']:
            for role in ('actors','parked_context'):
                for actor in state[role]:
                    pos=np.asarray(actor['position_local_xyz_m']);rotation=np.asarray(actor['city_rotation']);dim=np.asarray(actor['dimensions_lwh_m'])
                    corners=np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]])*dim
                    bodies.append(shapely.Polygon((corners@rotation.T+pos)[:,:2]));positions.append(pos)
                    observations.append(dict(log_id=data['log_id'],track_uuid=actor['track_uuid'],timestamp_ns=state['timestamp_ns'],role=role))
    tree=shapely.STRtree(bodies);rejected=[]
    for index,item in enumerate(read(a.canopy)['canopy_candidates']):
        xy=item['position_enu_m'];ids=tree.query(shapely.Point(xy[:2]),predicate='dwithin',distance=spec['constructed_trunk_radius_m'])
        ids=[int(k) for k in ids if abs(positions[int(k)][2]-xy[2])<spec['same_ground_layer_height_m']]
        if ids:
            rejected.append(dict(candidate=index,actor_label=('MAP_InferredCanopy_' if item.get('context_kind')=='inferred_outer' else 'MAP_ObservedCanopy_')+str(index),
                source_position_enu_m=xy,context_kind=item.get('context_kind','lidar_supported'),
                conflicting_source_observations=[observations[k] for k in ids],
                reason='Constructed trunk intersects a measured source vehicle body; a canopy cluster does not determine a surveyed trunk position. Raw canopy evidence retained.'))
    root=a.output/'landcover';root.mkdir(exist_ok=True)
    for name in ('ground_mask.png','ground_mask.json','parking_mask.png','parking.json'):shutil.copy2(a.landcover/name,root/name)
    patches=[]
    for patch in spec['reviewed_landcover_patches']:
        source=Path(patch['ground_root']);grid=read(source/'ground_mask.json')
        # Patches use the same reviewed local frame as the supplied city scene.
        if scene['coordinates']['origin_city_xyz']!=patch['origin_city_xyz']:
            raise ValueError('Reviewed land-cover patch needs source-frame registration before reuse')
        lower,upper=map(np.asarray,(grid['lower'],grid['upper']));size=grid['size']
        lo,hi=np.asarray(patch['coverage_lower_xy_m']),np.asarray(patch['coverage_upper_xy_m'])
        xs=np.linspace(lower[0],upper[0],size);ys=np.linspace(upper[1],lower[1],size)
        dx=np.minimum(xs-lo[0],hi[0]-xs);dy=np.minimum(ys-lo[1],hi[1]-ys)
        alpha=np.rint(np.clip(np.minimum(dx[None,:],dy[:,None])/patch['boundary_blend_m'],0,1)*255).astype(np.uint8)
        target=root/('reviewed_'+str(len(patches)));target.mkdir(exist_ok=True)
        cv2.imwrite(str(target/'coverage_mask.png'),alpha)
        patches.append(dict(patch,lower=grid['lower'],upper=grid['upper'],
            ground_mask=str((source/'ground_mask.png').resolve()),
            parking_mask=str(Path(patch['parking_mask']).resolve()),coverage_mask=str((target/'coverage_mask.png').resolve())))
    (root/'reviewed_patches.json').write_text(json.dumps(dict(patches=patches),indent=2)+'\n')
    grid=read(root/'ground_mask.json');lo=np.asarray(grid['lower']);hi=np.asarray(grid['upper']);size=grid['size']
    mask=np.full((size,size),255,np.uint8);regions=[];service=[]
    for lane in scene['fusion_map']['lanes']:
        if lane['source']=='osm_sumo' and not lane['internal'] and lane.get('source_tags',{}).get('highway')=='service':
            line=shapely.LineString(np.asarray(lane['centerline'])[:,:2]);region=line.buffer(lane['width_m']/2+.25,cap_style='flat',join_style='mitre');regions.append(region)
            service.append(dict(lane_id=lane['id'],edge=lane['edge_id'],tags=lane['source_tags']))
    region=shapely.union_all(regions)
    def pixels(ring):
        uv=(shapely.get_coordinates(ring)-lo)/(hi-lo);uv[:,1]=1-uv[:,1];return np.rint(uv*(size-1)).astype(np.int32)
    for poly in shapely.get_parts(region):
        if poly.geom_type!='Polygon':continue
        cv2.fillPoly(mask,[pixels(poly.exterior)],0)
        for ring in poly.interiors:cv2.fillPoly(mask,[pixels(ring)],255)
    cv2.imwrite(str(a.output/'inferred_paint_keep.png'),mask)
    result=dict(selection=str(a.selection),source_scene=str(a.scene),source_canopy=str(a.canopy),rejected_constructed_trunks=rejected,
        service_lanes_without_paint_evidence=service,inferred_paint_keep=dict(texture=str((a.output/'inferred_paint_keep.png').resolve()),lower=lo.tolist(),upper=hi.tolist()),
        ground_root=str(root.resolve()),pavement=spec['pavement'],reviewed_landcover_patches=patches,
        scope=__doc__,geometry_and_navigation_changed=False)
    (a.output/'feedback.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dict(rejected_constructed_trunks=len(rejected),source_actor_observations=len(observations),service_lanes=len(service),reviewed_patches=len(patches)),flush=True)


if __name__=='__main__':main()
