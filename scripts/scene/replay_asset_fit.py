"""Select native vehicle appearance and fit replay assets to support surfaces."""
import json,math
from pathlib import Path
import carla
import numpy as np
import shapely
from scripts.scene.render_dynamic_av2_view import actor_pose
from scripts.scene.build_assets import plane_heights


class Surface:
    def __init__(self,root):
        root=Path(root);roads=json.loads((root/'road_meshes.json').read_text());terrain=json.loads((root/'terrain_mesh.json').read_text())
        self.triangles=np.concatenate([np.asarray(m['vertices'])[m['faces']] for m in roads['meshes'] if m['faces']]+
            [np.asarray(terrain['vertices'])[terrain['faces']]])
        self.tree=shapely.STRtree(shapely.polygons(self.triangles[:,:,:2]))

    def heights(self,xy,reference_z_m=None):
        pairs=self.tree.query(shapely.points(xy),predicate='dwithin',distance=1e-7)
        heights=np.full(len(xy),-np.inf)
        values=plane_heights(self.triangles[pairs[1]],xy[pairs[0]])
        if reference_z_m is None:
            np.maximum.at(heights,pairs[0],values)
        elif len(values):
            reference=np.broadcast_to(reference_z_m,(len(xy),))
            order=np.lexsort((abs(values-reference[pairs[0]]),pairs[0]))
            _,first=np.unique(pairs[0][order],return_index=True)
            selected=order[first]
            heights[pairs[0][selected]]=values[selected]
        return heights

    def fit(self,row,actor):
        box=row['carla_box_pose'];position=np.array(box['center_xyz_m']);R=np.array(box['rotation_matrix'])
        yaw=math.atan2(R[1,0],R[0,0]);forward=np.array([math.cos(yaw),math.sin(yaw)]);side=np.array([-forward[1],forward[0]])
        ext=np.array([actor.bounding_box.extent.x,actor.bounding_box.extent.y,actor.bounding_box.extent.z])
        # Support locations approximate wheel footprints; exact source tire geometry is unavailable.
        offsets=np.array([sx*ext[0]*.65*forward+sy*ext[1]*.8*side for sx,sy in [(-1,-1),(-1,1),(1,-1),(1,1)]])
        source_bottom=position[2]-abs(R[2])@np.asarray(row['dimensions_lwh_m'])/2
        xy=position[:2]+offsets;z=self.heights(xy*np.array([1.,-1.]),source_bottom)
        if not np.isfinite(z).all():return actor_pose(box,actor.bounding_box),box,{'applied':False,'reason':'Surface coverage missing'}
        coef=np.linalg.lstsq(np.c_[offsets,np.ones(4)],z,rcond=None)[0]
        normal=np.array([-coef[0],-coef[1],1.]);normal/=np.linalg.norm(normal)
        f=np.r_[forward,coef[:2]@forward];f/=np.linalg.norm(f);right=np.cross(normal,f)
        rotation=np.column_stack([f,right,normal]);target=position.copy();target[2]=coef[2]+ext[2]/normal[2]+.005
        target_box={'center_xyz_m':target.tolist(),'rotation_matrix':rotation.tolist()}
        support=target+np.c_[offsets@np.c_[forward,side],np.full(4,-ext[2])]@rotation.T
        actual_ground=self.heights(support[:,:2]*[1.,-1.],support[:,2])
        if not np.isfinite(actual_ground).all():return actor_pose(box,actor.bounding_box),box,{'applied':False,'reason':'Support extends outside measured native surface'}
        gaps=support[:,2]-actual_ground
        return actor_pose(target_box,actor.bounding_box),target_box,{'applied':True,'source_center_delta_z_m':float(target[2]-position[2]),
            'source_bottom_minus_ground_m':float(source_bottom-coef[2]),
            'source_rotation_change_max':float(abs(rotation-R).max()),'support_gaps_m':gaps.tolist(),
            'surface_layer_selection':'Nearest exported surface to the original source cuboid bottom; final contact query uses fitted support height',
            'scope':'Native asset support fit; source XY/yaw preserved, source z/pitch/roll retained separately. Bounding support is not exact tire-contact measurement.'}


DIMENSION_MATCHED_CATEGORIES = ('REGULAR_VEHICLE', 'BOX_TRUCK', 'TRUCK', 'LARGE_VEHICLE')


def catalog(world):
    """Measure installed road-vehicle assets, including multi-axle trucks."""
    result=[]
    for bp in world.get_blueprint_library().filter('vehicle.*'):
        if not bp.has_attribute('number_of_wheels') or bp.get_attribute('number_of_wheels').as_int()<4:continue
        if bp.has_attribute('base_type') and bp.get_attribute('base_type').as_str().lower() not in ('car','van','truck','suv'):continue
        actor=world.try_spawn_actor(bp,carla.Transform(carla.Location(z=1500)))
        if actor is None:continue
        try:
            actor.set_simulate_physics(False);box=actor.bounding_box
            result.append({'blueprint':bp.id,'dimensions_lwh_m':[2*box.extent.x,2*box.extent.y,2*box.extent.z],
                           'base_type':bp.get_attribute('base_type').as_str() if bp.has_attribute('base_type') else None})
        finally:actor.destroy()
    return result


def choose(dimensions,assets,appearance=None,category='REGULAR_VEHICLE'):
    # Map installed CARLA models to their asset metadata.
    profiles=json.loads((Path(__file__).resolve().parents[2]/'configs/scene/native_vehicle_profiles.json').read_text())
    style=(appearance or {}).get('asset_family') or (appearance or {}).get('body_style')
    required=set((appearance or {}).get('required_body_features',[]))
    if category=='BOX_TRUCK':required.add('enclosed_cargo')
    if category in ('BOX_TRUCK', 'TRUCK', 'LARGE_VEHICLE'):
        style='box_truck' if category=='BOX_TRUCK' else 'large_vehicle'
    candidates=[a for a in assets if a['blueprint'] not in profiles['special_livery']
                and required.issubset(profiles.get('body_features',{}).get(a['blueprint'],[]))
                and (a['base_type']=='car' or a['blueprint'] in profiles.get(style or '',[]))]
    if style in profiles:
        styled=[a for a in candidates if a['blueprint'] in profiles[style]]
        if styled:candidates=styled
    target=np.array(dimensions)
    # Width/length are more reliable than height for partially observed boxes.
    return min(candidates,key=lambda a:float(np.sum(np.log(np.array(a['dimensions_lwh_m'])/target)**2*[1.,1.,.35])))['blueprint']
