"""Place native canopy assets at source-supported locations.

replacement_label_prefix selects an MAP_ObservedCanopy_ or MAP_InferredCanopy_
subset; omitting it replaces the whole canopy set.
"""
import argparse,json,os,shutil,subprocess,math
from pathlib import Path


def configure(source,target,evidence,output,assets,radius_mode,in_place=False):
    import unreal
    lib,level=unreal.EditorAssetLibrary,unreal.EditorLevelLibrary
    if in_place:
        if target!=source:raise ValueError('In-place context must target the supplied map')
        level.load_level(source)
    else:
        if lib.does_asset_exist(target):raise ValueError('Use a new map name')
        if not level.new_level_from_template(target,source):raise RuntimeError('Could not copy candidate')
    evidence=json.loads(Path(evidence).read_text());trees={a.get_actor_label():a for a in level.get_all_level_actors() if a.get_actor_label().startswith(('MAP_ObservedCanopy_','MAP_InferredCanopy_'))};rows=[]
    previous=[{'label':a.get_actor_label(),'location_cm':str(a.get_actor_location()),'asset':a.get_component_by_class(unreal.StaticMeshComponent).get_editor_property('static_mesh').get_path_name()} for a in trees.values()]
    prefix=evidence.get('replacement_label_prefix')
    selected={label:actor for label,actor in trees.items() if prefix is None or label.startswith(prefix)}
    previous=[row for row in previous if row['label'] in selected]
    for actor in selected.values():level.destroy_actor(actor)
    hidden_foliage=None
    def leafless_material():
        nonlocal hidden_foliage
        if hidden_foliage is not None:return hidden_foliage
        folder=target.rsplit('/Maps/',1)[0]+'/Appearance/Canopy';path=folder+'/source_bare_foliage'
        tools=unreal.AssetToolsHelpers.get_asset_tools();edit=unreal.MaterialEditingLibrary
        m=lib.load_asset(path) if lib.does_asset_exist(path)else tools.create_asset('source_bare_foliage',folder,unreal.Material,unreal.MaterialFactoryNew())
        edit.delete_all_material_expressions(m);m.set_editor_property('blend_mode',unreal.BlendMode.BLEND_MASKED)
        zero=edit.create_material_expression(m,unreal.MaterialExpressionConstant,0,0);zero.r=0.
        edit.connect_material_property(zero,'',unreal.MaterialProperty.MP_OPACITY_MASK)
        edit.recompile_material(m);lib.save_loaded_asset(m,only_if_is_dirty=False);hidden_foliage=m;return m
    for i,item in enumerate(evidence['canopy_candidates']):
        asset=item.get('asset',assets[i%len(assets)])
        if not asset.startswith('/Game/'):asset='/Game/Carla/Static/Vegetation/Trees/'+asset
        mesh=lib.load_asset(asset);bounds=mesh.get_bounds();o,e=bounds.origin,bounds.box_extent
        zscale=item['target_height_m']*100/(2*e.z)
        # Use the configured crown-width convention.
        extent=math.hypot(e.x,e.y) if radius_mode=='diagonal' else max(e.x,e.y)
        xy_scale=item['crown_radius_m']*100/extent
        yaw=item.get('yaw_deg',i*47%360);angle=math.radians(yaw);x,y,z=item['position_enu_m']
        ox=o.x*math.cos(angle)-o.y*math.sin(angle);oy=o.x*math.sin(angle)+o.y*math.cos(angle)
        position=unreal.Vector(100*x-ox*xy_scale,-100*y-oy*xy_scale,100*z-(o.z-e.z)*zscale)
        actor=level.spawn_actor_from_object(mesh,position,unreal.Rotator(yaw=yaw));actor.set_actor_label((prefix or ('MAP_InferredCanopy_' if item.get('context_kind')=='inferred_outer' else 'MAP_ObservedCanopy_'))+str(i));actor.set_actor_scale3d(unreal.Vector(xy_scale,xy_scale,zscale))
        component=actor.get_component_by_class(unreal.StaticMeshComponent);hidden_slots=[]
        if item.get('native_foliage_mode')=='hide_foliage_slots':
            for slot_index,slot in enumerate(mesh.get_editor_property('static_materials')):
                name=str(slot.get_editor_property('material_slot_name'))
                original=component.get_material(slot_index)
                label=(name+' '+(original.get_name() if original else '')).lower()
                if any(key in label for key in ('leaf','leaves','impostor')):
                    component.set_material(slot_index,leafless_material());hidden_slots.append(name)
            if not hidden_slots:raise ValueError('Selected bare-tree proxy has no identified foliage slots: '+asset)
            component.set_forced_lod_model(1)
        item_native=dict(source_id=item.get('id'),appearance_condition=item.get('appearance_condition'),
            native_foliage_mode=item.get('native_foliage_mode','retained_native_foliage'),
            hidden_foliage_slots=hidden_slots,source_support=item.get('native_support_method'),
            native_approximation=item.get('native_approximation'))
        rows.append({**item_native,'candidate':i,'label':actor.get_actor_label(),'context_kind':item.get('context_kind','lidar_supported'),'asset':asset,'height_m':2*e.z*zscale/100,'horizontal_half_width_m':max(e.x,e.y)*xy_scale/100,'source_position_enu_m':item['position_enu_m']})
    if not level.save_current_level():raise RuntimeError('Could not save canopy candidate')
    Path(output).write_text(json.dumps({'source_map':source,'map':target,'crown_radius_mode':radius_mode,'trees':rows,'previous_trees':previous,'replacement_label_prefix':prefix,'unselected_existing_trees_retained':len(trees)-len(selected),'source_evidence':str(evidence.get('log','')),'outer_completion':evidence.get('outer_completion'),'scope':'Native crown shapes at supplied candidate positions and heights. LiDAR-supported and inferred outer positions are labeled separately; no measured individual species or exact crown boundary claim.'},indent=2)+'\n')


def configure_leaf_layers(map_path, evidence_file, output):
    """Fit noncolliding leaf sections to selected ENU envelopes.

    Per-actor overrides apply to named leaf components and their materials.
    """
    import unreal
    lib,level=unreal.EditorAssetLibrary,unreal.EditorLevelLibrary
    if level.get_editor_world().get_path_name().split('.')[0]!=map_path:
        level.load_level(map_path)
    spec=json.loads(Path(evidence_file).read_text());prefix=spec['replacement_label_prefix']
    if not prefix.startswith(('MAP_ObservedCanopy_','MAP_InferredCanopy_')):
        raise ValueError('Leaf layers require a canopy label prefix')
    actors={a.get_actor_label():a for a in level.get_all_level_actors()}
    for label,a in list(actors.items()):
        if label.startswith(prefix):level.destroy_actor(a)
    tools=unreal.AssetToolsHelpers.get_asset_tools();edit=unreal.MaterialEditingLibrary
    folder=map_path.rsplit('/Maps/',1)[0]+'/Appearance/Canopy';hidden_path=folder+'/hidden_leaf_layer_sections'
    hidden=lib.load_asset(hidden_path)if lib.does_asset_exist(hidden_path)else tools.create_asset('hidden_leaf_layer_sections',folder,unreal.Material,unreal.MaterialFactoryNew())
    edit.delete_all_material_expressions(hidden);hidden.set_editor_property('blend_mode',unreal.BlendMode.BLEND_MASKED)
    zero=edit.create_material_expression(hidden,unreal.MaterialExpressionConstant,0,0);zero.r=0.
    edit.connect_material_property(zero,'',unreal.MaterialProperty.MP_OPACITY_MASK);edit.recompile_material(hidden);lib.save_loaded_asset(hidden,only_if_is_dirty=False)
    cache={}
    def leaf_material(original,parameters):
        key=(original.get_path_name(),json.dumps(parameters,sort_keys=True))
        if key in cache:return cache[key]
        available={str(n)for n in edit.get_vector_parameter_names(original)}
        missing=set(parameters)-available
        if missing:raise ValueError('Requested source leaf parameters unavailable: '+str(sorted(missing)))
        path,_=tools.create_unique_asset_name(folder+'/'+original.get_name()+'_source_leaf','')
        material=lib.duplicate_asset(original.get_path_name(),path)
        for parameter,value in parameters.items():
            edit.set_material_instance_vector_parameter_value(material,parameter,unreal.LinearColor(*value))
            actual=edit.get_material_instance_vector_parameter_value(material,parameter)
            if max(abs(a-b)for a,b in zip([actual.r,actual.g,actual.b,actual.a],value))>1e-6:
                raise RuntimeError('Leaf parameter did not retain its supplied value: '+parameter)
        lib.save_loaded_asset(material,only_if_is_dirty=False);cache[key]=material;return material
    rows=[];changed=[]
    for item in spec.get('retint_existing',[]):
        a=actors[item['label']];c=a.get_component_by_class(unreal.StaticMeshComponent);slot=item['leaf_slot'];original=c.get_material(slot)
        material=leaf_material(original,item.get('vector_parameters',{}));before_lod=c.get_editor_property('forced_lod_model');c.set_material(slot,material);c.set_forced_lod_model(1)
        changed.append(dict(label=item['label'],component=a.get_name()+'::'+c.get_name(),slot=slot,previous_material=original.get_path_name(),material=material.get_path_name(),vector_parameters=item.get('vector_parameters',{}),previous_forced_lod=before_lod,forced_lod=1))
    for i,item in enumerate(spec['layers']):
        mesh=lib.load_asset(item['asset']);slot=item['leaf_slot'];materials=mesh.get_editor_property('static_materials')
        original=materials[slot].get_editor_property('material_interface');material=leaf_material(original,item.get('vector_parameters',{}))
        native_lo,native_hi=item['leaf_bounds_native_cm'];enu_lo,enu_hi=item['bounds_enu_m']
        desired_lo=[100*enu_lo[0],-100*enu_hi[1],100*enu_lo[2]];desired_hi=[100*enu_hi[0],-100*enu_lo[1],100*enu_hi[2]]
        scale=[(hi-lo)/(v-u)for lo,hi,u,v in zip(desired_lo,desired_hi,native_lo,native_hi)]
        if not all(x>0 for x in scale):raise ValueError('Leaf bounds require positive extents')
        location=[lo-u*s for lo,u,s in zip(desired_lo,native_lo,scale)]
        a=level.spawn_actor_from_object(mesh,unreal.Vector(*location));a.set_actor_label(prefix+str(i));a.set_actor_scale3d(unreal.Vector(*scale));c=a.get_component_by_class(unreal.StaticMeshComponent)
        for j in range(len(materials)):c.set_material(j,material if j==slot else hidden)
        c.set_forced_lod_model(1);c.set_collision_profile_name('NoCollision');c.set_editor_property('bAffectDistanceFieldLighting',False)
        rows.append(dict(source_id=item['id'],label=a.get_actor_label(),component=a.get_name()+'::'+c.get_name(),mesh=mesh.get_path_name(),leaf_slot=slot,leaf_slot_name=str(materials[slot].get_editor_property('material_slot_name')),leaf_material=material.get_path_name(),hidden_material=hidden.get_path_name(),hidden_slots=[j for j in range(len(materials))if j!=slot],leaf_bounds_native_cm=item['leaf_bounds_native_cm'],expected_bounds_enu_m=item['bounds_enu_m'],location_cm=location,scale=scale,forced_lod=1,collision='NoCollision',vector_parameters=item.get('vector_parameters',{}),source_group=item.get('source_group'),point_count=item.get('point_count')))
    if not level.save_current_level():raise RuntimeError('Could not save leaf-layer candidate')
    result=dict(map=map_path,layers=rows,retinted_existing=changed,evidence=str(evidence_file),scope='Explicit native leaf material-section envelopes only; hidden stem/impostor sections, no collision, forced highest detail. Source shape and layer decomposition approximate; material parameters require actual asset support.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-map',required=True);p.add_argument('--in-place',action='store_true');p.add_argument('--name',required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--assets',nargs='+',default=['SM_Oak_M_v7','SM_Maple_M_v3','SM_Maple_M_v3']);p.add_argument('--radius-mode',choices=['diagonal','axis'],default='diagonal');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    target=a.source_map if a.in_place else f'/Game/MAP{a.name}/Maps/{a.name}/{a.name}';project=Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4'
    def xodr(path):return project/'Content'/path.removeprefix('/Game/').rsplit('/',1)[0]/'OpenDrive'/(path.rsplit('/',1)[-1]+'.xodr')
    dest=xodr(target);dest.parent.mkdir(parents=True,exist_ok=True)
    if not a.in_place:shutil.copy2(xodr(a.source_map),dest)
    script=out/'import_ue.py';report=out/'import.json';script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.assets.revise_canopy_assets import configure\nconfigure(%r,%r,%r,%r,%r,%r,%r)\n'%(str(Path.cwd()),a.source_map,target,str(a.evidence.resolve()),str(report),a.assets,a.radius_mode,a.in_place))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for key in ['CONFIG','CACHE','DATA']:
        d=out/'runtime'/key.lower();d.mkdir(parents=True,exist_ok=True);env['XDG_'+key+'_HOME']=str(d)
    cmd=['taskset','-c',os.environ['MAP_CPUSET'],str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),str(project/'CarlaUE4.uproject'),'-run=pythonscript','-script='+str(script),'-nullrhi','-unattended','-nosourcecontrol','-nosound','-nocrashreports']
    with (out/'editor.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    if report.exists():
        result=json.loads(report.read_text());result['editor_returncode']=r.returncode
        report.write_text(json.dumps(result,indent=2)+'\n')
    if r.returncode or not report.exists():raise RuntimeError('Canopy candidate build failed')
    print('Saved',target,'editor exit',r.returncode)


if __name__=='__main__':main()
