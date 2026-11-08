"""Replace only selected outer building masses by source-conditioned assets."""
import argparse,json,os,re,subprocess
from pathlib import Path


def configure(map_path,root,materials_file,output,textures_only=False,reuse_materials=False,source_assets=None,reuse_meshes=False):
    import unreal,math
    lib,level=unreal.EditorAssetLibrary,unreal.EditorLevelLibrary
    if level.get_editor_world().get_path_name().split('.')[0] != map_path:
        level.load_level(map_path)
    root=Path(root);assets=json.loads((root/'architecture.json').read_text(encoding='utf-8'));specs=json.loads(Path(materials_file).read_text(encoding='utf-8'))['materials']
    plan=json.loads(Path(assets['plan']).read_text(encoding='utf-8'))
    source_objects={r['source_id']:r for r in json.loads(Path(source_assets).read_text())['objects'] if r['kind']=='building'} if source_assets else {}
    replace_complete=bool(plan.get('replace_complete_building_set'))
    reuse_paths=plan.get('material_reuse_paths',{})
    preserve_materials=bool(plan.get('preserve_native_materials',plan.get('previous_native_import')))
    previous={}
    if plan.get('previous_native_import'):
        previous_path=Path(plan['previous_native_import'])
        if not previous_path.is_absolute():
            previous_path=Path(__file__).resolve().parents[2]/previous_path
        previous={r['source_id']:r['new_mesh'] for r in json.loads(previous_path.read_text(encoding='utf-8'))['buildings']}
    package=map_path.rsplit('/Maps/',1)[0];tools=unreal.AssetToolsHelpers.get_asset_tools();edit=unreal.MaterialEditingLibrary
    def import_asset(filename,folder,name=None,mesh=False):
        task=unreal.AssetImportTask();task.filename=str(filename);task.destination_path=folder;task.automated=True;task.save=True;task.replace_existing=True
        if name:task.destination_name=name
        if mesh:
            options=unreal.FbxImportUI();options.import_mesh=True;options.import_materials=False;options.import_textures=False
            options.mesh_type_to_import=unreal.FBXImportType.FBXIT_STATIC_MESH;data=options.static_mesh_import_data
            data.combine_meshes=False;data.convert_scene=True;data.convert_scene_unit=True;data.transform_vertex_to_absolute=True;data.generate_lightmap_u_vs=False;data.auto_generate_collision=False;task.options=options
        tools.import_asset_tasks([task]);return [lib.load_asset(p) for p in task.imported_object_paths]
    if textures_only:
        updated=[]
        for name,spec in specs.items():
            texture=import_asset(spec['texture'],package+'/Appearance/Frontage',name+'_basecolor')[0]
            texture.set_editor_property('srgb',True);texture.set_editor_property('address_x',unreal.TextureAddress.TA_WRAP);texture.set_editor_property('address_y',unreal.TextureAddress.TA_WRAP)
            lib.save_loaded_asset(texture,only_if_is_dirty=False);updated.append(texture.get_path_name())
        Path(output).write_text(json.dumps({'map':map_path,'textures_only':True,'updated':updated,'materials':specs},indent=2)+'\n')
        return
    def material(name,color=None,texture=None,roughness=.9,period=2.):
        path=package+'/Appearance/Frontage/'+name
        if name in reuse_paths:
            retained=lib.load_asset(reuse_paths[name])
            if not isinstance(retained,unreal.Material):raise ValueError('Recorded native appearance material unavailable: '+reuse_paths[name])
            return retained
        if reuse_materials:
            retained=lib.load_asset(path)
            if not isinstance(retained,unreal.Material):raise ValueError('Missing reusable native material: '+path)
            return retained
        m=lib.load_asset(path) if lib.does_asset_exist(path) else tools.create_asset(name,package+'/Appearance/Frontage',unreal.Material,unreal.MaterialFactoryNew())
        edit.delete_all_material_expressions(m)
        if texture:
            t=import_asset(texture,package+'/Appearance/Frontage',name+'_basecolor')[0]
            t.set_editor_property('srgb',True);t.set_editor_property('address_x',unreal.TextureAddress.TA_WRAP);t.set_editor_property('address_y',unreal.TextureAddress.TA_WRAP)
            lib.save_loaded_asset(t,only_if_is_dirty=False)
            node=edit.create_material_expression(m,unreal.MaterialExpressionTextureSample,0,0);node.set_editor_property('texture',t)
            uv=edit.create_material_expression(m,unreal.MaterialExpressionTextureCoordinate,0,0)
            uv.set_editor_property('u_tiling',2./period);uv.set_editor_property('v_tiling',2./period)
            edit.connect_material_expressions(uv,'',node,'Coordinates')
            edit.connect_material_property(node,'RGB',unreal.MaterialProperty.MP_BASE_COLOR)
        else:
            node=edit.create_material_expression(m,unreal.MaterialExpressionConstant3Vector,0,0);node.set_editor_property('constant',unreal.LinearColor(*color,1.))
            edit.connect_material_property(node,'',unreal.MaterialProperty.MP_BASE_COLOR)
        r=edit.create_material_expression(m,unreal.MaterialExpressionConstant,0,0);r.r=roughness;edit.connect_material_property(r,'',unreal.MaterialProperty.MP_ROUGHNESS)
        m.set_editor_property('two_sided',True);edit.recompile_material(m);lib.save_loaded_asset(m,only_if_is_dirty=False);return m
    used_families={row['style']['family'] for row in assets['buildings']}
    walls={name:material(name,texture=spec['texture'],period=spec['period_m']) for name,spec in specs.items() if name in used_families} if not preserve_materials else {}
    common={k:material(k,c,roughness=.3 if k=='glass' else .9) for k,c in {
        'roof':(.12,.10,.075),'trim':(.68,.69,.63),'glass':(.08,.12,.14),'door':(.22,.17,.12),'foundation':(.36,.35,.31)}.items()} if not preserve_materials else {}
    existing={};by_mesh_path={};by_mesh_name={};complete_targets=[]
    retained_slot_materials={};retained_primitive_materials={}
    for actor in level.get_all_level_actors():
        for c in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=c.get_editor_property('static_mesh')
            if mesh and '/Static/Building/' in mesh.get_path_name() and c.is_visible():
                complete_targets.append(c)
                # Reimport can replace this same mesh asset and reset its slots.
                # Capture actual component materials before importing any FBX.
                retained_slot_materials[c.get_path_name()]={
                    str(slot.get_editor_property('material_slot_name')):c.get_material(i)
                    for i,slot in enumerate(mesh.get_editor_property('static_materials'))}
                retained_primitive_materials[c.get_path_name()]=c.get_material(0)
                by_mesh_path[mesh.get_path_name()]=c
                by_mesh_name[mesh.get_name()]=c
                p=mesh.get_bounds().origin
                key=(int(math.floor(p.x/20.)),int(math.floor(p.y/20.)))
                existing.setdefault(key,[]).append(c)
    existing_buckets={c.get_path_name():bucket for bucket in existing.values()for c in bucket}
    source_coverage=None
    if source_objects:
        source_rows=[];missing_sources=[]
        for source_id,row in source_objects.items():
            name=re.sub(r'[^A-Za-z0-9_]','_',row['name'])
            candidates=[by_mesh_name[n] for n in (name,'map_assets_'+name) if n in by_mesh_name]
            if len(candidates)!=1:
                missing_sources.append(source_id);continue
            bounds=candidates[0].get_editor_property('static_mesh').get_bounds()
            origin,extent=bounds.origin,bounds.box_extent
            actual=[origin.x-extent.x,origin.y-extent.y,origin.z-extent.z,
                origin.x+extent.x,origin.y+extent.y,origin.z+extent.z]
            lo,hi=row['bounds_m']['min'],row['bounds_m']['max']
            expected=[lo[0]*100,-hi[1]*100,lo[2]*100,hi[0]*100,-lo[1]*100,hi[2]*100]
            source_rows.append(dict(source_id=source_id,bounds_error_cm=max(abs(x-y)for x,y in zip(actual,expected))))
        source_coverage=dict(source_assets=source_assets,expected=len(source_objects),present=len(source_rows),
            missing=missing_sources,maximum_bounds_error_cm=max(r['bounds_error_cm'] for r in source_rows),
            records=source_rows,reopened_saved_map=True)
        source_coverage['passed']=not missing_sources and source_coverage['maximum_bounds_error_cm']<=.2
        Path(str(output)+'.source_masses.json').write_text(json.dumps(source_coverage,indent=2)+'\n')
        if not source_coverage['passed']:raise ValueError('Original native source masses incomplete; inspect source_masses report')
    if reuse_meshes:
        imported=[lib.load_asset(p) for p in lib.list_assets(package+'/Static/Building/Frontage',recursive=False,include_folder=False)]
        imported=[m for m in imported if isinstance(m,unreal.StaticMesh)]
    else:
        imported=[m for m in import_asset(root/'frontage.fbx',package+'/Static/Building/Frontage',mesh=True) if isinstance(m,unreal.StaticMesh)]
    imported_by_name={m.get_name():m for m in imported}
    records=[]
    for index,row in enumerate(assets['buildings']):
        meshes=[imported_by_name[k] for k in (row['mesh'],'frontage_'+row['mesh']) if k in imported_by_name]
        if len(meshes)!=1:raise ValueError('Cannot match imported building: '+row['mesh'])
        mesh=meshes[0];x,y=row['old_bounds_center_xy_m'];matches=[]
        # These detailed facade components already exclude distance-field
        # lighting below. Avoid generating that unused volume during import.
        settings=unreal.EditorStaticMeshLibrary.get_lod_build_settings(mesh,0)
        settings.distance_field_resolution_scale=0.
        unreal.EditorStaticMeshLibrary.set_lod_build_settings(mesh,0,settings)
        bx,by=int(math.floor(100*x/20.)),int(math.floor(-100*y/20.))
        nearby=[c for dx in (-1,0,1) for dy in (-1,0,1) for c in existing.get((bx+dx,by+dy),[])]
        for c in nearby:
            bounds=c.get_editor_property('static_mesh').get_bounds();p=bounds.origin
            matches.append((math.hypot(p.x-100*x,p.y+100*y),c))
        if replace_complete:
            replaced=[];old=None;error=None
        elif previous:
            replaced=([by_mesh_name[name]for name in row['previous_mesh_names']]if row.get('previous_mesh_names')else
                [by_mesh_path[previous[key]]for key in row.get('previous_source_ids',[row['source_id']])])
            old=replaced[0]
            center=old.get_editor_property('static_mesh').get_bounds().origin
            error=math.hypot(center.x-100*x,center.y+100*y)
        elif source_objects:
            source=source_objects[row['source_id']]
            name=re.sub(r'[^A-Za-z0-9_]','_',source['name'])
            candidates=[by_mesh_name[n] for n in (name,'map_assets_'+name) if n in by_mesh_name]
            if len(candidates)!=1:raise ValueError('Source mass identity unavailable: '+row['source_id'])
            old=candidates[0];replaced=[old]
            center=old.get_editor_property('static_mesh').get_bounds().origin
            error=math.hypot(center.x-100*x,center.y+100*y)
        else:
            if not matches:raise ValueError('Source footprint match unavailable: '+row['source_id'])
            error,old=min(matches,key=lambda x:x[0])
            if error>.2:raise ValueError('Selected footprint does not match retained mass: '+row['source_id'])
            replaced=[old]
        slots=[];retained_materials={}
        if preserve_materials and row.get('retain_primitive_material'):
            retained_materials={key:retained_primitive_materials[old.get_path_name()]for key in row['material_slots']}
        elif preserve_materials:
            for name,retained in retained_slot_materials[old.get_path_name()].items():
                key=next(k for k in row['material_slots'] if name==k or name.endswith('_'+k))
                retained_materials[key]=retained
        overrides={key:material('source_'+re.sub(r'[^A-Za-z0-9_]','_',row['source_id'])+'_'+key,**spec)
                   for key,spec in row.get('material_overrides',{}).items()}
        for i,slot in enumerate(mesh.get_editor_property('static_materials')):
            name=str(slot.get_editor_property('material_slot_name'));key=next((k for k in row['material_slots'] if name==k or name.endswith('_'+k)),None)
            if key is None:raise ValueError('Unknown architecture material slot: '+name)
            assigned=retained_materials[key] if preserve_materials else (walls[row['style']['family']] if key=='wall' else common[key])
            assigned=overrides.get(key,assigned)
            mesh.set_material(i,assigned);slots.append({'slot':name,'material':assigned.get_path_name()})
        mesh.get_editor_property('body_setup').set_editor_property('collision_trace_flag',unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
        lib.save_loaded_asset(mesh,only_if_is_dirty=False)
        actor=level.spawn_actor_from_object(mesh,unreal.Vector(0,0,0));actor.set_actor_label('MAP_Frontage_'+row['source_id'].replace(':','_'))
        component=actor.get_component_by_class(unreal.StaticMeshComponent);component.set_editor_property('bAffectDistanceFieldLighting',False)
        component.set_collision_profile_name('BlockAll')
        for part in replaced:
            part.set_visibility(False);part.set_hidden_in_game(True);part.set_collision_profile_name('NoCollision')
            center=part.get_editor_property('static_mesh').get_bounds().origin
            existing_buckets[part.get_path_name()].remove(part)
        records.append({'source_id':row['source_id'],'old_mesh':old.get_editor_property('static_mesh').get_path_name() if old else None,
                        'new_mesh':mesh.get_path_name(),'style':row['style'],'slots':slots,'footprint_center_error_cm':error,
                        'replaced_source_ids':row.get('previous_source_ids',[row['source_id']]),
                        'replaced_meshes':[part.get_editor_property('static_mesh').get_path_name()for part in replaced],
                        'source_match':'complete current source solid identity' if replace_complete else 'previous native import identity' if previous else 'recorded source asset identity' if source_objects else 'source footprint center'})
    retired=[]
    if replace_complete:
        for c in complete_targets:
            retired.append(dict(component=c.get_path_name(),mesh=c.get_editor_property('static_mesh').get_path_name()))
            c.set_visibility(False);c.set_hidden_in_game(True);c.set_collision_profile_name('NoCollision')
        # Retire source masses after importing their replacements.
        if len(records)!=len(assets['buildings']):
            raise RuntimeError('Complete building replacement did not install every exported source solid')
    support=[];actors={a.get_actor_label():a for a in level.get_all_level_actors()}
    for row in plan.get('retained_native_support',[]):
        actor=actors[row['label']];center,extent=actor.get_actor_bounds(False)
        bottom=(center.z-extent.z)/100;delta=row['target_bottom_z_m']-bottom
        position=actor.get_actor_location();position.z+=delta*100;actor.set_actor_location(position,False,False)
        support.append(dict(row,previous_bottom_z_m=bottom,delta_z_m=delta))
    for row in plan.get('excluded_context_trees',[]):
        actor=actors[row['label']]
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            component.set_visibility(False);component.set_hidden_in_game(True);component.set_collision_profile_name('NoCollision')
    if not level.save_current_level():raise RuntimeError('Frontage save failed')
    Path(output).write_text(json.dumps({'map':map_path,'buildings':records,'retained_native_support':support,
        'materials':specs,'native_materials_retained':preserve_materials or bool(reuse_paths),'unchanged_material_assets_reused':reuse_materials or bool(reuse_paths),'unchanged_geometry_assets_reused':reuse_meshes,'source_assets':source_assets,'source_coverage':source_coverage,'excluded_context_trees':plan.get('excluded_context_trees',[]),
        'scope':assets['scope'],'native_validation':'pending',
        'complete_building_replacement':replace_complete,'retired_building_components':retired,
        'material_reuse_paths':reuse_paths,'expected_exported_buildings':len(assets['buildings'])},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map',required=True)
    p.add_argument('--source-assets',type=Path,help='Exact original building source identity manifest')
    p.add_argument('--reuse-meshes',action='store_true',help='Reuse explicitly unchanged already imported detailed meshes')
    p.add_argument('--reuse-materials',action='store_true',help='Reuse explicitly unchanged materials already created in the selected owned package')
    p.add_argument('--textures-only',action='store_true',help='Apply a reviewed material feedback without rebuilding geometry')
    for name in ['root','materials','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);script=out/'architecture_ue.py';report=out/'architecture_import.json'
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.assets.import_frontage_architecture import configure\nconfigure(%r,%r,%r,%r,%r,%r,%r,%r)\n'%(str(Path.cwd()),a.map,str(a.root.resolve()),str(a.materials.resolve()),str(report),a.textures_only,a.reuse_materials,str(a.source_assets.resolve()) if a.source_assets else None,a.reuse_meshes))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for key in ['CONFIG','CACHE','DATA']:
        directory=out/'runtime'/key.lower();directory.mkdir(parents=True,exist_ok=True);env['XDG_'+key+'_HOME']=str(directory)
    command=['taskset','-c',os.environ['MAP_CPUSET'],str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),str(Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4/CarlaUE4.uproject')]
    command+=['-run=pythonscript','-script='+str(script),'-nullrhi']
    command+=['-unattended','-nosourcecontrol','-nosound','-nocrashreports',
        '-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False']
    with (out/'editor.log').open('w') as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if report.exists():
        saved=json.loads(report.read_text());saved['editor_returncode']=r.returncode
        report.write_text(json.dumps(saved,indent=2)+'\n')
    if r.returncode!=0 or not report.exists():raise RuntimeError('Architecture import failed; inspect editor.log')
    print({'map':a.map,'editor_exit':r.returncode},flush=True)


if __name__=='__main__':main()
