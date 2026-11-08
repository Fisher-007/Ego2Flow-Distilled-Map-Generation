"""Clone a native map and replace selected surface and context assets.

Optional inputs update OpenDRIVE and remove inferred vegetation identified
by the lane-clearance report.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def configure_in_editor(source, target, assets_file, output, resume_imported=False, context_clearance_file=None):
    import unreal
    library,level=unreal.EditorAssetLibrary,unreal.EditorLevelLibrary
    if library.does_asset_exist(target) and not resume_imported:
        raise ValueError('Use a new map name for this surface replacement')
    assets=json.loads(Path(assets_file).read_text())
    if not assets.get('surfaces_only'):
        raise ValueError('Supply a surface-only FBX export')
    if resume_imported:
        if not library.does_asset_exist(target) or not level.load_level(target):
            raise RuntimeError('Resume requires the existing incomplete surface map')
    elif not level.new_level_from_template(target,source):
        raise RuntimeError('Could not clone source map')
    name=target.rsplit('/',1)[-1];package='/Game/MAP'+name
    def transform_values(component):
        value = component.get_world_transform()
        return {name: [getattr(getattr(value, name), axis) for axis in axes]
                for name, axes in (('translation', 'xyz'), ('rotation', 'xyzw'), ('scale3d', 'xyz'))}

    def state(component):
        return {'mesh':component.get_editor_property('static_mesh').get_path_name(),
                'transform':transform_values(component),
                'materials':[component.get_material(i).get_path_name() for i in range(component.get_num_materials())]}

    context_candidates={}
    context_clearance_m=0.
    if context_clearance_file:
        context_feedback=json.loads(Path(context_clearance_file).read_text())
        context_candidates={r['actor_label']:r for r in context_feedback['rejected_constructed_trunks']}
        context_clearance_m=context_feedback['clearance_m']
    retired_context=[]
    targets={}
    components=[]
    for actor in level.get_all_level_actors():
        label=actor.get_actor_label()
        if label in context_candidates:
            condition=context_candidates[label]
            origin,extent=actor.get_actor_bounds(False)
            bottom,top=(origin.z-extent.z)/100.,(origin.z+extent.z)/100.
            if any(r['surface_z_m']<=top and r['surface_z_m']+context_clearance_m>=bottom for r in condition['contacts']):
                retired_context.append(dict(condition,native_vertical_bounds_m=[bottom,top]))
                level.destroy_actor(actor)
                continue
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=component.get_editor_property('static_mesh')
            if mesh:
                components.append(component)
                if component.is_visible() and re.fullmatch(r'map_assets_(?:road|terrain|road_edge)(?:_\d+)?',mesh.get_name()):
                    targets[mesh.get_name()]=component
    kind_of=lambda name:re.fullmatch(r'map_assets_(road|terrain|road_edge)(?:_\d+)?',name).group(1)
    if {kind_of(name) for name in targets}!={'road','terrain','road_edge'}:
        raise ValueError('Expected existing road, terrain and edge surfaces')
    selected_kinds=set(assets.get('surface_kinds',['road','terrain','road_edge']))
    # Additive support patches have no retired tile. Their material still
    # comes from the existing visible surface family in the source world.
    reference_materials={}
    for mesh_name,component in targets.items():
        reference_materials.setdefault(kind_of(mesh_name),[component.get_material(i) for i in range(component.get_num_materials())])
    targets={name:component for name,component in targets.items() if kind_of(name) in selected_kinds}
    if 'replace_surface_names'in assets:
        selected_names=set(assets['replace_surface_names'])
        if selected_names-targets.keys():
            missing=sorted(selected_names-targets.keys())
            Path(output).with_name('surface_selection_failure.json').write_text(json.dumps(dict(source_map=source,missing=missing,visible_surface_names=sorted(targets)),indent=2)+'\n')
            raise ValueError('Requested road patch lacks installed source tiles: '+', '.join(missing))
        targets={name:component for name,component in targets.items()if name in selected_names}
    material_by_kind={}
    for mesh_name,component in targets.items():
        material_by_kind.setdefault(kind_of(mesh_name),[component.get_material(i) for i in range(component.get_num_materials())])
    selected_component_paths={c.get_path_name() for c in targets.values()}
    untouched=[c for c in components if c.get_path_name() not in selected_component_paths]
    before={c.get_path_name():state(c) for c in untouched}
    task=unreal.AssetImportTask()
    task.filename=assets['fbx'];task.destination_path=package+'/SurfaceImport'
    task.automated=True;task.replace_existing=False;task.save=True
    options=unreal.FbxImportUI()
    options.import_mesh=True;options.import_materials=False;options.import_textures=False
    options.import_as_skeletal=False;options.mesh_type_to_import=unreal.FBXImportType.FBXIT_STATIC_MESH
    data=options.static_mesh_import_data
    data.combine_meshes=False;data.convert_scene=True;data.convert_scene_unit=True
    data.transform_vertex_to_absolute=True;data.generate_lightmap_u_vs=False
    data.auto_generate_collision=False
    task.options=options
    if not resume_imported:
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        imported={library.load_asset(path).get_name():path for path in task.imported_object_paths
                  if isinstance(library.load_asset(path),unreal.StaticMesh)}
    records=[]
    omitted=[]
    updated=set()
    for entry in assets['objects']:
        kind=entry['kind'];asset_name='map_assets_'+entry['name']
        component=targets.get(asset_name)
        old=component.get_editor_property('static_mesh') if component else None
        transform=transform_values(component) if component else None
        materials=material_by_kind.get(kind,reference_materials[kind])
        category='Road' if kind=='road' else 'Terrain'
        path=package+'/Static/'+category+'/'+name+'/'+asset_name
        precision=entry.get('native_precision')
        if resume_imported:
            original=None if library.does_asset_exist(path) else package+'/SurfaceImport/'+asset_name
            missing=original is not None and not library.does_asset_exist(original)
        else:
            original=imported.get(asset_name)
            missing=original is None
        if missing:
            # Skip source surfaces that collapse at native coordinate precision.
            from map_generation.export_geometry import missing_surface_omission
            reason=missing_surface_omission(entry)
            if reason is not None:
                omitted.append(dict(asset=asset_name,kind=kind,faces=entry['faces'],
                    native_precision=precision,source_bounds_m=entry['bounds_m'],reason=reason))
                continue
            raise RuntimeError('Missing imported nonzero surface: '+asset_name)
        if original is not None and not library.rename_asset(original,path):
            raise RuntimeError('Could not move imported surface to semantic folder')
        mesh=library.load_asset(path)
        if entry.get('material_kinds'):
            # FBX may reorder sections by first use. Bind by the imported
            # material name, using this source world's existing appearance.
            materials=[]
            for slot in mesh.get_editor_property('static_materials'):
                slot_kind=str(slot.get_editor_property('imported_material_slot_name'))
                if slot_kind not in entry['material_kinds']:
                    raise ValueError('Unexpected imported surface material slot: '+slot_kind)
                materials.append(reference_materials[slot_kind][0])
        settings=unreal.EditorStaticMeshLibrary.get_lod_build_settings(mesh,0)
        if settings.distance_field_resolution_scale!=0.:
            settings.distance_field_resolution_scale=0.
            unreal.EditorStaticMeshLibrary.set_lod_build_settings(mesh,0,settings)
        low,high=entry['bounds_m']['min'],entry['bounds_m']['max']
        bounds=mesh.get_bounds()
        expected=[(low[i]+high[i])*50*(1 if i!=1 else -1) for i in range(3)]+[(high[i]-low[i])*50 for i in range(3)]
        actual=[bounds.origin.x,bounds.origin.y,bounds.origin.z,bounds.box_extent.x,bounds.box_extent.y,bounds.box_extent.z]
        source_bounds_error=max(abs(a-b) for a,b in zip(actual,expected))
        comparison='source_bounds'
        if precision and precision['bounds_cm'] is not None:
            # Check imported bounds against the source envelope and nondegenerate extent.
            inner=precision['bounds_cm']
            outer_low=[100*low[0],-100*high[1],100*low[2]]
            outer_high=[100*high[0],-100*low[1],100*high[2]]
            actual_low=[actual[i]-actual[i+3] for i in range(3)]
            actual_high=[actual[i]+actual[i+3] for i in range(3)]
            error=max([0.]+[v for i in range(3) for v in (
                outer_low[i]-actual_low[i],actual_high[i]-outer_high[i],
                actual_low[i]-inner['min'][i],inner['max'][i]-actual_high[i])])
            comparison='raw_source_enclosure_and_nondegenerate_source_coverage'
        else:
            error=max(abs(a-b) for a,b in zip(actual,expected))
        if error>.1:
            Path(output).with_name('surface_failure.json').write_text(json.dumps(dict(
                asset=path,expected_bounds_cm=expected,actual_bounds_cm=actual,
                maximum_bounds_difference_cm=error),indent=2)+'\n')
            raise ValueError('Imported surface bounds differ by more than 1 mm: '+asset_name)
        body=mesh.get_editor_property('body_setup')
        if body.get_editor_property('collision_trace_flag')!=unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE:
            body.set_editor_property('collision_trace_flag',unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
        for i,material in enumerate(materials):
            if mesh.get_material(i)!=material:mesh.set_material(i,material)
        if component:
            component.set_static_mesh(mesh)
        else:
            actor=level.spawn_actor_from_object(mesh,unreal.Vector(0,0,0))
            actor.set_actor_label('MAP_Surface_'+entry['name'])
            component=actor.get_component_by_class(unreal.StaticMeshComponent)
        origin=entry.get('asset_origin_enu_m')
        if origin is not None:
            component.set_world_location(unreal.Vector(origin[0]*100,-origin[1]*100,origin[2]*100),False,False)
            placed=transform_values(component)
            expected_position=[origin[0]*100,-origin[1]*100,origin[2]*100]
            if max(abs(x-y)for x,y in zip(placed['translation'],expected_position))>.1:
                raise RuntimeError('Surface world-origin placement differs by more than 1 mm')
            if placed['rotation']!=[0.,0.,0.,1.] or placed['scale3d']!=[1.,1.,1.]:
                raise RuntimeError('Centered surface requires the exported unit scale and axis frame')
        updated.add(asset_name)
        for i,material in enumerate(materials):component.set_material(i,material)
        component.set_editor_property('bAffectDistanceFieldLighting',False)
        if origin is None and transform is not None and transform_values(component)!=transform:
            raise RuntimeError('Surface transform changed')
        library.save_loaded_asset(mesh,only_if_is_dirty=True)
        records.append({'kind':kind,'component':component.get_path_name(),'source_mesh':old.get_path_name() if old else None,
                        'new_mesh':mesh.get_path_name(),'bounds_error_cm':error,
                        'raw_source_bounds_error_cm':source_bounds_error,'native_precision':precision,
                        'bounds_comparison':comparison,
                        'materials':[m.get_path_name() for m in materials],
                        'collision':str(unreal.EditorStaticMeshLibrary.get_collision_complexity(mesh)),
                        'asset_origin_enu_m':origin,
                        'transform_unchanged':transform is not None and transform_values(component)==transform,
                        'new_component_at_export_origin':transform is None})
        if len(records)%128==0:
            Path(output).with_name('surface_progress.json').write_text(json.dumps(dict(
                configured=len(records),expected=len(assets['objects']),last_asset=asset_name,
                omitted_degenerate=len(omitted),complete=False),indent=2)+'\n')
    retired=[]
    for asset_name,component in targets.items():
        if asset_name not in updated:
            component.set_visibility(False);component.set_hidden_in_game(True)
            component.set_collision_profile_name('NoCollision');retired.append(asset_name)
    if any(state(c)!=before[c.get_path_name()] for c in untouched):
        raise RuntimeError('An unselected static component changed during surface replacement')
    if not level.save_current_level():raise RuntimeError('Could not save surface variant')
    Path(output).with_name('surface_progress.json').write_text(json.dumps(dict(
        configured=len(records),expected=len(assets['objects']),omitted_degenerate=len(omitted),complete=True),indent=2)+'\n')
    Path(output).write_text(json.dumps({'source_map':source,'map':target,'assets':assets_file,
        'surfaces':records,'selected_surface_kinds':sorted(selected_kinds),
        'omitted_degenerate_surfaces':omitted,
        'retired_surface_assets':retired,'unchanged_static_components':len(untouched),
        'context_clearance_file':context_clearance_file,'retired_inferred_context':retired_context,
        'resumed_imported_assets':resume_imported,
        'passed':True,'scope':'Selected surface kinds replaced, including spatial partitions; unselected surfaces, buildings, paint and signals retained. Native collision/render validation pending.'},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--source-map',required=True)
    p.add_argument('--name',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--xodr',type=Path,help='Reviewed OpenDRIVE matching the supplied surfaces; default retains source XODR')
    p.add_argument('--resume-imported',action='store_true',help='Finish an incomplete clone using its already imported meshes')
    p.add_argument('--context-clearance',type=Path,help='Final-lane inferred vegetation review; confirm actual instance heights during installation')
    a=p.parse_args();output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    project=Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4'
    target=f'/Game/MAP{a.name}/Maps/{a.name}/{a.name}'
    src=project/'Content'/a.source_map.removeprefix('/Game/').rsplit('/',1)[0]/'OpenDrive'/(a.source_map.rsplit('/',1)[-1]+'.xodr')
    dst=project/'Content'/target.removeprefix('/Game/').rsplit('/',1)[0]/'OpenDrive'/(a.name+'.xodr')
    if a.resume_imported:
        if not dst.exists() or dst.read_bytes()!=(a.xodr if a.xodr else src).read_bytes():
            raise ValueError('Resume requires the same previously installed navigation')
    else:
        if dst.exists():raise ValueError('Use a new output map')
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.xodr if a.xodr else src,dst)
    result=output/'surface_import.json';script=output/'replace_surfaces_ue.py'
    script.write_text("import importlib.util,sys\nsys.path.insert(0,%r)\ns=importlib.util.spec_from_file_location('surfaces',%r)\nm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nm.configure_in_editor(%r,%r,%r,%r,%r,context_clearance_file=%r)\n"%(str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve()),a.source_map,target,str(a.assets.resolve()),str(result),a.resume_imported,str(a.context_clearance.resolve()) if a.context_clearance else None))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for key in ('CONFIG','CACHE','DATA'):
        path=output/'runtime'/key.lower();path.mkdir(parents=True,exist_ok=True);env['XDG_'+key+'_HOME']=str(path)
    cmd=['taskset','-c',os.environ['MAP_CPUSET'],str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),str(project/'CarlaUE4.uproject'),'-run=pythonscript','-script='+str(script),'-nullrhi','-unattended','-nosourcecontrol','-nosound','-nocrashreports','-abslog='+str(output/'editor.log'),
        '-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False']
    start=time.monotonic()
    with (output/'editor.stdout.log').open('w') as log:
        proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not result.exists():raise RuntimeError('Surface import failed; inspect editor log')
    report=json.loads(result.read_text());report.update(editor_returncode=proc.returncode,editor_wall_s=time.monotonic()-start,
        source_xodr=str(src),supplied_xodr=str(a.xodr.resolve()) if a.xodr else None,
        target_xodr=str(dst),opendrive_byte_identical=src.read_bytes()==dst.read_bytes())
    report['navigation_matches_supplied'] = dst.read_bytes() == (a.xodr if a.xodr else src).read_bytes()
    result.write_text(json.dumps(report,indent=2)+'\n')
    print({k:report[k] for k in ('map','passed','editor_returncode','navigation_matches_supplied')},flush=True)
    if proc.returncode:
        raise RuntimeError('Surface editor exited nonzero; preserve its save report and independently verify persisted state')


if __name__=='__main__':main()
