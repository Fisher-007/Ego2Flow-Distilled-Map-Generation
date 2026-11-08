"""Install source-selected RoadLine meshes or verify their persisted native bindings."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time



def exported_paint_object(mesh_name, fbx, asset_report):
    """Match the exported name after UE replaces Blender's dotted suffix."""
    rows=json.loads(Path(asset_report).read_text())['objects']
    matches=[row for row in rows if any(mesh_name in (name, name.replace('.', '_'))
             for name in (row['name'], Path(fbx).stem+'_'+row['name']))]
    if len(matches)!=1:
        raise ValueError('Imported paint has no unique export object: '+mesh_name)
    return matches[0]


def exported_origin(mesh_name, fbx, asset_report):
    """Resolve an exported object's local ENU origin, including UE's FBX prefix."""
    if not asset_report:
        return [0., 0., 0.]
    row=exported_paint_object(mesh_name, fbx, asset_report)
    return [float(v) for v in row.get('asset_origin_enu_m',[0.,0.,0.])]


def exported_material_key(row):
    """Keep observed colors per object and use the exported road paint palette."""
    material=row.get('material','')
    if material.startswith('observed_paint_'):
        return row['name']
    if material in ('road_mark_white', 'road_mark_yellow'):
        return material[len('road_mark_'):]
    return 'yellow' if 'yellow' in row['name'] else 'white'


def source_paint_material(unreal, package, color, asset_report):
    """Initialize missing native paint from its explicit exported material inputs."""
    if not asset_report:
        raise ValueError('First paint import requires an export material report')
    rows=json.loads(Path(asset_report).read_text())['objects']
    specs=[row['material_inputs'] for row in rows
           if exported_material_key(row)==color and 'material_inputs' in row]
    if not specs or any(spec!=specs[0] for spec in specs[1:]):
        raise ValueError('Missing or conflicting exported '+color+' paint material inputs')
    spec=specs[0];lib=unreal.EditorAssetLibrary;edit=unreal.MaterialEditingLibrary
    folder=package+'/Appearance/RoadPaint';name='source_paint_'+color
    path=folder+'/'+name
    material=lib.load_asset(path) if lib.does_asset_exist(path) else unreal.AssetToolsHelpers.get_asset_tools().create_asset(name,folder,unreal.Material,unreal.MaterialFactoryNew())
    edit.delete_all_material_expressions(material)
    node=edit.create_material_expression(material,unreal.MaterialExpressionConstant3Vector,-200,0)
    node.set_editor_property('constant',unreal.LinearColor(*spec['Base Color']))
    edit.connect_material_property(node,'',unreal.MaterialProperty.MP_BASE_COLOR)
    for index,(source,property_name) in enumerate([
            ('Roughness','MP_ROUGHNESS'),('Metallic','MP_METALLIC'),
            ('Specular IOR Level','MP_SPECULAR')]):
        node=edit.create_material_expression(material,unreal.MaterialExpressionConstant,-200,100+index*100)
        node.set_editor_property('r',float(spec[source]))
        edit.connect_material_property(node,'',getattr(unreal.MaterialProperty,property_name))
    edit.recompile_material(material);lib.save_loaded_asset(material,only_if_is_dirty=False)
    return material,dict(material=material.get_path_name(),source_report=str(asset_report),
                         exported_inputs=spec,scope='Explicit exported paint palette; native visual reflectance remains to be reviewed.')


def select_paint_material(unreal, package, mesh_name, fbx, asset_report, colors, initialized_materials):
    """Honor explicit source colors while reusing established white/yellow paint."""
    if asset_report:
        row=exported_paint_object(mesh_name, fbx, asset_report)
        if row.get('material','').startswith('observed_paint_'):
            key=row['name']
            if key not in initialized_materials:
                colors[key],initialized_materials[key]=source_paint_material(
                    unreal,package,key,asset_report)
            return colors[key]
        color=exported_material_key(row)
    else:
        color='yellow' if 'yellow' in mesh_name else 'white'
    if not colors.get(color):
        colors[color],initialized_materials[color]=source_paint_material(
            unreal,package,color,asset_report)
    return colors[color]


def native_origin_cm(origin):
    return [100.*origin[0], -100.*origin[1], 100.*origin[2]]


def configure(source,target,fbx,output,role='observed_av2_crosswalks',in_place=False,replace_kind=None,replace_report=None,asset_report=None):
    import unreal
    lib,level=unreal.EditorAssetLibrary,unreal.EditorLevelLibrary
    if in_place:
        if not level.load_level(target):raise RuntimeError('Could not load paint target')
    else:
        if lib.does_asset_exist(target):raise ValueError('Use a new map name')
        if not level.new_level_from_template(target,source):raise RuntimeError('Could not clone map')
    colors={};old=[];initialized_materials={}
    prior_meshes=({row['mesh'] for row in json.loads(Path(replace_report).read_text())['paint']}
                  if replace_report else set())
    found_prior=[]
    for actor in level.get_all_level_actors():
        for c in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=c.get_editor_property('static_mesh')
            if mesh and '/Static/RoadLine/' in mesh.get_path_name():
                # Existing inferred paint is replaceable by explicit identity;
                # source paint still supplies the established material palette.
                if 'outer_markings' not in mesh.get_name():
                    for color in ['white','yellow']:
                        if color in mesh.get_name():colors[color]=c.get_material(0)
                selected=(replace_kind and replace_kind in mesh.get_name()) or mesh.get_path_name() in prior_meshes
                if selected and c.is_visible():
                    old.append(c)
                    if mesh.get_path_name() in prior_meshes:found_prior.append(mesh.get_path_name())
    if prior_meshes and (set(found_prior)!=prior_meshes or len(found_prior)!=len(prior_meshes)):
        raise ValueError('Recorded previous source paint is not present exactly once before replacement')
    folder=target.rsplit('/Maps/',1)[0]+'/Static/RoadLine/'+role
    task=unreal.AssetImportTask();task.filename=fbx;task.destination_path=folder
    task.automated=True;task.replace_existing=False;task.save=True
    options=unreal.FbxImportUI();options.import_mesh=True;options.import_materials=False;options.import_textures=False
    options.import_as_skeletal=False;options.mesh_type_to_import=unreal.FBXImportType.FBXIT_STATIC_MESH
    data=options.static_mesh_import_data;data.combine_meshes=False;data.convert_scene=True;data.convert_scene_unit=True
    data.transform_vertex_to_absolute=True;data.generate_lightmap_u_vs=False;data.auto_generate_collision=False;task.options=options
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task]);records=[]
    for path in task.imported_object_paths:
        mesh=lib.load_asset(path)
        if not isinstance(mesh,unreal.StaticMesh):continue
        material=select_paint_material(unreal,target.rsplit('/Maps/',1)[0],
                                       mesh.get_name(),fbx,asset_report,colors,initialized_materials)
        settings=unreal.EditorStaticMeshLibrary.get_lod_build_settings(mesh,0)
        settings.distance_field_resolution_scale=0.
        unreal.EditorStaticMeshLibrary.set_lod_build_settings(mesh,0,settings)
        mesh.set_material(0,material);lib.save_loaded_asset(mesh,only_if_is_dirty=False)
        origin=exported_origin(mesh.get_name(),fbx,asset_report)
        placement=native_origin_cm(origin)
        actor=level.spawn_actor_from_object(mesh,unreal.Vector(*placement))
        c=actor.get_component_by_class(unreal.StaticMeshComponent)
        c.set_material(0,material);c.set_collision_profile_name('NoCollision')
        c.set_editor_property('cast_shadow',False);c.set_editor_property('bAffectDistanceFieldLighting',False)
        records.append({'mesh':mesh.get_path_name(),'actor':actor.get_path_name(),
                        'asset_origin_enu_m':origin,'expected_translation_cm':placement,
                        'material':material.get_path_name(),'collision':str(c.get_collision_profile_name()),'cast_shadow':False})
    for c in old:c.set_visibility(False);c.set_hidden_in_game(True);c.set_collision_profile_name('NoCollision')
    if not records or not level.save_current_level():raise RuntimeError('Paint import did not complete')
    Path(output).write_text(json.dumps({'map':target,'source_map':source,'paint':records,'passed':True,
        'paint_role':role,'asset_report':asset_report,'initialized_materials':initialized_materials,'replaced_components':len(old),'replace_kind':replace_kind,
        'replace_report':replace_report,'replaced_recorded_meshes':found_prior,
        'scope':'Source-selected non-colliding RoadLine meshes; designated old paint replaced; roads, facades and signal rules retained.'},indent=2)+'\n')


def verify_saved(target, source_report, output):
    """Reopen a saved map and inspect declared paint bindings without editing."""
    import unreal
    prior=json.loads(Path(source_report).read_text())
    if prior['map']!=target:raise ValueError('Paint report belongs to a different map')
    level=unreal.EditorLevelLibrary
    if not level.load_level(target):raise RuntimeError('Could not reopen saved paint map')
    expected={row['mesh']:row for row in prior['paint']}
    found={key:[]for key in expected};errors=[]
    for actor in level.get_all_level_actors():
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=component.get_editor_property('static_mesh')
            if not mesh or mesh.get_path_name()not in expected:continue
            key=mesh.get_path_name();material=component.get_material(0)
            transform=component.get_world_transform();values=[]
            for field,axes in [('translation','xyz'),('rotation','xyzw'),('scale3d','xyz')]:
                value=transform.get_editor_property(field)
                values.extend(float(value.get_editor_property(axis))for axis in axes)
            actual=dict(component=component.get_path_name(),mesh=key,
                material=material.get_path_name()if material else None,
                visible=component.is_visible(),hidden_in_game=component.get_editor_property('hidden_in_game'),
                collision=str(component.get_collision_profile_name()),
                cast_shadow=component.get_editor_property('cast_shadow'),transform=values)
            expected_translation=expected[key].get('expected_translation_cm',native_origin_cm(expected[key].get('asset_origin_enu_m',[0.,0.,0.])))
            actual['expected_translation_cm']=expected_translation
            actual['translation_error_m']=max(abs(values[i]-expected_translation[i]) for i in range(3))/100.
            found[key].append(actual)
            if (not actual['visible'] or actual['hidden_in_game'] or
                actual['material']!=expected[key]['material'] or
                actual['collision']!='NoCollision' or actual['cast_shadow'] or
                actual['translation_error_m']>.001 or values[3:]!=[0.,0.,0.,1.,1.,1.,1.]):
                errors.append(actual)
    wrong_counts={key:len(rows)for key,rows in found.items()if len(rows)!=1}
    report=dict(map=target,source_report=str(source_report),
        expected_meshes=len(expected),persisted_components=sum(map(len,found.values())),
        wrong_counts=wrong_counts,property_errors=errors,components=found,
        reopened_saved_map=True,map_modified=False,passed=not wrong_counts and not errors,
        scope='Exact declared paint asset identities, bindings, export-origin placement within 1 mm, visibility and no-collision state on map reopen. Pixel appearance and driving remain separate.')
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise RuntimeError('Saved paint verification failed; see report')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-map',required=True);p.add_argument('--name',required=True)
    p.add_argument('--fbx',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-report',type=Path,help='build_markings FBX report with per-object ENU origins and first-import paint material inputs; defaults to the adjacent fbx_report.json when present')
    p.add_argument('--verify-report',type=Path,help='Reopen and verify existing paint report without importing or saving')
    p.add_argument('--role',default='observed_av2_crosswalks',help='Record whether paint is observed or inferred')
    p.add_argument('--in-place',action='store_true',help='Install on an existing map')
    p.add_argument('--replace-kind',choices=['road_markings','road_crossings'],help='Retire the corresponding previous source paint')
    p.add_argument('--replace-report',type=Path,help='Retire exact previous source paint identities from its saved native import report')
    a=p.parse_args()
    if not a.verify_report and not a.fbx:p.error('--fbx is required for import')
    asset_report=a.asset_report
    if asset_report is None and a.fbx:
        sidecar=a.fbx.with_name('fbx_report.json')
        if sidecar.exists():asset_report=sidecar
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    project=Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4';target=f'/Game/MAP{a.name}/Maps/{a.name}/{a.name}'
    src=project/'Content'/a.source_map.removeprefix('/Game/').rsplit('/',1)[0]/'OpenDrive'/(a.source_map.rsplit('/',1)[-1]+'.xodr')
    dst=project/'Content'/target.removeprefix('/Game/').rsplit('/',1)[0]/'OpenDrive'/(a.name+'.xodr')
    if not a.in_place and not a.verify_report:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    report=out/('crosswalk_verification.json'if a.verify_report else 'crosswalk_import.json');script=out/'import_ue.py'
    prefix="import importlib.util\ns=importlib.util.spec_from_file_location('crosswalks',%r)\nm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"%str(Path(__file__).resolve())
    if a.verify_report:
        script.write_text(prefix+"m.verify_saved(%r,%r,%r)\n"%(target,str(a.verify_report.resolve()),str(report)))
    else:
        script.write_text(prefix+"m.configure(%r,%r,%r,%r,%r,%r,%r,%r,%r)\n"%(a.source_map,target,str(a.fbx.resolve()),str(report),a.role,a.in_place,a.replace_kind,str(a.replace_report.resolve())if a.replace_report else None,str(asset_report.resolve())if asset_report else None))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for key in ['CONFIG','CACHE','DATA']:
        d=out/'runtime'/key.lower();d.mkdir(parents=True,exist_ok=True);env['XDG_'+key+'_HOME']=str(d)
    command=['taskset','-c',os.environ['MAP_CPUSET'],str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),str(project/'CarlaUE4.uproject'),'-run=pythonscript','-script='+str(script),'-nullrhi','-unattended','-nosourcecontrol','-nosound','-nocrashreports','-abslog='+str(out/'editor.log')]
    start=time.monotonic()
    with (out/'editor.stdout.log').open('w') as log:proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not report.exists():raise RuntimeError('Crosswalk import failed; see editor log')
    r=json.loads(report.read_text());r.update(target_xodr=str(dst),opendrive_byte_identical=src.read_bytes()==dst.read_bytes(),editor_returncode=proc.returncode,editor_wall_s=time.monotonic()-start)
    report.write_text(json.dumps(r,indent=2)+'\n')
    if proc.returncode:raise RuntimeError('Crosswalk editor exited nonzero; inspect saved report and editor log')
    print(report)


if __name__=='__main__':main()
