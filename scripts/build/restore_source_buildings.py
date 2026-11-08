"""Restore explicitly selected source building meshes omitted by native FBX import."""
import argparse,json,os,re,subprocess
from pathlib import Path

def configure(map_path,subset_file,output):
    import unreal
    lib,level=unreal.EditorAssetLibrary,unreal.EditorLevelLibrary
    level.load_level(map_path)
    subset=json.loads(Path(subset_file).read_text())
    source=json.loads(Path(subset['source_assets']).read_text())
    intended={r['source_id']:r for r in source['objects'] if r['kind']=='building'}
    names=lambda r:[re.sub(r'[^A-Za-z0-9_]','_',r['name']),'map_assets_'+re.sub(r'[^A-Za-z0-9_]','_',r['name'])]
    def components():
        result={}
        for actor in level.get_all_level_actors():
            for c in actor.get_components_by_class(unreal.StaticMeshComponent):
                mesh=c.get_editor_property('static_mesh')
                if mesh and '/Static/Building/' in mesh.get_path_name() and c.is_visible():
                    result[mesh.get_name()]=c
        return result
    before=components()
    selected={r['source_id'] for r in subset['objects']}
    absent={key for key,r in intended.items() if not any(n in before for n in names(r))}
    if absent!=selected:raise ValueError('Selected omissions differ from actual native source masses: '+str(sorted(absent^selected)))
    examples=[before[n] for r in intended.values() for n in names(r) if n in before]
    materials={c.get_material(0).get_path_name() for c in examples}
    if len(materials)!=1:raise ValueError('Original source masses have differing base materials; select them explicitly')
    material=lib.load_asset(next(iter(materials)))
    package=map_path.rsplit('/Maps/',1)[0];short=map_path.rsplit('/',1)[-1]
    task=unreal.AssetImportTask();task.filename=str(Path(subset['fbx']).resolve())
    task.destination_path=package+'/Static/Building/'+short
    task.automated=True;task.save=True;task.replace_existing=True
    options=unreal.FbxImportUI();options.import_mesh=True;options.import_materials=False;options.import_textures=False
    options.mesh_type_to_import=unreal.FBXImportType.FBXIT_STATIC_MESH
    data=options.static_mesh_import_data;data.combine_meshes=False;data.convert_scene=True;data.convert_scene_unit=True
    data.transform_vertex_to_absolute=True;data.generate_lightmap_u_vs=False;data.auto_generate_collision=False
    task.options=options;unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    imported={}
    for path in task.imported_object_paths:
        obj=lib.load_asset(path)
        if isinstance(obj,unreal.StaticMesh):imported[obj.get_name()]=obj
    records=[]
    for row in subset['objects']:
        candidates=[imported[n] for n in names(row) if n in imported]
        if len(candidates)!=1:raise ValueError('Restored mesh unavailable: '+row['source_id'])
        mesh=candidates[0]
        for i in range(len(mesh.get_editor_property('static_materials'))):mesh.set_material(i,material)
        mesh.get_editor_property('body_setup').set_editor_property('collision_trace_flag',unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
        lib.save_loaded_asset(mesh,only_if_is_dirty=False)
        actor=level.spawn_actor_from_object(mesh,unreal.Vector(0,0,0));actor.set_actor_label('MAP_Source_'+row['name'])
        c=actor.get_component_by_class(unreal.StaticMeshComponent)
        c.set_mobility(unreal.ComponentMobility.STATIC);c.set_collision_profile_name('BlockAll')
        bounds=mesh.get_bounds();origin=bounds.origin;extent=bounds.box_extent
        actual_min=[origin.x-extent.x,origin.y-extent.y,origin.z-extent.z]
        actual_max=[origin.x+extent.x,origin.y+extent.y,origin.z+extent.z]
        lo,hi=row['bounds_m']['min'],row['bounds_m']['max']
        expected_min=[lo[0]*100,-hi[1]*100,lo[2]*100]
        expected_max=[hi[0]*100,-lo[1]*100,hi[2]*100]
        error=max(abs(x-y)for x,y in zip(actual_min+actual_max,expected_min+expected_max))
        records.append(dict(source_id=row['source_id'],mesh=mesh.get_path_name(),bounds_error_cm=error,
            source_vertices_unchanged=True,transform='Identity at ENU-to-native source mesh coordinates'))
    after=components();missing=[key for key,row in intended.items() if not any(n in after for n in names(row))]
    report=dict(map=map_path,subset=subset_file,expected_source_buildings=len(intended),
        before_source_buildings=len(intended)-len(absent),restored=records,missing_after=missing,
        maximum_bounds_error_cm=max(r['bounds_error_cm']for r in records),material=material.get_path_name(),
        passed=not missing and all(r['bounds_error_cm']<=.2 for r in records),saved=False,
        scope='Restore actual absent source geometry; native runtime collision/render acceptance remains separate.')
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise ValueError('Source building restoration geometry failed')
    report['saved']=bool(level.save_current_level())
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['saved']:raise RuntimeError('Restored source map save failed')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',required=True)
    for key in ('subset','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    script=a.output/'restore_source_assets_ue.py';report=a.output/'restoration.json'
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.restore_source_buildings import configure\nconfigure(%r,%r,%r)\n'%(str(Path.cwd()),a.map,str(a.subset.resolve()),str(report.resolve())))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for key in ['CONFIG','CACHE','DATA']:
        d=a.output/'runtime'/key.lower();d.mkdir(parents=True,exist_ok=True);env['XDG_'+key+'_HOME']=str(d.resolve())
    command=['taskset','-c',os.environ['MAP_CPUSET'],str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),
        str(Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4/CarlaUE4.uproject'),'-run=pythonscript',
        '-script='+str(script.resolve()),'-nullrhi','-unattended','-nosourcecontrol','-nosound','-nocrashreports',
        '-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False']
    (a.output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (a.output/'editor.log').open('w')as f:result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode or not report.exists():raise RuntimeError('Source restoration failed; inspect editor.log')
    print(report.read_text(),flush=True)

if __name__=='__main__':main()
