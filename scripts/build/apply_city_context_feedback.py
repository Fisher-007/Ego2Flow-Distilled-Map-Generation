"""Apply reviewed ground, constructed-trunk and inferred-paint feedback to a copy."""
import argparse
import filecmp
import json
import os
from pathlib import Path
import shutil
import subprocess


def configure(source,target,feedback_file,report_file):
    import unreal
    from scripts.assets.surface_materials import configure_ground
    lib,level,edit=unreal.EditorAssetLibrary,unreal.EditorLevelLibrary,unreal.MaterialEditingLibrary
    if lib.does_asset_exist(target):raise ValueError('Use a new map for context feedback')
    if not level.new_level_from_template(target,source):raise RuntimeError('Map copy failed')
    feedback=json.loads(Path(feedback_file).read_text());package=target.rsplit('/Maps/',1)[0]
    rejected={row['actor_label']:row for row in feedback['rejected_constructed_trunks']}
    terrain=[];paint=[];removed=[]
    for actor in list(level.get_all_level_actors()):
        label=actor.get_actor_label()
        if label in rejected:
            removed.append(rejected[label]);level.destroy_actor(actor);continue
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=component.get_editor_property('static_mesh')
            if not mesh or not component.is_visible():continue
            if mesh.get_name().startswith('map_assets_terrain'):terrain.append(component)
            if '/Static/RoadLine/' in mesh.get_path_name() and 'outer_markings' in mesh.get_name():paint.append(component)
    pavement=feedback['pavement']
    ground=configure_ground(unreal,terrain[0],package,Path(feedback['ground_root']),grass_texture='T_Grass_v01_d',grass_gain=.65,
        asphalt_texture=pavement['texture'],asphalt_period_m=pavement['period_m'],asphalt_gain=1.)
    for component in terrain[1:]:component.set_material(0,terrain[0].get_material(0))
    spec=feedback['inferred_paint_keep'];task=unreal.AssetImportTask();task.filename=spec['texture']
    task.destination_path=package+'/Appearance/Context';task.destination_name='inferred_paint_keep';task.automated=True;task.save=True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task]);texture=lib.load_asset(task.imported_object_paths[0]);texture.set_editor_property('srgb',False)
    texture.set_editor_property('virtual_texture_streaming',False)
    for axis in ('address_x','address_y'):texture.set_editor_property(axis,unreal.TextureAddress.TA_CLAMP)
    lib.save_loaded_asset(texture,only_if_is_dirty=False)
    clones={}
    for component in paint:
        old=component.get_material(0);key=old.get_path_name()
        if key not in clones:
            material=lib.duplicate_asset(key,package+'/Appearance/Context/inferred_paint_'+str(len(clones)))
            material.set_editor_property('blend_mode',unreal.BlendMode.BLEND_MASKED)
            def node(cls):return edit.create_material_expression(material,cls,0,0)
            def link(a,output,b,input):edit.connect_material_expressions(a,output,b,input)
            world=node(unreal.MaterialExpressionWorldPosition);xy=node(unreal.MaterialExpressionComponentMask)
            for flag,val in [('r',True),('g',True),('b',False),('a',False)]:xy.set_editor_property(flag,val)
            link(world,'',xy,'')
            lo,hi=spec['lower'],spec['upper'];width,height=hi[0]-lo[0],hi[1]-lo[1]
            factor=node(unreal.MaterialExpressionConstant2Vector);factor.set_editor_property('r',1/(100*width));factor.set_editor_property('g',1/(100*height))
            multiply=node(unreal.MaterialExpressionMultiply);link(xy,'',multiply,'A');link(factor,'',multiply,'B')
            offset=node(unreal.MaterialExpressionConstant2Vector);offset.set_editor_property('r',-lo[0]/width);offset.set_editor_property('g',hi[1]/height)
            uv=node(unreal.MaterialExpressionAdd);link(multiply,'',uv,'A');link(offset,'',uv,'B')
            mask=node(unreal.MaterialExpressionTextureSample);mask.texture=texture;mask.set_editor_property('sampler_type',unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR);link(uv,'',mask,'')
            edit.connect_material_property(mask,'R',unreal.MaterialProperty.MP_OPACITY_MASK)
            edit.recompile_material(material);lib.save_loaded_asset(material,only_if_is_dirty=False);clones[key]=material
        component.set_material(0,clones[key])
    if not level.save_current_level():raise RuntimeError('Could not save context revision')
    Path(report_file).write_text(json.dumps(dict(source_map=source,map=target,feedback=feedback_file,
        terrain_components=len(terrain),ground=ground,removed_constructed_trunks=removed,
        inferred_paint_components=len(paint),paint_materials={k:v.get_path_name() for k,v in clones.items()},
        scope='Source-priority ground appearance, source-body clearance for constructed trunks and opacity of unobserved service-road paint. Physical road/terrain meshes, source paint, buildings and navigation retained. Native views and affected driving still require review.'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-map',required=True);p.add_argument('--name',required=True)
    p.add_argument('--conditions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    conditions=json.loads(a.conditions.read_text())
    selected=next(row['native_construction'] for row in conditions['native_objects'] if row.get('native_construction',{}).get('kind')=='city_context_feedback')
    a.feedback=Path(selected['feedback_file'])
    a.output.mkdir(parents=True,exist_ok=True);target=f'/Game/MAP{a.name}/Maps/{a.name}/{a.name}'
    project=Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4'
    def xodr(path):return project/'Content'/path.removeprefix('/Game/').rsplit('/',1)[0]/'OpenDrive'/(path.rsplit('/',1)[-1]+'.xodr')
    source=xodr(a.source_map);destination=xodr(target);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    script=a.output/'import_ue.py';report=a.output/'native_context.json'
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.apply_city_context_feedback import configure\nconfigure(%r,%r,%r,%r)\n'%(str(Path.cwd()),a.source_map,target,str(a.feedback.resolve()),str(report.resolve())))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for key in ('CONFIG','CACHE','DATA'):
        d=a.output/'runtime'/key.lower();d.mkdir(parents=True,exist_ok=True);env['XDG_'+key+'_HOME']=str(d.resolve())
    command=['taskset','-c',os.environ['MAP_CPUSET'],str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),str(project/'CarlaUE4.uproject'),'-run=pythonscript','-script='+str(script.resolve()),'-nullrhi','-unattended','-nosourcecontrol','-nosound','-nocrashreports']
    with (a.output/'editor.log').open('w') as log:run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not report.exists():raise RuntimeError('Context assembly did not produce its save report')
    data=json.loads(report.read_text());data.update(graph_conditions=str(a.conditions),editor_returncode=run.returncode,opendrive_byte_identical=filecmp.cmp(source,destination,shallow=False));report.write_text(json.dumps(data,indent=2)+'\n')
    print(dict(map=target,editor_returncode=run.returncode,removed_trunks=len(data['removed_constructed_trunks']),terrain_components=data['terrain_components'],inferred_paint_components=data['inferred_paint_components']),flush=True)


if __name__=='__main__':main()
