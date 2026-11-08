"""Bind reviewed generated materials to every exported native surface tile."""
import argparse,json,os,subprocess
from pathlib import Path

def configure(map_path,root,materials_file,assets_file,output):
    import unreal
    from scripts.assets.surface_materials import configure_ground,link
    lib,level=unreal.EditorAssetLibrary,unreal.EditorLevelLibrary
    level.load_level(map_path)
    specs=json.loads(Path(materials_file).read_text())['materials']
    manifest=json.loads(Path(assets_file).read_text())
    selected={'map_assets_'+r['name']:r['kind'] for r in manifest['objects'] if r['kind']in ('road','terrain','road_edge')}
    components={k:[]for k in ('road','terrain','road_edge')};found=set();before={}
    def geometry_state(component):
        transform=component.get_world_transform()
        values=[]
        for key,axes in [('translation','xyz'),('rotation','xyzw'),('scale3d','xyz')]:
            value=transform.get_editor_property(key)
            values.extend(float(value.get_editor_property(axis))for axis in axes)
        return (component.get_editor_property('static_mesh').get_path_name(),values)

    for actor in level.get_all_level_actors():
        for c in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=c.get_editor_property('static_mesh')
            if mesh and mesh.get_name()in selected:
                components[selected[mesh.get_name()]].append(c);found.add(mesh.get_name())
                before[c.get_path_name()]=geometry_state(c)
    if found!=set(selected):raise ValueError('Missing exported native surface tiles: '+str(sorted(set(selected)-found)))
    package=map_path.rsplit('/Maps/',1)[0];folder=package+'/Appearance/GeneratedGround'
    edit,tools=unreal.MaterialEditingLibrary,unreal.AssetToolsHelpers.get_asset_tools()
    textures={}
    for key in ('asphalt','dormant_grass','sidewalk_concrete'):
        if specs[key].get('unreal_texture'):
            tex=lib.load_asset(specs[key]['unreal_texture'])
            if not isinstance(tex,unreal.Texture):
                raise ValueError('Missing reviewed native surface texture: '+specs[key]['unreal_texture'])
            textures[key]=tex
            continue
        task=unreal.AssetImportTask();task.filename=specs[key]['texture'];task.destination_path=folder
        task.destination_name=key+'_basecolor';task.automated=True;task.replace_existing=True;task.save=True
        tools.import_asset_tasks([task]);tex=lib.load_asset(task.imported_object_paths[0])
        tex.set_editor_property('srgb',True);tex.set_editor_property('virtual_texture_streaming',False)
        tex.set_editor_property('address_x',unreal.TextureAddress.TA_WRAP);tex.set_editor_property('address_y',unreal.TextureAddress.TA_WRAP)
        lib.save_loaded_asset(tex,only_if_is_dirty=False);textures[key]=tex
    def material(key):
        name=key+'_world';path=folder+'/'+name
        m=lib.load_asset(path) if lib.does_asset_exist(path)else tools.create_asset(name,folder,unreal.Material,unreal.MaterialFactoryNew())
        edit.delete_all_material_expressions(m)
        node=lambda cls:edit.create_material_expression(m,cls,0,0)
        world=node(unreal.MaterialExpressionWorldPosition);mask=node(unreal.MaterialExpressionComponentMask)
        for k,v in [('r',True),('g',True),('b',False),('a',False)]:mask.set_editor_property(k,v)
        link(edit,world,'',mask,'');scale=node(unreal.MaterialExpressionMultiply)
        scale.set_editor_property('const_b',1./(100*specs[key]['period_m']));link(edit,mask,'',scale,'A')
        sample=node(unreal.MaterialExpressionTextureSample);sample.set_editor_property('texture',textures[key])
        link(edit,scale,'',sample,'')
        # Generated texture color is retained; native reflectance calibration
        # is an explicit city condition, separate from model output lineage.
        tone=node(unreal.MaterialExpressionMultiply)
        tone.set_editor_property('const_b',float(specs['asphalt'].get('native_basecolor_gain',1.)) if key=='asphalt' else 1.)
        link(edit,sample,'RGB',tone,'A');edit.connect_material_property(tone,'',unreal.MaterialProperty.MP_BASE_COLOR)
        for k,v in [('ROUGHNESS',.95),('SPECULAR',.15)]:
            n=node(unreal.MaterialExpressionConstant);n.set_editor_property('r',v);edit.connect_material_property(n,'',getattr(unreal.MaterialProperty,'MP_'+k))
        edit.recompile_material(m);lib.save_loaded_asset(m,only_if_is_dirty=False);return m
    mats={'road':material('asphalt'),'road_edge':material('sidewalk_concrete')}
    mask_report=configure_ground(unreal,components['terrain'][0],package,Path(root),
        grass_texture=textures['dormant_grass'].get_path_name(),grass_gain=1.,
        grass_period_m=specs['dormant_grass']['period_m'],
        concrete_texture=textures['sidewalk_concrete'].get_path_name(),concrete_period_m=specs['sidewalk_concrete']['period_m'],
        asphalt_texture=textures['asphalt'].get_path_name(),asphalt_period_m=specs['asphalt']['period_m'],asphalt_gain=float(specs['asphalt'].get('native_basecolor_gain',1.)),asphalt_desaturate=False)
    mats['terrain']=components['terrain'][0].get_material(0)
    for kind,rows in components.items():
        for c in rows:c.set_material(0,mats[kind])
    for rows in components.values():
        for c in rows:
            now=geometry_state(c)
            if now!=before[c.get_path_name()]:raise RuntimeError('Material binding altered source geometry: '+str(dict(component=c.get_path_name(),before=before[c.get_path_name()],after=now)))
    if not level.save_current_level():raise RuntimeError('Could not save native material bindings')
    Path(output).write_text(json.dumps(dict(map=map_path,components={k:len(v)for k,v in components.items()},
        materials={k:v.get_path_name()for k,v in mats.items()},source_materials=specs,ground_mask=mask_report,
        native_basecolor_gains=dict(asphalt=float(specs['asphalt'].get('native_basecolor_gain',1.))),
        exported_geometry_and_transforms_unchanged=True,scope='Explicitly selected generated or reused native base-color assets with recorded metric period; roughness .95 and specular .15 are inferred dry-surface rendering defaults. Native base-color gains are explicit inferred reflectance conditions shared by road and parking pavement; texture files and metric sampling remain unchanged. No measured albedo or geometry reconstruction claim.'),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map',required=True)
    for k in ['root','materials','assets','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    script=out/'materials_ue.py';report=out/'bindings.json'
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.configure_city_surface_materials import configure\nconfigure(%r,%r,%r,%r,%r)\n'%(str(Path.cwd()),a.map,str(a.root.resolve()),str(a.materials.resolve()),str(a.assets.resolve()),str(report)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for key in ('CONFIG','CACHE','DATA'):
        d=out/'runtime'/key.lower();d.mkdir(parents=True,exist_ok=True);env['XDG_'+key+'_HOME']=str(d)
    command=['taskset','-c',os.environ['MAP_CPUSET'],str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),str(Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4/CarlaUE4.uproject'),'-run=pythonscript','-script='+str(script),'-nullrhi','-unattended','-nosourcecontrol','-nosound','-nocrashreports']
    with (out/'editor.log').open('w')as f:proc=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
    if report.exists():
        result=json.loads(report.read_text());result['editor_returncode']=proc.returncode
        report.write_text(json.dumps(result,indent=2)+'\n')
    if proc.returncode or not report.exists():raise RuntimeError('Native ground material binding failed; inspect editor.log')
    print(dict(report=str(report),editor_exit=proc.returncode))
if __name__=='__main__':main()
