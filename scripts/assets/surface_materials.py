"""Create native ground materials from source-bound textures and masks."""
import json
from pathlib import Path


def link(edit, source, output_name, target, input_name):
    if not edit.connect_material_expressions(source, output_name, target, input_name):
        raise RuntimeError('Material pin connection failed: '+source.get_class().get_name()+' -> '+target.get_class().get_name()+':'+input_name)



def configure_ground(unreal, component, package, root, grass_texture='T_Grass01_d', grass_gain=1., asphalt_texture=None, asphalt_period_m=3., asphalt_gain=.35, grass_period_m=3., concrete_texture=None, concrete_period_m=3., asphalt_desaturate=True, concrete_gain=1.):
    # Combine the grass texture with the inferred sidewalk mask.
    lib=unreal.EditorAssetLibrary
    edit=unreal.MaterialEditingLibrary;tools=unreal.AssetToolsHelpers.get_asset_tools()
    mask_spec=json.loads((root/'ground_mask.json').read_text())
    task=unreal.AssetImportTask();task.filename=str(root/'ground_mask.png');task.destination_path=package+'/Appearance/Context';task.automated=True;task.replace_existing=True;task.save=True
    tools.import_asset_tasks([task]);mask_tex=lib.load_asset(task.imported_object_paths[0]);mask_tex.set_editor_property('srgb',False)
    # Use ordinary linear samplers for world-space masks at every resolution.
    mask_tex.set_editor_property('virtual_texture_streaming',False)
    lib.save_loaded_asset(mask_tex,only_if_is_dirty=False)
    path=package+'/Appearance/Context/observed_streetside'
    material=lib.load_asset(path) if lib.does_asset_exist(path) else tools.create_asset('observed_streetside',package+'/Appearance/Context',unreal.Material,unreal.MaterialFactoryNew())
    edit.delete_all_material_expressions(material)
    def node(cls):return edit.create_material_expression(material,cls,0,0)
    def connect(a,b,key):link(edit,a,'',b,key)
    world=node(unreal.MaterialExpressionWorldPosition);xy=node(unreal.MaterialExpressionComponentMask);xy.set_editor_property('r',True);xy.set_editor_property('g',True);xy.set_editor_property('b',False);xy.set_editor_property('a',False);connect(world,xy,'')
    scale=node(unreal.MaterialExpressionMultiply);scale.set_editor_property('const_b',1/300.);connect(xy,scale,'A')
    def sample(path, period_m=3.):
        n=node(unreal.MaterialExpressionTextureSample);n.set_editor_property('texture',lib.load_asset(path))
        if not n.texture.get_editor_property('srgb'):n.set_editor_property('sampler_type',unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
        texture_scale=scale
        if period_m!=3.:
            texture_scale=node(unreal.MaterialExpressionMultiply);texture_scale.set_editor_property('const_b',1/(100*period_m));connect(xy,texture_scale,'A')
        connect(texture_scale,n,'');return n
    grass=sample(grass_texture if grass_texture.startswith('/Game/') else '/Game/Carla/Static/GenericMaterials/Ground/Textures/Grass/'+grass_texture, grass_period_m)
    grass_tone=node(unreal.MaterialExpressionMultiply);grass_tone.set_editor_property('const_b',grass_gain);link(edit,grass,'RGB',grass_tone,'A')
    concrete=sample(concrete_texture or '/Game/Carla/Static/GenericMaterials/Concrete/T_Concrete_01_d', concrete_period_m)
    concrete_value,concrete_channel=concrete,'RGB'
    if concrete_gain != 1.:
        concrete_value=node(unreal.MaterialExpressionMultiply)
        concrete_value.set_editor_property('const_b',float(concrete_gain))
        link(edit,concrete,'RGB',concrete_value,'A');concrete_channel=''
    # World XY uses centimeters and mirrored northing, matching raster rows.
    lower,upper=mask_spec['lower'],mask_spec['upper'];width=upper[0]-lower[0];height=upper[1]-lower[1]
    uv_scale=node(unreal.MaterialExpressionConstant2Vector);uv_scale.set_editor_property('r',1/(100*width));uv_scale.set_editor_property('g',1/(100*height))
    mul=node(unreal.MaterialExpressionMultiply);connect(xy,mul,'A');connect(uv_scale,mul,'B')
    shift=node(unreal.MaterialExpressionConstant2Vector);shift.set_editor_property('r',-lower[0]/width);shift.set_editor_property('g',upper[1]/height)
    add=node(unreal.MaterialExpressionAdd);connect(mul,add,'A');connect(shift,add,'B')
    mask=node(unreal.MaterialExpressionTextureSample);mask.texture=mask_tex;mask.set_editor_property('sampler_type',unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR);connect(add,mask,'')
    blend=node(unreal.MaterialExpressionLinearInterpolate);link(edit,grass_tone,'',blend,'A');link(edit,concrete_value,concrete_channel,blend,'B');link(edit,mask,'R',blend,'Alpha')
    if (root/'parking_mask.png').exists():
        task=unreal.AssetImportTask();task.filename=str(root/'parking_mask.png');task.destination_path=package+'/Appearance/Context';task.automated=True;task.replace_existing=True;task.save=True
        tools.import_asset_tasks([task]);tex=lib.load_asset(task.imported_object_paths[0]);tex.set_editor_property('srgb',False)
        tex.set_editor_property('virtual_texture_streaming',False);lib.save_loaded_asset(tex,only_if_is_dirty=False)
        parking_mask=node(unreal.MaterialExpressionTextureSample);parking_mask.texture=tex;parking_mask.set_editor_property('sampler_type',unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR);connect(add,parking_mask,'')
        asphalt=sample(asphalt_texture or '/Game/Carla/Static/GenericMaterials/Asphalt/Textures/Asphalt01/T_Asphalt01_d', asphalt_period_m)
        gray=node(unreal.MaterialExpressionDesaturation) if asphalt_desaturate else asphalt
        if asphalt_desaturate:link(edit,asphalt,'RGB',gray,'')
        tone=node(unreal.MaterialExpressionMultiply);tone.set_editor_property('const_b',asphalt_gain)
        link(edit,gray,'' if asphalt_desaturate else 'RGB',tone,'A')
        final=node(unreal.MaterialExpressionLinearInterpolate);connect(blend,final,'A');connect(tone,final,'B');link(edit,parking_mask,'R',final,'Alpha');blend=final
        mask_spec['parking']=json.loads((root/'parking.json').read_text())
    # Retain local land-cover resolution over the city background mask.
    patches_file=root/'reviewed_patches.json'
    if patches_file.exists():
        patch_reports=[]
        for index,patch in enumerate(json.loads(patches_file.read_text())['patches']):
            lower,upper=patch['lower'],patch['upper']
            width,height=upper[0]-lower[0],upper[1]-lower[1]
            factor=node(unreal.MaterialExpressionConstant2Vector)
            factor.set_editor_property('r',1/(100*width));factor.set_editor_property('g',1/(100*height))
            multiply=node(unreal.MaterialExpressionMultiply);connect(xy,multiply,'A');connect(factor,multiply,'B')
            offset=node(unreal.MaterialExpressionConstant2Vector)
            offset.set_editor_property('r',-lower[0]/width);offset.set_editor_property('g',upper[1]/height)
            local_uv=node(unreal.MaterialExpressionAdd);connect(multiply,local_uv,'A');connect(offset,local_uv,'B')
            def patch_sample(key):
                task=unreal.AssetImportTask();task.filename=patch[key]
                task.destination_path=package+'/Appearance/Context/Reviewed'+str(index)
                task.destination_name=key;task.automated=True;task.save=True
                tools.import_asset_tasks([task]);texture=lib.load_asset(task.imported_object_paths[0])
                texture.set_editor_property('srgb',False)
                texture.set_editor_property('virtual_texture_streaming',False)
                for axis in ('address_x','address_y'):texture.set_editor_property(axis,unreal.TextureAddress.TA_CLAMP)
                lib.save_loaded_asset(texture,only_if_is_dirty=False)
                expression=node(unreal.MaterialExpressionTextureSample);expression.texture=texture
                expression.set_editor_property('sampler_type',unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
                connect(local_uv,expression,'');return expression
            local_ground,local_parking,coverage=[patch_sample(key) for key in ('ground_mask','parking_mask','coverage_mask')]
            def patch_tone(source, channel, gain):
                tone_node=node(unreal.MaterialExpressionMultiply)
                tone_node.set_editor_property('const_b',float(gain))
                link(edit,source,channel,tone_node,'A')
                return tone_node
            patch_grass=patch_tone(grass_tone,'',patch.get('grass_gain',1.))
            patch_concrete=patch_tone(concrete_value,concrete_channel,patch.get('concrete_gain',1.))
            local_color=node(unreal.MaterialExpressionLinearInterpolate)
            connect(patch_grass,local_color,'A');connect(patch_concrete,local_color,'B');link(edit,local_ground,'R',local_color,'Alpha')
            local_paved=node(unreal.MaterialExpressionLinearInterpolate)
            connect(local_color,local_paved,'A');connect(tone,local_paved,'B');link(edit,local_parking,'R',local_paved,'Alpha')
            selected=node(unreal.MaterialExpressionLinearInterpolate)
            connect(blend,selected,'A');connect(local_paved,selected,'B');link(edit,coverage,'R',selected,'Alpha')
            blend=selected;patch_reports.append(patch)
        mask_spec['reviewed_patches']=patch_reports
    edit.connect_material_property(blend,'',unreal.MaterialProperty.MP_BASE_COLOR)
    for prop,val in [('ROUGHNESS',.95),('SPECULAR',.15)]:
        n=node(unreal.MaterialExpressionConstant);n.set_editor_property('r',val);edit.connect_material_property(n,'',getattr(unreal.MaterialProperty,'MP_'+prop))
    edit.recompile_material(material);lib.save_loaded_asset(material,only_if_is_dirty=False)
    component.set_material(0,material)
    return mask_spec

