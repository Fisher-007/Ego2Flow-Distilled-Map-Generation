"""Place graph-selected source signs using existing CARLA sign meshes."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

def load_observed_face(unreal, spec, folder):
    """Import the measured UV panel and source photograph with native materials."""
    source = spec["source_face_asset"]
    name = spec["id"].replace(":", "_")
    lib = unreal.EditorAssetLibrary
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    imports = []
    for kind, filename in (("face", source["mesh"]), ("texture", source["texture"])):
        task = unreal.AssetImportTask()
        task.filename = filename
        task.destination_path = folder
        task.destination_name = name + "_" + kind
        task.automated = True
        task.replace_existing = True
        task.save = True
        if kind == "face":
            options = unreal.FbxImportUI()
            options.set_editor_property("import_mesh", True)
            options.set_editor_property("import_materials", False)
            options.set_editor_property("import_textures", False)
            options.set_editor_property("import_as_skeletal", False)
            options.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_STATIC_MESH)
            options.static_mesh_import_data.set_editor_property("convert_scene", False)
            options.static_mesh_import_data.set_editor_property("import_uniform_scale", 1.)
            task.options = options
        tools.import_asset_tasks([task])
        imports.append(lib.load_asset(task.imported_object_paths[0]))
    mesh, texture = imports
    extent = mesh.get_bounds().box_extent
    if extent.x <= 0 or extent.z <= 0:
        raise ValueError("Observed panel import did not preserve horizontal X and vertical Z axes")
    texture.set_editor_property("srgb", True)
    texture.set_editor_property("virtual_texture_streaming", False)
    texture.set_editor_property("address_x", unreal.TextureAddress.TA_CLAMP)
    texture.set_editor_property("address_y", unreal.TextureAddress.TA_CLAMP)
    lib.save_loaded_asset(texture, only_if_is_dirty=False)
    material_name = name + "_material"
    path = folder + "/" + material_name
    material = lib.load_asset(path) if lib.does_asset_exist(path) else tools.create_asset(
        material_name, folder, unreal.Material, unreal.MaterialFactoryNew())
    edit = unreal.MaterialEditingLibrary
    edit.delete_all_material_expressions(material)
    material.set_editor_property("two_sided", True)
    sample = edit.create_material_expression(material, unreal.MaterialExpressionTextureSample, 0, 0)
    sample.set_editor_property("texture", texture)
    edit.connect_material_property(sample, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    for property_name, value in (("ROUGHNESS", .8), ("SPECULAR", .1)):
        constant = edit.create_material_expression(material, unreal.MaterialExpressionConstant, 0, 0)
        constant.set_editor_property("r", value)
        edit.connect_material_property(constant, "", getattr(unreal.MaterialProperty, "MP_" + property_name))
    if spec.get('emissive_gain', 0.):
        gain = edit.create_material_expression(material, unreal.MaterialExpressionConstant, 0, 0)
        gain.set_editor_property('r', spec['emissive_gain'])
        multiply = edit.create_material_expression(material, unreal.MaterialExpressionMultiply, 0, 0)
        edit.connect_material_expressions(sample, 'RGB', multiply, 'A')
        edit.connect_material_expressions(gain, '', multiply, 'B')
        edit.connect_material_property(multiply, '', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    edit.recompile_material(material)
    lib.save_loaded_asset(material, only_if_is_dirty=False)
    mesh.set_material(0, material)
    lib.save_loaded_asset(mesh, only_if_is_dirty=False)
    return mesh


def load_observed_back(unreal, spec, folder):
    """Import the finite measured-outline solid, retaining its exact collision."""
    source = spec["source_face_asset"]
    task = unreal.AssetImportTask()
    task.filename = source["backing_mesh"]
    task.destination_path = folder
    task.destination_name = spec["id"].replace(":", "_") + "_back"
    task.automated = True
    task.replace_existing = True
    task.save = True
    options = unreal.FbxImportUI()
    for key, value in (("import_mesh", True), ("import_materials", False),
                       ("import_textures", False), ("import_as_skeletal", False)):
        options.set_editor_property(key, value)
    options.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_STATIC_MESH)
    options.static_mesh_import_data.set_editor_property("convert_scene", False)
    options.static_mesh_import_data.set_editor_property("import_uniform_scale", 1.)
    task.options = options
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    mesh = unreal.EditorAssetLibrary.load_asset(task.imported_object_paths[0])
    mesh.get_editor_property("body_setup").set_editor_property(
        "collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
    unreal.EditorAssetLibrary.save_loaded_asset(mesh, only_if_is_dirty=False)
    return mesh


def measured_frame(unreal, spec, yaw):
    """Use the measured sign frame, defaulting to vertical placement."""
    if "up_xyz" not in spec:
        return unreal.Rotator(yaw=yaw), None
    axes = [[v[0], -v[1], v[2]] for v in
            (spec["tangent_xyz"], spec["front_normal_xyz"], spec["up_xyz"])]
    rotation = unreal.MathLibrary.make_rotation_from_axes(
        *[unreal.Vector(*v) for v in axes])
    return rotation, axes


def frame_offset(unreal, axes, origin, scale):
    return unreal.Vector(*[sum(axes[j][i]*getattr(origin, 'xyz'[j])*scale[j]
                              for j in range(3)) for i in range(3)])


def actor_axes(actor):
    return [[v.x, v.y, v.z] for v in (actor.get_actor_forward_vector(),
            actor.get_actor_right_vector(), actor.get_actor_up_vector())]


def configure(map_path,conditions_file,output):
    import unreal
    lib,level=unreal.EditorAssetLibrary,unreal.EditorLevelLibrary
    level.load_level(map_path)
    conditions=json.loads(Path(conditions_file).read_text())
    specs=[r['native_construction']['settings'] for r in conditions['native_objects']
           if r['choice'] in ('retain','enhance') and r.get('native_construction',{}).get('kind')=='source_sign_face']
    paths=dict(no_entry='/Game/Carla/Static/TrafficSign/PostSigns/SM_DoNoEnter',
               one_way_right='/Game/Carla/Static/TrafficSign/OneWay/SM_oneWayR')
    labels={'MAP_SourceSign_'+s['id'].replace(':','_') for s in specs}
    for actor in list(level.get_all_level_actors()):
        if any((actor.get_actor_label() in (label, label+'_pole', label+'_back') or actor.get_actor_label().startswith(label+'_support_')) for label in labels):
            level.destroy_actor(actor)
    records=[]
    for spec in specs:
        source_face=spec.get('source_face_asset')
        category = 'TrafficLight' if spec['kind'] in ('pedestrian_stop_display', 'pedestrian_source_display') else 'TrafficSign'
        folder=map_path.rsplit('/Maps/',1)[0]+'/Static/'+category+'/ObservedControls'
        mesh=load_observed_face(unreal,spec,folder) if source_face else lib.load_asset(paths[spec['kind']])
        b=mesh.get_bounds();o,e=b.origin,b.box_extent
        center=list(spec['center_local_xyz_m']);right=spec['tangent_xyz']
        width=spec['width_m']
        fit=spec.get('asset_fit',{})
        center=[center[i]+right[i]*width*fit.get('center_right_offset_fraction',0.) for i in range(3)]
        width*=fit.get('width_multiplier',1.)
        sx=100*width/(2*e.x);sz=100*spec['height_m']/(2*e.z)
        yaw=-math.degrees(math.atan2(right[1],right[0]));angle=math.radians(yaw)
        x,y,z=center;normal=spec['front_normal_xyz']
        offset=unreal.Vector(sx*o.x*math.cos(angle)-o.y*math.sin(angle),
                             sx*o.x*math.sin(angle)+o.y*math.cos(angle),sz*o.z)
        rotation, axes = measured_frame(unreal, spec, yaw)
        if axes is not None:
            offset = frame_offset(unreal, axes, o, [sx, 1., sz])
        label='MAP_SourceSign_'+spec['id'].replace(':','_')
        actor=level.spawn_actor_from_object(mesh,unreal.Vector(100*x,-100*y,100*z)-offset,rotation)
        actor.set_actor_label(label);actor.set_actor_scale3d(unreal.Vector(sx,1,sz))
        actor.get_component_by_class(unreal.StaticMeshComponent).set_collision_profile_name('NoCollision' if source_face else 'BlockAll')
        if source_face:
            backing_mesh = load_observed_back(unreal, spec, folder) if source_face.get('backing_mesh') else lib.load_asset('/Engine/BasicShapes/Cube')
            back_center = [x-normal[0]*.015, y-normal[1]*.015,
                           z-normal[2]*.015 if axes is not None else z]
            back_scale = [sx, 1., sz] if source_face.get('backing_mesh') else [width, .02, spec['height_m']]
            back_position = unreal.Vector(100*back_center[0], -100*back_center[1], 100*back_center[2])
            if axes is not None:
                back_position -= frame_offset(unreal, axes, backing_mesh.get_bounds().origin, back_scale)
            back=level.spawn_actor_from_object(backing_mesh,back_position,rotation)
            back.set_actor_label(label+'_back')
            back.set_actor_scale3d(unreal.Vector(*back_scale))
            back.get_component_by_class(unreal.StaticMeshComponent).set_collision_profile_name('BlockAll')
        if spec.get('support_pole', True) and (spec['kind']=='no_entry' or source_face):
            pole=lib.load_asset('/Engine/BasicShapes/Cylinder');base=spec['ground_z_m'];top=spec.get('pole_top_z_m',z+spec['height_m']/2)
            support=level.spawn_actor_from_object(pole,unreal.Vector(100*(x-normal[0]*.055),-100*(y-normal[1]*.055),50*(base+top)))
            support.set_actor_label(label+'_pole');support.set_actor_scale3d(unreal.Vector(.055,.055,top-base))
        supports = []
        if spec.get('support_segments'):
            from scripts.build.install_observed_signal_heads import place_mast
            supports = place_mast(unreal, dict(signal_id=spec['id'],
                support_segments=spec['support_segments'],
                mast_support=dict(ground_support_evidence=spec.get('support_reference'))),
                label_prefix=label+'_support')
        records.append(dict(source_id=spec['id'],asset=mesh.get_path_name(),actor_label=label,
            support_segments=supports,
            position_local_xyz_m=center,width_m=width,height_m=spec['height_m'],yaw_carla_deg=yaw,
            measured_axes_carla=axes, backing_center_local_xyz_m=back_center if source_face else None,
            backing_asset=backing_mesh.get_path_name() if source_face else None,
            backing_outline_uv=source_face.get('face_outline_uv') if source_face else None,
            source_measurement=spec,native_visibility='pending',traffic_rule_modified=False,source_texture=source_face,physical_support=dict(panel='Measured-outline 2 cm solid' if source_face and source_face.get('backing_mesh') else 'Existing rectangular or CARLA body', pole=bool(spec.get('support_pole',True)), support_reference=spec.get('support_reference'), face_collision='NoCollision' if source_face else 'BlockAll')))
    if not level.save_current_level():raise RuntimeError('Source sign installation could not be saved')
    Path(output).write_text(json.dumps(dict(map=map_path,conditions=conditions_file,objects=records,
        scope='Existing CARLA meshes at source-measured sign faces. Native viewpoint review follows; existing road permissions and controller bindings retained.'),indent=2)+'\n')


def verify(map_path, report_path, output):
    """Reopen saved source faces and measure their transforms and asset bindings."""
    import unreal
    level = unreal.EditorLevelLibrary
    if not level.load_level(map_path):
        raise RuntimeError("Cannot reopen source-sign map")
    expected = json.loads(Path(report_path).read_text())
    actors = {}
    for actor in level.get_all_level_actors():
        actors.setdefault(actor.get_actor_label(), []).append(actor)
    rows, errors = [], []
    for item in expected["objects"]:
        label = item["actor_label"]
        found = actors.get(label, [])
        if len(found) != 1:
            errors.append(dict(label=label, reason="saved face actor count", actual=len(found)))
            continue
        actor = found[0]
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        mesh = component.get_editor_property("static_mesh")
        bounds = mesh.get_bounds()
        scale, position = actor.get_actor_scale3d(), actor.get_actor_location()
        yaw = actor.get_actor_rotation().yaw
        angle = math.radians(yaw)
        origin = bounds.origin
        actual = [(position.x + scale.x*origin.x*math.cos(angle)-scale.y*origin.y*math.sin(angle))/100,
                  -(position.y + scale.x*origin.x*math.sin(angle)+scale.y*origin.y*math.cos(angle))/100,
                  (position.z + scale.z*origin.z)/100]
        frame_error = None
        if item.get("measured_axes_carla") is not None:
            axes = actor_axes(actor)
            offset = frame_offset(unreal, axes, origin, [scale.x, scale.y, scale.z])
            actual = [(position.x+offset.x)/100, -(position.y+offset.y)/100,
                      (position.z+offset.z)/100]
            frame_error = max(abs(a-b) for actual_axis, expected_axis in
                zip(axes,item["measured_axes_carla"]) for a,b in zip(actual_axis,expected_axis))
        center_error = max(abs(a-b) for a,b in zip(actual,item["position_local_xyz_m"]))
        size_error = max(abs(2*bounds.box_extent.x*scale.x/100-item["width_m"]),
                         abs(2*bounds.box_extent.z*scale.z/100-item["height_m"]))
        row = dict(label=label, asset=mesh.get_path_name(), center_error_m=center_error,
                   size_error_m=size_error, yaw_error_deg=abs((yaw-item["yaw_carla_deg"]+180)%360-180),
                   face_collision=str(component.get_collision_profile_name()),
                   material=component.get_material(0).get_path_name() if component.get_material(0) else None)
        row["frame_component_error"] = frame_error
        orientation_failed = frame_error>1e-5 if frame_error is not None else row["yaw_error_deg"]>1e-3
        if row["asset"] != item["asset"] or max(center_error,size_error)>1e-3 or orientation_failed:
            errors.append(dict(label=label, reason="saved face transform or asset", actual=row))
        if item.get("backing_asset"):
            backs = actors.get(label+"_back", [])
            if len(backs) != 1:
                errors.append(dict(label=label, reason="saved backing actor count", actual=len(backs)))
            else:
                back = backs[0].get_component_by_class(unreal.StaticMeshComponent)
                body = back.get_editor_property("static_mesh")
                row["backing"] = dict(asset=body.get_path_name(), collision=str(back.get_collision_profile_name()),
                                     collision_trace=str(body.get_editor_property("body_setup").get_editor_property("collision_trace_flag")))
                if item.get("measured_axes_carla") is not None:
                    back_actor = backs[0]
                    back_axes = actor_axes(back_actor)
                    back_scale, back_position = back_actor.get_actor_scale3d(), back_actor.get_actor_location()
                    offset = frame_offset(unreal, back_axes, body.get_bounds().origin,
                                          [back_scale.x,back_scale.y,back_scale.z])
                    back_center = [(back_position.x+offset.x)/100, -(back_position.y+offset.y)/100,
                                   (back_position.z+offset.z)/100]
                    row["backing"]["center_error_m"] = max(abs(a-b) for a,b in
                        zip(back_center,item["backing_center_local_xyz_m"]))
                    row["backing"]["frame_component_error"] = max(abs(a-b) for actual_axis, expected_axis in
                        zip(back_axes,item["measured_axes_carla"]) for a,b in zip(actual_axis,expected_axis))
                    if row["backing"]["center_error_m"]>1e-3 or row["backing"]["frame_component_error"]>1e-5:
                        errors.append(dict(label=label, reason="saved measured backing transform", actual=row["backing"]))
                if row["backing"]["asset"] != item["backing_asset"] or row["backing"]["collision"]!="BlockAll":
                    errors.append(dict(label=label, reason="saved backing asset or collision", actual=row["backing"]))
        for support in item.get("support_segments", []):
            if len(actors.get(support["label"],[]))!=1:
                errors.append(dict(label=support["label"], reason="saved support actor missing or duplicated"))
        if item.get("source_texture") and (row["face_collision"]!="NoCollision" or row["material"] is None):
            errors.append(dict(label=label, reason="saved source face material or collision", actual=row))
        rows.append(row)
    result = dict(map=map_path, saved_report=report_path, objects=rows, errors=errors,
                  passed=not errors, scope="Independent saved-level reopen, measured face transforms, asset/material/collision bindings and finite mounting presence. Source-view visibility and actual driving remain separate.")
    Path(output).write_text(json.dumps(result,indent=2)+"\n")
    if errors:
        raise RuntimeError("Saved source-sign verification failed: "+str(len(errors)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map',required=True)
    p.add_argument('--conditions',type=Path);p.add_argument('--verify-report',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    entry='verify' if a.verify_report else 'configure'
    source=a.verify_report or a.conditions
    if source is None:p.error('Provide --conditions or --verify-report')
    script=out/'install_ue.py';report=out/('verification.json' if a.verify_report else 'native_signs.json')
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.install_source_signs import %s\n%s(%r,%r,%r)\n'%(
        str(Path.cwd()),entry,entry,a.map,str(source.resolve()),str(report)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    command=['taskset','-c',os.environ['MAP_CPUSET'],str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),
        str(Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4/CarlaUE4.uproject'),'-run=pythonscript','-script='+str(script),
        '-nullrhi','-unattended','-nosourcecontrol','-nosound','-nocrashreports']
    with (out/'editor.log').open('w') as log:run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not report.exists():raise RuntimeError('Source sign import failed; inspect editor.log')
    data=json.loads(report.read_text());data['editor_returncode']=run.returncode
    report.write_text(json.dumps(data,indent=2)+'\n');print(report,flush=True)
    if run.returncode:raise RuntimeError('Unreal exited '+str(run.returncode)+'; saved report requires independent reopen verification')


if __name__=='__main__':main()
