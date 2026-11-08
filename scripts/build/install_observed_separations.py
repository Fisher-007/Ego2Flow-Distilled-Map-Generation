"""Install finite source-selected assets using the existing FBX/ENU conventions."""
import argparse
import json
import os
from pathlib import Path
import subprocess


SEMANTIC_FOLDERS = {
    'Building','Fence','Pedestrian','Pole','Other','Road','RoadLine','SideWalk',
    'TrafficSign','Vegetation','Car','Wall','Sky','Ground','Bridge','RailTrack',
    'GuardRail','TrafficLight','Static','Dynamic','Water','Terrain','Truck',
    'Motorcycle','Bicycle','Bus','Rider','Train'}


def semantic_category(folder):
    """Return the actual CARLA fourth asset-path segment; choices stay in conditions."""
    category = folder[len('Static/'):] if folder.startswith('Static/') else folder
    if category not in SEMANTIC_FOLDERS:
        raise ValueError('Specify an actual CARLA semantic category: '+folder)
    return category


def relocate_semantics(map_path, reports, categories_file, output):
    """Move selected saved assets to reviewed semantic folders without reimport."""
    import unreal
    lib, level = unreal.EditorAssetLibrary, unreal.EditorLevelLibrary
    if not level.load_level(map_path):
        raise RuntimeError('Could not load semantic correction target')
    categories = json.loads(Path(categories_file).read_text())['categories']
    package = map_path.rsplit('/Maps/',1)[0]
    actors = {}
    for actor in level.get_all_level_actors():
        actors.setdefault(actor.get_actor_label(),[]).append(actor)
    records, changes = [], []
    for filename in reports:
        previous = json.loads(Path(filename).read_text())
        for row in previous['objects']:
            row = dict(row)
            if row['name'] in categories:
                category = semantic_category(categories[row['name']])
                matches = actors.get(row['actor_label'],[])
                if len(matches) != 1:
                    raise ValueError('Expected one saved facility: '+row['actor_label'])
                component = matches[0].get_component_by_class(unreal.StaticMeshComponent)
                mesh = component.get_editor_property('static_mesh')
                before = mesh.get_path_name()
                if before != row['asset']:
                    raise ValueError('Saved facility identity differs from source report: '+row['name'])
                destination = package+'/Static/'+category+'/Observed/'+row['name']
                if before.split('.')[0] != destination:
                    if lib.does_asset_exist(destination):
                        raise ValueError('Semantic destination already occupied: '+destination)
                    if not lib.rename_asset(before,destination):
                        raise RuntimeError('Could not relocate selected facility: '+before)
                    mesh = lib.load_asset(destination)
                    component.set_static_mesh(mesh)
                    lib.save_loaded_asset(mesh,only_if_is_dirty=False)
                row.update(asset=mesh.get_path_name(),semantic_folder=category)
                changes.append(dict(name=row['name'],previous_asset=before,
                    current_asset=row['asset'],semantic_folder=category,
                    category_at_carla_segment_4=row['asset'].split('/')[4]))
            records.append(row)
    if not level.save_current_level():
        raise RuntimeError('Could not save semantic correction')
    Path(output).write_text(json.dumps(dict(map=map_path,objects=records,
        semantic_changes=changes,conditions=categories_file,source_reports=reports,
        scope='Existing asset relocation only; source geometry, transforms, materials and collision retained. Runtime semantic pixels are separate.'),indent=2)+'\n')


def native_material(unreal, folder, name, rgba, two_sided):
    lib = unreal.EditorAssetLibrary
    path = folder + '/' + name
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = lib.load_asset(path) if lib.does_asset_exist(path) else tools.create_asset(
        name, folder, unreal.Material, unreal.MaterialFactoryNew())
    edit = unreal.MaterialEditingLibrary
    edit.delete_all_material_expressions(mat)
    mat.set_editor_property('two_sided', two_sided)
    color = edit.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, 0, 0)
    color.set_editor_property('constant', unreal.LinearColor(*rgba))
    edit.connect_material_property(color, '', unreal.MaterialProperty.MP_BASE_COLOR)
    for prop, value in (('ROUGHNESS', .85), ('SPECULAR', .1)):
        node = edit.create_material_expression(mat, unreal.MaterialExpressionConstant, 0, 0)
        node.set_editor_property('r', value)
        edit.connect_material_property(node, '', getattr(unreal.MaterialProperty, 'MP_'+prop))
    edit.recompile_material(mat)
    lib.save_loaded_asset(mat, only_if_is_dirty=False)
    return mat


def configure(map_path, manifests, output):
    import unreal
    lib, level = unreal.EditorAssetLibrary, unreal.EditorLevelLibrary
    if not level.load_level(map_path):
        raise RuntimeError('Could not open the selected native map')
    package = map_path.rsplit('/Maps/', 1)[0]
    records = []
    for filename in manifests:
        assets = json.loads(Path(filename).read_text())
        task = unreal.AssetImportTask()
        task.filename = assets['fbx']
        task.destination_path = package+'/ObservedImport/'+Path(assets['fbx']).stem
        task.automated = True
        task.replace_existing = True
        task.save = True
        options = unreal.FbxImportUI()
        options.import_mesh = True
        options.import_materials = False
        options.import_textures = False
        options.import_as_skeletal = False
        options.mesh_type_to_import = unreal.FBXImportType.FBXIT_STATIC_MESH
        data = options.static_mesh_import_data
        data.combine_meshes = False
        data.convert_scene = True
        data.convert_scene_unit = True
        data.transform_vertex_to_absolute = True
        data.generate_lightmap_u_vs = False
        data.auto_generate_collision = False
        task.options = options
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        imported = [lib.load_asset(path) for path in task.imported_object_paths]
        imported = [mesh for mesh in imported if isinstance(mesh, unreal.StaticMesh)]
        for entry in assets['objects']:
            matches = [mesh for mesh in imported if mesh.get_name() == entry['name']
                       or mesh.get_name().endswith('_'+entry['name'])]
            if len(matches) != 1:
                raise RuntimeError('Expected one imported source mesh: '+entry['name'])
            mesh = matches[0]
            category = semantic_category(entry.get('semantic_category',entry['semantic_folder']))
            destination = package+'/Static/'+category+'/Observed/'+entry['name']
            if lib.does_asset_exist(destination):
                # Replace the named source-selected asset in the target map.
                existing = lib.load_asset(destination)
                if existing.get_path_name() != mesh.get_path_name():
                    if not lib.delete_asset(destination):
                        raise RuntimeError('Could not replace selected observed asset')
            if mesh.get_path_name().split('.')[0] != destination:
                if not lib.rename_asset(mesh.get_path_name(), destination):
                    raise RuntimeError('Could not move observed asset to semantic folder')
            mesh = lib.load_asset(destination)
            low, high = entry['local_bounds_m']
            bounds = mesh.get_bounds()
            expected = [(low[i]+high[i])*50*(1 if i!=1 else -1) for i in range(3)]
            expected += [(high[i]-low[i])*50 for i in range(3)]
            actual = [bounds.origin.x, bounds.origin.y, bounds.origin.z,
                      bounds.box_extent.x, bounds.box_extent.y, bounds.box_extent.z]
            error = max(abs(x-y) for x,y in zip(actual, expected))
            if error > .1:
                raise ValueError('Observed asset bounds changed by more than 1 mm: '+entry['name'])
            mesh.get_editor_property('body_setup').set_editor_property(
                'collision_trace_flag', unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
            mat = native_material(unreal, package+'/Appearance/ObservedSeparators',
                                  entry['name']+'_material', entry['color_rgba'], entry.get('two_sided', False))
            mesh.set_material(0, mat)
            lib.save_loaded_asset(mesh, only_if_is_dirty=False)
            label = 'MAP_ObservedSeparator_'+entry['name']
            for actor in list(level.get_all_level_actors()):
                if actor.get_actor_label() == label:
                    level.destroy_actor(actor)
            origin = entry['asset_origin_enu_m']
            actor = level.spawn_actor_from_object(mesh, unreal.Vector(origin[0]*100, -origin[1]*100, origin[2]*100))
            actor.set_actor_label(label)
            component = actor.get_component_by_class(unreal.StaticMeshComponent)
            component.set_collision_profile_name(entry.get('collision', 'BlockAll'))
            component.set_editor_property('cast_shadow', entry.get('cast_shadow', True))
            records.append(dict(name=entry['name'], actor_label=label, source_id=entry.get('source_id'),
                asset=mesh.get_path_name(), asset_origin_enu_m=origin,
                local_bounds_error_cm=error, semantic_folder=category,
                collision=entry.get('collision', 'BlockAll'), material=mat.get_path_name(),
                source_manifest=filename, native_visibility='pending'))
    if not level.save_current_level():
        raise RuntimeError('Observed asset installation could not be saved')
    Path(output).write_text(json.dumps(dict(map=map_path, objects=records,
        scope='Source-selected native asset import/save only. Independent reopen, physical clearance and source-camera rendering are separate.'), indent=2)+'\n')


def verify(map_path, installation, output):
    import unreal
    level = unreal.EditorLevelLibrary
    if not level.load_level(map_path):
        raise RuntimeError('Could not reopen the selected native map')
    expected = json.loads(Path(installation).read_text())['objects']
    actors = {a.get_actor_label(): a for a in level.get_all_level_actors()}
    rows = []
    for item in expected:
        actor = actors.get(item['actor_label'])
        component = actor.get_component_by_class(unreal.StaticMeshComponent) if actor else None
        mesh = component.get_editor_property('static_mesh') if component else None
        origin = item['asset_origin_enu_m']
        p = actor.get_actor_location() if actor else None
        error = max(abs(a-b) for a,b in zip([p.x,p.y,p.z], [100*origin[0],-100*origin[1],100*origin[2]])) if p else None
        actual_collision = str(component.get_collision_profile_name()) if component else None
        passed = bool(mesh and mesh.get_path_name() == item['asset'] and error <= .1
                      and actual_collision == item['collision'] and component.is_visible())
        rows.append(dict(actor_label=item['actor_label'], present=actor is not None,
            mesh=mesh.get_path_name() if mesh else None, origin_error_cm=error,
            collision=actual_collision, passed=passed))
    report = dict(map=map_path, installation=installation, objects=rows,
        passed=all(r['passed'] for r in rows), scope='Independent persisted identity, origin, collision profile and visibility flags; actual rendering/driving remain separate.')
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise ValueError('Observed asset persistence mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', required=True)
    p.add_argument('--assets', nargs='+', type=Path)
    p.add_argument('--verify-report', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    function = 'verify' if a.verify_report else 'configure'
    source = str(a.verify_report.resolve()) if a.verify_report else [str(x.resolve()) for x in a.assets]
    report = out/('persistence.json' if a.verify_report else 'native_assets.json')
    script = out/'assets_ue.py'
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.install_observed_separations import %s\n%s(%r,%r,%r)\n' %
                      (str(Path.cwd()), function, function, a.map, source, str(report)))
    command = ['taskset','-c',os.environ['MAP_CPUSET'],
        str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),
        str(Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4/CarlaUE4.uproject'),
        '-run=pythonscript','-script='+str(script),'-nullrhi','-unattended',
        '-nosourcecontrol','-nosound','-nocrashreports']
    with (out/'editor.log').open('w') as log:
        run = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'),
                             stdout=log, stderr=subprocess.STDOUT)
    print(dict(editor_exit_code=run.returncode, saved_report=report.exists(), report=str(report)), flush=True)
    if run.returncode or not report.exists():
        raise RuntimeError('Native observed asset operation failed; preserve editor.log and any saved report')


if __name__ == '__main__':
    main()
