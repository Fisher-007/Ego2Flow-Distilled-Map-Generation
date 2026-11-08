"""Apply source-identified height adjustments to selected map assets."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def configure(map_path, bindings_path, output):
    import unreal
    level = unreal.EditorLevelLibrary
    if not level.load_level(map_path):
        raise RuntimeError('Could not load context binding target')
    data = json.loads(Path(bindings_path).read_text())
    if not data['passed']:
        raise ValueError('Context source geometry needs resolution before native height binding')
    by_path, by_name, actors = {}, {}, {}
    for actor in level.get_all_level_actors():
        actors[actor.get_actor_label()] = actor
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh = component.get_editor_property('static_mesh')
            if mesh and component.is_visible():
                by_path.setdefault(mesh.get_path_name(), []).append(component)
                by_name.setdefault(mesh.get_name(), []).append(component)
    def values(component):
        t = component.get_world_transform()
        return dict(translation=[getattr(t.translation, k) for k in 'xyz'],
                    scale=[getattr(t.scale3d, k) for k in 'xyz'],
                    rotation=[getattr(t.rotation, k) for k in 'xyzw'],
                    materials=[component.get_material(i).get_path_name()
                               for i in range(component.get_num_materials())])
    buildings = []
    for row in data['buildings']:
        matches = (by_path.get(row['mesh_path'], []) if row['mesh_path'] else
                   [c for name in row['mesh_names'] for c in by_name.get(name, [])])
        if len(matches) != 1:
            raise ValueError('Expected one visible native building for ' + row['source_id'])
        component = matches[0]
        before = values(component)
        if max(abs(x) for x in before['rotation'][:2]) > 1e-7:
            raise ValueError('Vertical source binding requires an upright source building')
        factor, offset = row['z_scale'], row['z_offset_m']
        location, scale = before['translation'], before['scale']
        component.set_world_scale3d(unreal.Vector(scale[0], scale[1], scale[2] * factor))
        component.set_world_location(unreal.Vector(location[0], location[1],
            location[2] * factor + 100 * offset), False, False)
        after = values(component)
        if (after['translation'][:2] != before['translation'][:2] or
            after['scale'][:2] != before['scale'][:2] or
            after['rotation'] != before['rotation'] or after['materials'] != before['materials']):
            raise ValueError('Context binding changed building XY or appearance')
        buildings.append(dict(source_id=row['source_id'], component=component.get_path_name(),
                              before=before, after=after))
    canopy = []
    for row in data['canopy']:
        actor = actors[row['actor_label']]
        before_position = actor.get_actor_location()
        before_scale = actor.get_actor_scale3d()
        center, extent = actor.get_actor_bounds(False)
        old_bottom, old_top = (center.z - extent.z) / 100, (center.z + extent.z) / 100
        bottom, top = row['target_bottom_z_m'], row['target_top_z_m']
        factor = (top - bottom) / (old_top - old_bottom)
        actor.set_actor_scale3d(unreal.Vector(before_scale.x, before_scale.y,
                                               before_scale.z * factor))
        center, extent = actor.get_actor_bounds(False)
        delta = bottom * 100 - (center.z - extent.z)
        actor.set_actor_location(unreal.Vector(before_position.x, before_position.y,
                                               before_position.z + delta), False, False)
        center, extent = actor.get_actor_bounds(False)
        actual = [(center.z - extent.z) / 100, (center.z + extent.z) / 100]
        if max(abs(actual[0] - bottom), abs(actual[1] - top)) > .002:
            raise ValueError('Native canopy does not realize recorded support and crown top')
        canopy.append(dict(actor_label=row['actor_label'], previous_bounds_z_m=[old_bottom, old_top],
            target_bounds_z_m=[bottom, top], actual_bounds_z_m=actual,
            observed_top_retained=row['observed_top_retained'],
            inferred_height_retained=row['inferred_height_retained']))
    if not level.save_current_level():
        raise RuntimeError('Could not save current source context height bindings')
    report = dict(map=map_path, bindings=bindings_path, buildings=buildings, canopy=canopy,
        saved=True, passed=True,
        scope='Recorded inferred vertical support realized; native source XY, identities, materials and source crown tops retained. Source signs/heads are installed separately from bound conditions; visual and driving review remain pending.')
    Path(output).write_text(json.dumps(report, indent=2) + '\n')



def verify_saved(map_path, report_path, output):
    import unreal
    data = json.loads(Path(report_path).read_text())
    if data['map'] != map_path or not data['saved']:
        raise ValueError('Context report is not a saved result for this map')
    level = unreal.EditorLevelLibrary
    if not level.load_level(map_path):
        raise RuntimeError('Could not reopen saved context target')
    components, actors = {}, {}
    for actor in level.get_all_level_actors():
        actors[actor.get_actor_label()] = actor
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            components[component.get_path_name()] = component
    errors = []
    for row in data['buildings']:
        component = components.get(row['component'])
        if component is None:
            errors.append(dict(source_id=row['source_id'], error='Missing component'))
            continue
        t = component.get_world_transform()
        expected = row['after']
        actual = dict(translation=[getattr(t.translation, k) for k in 'xyz'],
                      scale=[getattr(t.scale3d, k) for k in 'xyz'],
                      rotation=[getattr(t.rotation, k) for k in 'xyzw'],
                      materials=[component.get_material(i).get_path_name()
                                 for i in range(component.get_num_materials())])
        for key, tolerance in [('translation', .1), ('scale', 1e-5), ('rotation', 1e-5)]:
            if max(abs(a-b) for a,b in zip(actual[key],expected[key])) > tolerance:
                errors.append(dict(source_id=row['source_id'], field=key,
                                   expected=expected[key], actual=actual[key]))
        if actual['materials'] != expected['materials'] or not component.is_visible():
            errors.append(dict(source_id=row['source_id'], error='Appearance/visibility mismatch'))
    max_canopy_error = 0.
    for row in data['canopy']:
        actor = actors.get(row['actor_label'])
        if actor is None:
            errors.append(dict(actor_label=row['actor_label'], error='Missing canopy'))
            continue
        center, extent = actor.get_actor_bounds(False)
        bounds = [(center.z-extent.z)/100, (center.z+extent.z)/100]
        error = max(abs(a-b) for a,b in zip(bounds,row['target_bounds_z_m']))
        max_canopy_error = max(max_canopy_error,error)
        if error > .002:
            errors.append(dict(actor_label=row['actor_label'], target=row['target_bounds_z_m'], actual=bounds))
    report = dict(map=map_path, saved_report=report_path, read_only=True,
                  buildings=len(data['buildings']), canopy=len(data['canopy']),
                  max_canopy_bound_error_m=max_canopy_error, errors=errors,
                  passed=not errors, scope='Independent saved native context transforms and support bounds; visual and driving review remain separate.')
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    if errors:
        raise RuntimeError('Saved native context differs from recorded support binding')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bindings', type=Path)
    mode.add_argument('--verify-report', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    function = 'verify_saved' if a.verify_report else 'configure'
    source = a.verify_report or a.bindings
    script = out / 'bind_context_ue.py'
    report = out / ('context_verification.json' if a.verify_report else 'native_context_heights.json')
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.apply_context_height_bindings import %s\n%s(%r,%r,%r)\n' %
                     (str(Path.cwd()), function, function, a.map, str(source.resolve()), str(report)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    for key in ('CONFIG', 'CACHE', 'DATA'):
        path = out / 'runtime' / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env['XDG_' + key + '_HOME'] = str(path)
    command = ['taskset', '-c', os.environ['MAP_CPUSET'],
        str(Path(os.environ['UE4_ROOT']) / 'Engine/Binaries/Linux/UE4Editor'),
        str(Path(os.environ['CARLA_ROOT']) / 'Unreal/CarlaUE4/CarlaUE4.uproject'),
        '-run=pythonscript', '-script=' + str(script), '-nullrhi', '-unattended',
        '-nosourcecontrol', '-nosound', '-nocrashreports']
    with (out / 'editor.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if report.exists():
        data = json.loads(report.read_text())
        data['editor_returncode'] = result.returncode
        report.write_text(json.dumps(data, indent=2) + '\n')
    if result.returncode or not report.exists():
        raise RuntimeError('Native context height application failed; inspect actual editor result')
    print(report, flush=True)


if __name__ == '__main__':
    main()
