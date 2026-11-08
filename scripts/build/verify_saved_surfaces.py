"""Reopen saved source surfaces and verify their exact declared native bindings."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def verify(map_path, source_report, output):
    import unreal
    report = json.loads(Path(source_report).read_text())
    assets = json.loads(Path(report['assets']).read_text())
    entries = {'map_assets_' + x['name']: x for x in assets['objects']}
    expected = {x['new_mesh']: x for x in report['surfaces']}
    level = unreal.EditorLevelLibrary
    if not level.load_level(map_path):
        raise RuntimeError('Could not reopen saved native surface map')
    found = {key: [] for key in expected}
    errors = []
    for actor in level.get_all_level_actors():
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh = component.get_editor_property('static_mesh')
            if not mesh or mesh.get_path_name() not in expected:
                continue
            key = mesh.get_path_name()
            row, entry = expected[key], entries[mesh.get_name()]
            t = component.get_world_transform()
            origin = entry.get('asset_origin_enu_m', [0., 0., 0.])
            position = [t.translation.x, t.translation.y, t.translation.z]
            expected_position = [100 * origin[0], -100 * origin[1], 100 * origin[2]]
            position_error = max(abs(x - y) for x, y in zip(position, expected_position))
            materials = [component.get_material(i).get_path_name()
                         for i in range(component.get_num_materials())]
            bounds = mesh.get_bounds()
            center = [bounds.origin.x, bounds.origin.y, bounds.origin.z]
            extent = [bounds.box_extent.x, bounds.box_extent.y, bounds.box_extent.z]
            actual_lo = [center[i] - extent[i] for i in range(3)]
            actual_hi = [center[i] + extent[i] for i in range(3)]
            lo, hi = entry['bounds_m']['min'], entry['bounds_m']['max']
            outer_lo = [lo[0] * 100, -hi[1] * 100, lo[2] * 100]
            outer_hi = [hi[0] * 100, -lo[1] * 100, hi[2] * 100]
            inner = (entry.get('native_precision') or {}).get('bounds_cm')
            if inner is None:
                inner = dict(min=outer_lo, max=outer_hi)
            bounds_error = max([0.] + [v for i in range(3) for v in (
                outer_lo[i] - actual_lo[i], actual_hi[i] - outer_hi[i],
                actual_lo[i] - inner['min'][i], inner['max'][i] - actual_hi[i])])
            complex_collision = (mesh.get_editor_property('body_setup').get_editor_property('collision_trace_flag')
                                 == unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
            unit_frame = ([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w] == [0., 0., 0., 1.]
                          and [t.scale3d.x, t.scale3d.y, t.scale3d.z] == [1., 1., 1.])
            actual = dict(mesh=key, component=component.get_path_name(),
                position_error_cm=position_error, bounds_error_cm=bounds_error,
                materials=materials, complex_collision=complex_collision,
                visible=component.is_visible(), hidden_in_game=component.get_editor_property('hidden_in_game'),
                collision_profile=str(component.get_collision_profile_name()), unit_frame=unit_frame)
            found[key].append(actual)
            if (position_error > .1 or bounds_error > .1 or materials != row['materials']
                or not complex_collision or not unit_frame or not actual['visible']
                or actual['hidden_in_game'] or actual['collision_profile'] == 'NoCollision'):
                errors.append(actual)
    counts = {k: len(v) for k, v in found.items() if len(v) != 1}
    result = dict(map=map_path, source_report=source_report,
        expected_surfaces=len(expected), persisted_surfaces=sum(map(len, found.values())),
        wrong_counts=counts, property_errors=errors, surfaces=found,
        reopened_saved_map=True, map_modified=False, passed=not counts and not errors,
        scope='Saved declared surface identities, centered placement, source bounds, materials and complex collision; native rendering and driving remain separate.')
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise RuntimeError('Saved native surface verification failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--map')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    previous = json.loads(a.report.read_text())
    target = a.map or previous['map']
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    script, result = out / 'verify_ue.py', out / 'surface_verification.json'
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.verify_saved_surfaces import verify\nverify(%r,%r,%r)\n' %
                     (str(Path.cwd()), target, str(a.report.resolve()), str(result)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    for key in ('CONFIG', 'CACHE', 'DATA'):
        path = out / 'runtime' / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env['XDG_' + key + '_HOME'] = str(path)
    cmd = ['taskset', '-c', os.environ['MAP_CPUSET'],
        str(Path(os.environ['UE4_ROOT']) / 'Engine/Binaries/Linux/UE4Editor'),
        str(Path(os.environ['CARLA_ROOT']) / 'Unreal/CarlaUE4/CarlaUE4.uproject'),
        '-run=pythonscript', '-script=' + str(script), '-nullrhi', '-unattended',
        '-nosourcecontrol', '-nosound', '-nocrashreports',
        '-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False']
    with (out / 'editor.log').open('w') as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not result.exists():
        raise RuntimeError('Saved surface verification produced no report')
    data = json.loads(result.read_text())
    data['editor_returncode'] = proc.returncode
    project = Path(os.environ['CARLA_ROOT']) / 'Unreal/CarlaUE4'
    xodr = project / 'Content' / target.removeprefix('/Game/').rsplit('/', 1)[0] / 'OpenDrive' / (target.rsplit('/', 1)[-1] + '.xodr')
    expected = Path(previous.get('supplied_xodr') or previous['source_xodr'])
    data['navigation_matches_supplied'] = xodr.read_bytes() == expected.read_bytes()
    data['passed'] = data['passed'] and data['navigation_matches_supplied']
    result.write_text(json.dumps(data, indent=2) + '\n')
    if proc.returncode or not data['passed']:
        raise RuntimeError('Saved native surface verification did not complete successfully')
    print({k: data[k] for k in ('expected_surfaces', 'persisted_surfaces', 'passed', 'navigation_matches_supplied', 'editor_returncode')}, flush=True)


if __name__ == '__main__':
    main()
