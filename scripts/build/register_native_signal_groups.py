"""Persist existing native signal groups in CARLA's global traffic manager."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

MANAGER_LABEL = 'MAP_NativeSignalManager'


def create_native_manager(unreal):
    """Spawn the standard manager Blueprint without generating editor signals."""
    # A native-class ActorFactory accesses the viewport in UE4 commandlets.
    # Reuse the Blueprint factory path used by observed source signal groups.
    library = unreal.EditorAssetLibrary
    package = '/Game/MAPDevelopment/SourceControls'
    name = 'BP_NativeSignalManager'
    asset_path = package + '/' + name
    blueprint = library.load_asset(asset_path) if library.does_asset_exist(asset_path) else None
    if blueprint is None:
        factory = unreal.BlueprintFactory()
        factory.set_editor_property('parent_class', unreal.TrafficLightManager)
        blueprint = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, package, unreal.Blueprint, factory)
        if blueprint is None or not library.save_loaded_asset(blueprint):
            raise RuntimeError('Cannot create native manager Blueprint')
    manager = unreal.EditorLevelLibrary.spawn_actor_from_object(blueprint, unreal.Vector())
    manager.set_actor_label(MANAGER_LABEL)
    return manager


def inspect_registry(unreal, manager):
    records = []
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if not hasattr(actor, 'get_traffic_light_component'):
            continue
        component = actor.get_traffic_light_component()
        if component is None:
            continue
        controller, group = component.get_controller(), component.get_group()
        if controller is None or group is None:
            raise RuntimeError('Serialized signal has no controller/group: ' + actor.get_actor_label())
        signal_id, controller_id = component.get_sign_id(), controller.get_controller_id()
        junction_id = group.get_junction_id()
        records.append(dict(signal_id=signal_id, controller_id=controller_id,
            junction_id=junction_id, actor=actor.get_actor_label(),
            group_registered=manager.get_traffic_group(junction_id) == group,
            controller_registered=manager.get_controller(controller_id) == controller,
            component_registered=manager.get_traffic_sign(signal_id) == component))
    return dict(manager=manager.get_actor_label(), component_count=len(records),
        group_count=len({r['junction_id'] for r in records}), components=records,
        passed=bool(records) and all(r['group_registered'] and r['controller_registered']
                                    and r['component_registered'] for r in records))


def register_current_level(unreal, opendrive_triggers=False):
    """Register saved signal groups through the CARLA traffic-manager API."""
    level = unreal.EditorLevelLibrary
    managers = [a for a in level.get_all_level_actors() if isinstance(a, unreal.TrafficLightManager)]
    # Refresh the manager registry after replacing signal groups.
    for actor in managers:
        if actor.get_actor_label() == MANAGER_LABEL:
            level.destroy_actor(actor)
    managers = [a for a in level.get_all_level_actors() if isinstance(a, unreal.TrafficLightManager)]
    if len(managers) > 1:
        raise RuntimeError('Multiple native traffic managers in the loaded level')
    if managers:
        manager = managers[0]
    else:
        manager = create_native_manager(unreal)
    for actor in level.get_all_level_actors():
        if not hasattr(actor, 'get_traffic_light_component'):
            continue
        component = actor.get_traffic_light_component()
        if component is not None:
            if opendrive_triggers:
                tags = list(actor.get_editor_property('tags'))
                tag = unreal.Name('MAP_UseOpenDriveTriggers')
                if tag not in tags:
                    actor.set_editor_property('tags', tags + [tag])
            manager.register_light_component_generated(component)
    result = inspect_registry(unreal, manager)
    if not result['passed']:
        raise RuntimeError('Native manager registration does not match serialized signal objects')
    result['automatic_generation_preserved'] = True
    result['opendrive_triggers_requested'] = opendrive_triggers
    result['scope'] = 'Registers retained serialized groups; CARLA still creates the remaining OpenDRIVE controls at runtime.'
    return result


def configure(map_path, output, source_map=None, verify_only=False, opendrive_triggers=False):
    import unreal
    level = unreal.EditorLevelLibrary
    if source_map:
        if unreal.EditorAssetLibrary.does_asset_exist(map_path):
            raise ValueError('Clone into a new map package')
        if not level.new_level_from_template(map_path, source_map):
            raise RuntimeError('Cannot clone map')
    elif not level.load_level(map_path):
        raise RuntimeError('Cannot load map')
    if verify_only:
        managers = [a for a in level.get_all_level_actors() if isinstance(a, unreal.TrafficLightManager)]
        if len(managers) != 1:
            raise RuntimeError('Expected one persisted traffic manager')
        report = inspect_registry(unreal, managers[0])
    else:
        report = register_current_level(unreal, opendrive_triggers)
        if not level.save_current_level():
            raise RuntimeError('Cannot save registered native controls')
    report.update(map=map_path, source_map=source_map, verify_only=verify_only)
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError('Persisted native registry verification failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True)
    parser.add_argument('--source-map')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--opendrive-triggers', action='store_true',
        help='Use official lane-validity triggers for saved displays; requires native/patches/carla_saved_signal_triggers.patch')
    args = parser.parse_args()
    if args.verify_only and args.source_map:
        raise ValueError('Reopen verification cannot clone a map')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.source_map:
        content = Path(os.environ['CARLA_ROOT']) / 'Unreal/CarlaUE4/Content'
        def navigation(name):
            package = content / name.removeprefix('/Game/')
            return package.parent / 'OpenDrive' / (package.name + '.xodr')
        src, dst = navigation(args.source_map), navigation(args.map)
        if dst.exists():
            raise ValueError('Use a new navigation destination')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    script, report = output / 'register_ue.py', output / 'registry.json'
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.register_native_signal_groups import configure\nconfigure(%r,%r,%r,%r,%r)\n' % (
        str(Path.cwd()), args.map, str(report), args.source_map, args.verify_only, args.opendrive_triggers))
    command = ['taskset', '-c', os.environ['MAP_CPUSET'],
        str(Path(os.environ['UE4_ROOT']) / 'Engine/Binaries/Linux/UE4Editor'),
        str(Path(os.environ['CARLA_ROOT']) / 'Unreal/CarlaUE4/CarlaUE4.uproject'),
        '-run=pythonscript', '-script=' + str(script), '-nullrhi', '-unattended',
        '-nosourcecontrol', '-nosound', '-nocrashreports']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    with (output / 'editor.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not report.exists():
        raise RuntimeError('No native registry report; inspect editor.log')
    data = json.loads(report.read_text())
    data['editor_returncode'] = result.returncode
    if args.source_map:
        data['navigation_byte_identical'] = src.read_bytes() == dst.read_bytes()
    report.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2), flush=True)
    if result.returncode:
        raise RuntimeError('Unreal exited %s; any saved state still requires independent reopen' % result.returncode)


if __name__ == '__main__':
    main()
