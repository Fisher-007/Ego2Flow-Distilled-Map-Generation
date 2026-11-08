#!/usr/bin/env python3
"""Configure and run the map construction workflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
# Prefer modules from this checkout.
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

from map_generation.city_workflow import CityWorkflow, read_json, write_json
from map_testing.adapters import check_adapter, configure

SOFTWARE_KEYS = (
    'map_python', 'scenario_dreamer_python', 'scenario_dreamer_source',
    'scenario_dreamer_data', 'scenario_dreamer_checkpoint', 'scenario_dreamer_autoencoder',
    'cosmos_python', 'cosmos_model', 'cosmos_guardrail_root',
    'blender', 'netconvert', 'sumo_home', 'ffmpeg', 'carla_root', 'ue_root',
)


def resolve_path(value, base):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return os.path.abspath(base / path if not path.is_absolute() else path)


def load_config(path):
    path = Path(path).resolve()
    raw = read_json(path)
    software = dict(raw['software'])
    software['map_python'] = software.get('map_python') or sys.executable
    software = {key: resolve_path(software[key], path.parent) for key in SOFTWARE_KEYS}
    source = dict(raw['input'])
    source['data_root'] = resolve_path(source['data_root'], path.parent)
    for key in ('city_config', 'catalog', 'osm', 'terrain_dem'):
        if source.get(key):
            source[key] = resolve_path(source[key], path.parent)
    name = raw['map_name']
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
        raise ValueError('map_name must start with a letter and use letters, digits or underscores')
    if source.get('city') not in {'ATX', 'DTW', 'MIA', 'PAO', 'PIT', 'WDC'}:
        raise ValueError('input.city must be a supported AV2 city code')
    if not source.get('city_config'):
        bounds = source.get('bbox_city_xy')
        if not isinstance(bounds, list) or len(bounds) != 4 or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
            raise ValueError('Supply input.bbox_city_xy [xmin,ymin,xmax,ymax] in AV2 city metres, or input.city_config')
    if not isinstance(source.get('area_count'), int) or source['area_count'] < 1:
        raise ValueError('input.area_count must be a positive integer')
    agent = dict(raw['agent'], provider='http')
    for role in ('decision', 'static_review', 'dynamic_review'):
        setting = {**agent, **agent.get('roles', {}).get(role, {})}
        if not setting.get('base_url') or not setting.get('model'):
            raise ValueError(f'API endpoint and model are required for {role}')
        if setting.get('provider') != 'http':
            raise ValueError('This entry point uses the HTTP API for all three roles')
        if setting.get('response_mode', 'json_schema') not in {'json_schema', 'tool_call'}:
            raise ValueError('response_mode must be json_schema or tool_call')
        if 'api_key' in setting:
            raise ValueError('Use api_key_env with an environment-variable name')
    resource = dict(raw['resources'])
    cpu_set = set()
    for interval in resource['cpus'].split(','):
        limits = list(map(int, interval.split('-')))
        if len(limits) == 1:
            cpu_set.add(limits[0])
        elif len(limits) == 2 and limits[0] <= limits[1]:
            cpu_set.update(range(limits[0], limits[1] + 1))
        else:
            raise ValueError('resources.cpus must be a CPU list such as 0-3 or 0-3,8-11')
    if not cpu_set or min(cpu_set) < 0:
        raise ValueError('resources.cpus must contain nonnegative CPU numbers')
    if not isinstance(resource['gpu'], int) or resource['gpu'] < 0:
        raise ValueError('resources.gpu must be a nonnegative GPU index')
    if not isinstance(resource['port'], int) or not 1024 <= resource['port'] <= 65525:
        raise ValueError('resources.port must leave space for port+1, port+2 and port+10')
    run = resolve_path(raw['output'], path.parent)
    environment = dict(raw.get('environment', {}))
    environment.update(
        CARLA_ROOT=software['carla_root'], UE4_ROOT=software['ue_root'],
        MAP_BLENDER=software['blender'], SUMO_HOME=software['sumo_home'],
        PROJECT_ROOT=software['scenario_dreamer_source'],
        SCRATCH_ROOT=software['scenario_dreamer_data'], DATASET_ROOT=software['scenario_dreamer_data'],
        CONFIG_PATH=str(Path(software['scenario_dreamer_source']) / 'cfgs'),
        MAP_SD_CHECKPOINT=software['scenario_dreamer_checkpoint'],
        MAP_SD_AUTOENCODER=software['scenario_dreamer_autoencoder'],
        WANDB_MODE='offline', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1',
    )
    task = (f'Construct the complete declared map {name} for {source["city"]} using the supplied input. '
            f'Inspect {source["area_count"]} distinct source-supported areas and their connecting roads. '
            'Use the information graph, protected road structure, conditional structure and appearance generation, '
            'native compilation, and independent static/dynamic feedback. Follow the supplied construction instructions. '
            'Use the requested region and sources; do not change the task scope to obtain a pass. '
            'Deliver the native map, input/generation manifests, loading instructions and observed limitations.')
    if raw.get('instructions'):
        task += '\nUser input: ' + raw['instructions']
    return dict(run_id=name, map_name=name, task=task, workspace=str(ROOT),
                input=source, data_root=source['data_root'], output_root=run,
                python=software['map_python'], agent=agent, resources={'map': resource},
                environment=environment,
                dynamic_policy=configure(raw.get('dynamic_policy'), path.parent), **software)


def doctor(config, *, check_api_key=True, native_ready=True):
    """Check the installed interfaces without loading generator weights."""
    failures = []
    try:
        check_adapter(config.get('dynamic_policy'))
    except ValueError as error:
        failures.append('dynamic_policy: ' + str(error))
    resource = config['resources']['map']
    affinity = os.sched_getaffinity(0)
    requested = set()
    for part in resource['cpus'].split(','):
        ends = [int(x) for x in part.split('-')]
        requested.update(range(ends[0], ends[-1] + 1))
    if not requested <= affinity:
        failures.append(f'CPUs outside this process allocation: {sorted(requested - affinity)}')
    for command in ('taskset', 'patch'):
        if shutil.which(command) is None:
            failures.append(f'Missing executable on PATH: {command}')
    executables = ('map_python', 'scenario_dreamer_python', 'cosmos_python', 'blender', 'netconvert', 'ffmpeg')
    for key in SOFTWARE_KEYS:
        p = Path(config[key])
        if not p.exists():
            failures.append(f'{key}: missing {p}')
        elif key in executables and (not p.is_file() or not os.access(p, os.X_OK)):
            failures.append(f'{key}: not executable: {p}')
    for key in ('city_config', 'catalog', 'osm', 'terrain_dem'):
        if config['input'].get(key) and not Path(config['input'][key]).is_file():
            failures.append(f'input.{key}: missing file')
    if not config['input'].get('allow_download') and not Path(config['data_root']).is_dir():
        failures.append('Local data_root is missing and input.allow_download is false')
    expected = [
        Path(config['carla_root']) / 'Unreal/CarlaUE4/CarlaUE4.uproject',
        Path(config['carla_root']) / 'Util/BuildTools/Import.py',
        Path(config['ue_root']) / 'Engine/Binaries/Linux/UE4Editor',
        Path(config['scenario_dreamer_source']) / 'cfgs/config.yaml',
    ]
    if native_ready:
        expected.append(Path(config['carla_root']) / 'Unreal/CarlaUE4/Plugins/MapBuilderEditorTools/MapBuilderEditorTools.uplugin')
        receipt = Path(config['carla_root']) / 'map_builder_native_setup.json'
        if not receipt.is_file() or read_json(receipt).get('status') != 'built':
            failures.append('Native adapter is not prepared: run prepare-native with this configuration')
    for p in expected:
        if not p.is_file():
            failures.append(f'Missing dependency file: {p}')
    if check_api_key:
        for role in ('decision', 'static_review', 'dynamic_review'):
            setting = {**config['agent'], **config['agent'].get('roles', {}).get(role, {})}
            key_name = setting.get('api_key_env', 'MAP_AGENT_API_KEY')
            if key_name and not os.environ.get(key_name):
                failures.append(f'{role}: environment variable {key_name} is not set')
    checks = {
        'map_python': 'import numpy, scipy, shapely, pyarrow, av2, carla, scenariogeneration, rasterio, trimesh, orjson, osqp, sumolib, requests, PIL, matplotlib, lxml',
        'scenario_dreamer_python': 'import torch, hydra, pytorch_lightning, torch_geometric, shapely; from models.scenario_dreamer_ldm import ScenarioDreamerLDM',
        'cosmos_python': 'import torch; from diffusers import Cosmos3OmniModularPipeline, UniPCMultistepScheduler; from cosmos_guardrail import cosmos_guardrail',
    }
    env = {**os.environ, **config['environment'], 'CUDA_VISIBLE_DEVICES': '',
           'PYTHONPATH': str(ROOT) + os.pathsep + config['scenario_dreamer_source'],
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    for key, script in checks.items():
        if Path(config[key]).is_file() and os.access(config[key], os.X_OK):
            try:
                result = subprocess.run(['taskset', '-c', resource['cpus'], config[key], '-c', script],
                                        env=env, capture_output=True, text=True, timeout=60)
                if result.returncode:
                    failures.append(f'{key}: {result.stderr.strip()[-1800:]}')
            except subprocess.TimeoutExpired:
                failures.append(f'{key}: import check exceeded 60 seconds')
    for item in failures:
        print('MISSING: ' + item, flush=True)
    if not failures:
        print('Environment checks passed. Model generation and native map quality are checked during construction.', flush=True)
    return not failures


def prepare_native(config):
    """Install native adapters into the user-supplied dedicated CARLA source build."""
    carla = Path(config['carla_root'])
    resource = config['resources']['map']
    output = carla / 'map_builder_setup_logs'
    output.mkdir(parents=True, exist_ok=True)
    applied = []
    progress = output / 'patches.json'
    previous = read_json(progress) if progress.is_file() else []
    for name in ('carla_generic_mesh_sensor_trace.patch', 'carla_generic_mesh_distance_field_lighting.patch', 'carla_saved_signal_triggers.patch'):
        patch = ROOT / 'native/patches' / name
        command = ['taskset', '-c', resource['cpus'], 'patch', '--batch', '-d', str(carla), '-p1', '-i', str(patch)]
        if name in previous:
            applied.append({'patch': name, 'action': 'previously_applied'})
            continue
        forward = subprocess.run([*command, '--forward', '--dry-run'], capture_output=True, text=True)
        if forward.returncode == 0:
            result = subprocess.run([*command, '--forward'], capture_output=True, text=True)
            (output / (name + '.log')).write_text(result.stdout + result.stderr)
            result.check_returncode()
            applied.append({'patch': name, 'action': 'applied'})
        else:
            reverse = subprocess.run([*command, '--reverse', '--dry-run'], capture_output=True, text=True)
            if reverse.returncode:
                (output / (name + '.log')).write_text(forward.stdout + forward.stderr + reverse.stdout + reverse.stderr)
                raise RuntimeError(f'Patch does not match the CARLA source: {name}; inspect {output}')
            applied.append({'patch': name, 'action': 'already_applied'})
        previous.append(name)
        write_json(progress, previous)
    env = {**os.environ, **config['environment'], 'MAP_CPUSET': resource['cpus'], 'PYTHONPATH': str(ROOT)}
    command = ['taskset', '-c', resource['cpus'], config['map_python'], '-m', 'scripts.build.prepare_editor_asset_tools',
               '--parallel-actions', str(resource.get('threads', 2)), '--output', str(output / 'editor')]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    write_json(carla / 'map_builder_native_setup.json', {'status': 'built', 'patches': applied, 'log_directory': str(output)})


def initialize(config):
    run = Path(config['output_root'])
    run.mkdir(parents=True, exist_ok=True)
    workspace = run / 'workspace'
    if workspace.exists():
        raise ValueError(f'Uninitialized workspace already exists at {workspace}; choose a new output directory')
    workspace.mkdir()
    # Copy source directories into the run workspace.
    for name in ('map_generation', 'map_testing', 'scripts', 'native', 'examples'):
        shutil.copytree(ROOT / name, workspace / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'configs/workflows', workspace / 'configs/workflows')
    config = dict(config, workspace=str(workspace))
    config['environment'] = dict(config['environment'], PYTHONPATH=str(workspace))
    native = run / 'native/carla'
    env = {**os.environ, **config['environment']}
    subprocess.run(['taskset', '-c', config['resources']['map']['cpus'], config['map_python'], '-m',
                    'scripts.build.prepare_native_view', '--source-carla', config['carla_root'], '--output', str(native)],
                   cwd=workspace, env=env, check=True)
    config['installed_carla_root'] = config['carla_root']
    config['carla_root'] = str(native)
    config['environment']['CARLA_ROOT'] = str(native)
    config['environment']['MAP_RUNTIME_DIR'] = str(run / 'runtime')
    config['environment']['MAP_DDC_DIR'] = str(run / 'derived_data')
    write_json(run / 'resolved_config.json', config)
    return CityWorkflow.create(run, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'doctor', 'prepare-native', 'status', 'respond', 'reopen', 'stop'])
    parser.add_argument('--config', type=Path, help='JSON configuration; used for new runs and environment checks')
    parser.add_argument('--run', type=Path, help='Existing run directory; resume uses its saved configuration')
    parser.add_argument('--instruction', type=Path, help='Text file for respond/reopen')
    parser.add_argument('--max-actions', type=int, default=0, help='0 continues to completion; positive value pauses after this many actions')
    args = parser.parse_args()
    if args.max_actions < 0:
        parser.error('--max-actions must be nonnegative')
    if args.action in {'doctor', 'prepare-native'} or (args.action == 'run' and args.run is None):
        if not args.config:
            parser.error('--config is required')
        config = load_config(args.config)
        run = Path(config['output_root'])
    else:
        if not args.run:
            parser.error('--run is required for an existing run')
        run = args.run.resolve()
        config = CityWorkflow(run).config
    if args.action == 'doctor':
        raise SystemExit(0 if doctor(config) else 1)
    if args.action == 'prepare-native':
        prepare_native(config)
        return
    if args.action == 'run' and not (run / 'state.json').is_file():
        if not doctor(config):
            raise SystemExit(1)
        workflow = initialize(config)
        config = workflow.config
    elif args.action == 'run':
        config = CityWorkflow(run).config
    if args.action == 'stop':
        import fcntl
        workflow = CityWorkflow(run)
        with (run / 'runner.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for identity in workflow.state['jobs']:
                workflow.tool('stop', {'job_id': identity, 'reason': 'User stopped this map run'})
            workflow.save()
        print('Stopped jobs owned by this run. Saved construction state is retained.')
        return
    action = 'resume' if args.action == 'run' else args.action
    argv = [config['python'], '-m', 'scripts.workflow.run', action, '--run', str(run)]
    if action == 'resume':
        argv += ['--max-actions', str(args.max_actions)]
    if action in {'respond', 'reopen'}:
        if not args.instruction:
            parser.error('--instruction is required')
        argv += ['--instruction', str(args.instruction.resolve())]
    env = {**os.environ, **config.get('environment', {})}
    completed = subprocess.run(argv, cwd=config['workspace'], env=env)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    status = read_json(run / 'state.json')['status']
    if status == 'complete':
        state = read_json(run / 'state.json')
        write_json(run / 'result.json', {'status': status, 'revision': state['revision'],
                   'map_name': config['map_name'], 'artifacts': state['artifacts']})
        print('Map delivery index: ' + str(run / 'result.json'))
    elif status == 'blocked':
        print('Construction needs input; inspect status and the last finding, then use respond.')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
