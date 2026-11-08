#!/usr/bin/env python3
"""Run one registered construction tool with the configured software and resources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_map import ROOT, load_config
from map_generation.city_workflow import read_json, write_json


def registry(workspace):
    return read_json(Path(workspace) / 'configs/workflows/city_tools.json')['tools']


def launch_spec(config, identity, arguments):
    workspace = Path(config['workspace'])
    tool = next((row for row in registry(workspace) if row['id'] == identity), None)
    if tool is None:
        raise ValueError('Unknown tool; use run_tool.py list: ' + identity)
    entry = tool['entrypoint']
    arguments = list(arguments)
    # Fill omitted software-path arguments from the build configuration.
    defaults = {
        'generate_corridor': {'--source-dir': config['scenario_dreamer_source']},
        'generate_access_group': {'--source-dir': config['scenario_dreamer_source']},
        'cosmos_transfer': {'--model': config['cosmos_model'], '--guardrail-root': config['cosmos_guardrail_root']},
        'build_coarse': {'--netconvert': config['netconvert'], '--sumo-home': config['sumo_home'],
                         '--data-root': config['data_root']},
    }
    for flag, value in defaults.get(identity, {}).items():
        if not any(arg == flag or arg.startswith(flag + '=') for arg in arguments):
            arguments.extend([flag, value])
    if entry['kind'] == 'module':
        python_key = {'map': 'map_python', 'scenario_dreamer': 'scenario_dreamer_python',
                      'cosmos': 'cosmos_python'}[entry['environment']]
        command = [config[python_key], '-m', entry['target'], *arguments]
    elif entry['kind'] == 'blender':
        command = [config['blender'], '--background', '--threads',
                   str(config['resources']['map'].get('threads', 2)), '--python-exit-code', '1',
                   '--python', str(workspace / entry['target']), '--', *arguments]
    elif entry['kind'] == 'shell':
        command = ['bash', str(workspace / entry['target']), *arguments]
    else:
        raise ValueError('Unsupported tool entrypoint: ' + entry['kind'])
    resource = config['resources']['map']
    environment = {**config.get('environment', {}), 'PYTHONPATH': str(workspace),
                   'MAP_CPUSET': resource['cpus'], 'MAP_GPU': str(resource['gpu']),
                   'CUDA_VISIBLE_DEVICES': str(resource['gpu']), 'MAP_RPC_PORT': str(resource['port']),
                   'OMP_NUM_THREADS': str(resource.get('threads', 2)), 'OPENBLAS_NUM_THREADS': '1'}
    return dict(tool=identity, argv=['taskset', '-c', resource['cpus'], *command],
                cwd=str(workspace), environment=environment, resources=resource)


def main():
    argv = sys.argv[1:]
    separator = argv.index('--') if '--' in argv else len(argv)
    forwarded = argv[separator + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'run'])
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', type=Path, help='New configuration, before a workflow is initialized')
    source.add_argument('--run', type=Path, help='Existing workflow; use its private workspace and saved paths')
    parser.add_argument('--tool', help='Registered tool ID; pass its arguments after --')
    parser.add_argument('--stage', help='Filter list by workflow stage')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without running it or checking installed models')
    parser.add_argument('--record-dir', type=Path, help='New directory for command, log and process result')
    args = parser.parse_args(argv[:separator])
    if args.action == 'list':
        workspace = read_json(args.run / 'state.json')['config']['workspace'] if args.run else ROOT
        rows = registry(workspace)
        print(json.dumps([row for row in rows if not args.stage or args.stage in row['stages']], indent=2))
        return
    if not args.tool or not (args.config or args.run):
        parser.error('run requires --tool and either --config or --run')
    config = read_json(args.run / 'state.json')['config'] if args.run else load_config(args.config)
    spec = launch_spec(config, args.tool, forwarded)
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return
    environment = {**os.environ, **spec['environment']}
    if args.record_dir:
        args.record_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.record_dir / 'command.json', spec)
    result = {'status': 'running'}
    started = time.monotonic()
    try:
        if args.record_dir:
            with (args.record_dir / 'output.log').open('w') as log:
                completed = subprocess.run(spec['argv'], cwd=spec['cwd'], env=environment,
                                           stdout=log, stderr=subprocess.STDOUT)
        else:
            completed = subprocess.run(spec['argv'], cwd=spec['cwd'], env=environment)
        result.update(exit_code=completed.returncode,
                      status='completed' if completed.returncode == 0 else 'failed')
    except BaseException as error:
        result.update(status='failed', error=str(error) or type(error).__name__)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        if args.record_dir:
            write_json(args.record_dir / 'result.json', result)
            print('Process record: ' + str(args.record_dir.resolve()), flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
