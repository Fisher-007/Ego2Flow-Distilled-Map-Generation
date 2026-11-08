"""Execute external model adapters through JSON requests and subprocesses."""
from pathlib import Path
import os
import shutil
import subprocess
import time

from map_generation.city_workflow import read_json, write_json


def absolute(value, base):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return os.path.abspath(base / path if not path.is_absolute() else path)


def configure(value, base):
    if not value:
        return None
    result = dict(value)
    result['cwd'] = absolute(value.get('cwd', '.'), base)
    if value.get('config'):
        result['config'] = absolute(value['config'], base)
    command = list(value.get('command', []))
    if command and '/' in command[0]:
        command[0] = absolute(command[0], base)
    result['command'] = command
    return result


def check_adapter(adapter):
    if not adapter or not adapter.get('name') or not adapter.get('command'):
        raise ValueError('Configure an installed adapter with name, command, cwd and config')
    command = adapter['command']
    if not isinstance(command, list) or not all(isinstance(v, str) and v for v in command):
        raise ValueError('adapter.command must be a nonempty argument array')
    if not shutil.which(command[0]):
        raise ValueError('Adapter executable is unavailable: ' + command[0])
    if not Path(adapter['cwd']).is_dir():
        raise ValueError('Adapter cwd does not exist: ' + adapter['cwd'])
    if len(command) > 1 and command[1].endswith('.py'):
        script = Path(absolute(command[1], Path(adapter['cwd'])))
        if not script.is_file():
            raise ValueError('Adapter script is missing: ' + str(script))
    if not adapter.get('config') or not Path(adapter['config']).is_file():
        raise ValueError('Adapter config file is missing')


def evidence_paths(items, base):
    if not isinstance(items, list) or not items:
        raise ValueError('The adapter must return actual evidence paths')
    paths = [absolute(item, base) for item in items]
    for path in paths:
        if not Path(path).is_file():
            raise ValueError('Missing evidence file: ' + path)
    return paths


def execute(adapter, request, directory, resources, environment=None):
    """Execute an argument array and record outputs in a new directory."""
    check_adapter(adapter)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    request = dict(request, adapter_config=adapter['config'], adapter_name=adapter['name'])
    request_path = directory / 'request.json'
    result_path = directory / 'adapter_result.json'
    write_json(request_path, request)
    argv = ['taskset', '-c', resources['cpus'], *adapter['command'],
            '--request', str(request_path), '--output', str(result_path)]
    env = {**os.environ, **(environment or {}),
           'CUDA_VISIBLE_DEVICES': str(resources['gpu']), 'MAP_CPUSET': resources['cpus'],
           'MAP_GPU': str(resources['gpu']), 'MAP_RPC_PORT': str(resources['port']),
           'OMP_NUM_THREADS': str(resources.get('threads', 2)), 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1'}
    receipt = dict(argv=argv, cwd=adapter['cwd'], resources=resources, status='running')
    write_json(directory / 'execution.json', receipt)
    started = time.monotonic()
    try:
        with (directory / 'output.log').open('w') as stream:
            completed = subprocess.run(argv, cwd=adapter['cwd'], env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
        receipt.update(exit_code=completed.returncode,
                       status='completed' if completed.returncode == 0 else 'failed')
        if completed.returncode:
            raise RuntimeError(f'Adapter exited {completed.returncode}; inspect {directory / "output.log"}')
        result = read_json(result_path)
        if result.get('execution_status') != 'completed':
            raise ValueError('Adapter did not complete execution; inspect adapter_result.json')
        result['evidence'] = evidence_paths(result.get('evidence'), directory)
        return result
    except BaseException as error:
        receipt.update(status='failed', error=str(error) or type(error).__name__)
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        write_json(directory / 'execution.json', receipt)
