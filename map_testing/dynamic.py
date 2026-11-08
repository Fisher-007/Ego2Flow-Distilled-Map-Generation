"""Run a configured driving policy and register dynamic-review results."""
from pathlib import Path

from map_generation.city_workflow import read_json, write_json
from map_testing.adapters import absolute, check_adapter, evidence_paths, execute


def validate_result(result, map_name, revision, policy_name, base):
    if (result.get('kind'), result.get('execution_status')) != ('dynamic_policy', 'completed'):
        raise ValueError('A completed dynamic_policy execution is required')
    if result.get('map_name') != map_name or result.get('revision') != revision:
        raise ValueError('Dynamic policy evidence must refer to the current map and revision')
    policy = result.get('policy', {})
    if policy.get('name') != policy_name or not policy.get('checkpoint') or 'settings' not in policy:
        raise ValueError('Record the configured policy, actual checkpoint and inference settings')
    control = result.get('learned_control', {})
    if not control.get('request_ids') or control.get('control_steps', 0) <= 0:
        raise ValueError('The learned policy must actually control at least one vehicle')
    traffic = result.get('traffic', {})
    counts = ('requested', 'admitted', 'completed', 'unadmitted', 'in_system', 'terminated_unfinished')
    if any(type(traffic.get(key)) is not int or traffic[key] < 0 for key in counts):
        raise ValueError('Record nonnegative counts for the complete traffic demand ledger')
    if traffic['requested'] != traffic['admitted'] + traffic['unadmitted']:
        raise ValueError('Requested demand must include unadmitted demand')
    if traffic['admitted'] != traffic['completed'] + traffic['in_system'] + traffic['terminated_unfinished']:
        raise ValueError('Admitted demand must include unfinished and terminated vehicles')
    for key in ('observation_seconds', 'waiting_vehicle_seconds', 'in_system_vehicle_seconds'):
        if not isinstance(traffic.get(key), (int, float)) or not 0 <= traffic[key] < float('inf'):
            raise ValueError('Record finite traffic duration and cumulative waiting/service times')
    if traffic['observation_seconds'] <= 0 or traffic['admitted'] == 0:
        raise ValueError('Traffic observation must actually run with admitted vehicles')
    for key in ('model_failures', 'limitations'):
        if not isinstance(result.get(key), list):
            raise ValueError(f'Record {key} explicitly, including an empty list when applicable')
    evidence_paths(result.get('evidence'), base)


def run(run_directory, request_path, output):
    run_directory = Path(run_directory).resolve()
    state = read_json(run_directory / 'state.json')
    if state['stage'] != 'dynamic_review' or state['status'] != 'running':
        raise ValueError('Run the learned-policy check during an active dynamic_review')
    config = state['config']
    adapter = config.get('dynamic_policy')
    check_adapter(adapter)
    request_path = Path(request_path).resolve()
    request = read_json(request_path)
    name, revision = config['map_name'], state['revision']
    if request.get('map_name') != name or request.get('revision') != revision:
        raise ValueError('Request must identify the current map and revision')
    if not request.get('map_asset') or not request.get('scope') or request.get('observation_seconds', 0) <= 0:
        raise ValueError('Request needs map_asset, scope and positive observation_seconds')
    for key in ('runtime_manifest', 'scene', 'routes', 'scenario'):
        request[key] = absolute(request[key], request_path.parent)
        if not Path(request[key]).is_file():
            raise ValueError('Missing dynamic input: ' + key)
    current = state['artifacts']['runtime_manifest']['path']
    if Path(request['runtime_manifest']).resolve() != Path(current).resolve():
        raise ValueError('Use the current compiled runtime_manifest')
    request.update(kind='dynamic_policy', resources=config['resources']['map'],
                   carla_root=config['carla_root'])
    output = Path(output).resolve()
    result = execute(adapter, request, output, config['resources']['map'], config.get('environment'))
    validate_result(result, name, revision, adapter['name'], output)
    if result['traffic']['observation_seconds'] + 1e-6 < request['observation_seconds']:
        raise ValueError('The observation ended early; retain it as a partial run, not a completed check')
    result.update(request=str(output / 'request.json'), execution=str(output / 'execution.json'))
    write_json(output / 'result.json', result)
    return output / 'result.json'


def validate_workflow_artifact(path, config, revision):
    """Execution completion is necessary; the Reviewer still judges map quality."""
    adapter = config.get('dynamic_policy')
    if not adapter:
        raise ValueError('Configure dynamic_policy before completing dynamic review')
    path = Path(path)
    result = read_json(path)
    if result.get('kind') == 'dynamic_policy_reuse':
        if result.get('map_name') != config['map_name'] or result.get('revision') != revision:
            raise ValueError('Reuse report must identify the current map and revision')
        if not result.get('unchanged_scope'):
            raise ValueError('Explain why the change leaves the previous policy observation unaffected')
        evidence_paths(result.get('evidence'), path.parent)
        path = Path(absolute(result['previous_result'], path.parent))
        result = read_json(path)
        if result['revision'] >= revision:
            raise ValueError('Reuse must reference an earlier actual execution')
        revision = result['revision']
    validate_result(result, config['map_name'], revision, adapter['name'], path.parent)
    receipt = read_json(result['execution'])
    if receipt.get('status') != 'completed' or receipt.get('exit_code') != 0:
        raise ValueError('Dynamic policy execution has no successful adapter receipt')
