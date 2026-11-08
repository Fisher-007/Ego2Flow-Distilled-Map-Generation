"""Prepare paired inputs, run a planning model, and score its predictions."""
from pathlib import Path

from map_generation.city_workflow import read_json, write_json
from map_testing.adapters import absolute, check_adapter, configure, execute
from map_testing.scoring import case_index, score, validate_axis


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    for key in ('output', 'cases'):
        config[key] = absolute(config[key], path.parent)
    for key in ('prepare', 'predictor'):
        config[key] = configure(config.get(key), path.parent)
    return config


def check(config, state):
    if state.get('status') != 'complete':
        raise ValueError('Downstream tests require a completed map construction run')
    for key in ('prepare', 'predictor'):
        check_adapter(config.get(key))
    if not Path(config['cases']).is_file():
        raise ValueError('Supply cases: source logs, anchors and desired camera/time coverage')
    resource = config['resources']
    if not resource.get('cpus') or 'gpu' not in resource or 'port' not in resource:
        raise ValueError('Specify testing CPU/GPU allocation and a free simulator port')


def paired_inputs(real_path, sim_path, reference_path):
    paths = [Path(real_path), Path(sim_path), Path(reference_path)]
    real, sim, reference = [read_json(path) for path in paths]
    times = validate_axis(real)
    if validate_axis(sim) != times or validate_axis(reference) != times:
        raise ValueError('Paired input/reference horizons differ')
    indexes = [case_index(value) for value in (real, sim, reference)]
    if not indexes[0] or not (indexes[0].keys() == indexes[1].keys() == indexes[2].keys()):
        raise ValueError('Real/sim/reference must describe the same nonempty set of cases')
    for identity, left in indexes[0].items():
        right = indexes[1][identity]
        if 'trajectory_xy_m' in left or 'trajectory_xy_m' in right:
            raise ValueError('Keep future references outside prediction inputs')
        if {k: v for k, v in left.items() if k != 'views'} != {k: v for k, v in right.items() if k != 'views'}:
            raise ValueError('Pairing changed non-image context: ' + identity)
        if not left.get('views') or len(left['views']) != len(right.get('views', [])):
            raise ValueError('Pairing needs matching camera/time views')
        for a, b in zip(left['views'], right['views']):
            if not all(key in a for key in ('camera', 'timestamp_ns', 'intrinsics', 'camera_to_ego')):
                raise ValueError('Record source camera identity, exposure and calibration')
            if {k: v for k, v in a.items() if k != 'image'} != {k: v for k, v in b.items() if k != 'image'}:
                raise ValueError('Pairing changed camera calibration, order or source exposure time')
            if a['timestamp_ns'] > left['timestamp_ns']:
                raise ValueError('A prediction image is later than its causal anchor')
            for view, base in ((a, paths[0].parent), (b, paths[1].parent)):
                view['image'] = absolute(view['image'], base)
                if not Path(view['image']).is_file():
                    raise ValueError('Missing prediction image: ' + view['image'])
    return real, sim, reference


def run(config, run_directory):
    run_directory = Path(run_directory).resolve()
    state = read_json(run_directory / 'state.json')
    check(config, state)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(map_name=state['config']['map_name'], revision=state['revision'])
    common = dict(identity, artifacts=state['artifacts'], resources=config['resources'],
                  carla_root=state['config']['carla_root'])
    environment = {**state['config'].get('environment', {}),
                   'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
                   'CARLA_ROOT': state['config']['carla_root'],
                   'MAP_RUNTIME_DIR': str(output / 'runtime'), **config.get('environment', {})}
    progress = dict(identity, execution_status='running', scope='post_construction_paired', stage='prepare')
    write_json(output / 'config.json', config)
    write_json(output / 'result.json', progress)
    try:
        prepared = execute(config['prepare'], dict(common, kind='prepare_pair', cases=config['cases']),
                           output / 'prepare', config['resources'], environment)
        if any(prepared.get(key) != value for key, value in identity.items()):
            raise ValueError('Pair preparation returned a different map or revision')
        paths = [absolute(prepared[key], output / 'prepare') for key in ('real_inputs', 'sim_inputs', 'reference')]
        real, sim, reference = paired_inputs(*paths)
        write_json(output / 'reference.json', reference)
        predictions = []
        for side, bundle in (('real', real), ('sim', sim)):
            progress['stage'] = 'predict_' + side
            write_json(output / 'result.json', progress)
            inputs = output / (side + '_inputs.json')
            write_json(inputs, bundle)
            # Pass causal inputs to prediction; reference trajectories remain in scoring.
            request = dict(identity, kind='predict', side=side, inputs=str(inputs), resources=config['resources'])
            prediction = execute(config['predictor'], request, output / side,
                                 config['resources'], environment)
            if prediction.get('side') != side or any(prediction.get(k) != v for k, v in identity.items()):
                raise ValueError('Prediction must identify its input side, map and revision')
            predictions.append(prediction)
        progress['stage'] = 'score'
        write_json(output / 'result.json', progress)
        scores = score(*predictions, reference)
        write_json(output / 'scores.json', scores)
        progress.update(execution_status='completed', stage='finished', scores=str(output / 'scores.json'),
                        preparation_evidence=prepared['evidence'],
                        limitations=prepared.get('limitations', []))
    except BaseException as error:
        progress.update(execution_status='failed', error=str(error) or type(error).__name__)
        raise
    finally:
        write_json(output / 'result.json', progress)
    return output / 'result.json'
