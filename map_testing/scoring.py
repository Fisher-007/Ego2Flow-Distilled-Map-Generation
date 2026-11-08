"""Compute paired trajectory errors on a common future time grid."""
import math

FRAME = 'anchor_ego_forward_left_m'


def case_index(bundle):
    rows = bundle['cases']
    indexed = {row['case_id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate case_id')
    return indexed


def validate_axis(bundle):
    times = bundle.get('future_times_s', [])
    if (bundle.get('coordinate_frame') != FRAME or not times
            or any(not math.isfinite(t) or t <= 0 for t in times)
            or any(a >= b for a, b in zip(times, times[1:]))):
        raise ValueError('Supply increasing positive future_times_s in anchor_ego_forward_left_m')
    return times


def trajectory(row, count):
    points = row.get('trajectory_xy_m') if row else None
    if (not isinstance(points, list) or len(points) != count
            or any(not isinstance(p, list) or len(p) != 2
                   or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p) for p in points)):
        return None
    return points


def errors(prediction, reference):
    distance = [math.hypot(p[0] - r[0], p[1] - r[1]) for p, r in zip(prediction, reference)]
    cumulative = []
    total = 0.0
    for count, value in enumerate(distance, 1):
        total += value
        cumulative.append(total / count)
    return dict(ade_m=cumulative[-1], fde_m=distance[-1], error_by_time_m=distance,
                ade_by_horizon_m=cumulative)


def score(real, simulated, reference):
    times = validate_axis(reference)
    for value in (real, simulated):
        if validate_axis(value) != times:
            raise ValueError('Real/simulation/reference time grids differ')
    if not real.get('model') or real['model'] != simulated.get('model'):
        raise ValueError('Both sides must use the same recorded model, checkpoint and inference settings')
    if not all(key in real['model'] for key in ('name', 'checkpoint', 'settings')):
        raise ValueError('Model metadata needs name, checkpoint and settings')
    truth, left, right = map(case_index, (reference, real, simulated))
    if not truth or set(left) - truth.keys() or set(right) - truth.keys():
        raise ValueError('Predictions have unknown cases, or references are empty')
    rows = []
    for identity, label in truth.items():
        target = trajectory(label, len(times))
        if target is None:
            raise ValueError('Invalid future reference for ' + identity)
        predictions, failures = {}, {}
        for side, index in (('real', left), ('sim', right)):
            item = index.get(identity)
            points = trajectory(item, len(times))
            if item and item.get('valid') is True and points is not None:
                predictions[side] = points
            else:
                failures[side] = ((item or {}).get('error') or 'Missing, invalid or non-finite trajectory')
        row = dict(case_id=identity, failures=failures, real_vs_reference=None,
                   sim_vs_reference=None, sim_vs_real=None)
        for side, points in predictions.items():
            row[side + '_vs_reference'] = errors(points, target)
        if len(predictions) == 2:
            row['sim_vs_real'] = errors(predictions['sim'], predictions['real'])
        rows.append(row)
    def aggregate(selected):
        result = {}
        for metric in ('real_vs_reference', 'sim_vs_reference', 'sim_vs_real'):
            values = [r[metric] for r in selected if r[metric] is not None]
            result[metric] = dict(valid_cases=len(values),
                ade_m=sum(v['ade_m'] for v in values) / len(values) if values else None,
                fde_m=sum(v['fde_m'] for v in values) / len(values) if values else None,
                by_horizon=[dict(time_s=t,
                    ade_m=sum(v['ade_by_horizon_m'][i] for v in values) / len(values) if values else None,
                    fde_m=sum(v['error_by_time_m'][i] for v in values) / len(values) if values else None)
                    for i, t in enumerate(times)])
        return result
    paired = [r for r in rows if r['sim_vs_real'] is not None]
    return dict(coordinate_frame=FRAME, future_times_s=times, model=real['model'],
                requested_cases=len(rows), paired_valid_cases=len(paired), cases=rows,
                all_valid_summary=aggregate(rows), paired_cases_summary=aggregate(paired),
                interpretation='Open-loop trajectory errors with valid-case coverage. '
                               'Metrics are null when no valid cases are available.')
