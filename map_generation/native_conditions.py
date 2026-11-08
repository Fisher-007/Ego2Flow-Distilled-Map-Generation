"""Use reviewed graph decisions as native construction inputs."""
from copy import deepcopy


def selected_features(measurements, conditions):
    result = deepcopy(measurements)
    selected = {row['source_id']: row for row in conditions['native_objects']
                if row['choice'] in ('retain', 'enhance') and row.get('native_construction')}
    names = ('sign', 'streetside', 'support_region')
    if 'accesses' not in measurements:
        names += ('access',)
    for name in names:
        key = measurements[name]['id']
        if key not in selected:
            raise ValueError(f'No graph construction decision for {key}')
        result[name] = deepcopy(selected[key]['native_construction'])
    result['construction_decisions'] = [selected[result[name]['id']] for name in names]
    if 'accesses' in measurements:
        result['accesses'] = []
        for access in measurements['accesses']:
            decision = selected.get(access['id'])
            if decision is None:
                raise ValueError(f"No graph construction decision for {access['id']}")
            result['accesses'].append(deepcopy(decision['native_construction']))
            result['construction_decisions'].append(decision)
    return result


def selected_appearance(conditions):
    return {row['source_id'].removeprefix('object:'): deepcopy(row['appearance'])
            for row in conditions['native_objects']
            if row['choice'] in ('retain', 'enhance') and row.get('appearance')}


def observed_accesses(features):
    """Read single-access and multiple-access selections."""
    return features["accesses"] if "accesses" in features else [features["access"]]
