"""Build a window-wide review envelope and source-connected access candidates."""
import numpy as np


def coverage(history, config):
    normal = np.asarray(config['normal_xy'], dtype=float)
    tangent = np.array([-normal[1], normal[0]])
    positions = np.array([r['ego']['position_local_xyz_m'][:2] for r in history['frames']])
    s = positions @ tangent
    margin = config['coverage_margin_m']
    return {'tangent_range_m': [float(s.min()-margin), float(s.max()+margin)],
            'trajectory_range_m': [float(s.min()), float(s.max())],
            'source_frame_count': len(positions), 'margin_m': margin,
            'required_review_indices': sorted(set([0, len(positions)-1,
                *np.arange(0, len(positions), 40).tolist()]))}


def candidates(scene, history, config, accesses):
    from shapely.geometry import LineString, Point
    normal = np.asarray(config['normal_xy']);tangent = np.array([-normal[1], normal[0]])
    low, high = config['tangent_range_m']
    confirmed = {a['source_road']: a for a in accesses}
    main = [r for r in scene['roads'] if r['tags'].get('highway') not in
            ('service', 'footway', 'path', 'cycleway', 'steps')]
    positions = np.array([r['ego']['position_local_xyz_m'][:2] for r in history['frames']])
    result = []
    for road in scene['roads']:
        if road['tags'].get('highway') != 'service':
            continue
        line = np.array([v[:2] for v in road['reference_line']])
        for parent in main:
            shared = set(road['node_ids']) & set(parent['node_ids'])
            for node in shared:
                mouth = line[road['node_ids'].index(node)]
                if not low <= mouth @ tangent <= high:
                    continue
                parent_line = LineString([v[:2] for v in parent['reference_line']])
                if min(parent_line.distance(Point(p)) for p in positions) > 12:
                    continue
                match = confirmed.get(road['id'])
                result.append({'source_road': road['id'], 'parent_road': parent['id'],
                    'shared_node': node, 'mouth_local_xy_m': mouth.tolist(),
                    'review_status': 'image_confirmed' if match else 'needs_image_review',
                    'observed_feature': match['id'] if match else None,
                    'nearest_history_index': int(np.argmin(np.linalg.norm(positions-mouth,axis=1))),
                    'scope': 'OSM connection candidate; visibility, curb drop and vehicle permission need source review.'})
    return result
