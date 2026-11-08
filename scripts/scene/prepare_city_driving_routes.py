"""Select source-supported junction routes using geometry and observed ego heading."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import shapely
from av2.utils.io import read_city_SE3_ego

from map_generation.coordinates import source_log_registration, register_local_pose


def prepare(scene, data_root, output):
    lanes = {r['id']: r for r in scene['fusion_map']['lanes'] if 'passenger' in r['allowed_classes']}
    lines = {k: shapely.LineString(np.asarray(r['centerline'])[:, :2]) for k, r in lanes.items()}
    graph = nx.DiGraph()
    graph.add_nodes_from(lanes)
    graph.add_edges_from((k, t, dict(weight=lines[t].length)) for k, r in lanes.items()
                         for t in r['successors'] if t in lanes)
    reverse = graph.reverse(copy=True)
    for a, b in reverse.edges:
        reverse[a][b]['weight'] = lines[b].length
    ids = list(lanes); tree = shapely.STRtree([lines[k] for k in ids])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for area in scene['areas']:
        observation = min(area['source_observations'], key=lambda r: r['ego_distance_m'])
        log = data_root / 'av2/sensor' / observation['split'] / observation['log_id']
        poses = read_city_SE3_ego(log)
        stamp = min(poses, key=lambda t: abs(t - observation['nearest_timestamp_ns']))
        pose = poses[stamp]
        position = pose.translation - scene['coordinates']['origin_city_xyz']
        rotation = pose.rotation
        registration = source_log_registration(scene, log.name)
        if registration:
            position, rotation = register_local_pose(position, rotation, registration)
        heading = rotation[:2, 0]; heading /= np.linalg.norm(heading)
        center = np.array(area['center_city_xy']) - scene['coordinates']['origin_city_xyz'][:2]
        region = shapely.box(*area['bbox_local_xy'])
        candidates = [ids[i] for i in tree.query(region, predicate='intersects')
                      if lanes[ids[i]]['source'] == 'av2' and lanes[ids[i]].get('is_intersection')]
        choices = []
        for key in candidates:
            before = nx.single_source_dijkstra_path(reverse, key, cutoff=180., weight='weight')
            after = nx.single_source_dijkstra_path(graph, key, cutoff=180., weight='weight')
            entries = [(k, list(reversed(p))) for k, p in before.items()
                       if 45. < np.linalg.norm(np.array(lines[k].coords[0]) - center) < 145.
                       and not lanes[k].get('internal') and not lanes[k].get('is_intersection')]
            exits = [(k, p) for k, p in after.items()
                     if 45. < np.linalg.norm(np.array(lines[k].coords[-1]) - center) < 145.
                     and not lanes[k].get('internal') and not lanes[k].get('is_intersection')]
            for start, entry in entries:
                tangent = np.array(lines[start].interpolate(min(5., lines[start].length)).coords[0]) - np.array(lines[start].coords[0])
                tangent /= np.linalg.norm(tangent)
                direction_error = 1. - float(tangent @ heading)
                for end, departure in exits:
                    path = entry + departure[1:]
                    if len(set(path)) != len(path):
                        continue
                    length = sum(lines[k].length for k in path)
                    if not 90. < length < 340.:
                        continue
                    entry_xy, exit_xy = np.array(lines[start].coords[0]), np.array(lines[end].coords[-1])
                    if np.linalg.norm(exit_xy - entry_xy) < 60.:
                        continue
                    # Prefer the observed incoming direction and a compact
                    # traversal. All route choices precede native testing.
                    score = 100 * direction_error + .1 * length + .2 * lines[start].distance(shapely.Point(position[:2]))
                    choices.append((score, key, path, length, direction_error))
        if not choices:
            records.append(dict(area_id=area['id'], status='no_source_traversal_selected'))
            continue
        score, key, path, length, heading_error = min(choices)
        route = dict(lane_ids=path, positions=[p for k in path for p in lanes[k]['centerline']], landmarks=[])
        row = dict(area_id=area['id'], source_scene=scene['map_id'], observation=observation,
            selected_junction_lane=key, source_heading_cosine=1 - heading_error,
            length_m=length, status='source_selected_native_driving_pending', route=route)
        (output / (area['id'] + '.json')).write_text(json.dumps(row, separators=(',', ':')) + '\n')
        records.append({k: v for k, v in row.items() if k != 'route'})
        print(area['id'], round(length, 1), round(1 - heading_error, 3), flush=True)
    (output / 'review.json').write_text(json.dumps(dict(routes=records,
        scope=__doc__, source_road_geometry_and_permissions_unchanged=True), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'data-root', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    prepare(json.loads(a.scene.read_text()), a.data_root, a.output)


if __name__ == '__main__':
    main()
