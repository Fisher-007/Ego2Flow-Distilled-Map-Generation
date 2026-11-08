"""Connect area routes through passenger-permitted source successors.

A compact traversal tree is selected from the recorded ordered-area routes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.scene.check_city_driving_plans import source_routes


def prepare(scene, area_routes, output):
    sources = [source for _, source in source_routes(area_routes)]
    if len(sources) < 2:
        raise ValueError('Inter-Area checks need at least two prepared source routes')
    import networkx as nx
    import numpy as np

    lanes = {r['id']: r for r in scene['fusion_map']['lanes']
             if 'passenger' in r['allowed_classes']}
    lengths = {key: float(np.linalg.norm(np.diff(np.asarray(row['centerline'])[:, :2], axis=0), axis=1).sum())
               for key, row in lanes.items()}
    graph = nx.DiGraph()
    graph.add_nodes_from(lanes)
    graph.add_edges_from((key, target, dict(weight=lengths[target]))
                         for key, row in lanes.items() for target in row['successors'] if target in lanes)
    pairs, paths = [], {}
    for source in sources:
        start = source['route']['lane_ids'][-1]
        distances, reachable = nx.single_source_dijkstra(graph, start, weight='weight')
        for target in sources:
            if target['area_id'] == source['area_id']:
                continue
            goal = target['route']['lane_ids'][0]
            row = dict(source=source['area_id'], target=target['area_id'],
                       start_lane_id=start, goal_lane_id=goal, reachable=goal in reachable)
            if row['reachable']:
                chain = reachable[goal]
                row['length_m'] = lengths[start] + distances[goal]
                by_source = Counter()
                for key in chain:
                    by_source[lanes[key]['source']] += lengths[key]
                row['length_by_source_m'] = dict(by_source)
                paths[(row['source'], row['target'])] = chain
            pairs.append(row)
    neighbors = nx.Graph()
    neighbors.add_nodes_from(row['area_id'] for row in sources)
    for row in sorted((r for r in pairs if r['reachable']), key=lambda r: r['length_m']):
        if not neighbors.has_edge(row['source'], row['target']):
            neighbors.add_edge(row['source'], row['target'], weight=row['length_m'], directed_pair=row)
    tree = nx.minimum_spanning_tree(neighbors)
    selected = sorted((data['directed_pair'] for _, _, data in tree.edges(data=True)),
                      key=lambda r: (r['source'], r['target']))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, pair in enumerate(selected, 1):
        chain = paths[(pair['source'], pair['target'])]
        name = '%s_transfer_%02d' % (scene['coordinates']['city'].lower(), index)
        points = []
        for key in chain:
            for point in lanes[key]['centerline']:
                if not points or point != points[-1]:
                    points.append(point)
        route = dict(area_id=name, route_kind='inter_area_source_successor_check',
                     source_area=pair['source'], target_area=pair['target'],
                     source_scene=scene['map_id'], length_m=pair['length_m'],
                     length_by_source_m=pair['length_by_source_m'],
                     route=dict(lane_ids=chain, positions=points, landmarks=[]),
                     status='source_selected_native_driving_pending')
        (output / (name + '.json')).write_text(json.dumps(route, separators=(',', ':')) + '\n')
        records.append({key: value for key, value in route.items() if key != 'route'})
    report = dict(ordered_area_pairs=pairs, selected_routes=records,
                  unreachable_pairs=[row for row in pairs if not row['reachable']],
                  selected_tree_components=nx.number_connected_components(tree),
                  source_roads_and_formal_areas_changed=False, scope=__doc__)
    (output / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(dict(ordered_pairs=len(pairs), unreachable_pairs=len(report['unreachable_pairs']),
               selected_routes=len(records), selected_tree_components=report['selected_tree_components'],
               selected_length_m=sum(row['length_m'] for row in records)), flush=True)
    if not records:
        raise ValueError('No inter-Area route could be prepared; see review.json unreachable pairs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'area-routes', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    prepare(json.loads(a.scene.read_text()), a.area_routes, a.output)


if __name__ == '__main__':
    main()
