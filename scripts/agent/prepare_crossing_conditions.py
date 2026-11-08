"""Keep source pedestrian crossings through graph selection and native paint."""
import argparse
import json
from pathlib import Path

from map_generation.information_graph import add_node, add_edge, compile_conditions


def prepare(graph, scene, review):
    task = 'task:source_crossing_visibility'
    add_node(graph, dict(id=task, kind='simulation_task'))
    for asset in scene['assets']:
        if asset['kind'] != 'pedestrian_crossing' or asset['source'] != 'av2':
            continue
        key = asset['id']
        style = review.get('crossings', {}).get(key, review['display_default'])
        add_node(graph, dict(id=key, kind='source_crossing', content=asset,
            native_construction=dict(kind='pedestrian_crossing_paint', settings=style),
            task_status={task: dict(missing=['native_realization', 'paired_view_review'])}))
        add_edge(graph, task, 'depends_on', key)
        add_edge(graph, key, 'supported_by', 'evidence:source_crossing_geometry')
        graph['decisions'].append(dict(source_id=key, task=task, choice='retain',
            agent=review['agent'], reason=review['reason'],
            required_properties=['source_crossing_extent', 'surface_attached_paint',
                                 'pattern_provenance', 'native_visibility'],
            evidence_ids=['evidence:source_crossing_geometry']))
    add_node(graph, dict(id='evidence:source_crossing_geometry', kind='source_evidence',
        source=graph['source_scene'], field='assets[kind=pedestrian_crossing]',
        style_review=review))
    return compile_conditions(graph, scene, task)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'graph', 'review', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    scene, graph, review = [json.loads(path.read_text()) for path in (a.scene, a.graph, a.review)]
    conditions = prepare(graph, scene, review)
    a.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('graph.json', graph), ('conditions.json', conditions)]:
        (a.output/name).write_text(json.dumps(data, separators=(',', ':'))+'\n')
    print(dict(selected_crossings=len(conditions['native_objects']),
               native_realization_and_visual_review='pending'), flush=True)


if __name__ == '__main__':
    main()
