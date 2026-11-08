"""Apply reviewed control-to-source relations to the existing evidence graph.

Selections identify source approaches. Native realization resolves those source
IDs through the current compiler mapping and verifies actual stop behavior.
"""
import argparse
import json
from pathlib import Path


def bind(graph, evidence, selection):
    nodes = {n['id']: n for n in graph['nodes']}
    observations = {r['source_id']: r for r in evidence['controls']}
    task = 'task:city_controls'
    groups = {}
    reports = []
    for decision in selection['decisions']:
        key = decision['source_id']
        observation = observations[key]
        candidates = {r['source_lane_id']: r for r in observation['candidate_approaches']}
        chosen = decision['controlled_source_lane_ids']
        for lane in chosen:
            if lane not in candidates:
                raise ValueError('Selected source approach is absent from the reviewed candidate set: ' + lane)
        node = nodes[key]
        node['control_approach_review'] = decision
        node['task_status'][task] = dict(complete=False, missing=list(decision['pending']))
        for lane in chosen:
            relation = dict(source=key, relation='controls_source_approach', target=lane,
                            rule=decision['rule'], evidence=selection['evidence'])
            if relation not in graph['edges']:
                graph['edges'].append(relation)
            groups.setdefault(lane, []).append(key)
        graph['decisions'] = [d for d in graph['decisions'] if (d['source_id'], d['task']) != (key, task)]
        graph['decisions'].append(dict(source_id=key, task=task,
            choice='retain' if chosen else 'seek_evidence', agent=selection['agent'],
            reason=decision['reason'], evidence_ids=['evidence:' + key.split(':', 1)[1]],
            required_properties=['source_sign_face', 'controlled_source_approach', 'native_stop_response']))
        reports.append(dict(source_id=key, controlled_source_lane_ids=chosen,
            intersection_scope=decision['intersection_scope'], pending=decision['pending']))
    return graph, dict(observations=len(reports), bound_observations=sum(bool(r['controlled_source_lane_ids']) for r in reports),
        distinct_source_approaches=len(groups), approach_signs=groups, controls=reports,
        scope='Reviewed source control relations; duplicate sign heads share an approach. Native stop realization and any all-way expansion remain separate checks.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['graph', 'evidence', 'selection', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    graph, report = bind(json.loads(a.graph.read_text()), json.loads(a.evidence.read_text()), json.loads(a.selection.read_text()))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'graph.json').write_text(json.dumps(graph, separators=(',', ':')) + '\n')
    (a.output / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print({k: v for k, v in report.items() if k not in ['controls', 'approach_signs']}, flush=True)


if __name__ == '__main__':
    main()
