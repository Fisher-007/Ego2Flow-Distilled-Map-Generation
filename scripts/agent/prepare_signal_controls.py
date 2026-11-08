"""Compile source-backed controller groups after visual coverage feedback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from map_generation.information_graph import add_node, add_edge, compile_conditions
from map_generation.source_signal_controls import associate_heads, controller_nodes, apply_controls, external_phase_review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'metadata', 'network', 'graph', 'review', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--netconvert', type=Path, required=True)
    p.add_argument('--reuse-control-network', type=Path)
    a = p.parse_args(); read = lambda path: json.loads(path.read_text())
    scene, metadata, graph, review = map(read, (a.scene, a.metadata, a.graph, a.review))
    a.output.mkdir(parents=True, exist_ok=True)
    association = associate_heads(ET.parse(a.network).getroot(), metadata['physical_junctions'],
        review['max_source_path_m'], review['ambiguity_margin_m'])
    (a.output/'association.json').write_text(json.dumps(association, indent=2)+'\n')
    if association['overlapping_assignments']:
        raise ValueError('Ambiguous source controller assignments; inspect association.json')
    task, key = 'task:source_control_coverage', 'construction:source_joint_controls'
    add_node(graph, dict(id=task, kind='simulation_task'))
    add_node(graph, dict(id='evidence:source_control_loss', kind='source_and_native_evidence', content=review['evidence']))
    add_node(graph, dict(id=key, kind='native_control_condition',
        native_construction=dict(kind='source_joint_controls', settings=review, groups=association['groups']),
        task_status={task: dict(missing=['native_approach_coverage', 'native_head_visibility', 'red_green_driving'])}))
    add_edge(graph, task, 'depends_on', key); add_edge(graph, key, 'supported_by', 'evidence:source_control_loss')
    graph['decisions'].append(dict(source_id=key, task=task, choice='enhance', agent=review['agent'],
        reason=review['reason'], evidence_ids=['evidence:source_control_loss'],
        required_properties=['source_head_coverage', 'directed_junction_association', 'coordinated_phase_clock',
                             'retained_road_geometry_and_rules', 'native_approach_binding', 'native_vehicle_response']))
    conditions = compile_conditions(graph, scene, task)
    node_file = a.output/'controllers.nod.xml'; root = controller_nodes(association['groups'], review['phase_layout'])
    ET.indent(root); ET.ElementTree(root).write(node_file, encoding='utf-8', xml_declaration=True)
    network = a.reuse_control_network or a.output/'control.net.xml'
    command = [str(a.netconvert), '--sumo-net-file', str(a.network), '--node-files', str(node_file),
        '--tls.layout', review['phase_layout'], '--tls.allred.time', str(review['all_red_s']), '--output-file', str(network)]
    if not a.reuse_control_network:
        env = dict(os.environ)
        for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PROJ_LIB', 'PROJ_DATA', 'GDAL_DATA'):
            env.pop(name, None)
        with (a.output/'netconvert.log').open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    control = ET.parse(network).getroot()
    phase_review = external_phase_review(control, association['groups'], review.get('reviewed_coactive_entries'))
    (a.output/'phase_review.json').write_text(json.dumps(phase_review, indent=2)+'\n')
    if not phase_review['passed']:
        raise ValueError('Generated controller entry phases need review; see phase_review.json')
    scene['coarse_map'], coverage = apply_controls(scene['coarse_map'], control, association['groups'])
    coverage.update(network=str(network), reused_control_network=bool(a.reuse_control_network), command=command,
        original_network=str(a.network), graph_condition=key, native_validation='pending')
    for name, value in [('scene.json', scene), ('coverage.json', coverage), ('graph.json', graph), ('conditions.json', conditions)]:
        (a.output/name).write_text(json.dumps(value, separators=(',', ':'))+'\n')
    print(dict(association=association['summary'], phase_separation=phase_review['passed'],
        programs=coverage['program_count'], unmatched_controls=len(coverage['unmatched_controls'])), flush=True)


if __name__ == '__main__':
    main()
