"""Associate retained source signal heads with physical junctions and SUMO programs."""
from collections import defaultdict
import heapq
import math
import xml.etree.ElementTree as ET


def associate_heads(root, physical, max_path_m=80., ambiguity_margin_m=5.):
    nodes = {n.get('id'): n for n in root.findall('junction') if n.get('type') != 'internal'}
    adjacent = defaultdict(list)
    for edge in root.findall('edge'):
        a, b = edge.get('from'), edge.get('to')
        if a in nodes and b in nodes:
            length = math.hypot(float(nodes[a].get('x'))-float(nodes[b].get('x')),
                                float(nodes[a].get('y'))-float(nodes[b].get('y')))
            adjacent[a].append((b, length))
    complexes = {str(j['id']): j for j in physical
                 if len(j['movement_road_ids']) >= 3 and len(j['incoming_lane_ids']) >= 2}
    membership = defaultdict(set)
    for key, junction in complexes.items():
        for node in junction['source_sumo_junction_ids']:
            membership[node].add(key)
    groups, records = {}, []
    for key, node in nodes.items():
        if node.get('type') != 'traffic_light':
            continue
        queue, visited, found = [(0., key, [key])], set(), []
        while queue:
            distance, current, path = heapq.heappop(queue)
            if current in visited or distance > max_path_m:
                continue
            visited.add(current)
            if current in membership:
                found.extend((distance, group, path) for group in membership[current])
                continue
            for target, length in adjacent[current]:
                if target not in visited:
                    heapq.heappush(queue, (distance+length, target, path+[target]))
        found.sort()
        row = dict(source_signal_node=key, candidates=[dict(distance_m=d, physical_junction=g, path=path)
                                                     for d, g, path in found])
        if found:
            distance, group, path = found[0]
            if len({g for d, g, _ in found if d <= distance+ambiguity_margin_m}) == 1:
                row.update(selected_junction=group, distance_m=distance, path=path)
                entry = groups.setdefault(group, dict(head_nodes=[], source_nodes=set(), associations=[]))
                entry['head_nodes'].append(key); entry['source_nodes'].update(path); entry['associations'].append(row)
        records.append(row)
    owners = defaultdict(list)
    for key, group in groups.items():
        group['source_nodes'].update(n for n in complexes[key]['source_sumo_junction_ids'] if n in nodes)
        group['source_nodes'] = sorted(group['source_nodes']); group['head_nodes'].sort()
        for node in group['source_nodes']:
            owners[node].append(key)
    overlap = {key: value for key, value in owners.items() if len(value)>1}
    return dict(groups=groups, source_signals=records, overlapping_assignments=overlap,
        summary=dict(heads=len(records), associated=sum('selected_junction' in r for r in records), groups=len(groups)))


def controller_nodes(groups, layout='incoming'):
    root = ET.Element('nodes')
    for key, group in groups.items():
        for node in group['source_nodes']:
            ET.SubElement(root, 'node', id=node, type='traffic_light', tl='MAP_'+key,
                          tlType='static', tlLayout=layout)
    return root


def apply_controls(coarse, root, groups):
    """Read signal attributes only; keep every retained lane and connection."""
    edges = {e.get('id'): e for e in root.findall('edge')}
    programs = [{"id": p.get('id'), "program_id": p.get('programID'), "type": p.get('type'),
        "phases": [dict(state=q.get('state'), duration_s=float(q.get('duration'))) for q in p.findall('phase')],
        "source": "SUMO-generated source-junction incoming-edge phase plan; timing is inferred"}
        for p in root.findall('tlLogic')]
    controls = {}
    for element in root.findall('connection'):
        c = element.attrib
        if 'tl' not in c:
            continue
        source, target = c['from']+'_'+c['fromLane'], c['to']+'_'+c['toLane']
        value = dict(signal_id=c['tl'], signal_link_index=int(c['linkIndex']))
        if c['tl'].startswith('MAP_') and c['tl'][4:] in groups:
            key = c['tl'][4:]; nodes = set(groups[key]['source_nodes'])
            edge = edges[c['from']]
            value.update(physical_junction_id=int(key),
                         control_entry=edge.get('from') not in nodes and edge.get('to') in nodes)
        controls[(source, target)] = value
    connections, matched = [], set()
    for old in coarse['connections']:
        key = (old['from_lane'], old['to_lane'])
        row = dict(old, signal_id=None, signal_link_index=None)
        if key in controls:
            row.update(controls[key]); matched.add(key)
        connections.append(row)
    entries = [dict(from_lane=s, to_lane=t, **c) for (s,t),c in controls.items() if c.get('control_entry')]
    report = dict(controlled_source_connections=len(controls), matched_connections=len(matched),
        unmatched_controls=[dict(from_lane=s,to_lane=t,**controls[(s,t)]) for s,t in sorted(controls.keys()-matched)],
        external_entries=entries, retained_lane_geometry=True, retained_connection_sequence=True,
        source_groups=groups, program_count=len(programs))
    return dict(coarse, connections=connections, signal_programs=programs, control_groups=groups), report


def external_phase_review(root, groups, reviewed_coactive_entries=None):
    """Review source entry service and coordinated releases at each junction.

    OSM may split one incoming direction across several carriageway edges.
    Explicit source review can retain these SUMO co-releases; native merging
    and yielding behaviour remains a driving check.
    """
    reviewed_coactive_entries = reviewed_coactive_entries or {}
    edges = {e.get('id'): e for e in root.findall('edge')}
    indices = defaultdict(lambda: defaultdict(set))
    for c in root.findall('connection'):
        tls = c.get('tl', '')
        if tls.startswith('MAP_') and tls[4:] in groups:
            group = set(groups[tls[4:]]['source_nodes']); edge = edges[c.get('from')]
            if edge.get('from') not in group and edge.get('to') in group:
                indices[tls][edge.get('id')].add(int(c.get('linkIndex')))
    records = []
    for program in root.findall('tlLogic'):
        key = program.get('id')
        if key not in indices:
            continue
        reviewed = [set(r['edges']) for r in reviewed_coactive_entries.get(key, [])]
        phases = []
        served = set()
        for phase in program.findall('phase'):
            active = [edge for edge, ids in indices[key].items() if any(phase.get('state')[i] in 'Gg' for i in ids)]
            served.update(active); phases.append(dict(duration_s=float(phase.get('duration')), active_edges=active,
                entry_release_reviewed=len(active)<=1 or any(set(active)<=group for group in reviewed)))
        records.append(dict(controller=key, approaches=len(indices[key]), phases=phases,
            unserved_edges=sorted(indices[key].keys()-served),
            reviewed_coactive_entries=reviewed_coactive_entries.get(key, []),
            passed=all(p['entry_release_reviewed'] for p in phases) and len(served)==len(indices[key])))
    return dict(passed=bool(records) and all(r['passed'] for r in records), controllers=records,
        scope='Source external entry phase separation. Native association and physical clearance/driving are checked separately.')
