"""Traffic-information graph linking source facts, candidates, and agent decisions."""
from copy import deepcopy
import math

import networkx as nx


def add_node(data, node):
    existing = next((n for n in data['nodes'] if n['id'] == node['id']), None)
    if existing is None:
        data['nodes'].append(node)
    else:
        existing.update(node)


def add_edge(data, source, relation, target, **attributes):
    edge = next((e for e in data['edges'] if
                 (e['source'], e['relation'], e['target']) == (source, relation, target)), None)
    if edge is None:
        edge = dict(source=source, relation=relation, target=target)
        data['edges'].append(edge)
    edge.update(attributes)


def unit(a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length = math.hypot(dx, dy)
    return (dx/length, dy/length) if length else (0., 0.)


def enrich(data, scene, scene_path):
    """Add actual lane topology and directed OSM evidence, without assigning rules."""
    from .source_space_geometry import declared_shared_lane
    data = deepcopy(data)
    data['source_scene'] = str(scene_path)
    data['coordinates'] = scene['coordinates']
    # Bulk city enrichment retains the existing graph schema and decisions.
    node_index = {node['id']: node for node in data['nodes']}
    edge_index = {(edge['source'], edge['relation'], edge['target']): edge for edge in data['edges']}

    def put_node(node):
        if node['id'] in node_index:
            node_index[node['id']].update(node)
        else:
            node_index[node['id']] = node
            data['nodes'].append(node)

    def put_edge(source, relation, target, **attributes):
        key = (source, relation, target)
        if key not in edge_index:
            edge_index[key] = dict(source=source, relation=relation, target=target)
            data['edges'].append(edge_index[key])
        edge_index[key].update(attributes)

    lanes = {l['id']: l for l in scene.get('fusion_map', {}).get('lanes', scene['lanes'])}
    fine = {l['id']: l for l in scene['lanes']}
    for key, lane in lanes.items():
        # Source geometry stays in SceneIR; the graph stores an exact reference.
        inferred_osm = lane.get('source') == 'osm_sumo'
        put_node(dict(id=key, kind='source_lane', source=lane.get('source'),
                           scene_ref=key, measured=key in fine,
                           generation_eligible=lane.get('source') in ('boundary_interpolation', 'source_path_access_transition'),
                           immutable_fields=(['successors', 'source_tags', 'source_lane_id'] if inferred_osm else
                               ['successors', 'allowed_classes', 'lane_type', 'left_boundary', 'right_boundary']),
                           reviewed_import_fields=(['allowed_classes', 'width_m', 'centerline',
                               'left_boundary', 'right_boundary'] if inferred_osm else []),
                           source_use_review=lane.get('source_use_review', [])))
        for target in lane.get('successors', []):
            if target in lanes:
                put_edge(key, 'successor', target, status='source', evidence=[str(scene_path)])
    for road in scene['roads']:
        put_node(dict(id=road['id'], kind='source_road', source='OSM',
                           name=road['tags'].get('name'), scene_ref=road['id'],
                           declared_shared_physical_lane=declared_shared_lane(road['tags'])))
    source_roads={r['id'] for r in scene['roads']}
    for key,lane in lanes.items():
        if lane.get('source')!='osm_sumo' or lane.get('internal'):continue
        road_id='osm:way:'+lane.get('edge_id','').lstrip('-').split('#')[0]
        if road_id in source_roads:
            put_edge(key,'on_source_way',road_id,status='source',evidence=[str(scene_path)])
    roads_by_node = {}
    for road in scene['roads']:
        for node_id in road['node_ids']:
            roads_by_node.setdefault(node_id, []).append(road)
    for control in scene['controls']:
        put_node(dict(id=control['id'], kind='source_control', control_type=control['kind'],
                           **{k:v for k,v in control.items() if k not in ('id','kind')}))
        node_id = control['id'].rsplit(':', 1)[-1]
        for road in roads_by_node.get(node_id, []):
            put_edge(control['id'], 'on_source_way', road['id'], status='source',
                     evidence=[str(scene_path)], direction=control['tags'].get('direction'))
            direction = control['tags'].get('direction')
            if control['kind'] != 'stop' or direction not in ('forward', 'backward'):
                continue
            points = road['reference_line']; index = road['node_ids'].index(node_id)
            # Use OSM way order and direction tags to resolve ambiguous source orientations.
            tangent = unit(points[max(0,index-1)], points[min(len(points)-1,index+1)])
            if direction == 'backward':
                tangent = tuple(-v for v in tangent)
            for key, lane in fine.items():
                if lane.get('lane_type') != 'VEHICLE' or lane.get('is_intersection'):
                    continue
                line = lane['centerline']; end = line[-1]
                distance = math.dist(end[:2], control['position'][:2])
                heading = unit(line[-2], end)
                alignment = sum(a*b for a,b in zip(tangent,heading))
                enters = any(lanes.get(t,{}).get('is_intersection') for t in lane['successors'])
                if distance <= 15 and alignment >= .8 and enters:
                    put_edge(control['id'], 'candidate_controls', key,
                             status='candidate', evidence=[control['id'],road['id']],
                             endpoint_distance_m=distance, heading_cosine=alignment,
                             question='Confirm the approach using source views before assigning a native trigger.')
    controls = scene['controls']
    for obj in list(data['nodes']):
        if obj.get('category') != 'STOP_SIGN':
            continue
        for control in controls:
            distance = math.dist(obj['median_position_local_xyz_m'][:2],control['position'][:2])
            if control['kind']=='stop' and distance <= 15:
                put_edge(obj['id'], 'candidate_same_control', control['id'],
                         status='candidate', distance_m=distance,
                         evidence=['evidence:'+obj['track_uuid'],control['id']])
    data['scope'] = ('Task/evidence/traffic dependency graph. Source topology, measured geometry, '
                     'candidate associations and reviewed relations are distinct. '
                     'No candidate relation is executable traffic authority.')
    return data


def task_packet(data, ids, hops=2):
    """Traverse typed traffic dependencies, then attach their source evidence."""
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(n['id'] for n in data['nodes'])
    relations = {'successor','controls_lane','candidate_controls','candidate_same_control',
                 'on_source_way','attached_above','identifies_road','depends_on','requires_review','affects_view',
                 'realized_as','reviewed_by','accesses_region','supported_by_surface','appearance_from',
                 'candidate_regulates_road','bounds_access','supports_parking','occupies_surface',
                 'joins_surface','shares_height_profile','maintains_distribution'}
    for e in data['edges']:
        if e['relation'] in relations:
            graph.add_edge(e['source'],e['target'],key=e['relation'])
    selected=set(ids)
    for _ in range(hops):
        selected |= {other for key in list(selected) for other in
                     list(graph.successors(key))+list(graph.predecessors(key))}
    selected |= {e['target'] for e in data['edges'] if e['source'] in selected and e['relation']=='supported_by'}
    nodes={n['id']:n for n in data['nodes']}
    return {'nodes':[nodes[k] for k in sorted(selected)],
            'edges':[e for e in data['edges'] if e['source'] in selected and e['target'] in selected],
            'decisions':[d for d in latest_decisions(data).values() if d['source_id'] in selected],
            'selection':{'roots':ids,'dependency_hops':hops,'proximity_is_not_dependency':True}}


def latest_decisions(data):
    # Construction annotations remain in the graph but are not object selections.
    return {(d['task'],d['source_id']):d for d in data['decisions']
            if 'task' in d and 'source_id' in d}


def compile_conditions(data, scene, task=None):
    """Separate immutable source data, selected generation and unresolved evidence."""
    nodes={n['id']:n for n in data['nodes']}
    selected=[d for d in latest_decisions(data).values() if task is None or d['task']==task]
    lanes={l['id']:l for l in scene.get('fusion_map',{}).get('lanes',scene['lanes'])}
    predecessors={key:[] for key in lanes}
    for lane in lanes.values():
        for target in lane['successors']:
            if target in predecessors:predecessors[target].append(lane)
    roads=[]; native=[]; unresolved=[]
    for d in selected:
        key=d['source_id'];node=nodes[key]
        pending=node.get('task_status',{}).get(d['task'],{}).get('missing',[])
        if d['choice']=='seek_evidence' or pending:
            unresolved.append({**deepcopy(d),'pending_properties':pending})
        if node['kind']=='source_lane':
            if d['choice']=='enhance' and node.get('generation_eligible'):
                lane=lanes[key]
                roads.append({'source_id':key,'decision':d,'lane':deepcopy(lane),
                              'incoming':[deepcopy(l) for l in predecessors[key]],
                              'outgoing':[deepcopy(lanes[t]) for t in lane['successors'] if t in lanes]})
        else:
            native.append({'source_id':key,'choice':d['choice'],'required_properties':d['required_properties'],
                           'evidence_ids':d['evidence_ids'],'reason':d['reason'],
                           'content':node.get('content'),
                           'position_local_xyz_m':node.get('median_position_local_xyz_m'),
                           'dimensions_lwh_m':node.get('median_dimensions_lwh_m')})
            if node.get('native_construction'):
                native[-1]['native_construction']=deepcopy(node['native_construction'])
            if node.get('appearance'):
                native[-1]['appearance']=deepcopy(node['appearance'])
    generated={r['source_id'] for r in roads}
    height_only=sorted({key for row in native if row['choice'] in ('retain','enhance')
                        for key in row.get('native_construction',{}).get('mutable_height_lane_ids',[])})
    relations=[e for e in data['edges'] if e['relation']=='controls_lane' and e.get('status')=='confirmed']
    return {'source_scene':data['source_scene'],'task':task,'road':{'generate':roads,
                'retain_lane_ids':[k for k in lanes if k not in generated],
                'height_only_revision_lane_ids':height_only,
                'preserve_all_topology':True},
            'native_objects':native,'confirmed_control_relations':relations,
            'appearance':{'preserve':[r for r in native if r['required_properties']],
                          'rule':'Keep these objects and their required properties in native 3D. Cosmos supplies appearance proposals, never traffic authority or replacements for verified text.'},
            'unresolved':unresolved,
            'scope':'Compiled selection and preservation conditions for generation and native checks.'}


def check_road_candidate(source, candidate, conditions):
    before={l['id']:l for l in source['fusion_map']['lanes']}
    after={l['id']:l for l in candidate['fusion_map']['lanes']}
    height_only=set(conditions['road'].get('height_only_revision_lane_ids',[]))
    def retained(key):
        a,b=before.get(key),after.get(key)
        if a==b:return True
        if key not in height_only or not a or not b:return False
        if a.get('source')!='osm_sumo' and not key.startswith('seam:'):return False
        aa,bb=deepcopy(a),deepcopy(b)
        aa.pop('bridge_profile_revision',None);bb.pop('bridge_profile_revision',None)
        for field in ('centerline','left_boundary','right_boundary'):
            if field in aa:aa[field]=[p[:2]for p in aa[field]]
            if field in bb:bb[field]=[p[:2]for p in bb[field]]
        return aa==bb
    changed=[k for k in conditions['road']['retain_lane_ids'] if not retained(k)]
    topology=[k for k,l in before.items() if
              any(l.get(f)!=after.get(k,{}).get(f) for f in
                  ('successors','allowed_classes','lane_type','width_m','end_widths_m','speed_mps',
                   'left_neighbor','right_neighbor','left_mark_type','right_mark_type'))]
    boundary_changes=[];retained_boundary_gaps=[]
    generated={r['source_id'] for r in conditions['road']['generate']}
    for request in conditions['road']['generate']:
        key=request['source_id'];line=after.get(key,{}).get('centerline',[])
        if not line:
            boundary_changes.append(key);continue
        # Jointly generated neighbours share their accepted port. Outside the
        # selected repair set, the original source boundary remains fixed.
        def neighbour(l):return after[l['id']] if l['id'] in generated else l
        for own_end,other_end,neighbors in ((0,-1,request['incoming']),(-1,0,request['outgoing'])):
            for l in neighbors:
                other=neighbour(l);a,b=line[own_end],other['centerline'][other_end]
                gap=math.dist(a,b)
                if gap<=1e-5:continue
                old_a,old_b=before[key]['centerline'][own_end],before[l['id']]['centerline'][other_end]
                if math.dist(a,old_a)<=1e-5 and math.dist(b,old_b)<=1e-5:
                    retained_boundary_gaps.append(dict(lane_id=key,neighbor_id=l['id'],gap_m=gap,
                        status='Unchanged source port gap; native compiler bridge validation required'))
                elif key not in boundary_changes:boundary_changes.append(key)
    added=sorted(set(after)-set(before)); removed=sorted(set(before)-set(after))
    return {'retained_lane_changes':changed,'topology_or_permission_changes':topology,
            'boundary_changes':boundary_changes,'retained_boundary_gaps':retained_boundary_gaps,'added_lanes':added,'removed_lanes':removed,
            'passed':not(changed or topology or boundary_changes or added or removed)}


def attach_runtime_coverage(data, run, path):
    """Associate a driving run's configured routes with traffic-control approaches."""
    run_id='run:'+str(path)
    add_node(data,{'id':run_id,'kind':'runtime_observation','source':str(path),
                   'map':run['map'],'controller':run['controller'],
                   'scope':'Configured navigation context, not observed lane traversal or STOP compliance.'})
    lane_routes={}
    for route,nav in run['navigation_by_route'].items():
        for lane in nav.get('navigation_context',{}).get('source_lane_ids',[]):
            lane_routes.setdefault(lane,[]).append(route)
            add_edge(data,run_id,'planned_lane',lane,status='configured_not_observed')
    coverage=[]
    for e in list(data['edges']):
        if e['relation'] not in ('candidate_controls','controls_lane'):continue
        routes=lane_routes.get(e['target'],[])
        coverage.append({'control':e['source'],'approach_lane':e['target'],
                         'association_status':e.get('status'),'configured_routes':routes,
                         'coverage':'requires_trigger_and_behavior_check' if routes else 'outside_configured_routes',
                         'control_response_verified':False})
    data['runtime_reviews']=data.get('runtime_reviews',[])+[{'run':run_id,'coverage':coverage}]
    return {'run':str(path),'map':run['map'],'coverage':coverage,
            'scope':'A route pass or zero collisions cannot complete an untested traffic-control dependency.'}


def condition_appearance_case(case, conditions, scope='scene_proposal'):
    """Compile soft appearance conditions into a Cosmos prompt.

    Exact text and traffic-control geometry are represented by native assets.
    """
    case=deepcopy(case)
    requirements=[{'source_id':r['source_id'],'content':r['content'],
                   'preserve_properties':r['required_properties']}
                  for r in conditions['appearance']['preserve']]
    import json
    if scope=='material_only':
        instruction=(' Generate only the requested material sample. Do not paint text, road signs, '
                     'vehicles or lane markings into it. Source objects and lettering are separate native assets.')
    else:
        instruction=(' Preserve the recorded scene layout and road-control objects. '
                     'Do not invent traffic markings or change sign content. '
                     'Exact sign lettering is supplied by separate native assets. ')+json.dumps(requirements,ensure_ascii=False)
    if isinstance(case['prompt'],dict):
        case['prompt']['context']=case['prompt'].get('context','')+instruction
    else:
        case['prompt']+=instruction
    case['information_conditions']={'requirements':requirements,
        'generation_scope':scope,
        'unresolved':[d['source_id'] for d in conditions['unresolved']],
        'scope':'Soft appearance guidance only. Native assets and property review enforce retained information.'}
    return case


def review(data, realized, realization_path):
    """Write explicit property feedback back into the graph; preserve review history."""
    lookup={r['source_id']:r for r in realized['objects']}
    feedback=[]
    for (task,key),decision in latest_decisions(data).items():
        item=lookup.get(key,{})
        properties=item.get('verified_properties',{})
        missing=[p for p in decision['required_properties'] if not properties.get(p,False)]
        complete=not missing and bool(item) and decision['choice']!='seek_evidence'
        asset_id='realization:'+key
        review_id='review:'+str(len(data.setdefault('property_reviews',[])))
        add_node(data,{'id':asset_id,'kind':'native_realization','source_id':key,
                       'report':str(realization_path),'map':realized.get('map'),
                       'verified_properties':properties})
        add_edge(data,key,'realized_as',asset_id,status='reported' if item else 'absent')
        actions=[]
        if 'controlled_approach' in missing:
            actions.append({'action':'retrieve_control_relation','packet_roots':[key],
                            'need':'Directed road evidence, source sign face and native trigger; appearance alone is insufficient.'})
        if any(p in missing for p in ('native_visibility','readable_road_name')):
            actions.append({'action':'inspect_calibrated_views','packet_roots':[key],
                            'need':'Choose evidence viewpoints with a visible face, then inspect the corresponding native image.'})
        if 'occlusion_geometry' in missing:
            actions.append({'action':'measure_sightline_geometry','packet_roots':[key],
                            'need':'Recover source geometry before allowing deletion or shape simplification.'})
        if any(p in missing for p in ('open_drivable_surface','ramped_sidewalk','curb_continuity')):
            actions.append({'action':'inspect_access_surface','packet_roots':[key],
                            'need':'Check source entrance, road/surface connection, curb height profile and obstruction geometry; a land-cover mask is insufficient.'})
        if 'complete_window_coverage' in missing:
            actions.append({'action':'inspect_full_observation_window','packet_roots':[key],
                            'need':'Use the complete trajectory and view margin; inspect first/last frames and intermediate frontage. A repaired early entrance does not cover the whole clip.'})
        if 'image_access_review' in missing:
            actions.append({'action':'retrieve_access_views','packet_roots':[key],
                            'need':'Use the source-connected service-way mouth to retrieve camera evidence; distinguish a visible vehicle exit from an unseen or nonvehicle path before construction.'})
        if 'support_contact' in missing:
            actions.append({'action':'inspect_asset_support','packet_roots':[key],
                            'need':'Measure native asset support against the actual surface; preserve source box states separately from asset fitting.'})
        if any(p in missing for p in ('parking_occupancy_consistent','parking_semantics','pavement_occupancy')):
            actions.append({'action':'inspect_parking_surface_relation','packet_roots':[key],
                            'need':'Compare source parking tags, observed vehicle state, native body footprint and sidewalk boundary. Repair inferred surfaces/assets before moving a source-observed vehicle.'})
        if any(p in missing for p in ('same_level_overlap_resolved','fixed_connection_continuity','bridge_sidewalk_connection','visible_surface_continuity')):
            actions.append({'action':'inspect_shared_surface_interfaces','packet_roots':[key],
                            'need':'Measure overlapping same-level road triangle heights and road/sidewalk/bridge joins; update native geometry, paint and navigation elevation together. A driving pass alone is insufficient.'})
        if any(p in missing for p in ('source_context_retained','outer_distribution_continuity')):
            actions.append({'action':'inspect_frontage_sequence','packet_roots':[key],
                            'need':'Compare equal-length source and extension windows for stationary context, canopy, building form/setback, pavement and supported controls. Separate road-use transitions and occlusion from unexplained disappearance; retain source/inferred provenance. Do not add traffic rules to match style.'})
        if any(p in missing for p in ('continuous_ground_profile','building_ground_support')):
            actions.append({'action':'inspect_final_frontage_support','packet_roots':[key],
                            'need':'Inspect final road/terrain triangles and retained building foundations together. Report source footprint conflicts separately; retain XY and verified accesses while updating height support.'})
        if 'source_conditioned_architecture' in missing:
            actions.append({'action':'retrieve_frontage_style_evidence','packet_roots':[key],
                            'need':'Retrieve OSM footprint/use/height and reviewed local camera references. Distinguish observed architecture from inferred roof, window and material choices.'})
        if 'source_supported_color' in missing:
            actions.append({'action':'review_body_color_evidence','packet_roots':[key],
                            'need':'Inspect visible body samples across source views; reject occluder/window/shadow samples and keep one appearance per track.'})
        if 'readable_speed_value' in missing:
            actions.append({'action':'inspect_calibrated_views','packet_roots':[key],
                            'need':'Read source sign content and units independently of annotation categories, then verify native lettering and pose.'})
        row={'source_id':key,'task':task,'choice':decision['choice'],
             'source_evidence':decision['evidence_ids'],'native_realization':item,
             'missing_properties':missing,'complete_for_requested_properties':complete,
             'next_actions':actions,
             'next_action':'retain_current_realization' if complete else 'resolve_requested_properties'}
        data['property_reviews'].append(deepcopy(row))
        add_node(data,{'id':review_id,'kind':'property_review','report':str(realization_path),
                       'task':task,'missing_properties':missing,'complete':complete})
        add_edge(data,asset_id,'reviewed_by',review_id,status='reported')
        node=next(n for n in data['nodes'] if n['id']==key)
        node.setdefault('task_status',{})[task]={'complete':complete,'missing':missing,'review':review_id}
        node['realization']='reviewed' if all(v['complete'] for v in node['task_status'].values()) else 'needs_review'
        feedback.append(row)
    return {'scope':'Selected property checks with explicit evidence, not automatic scene fidelity or independent image review.',
            'feedback':feedback,'pending':sum(not r['complete_for_requested_properties'] for r in feedback)}
