"""Attach reviewed city observations and actual fusion feedback to the shared graph."""

import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, box
from shapely.strtree import STRtree

from map_generation.information_graph import enrich, compile_conditions


def control_graph(root, review_path):
    """Retain calibrated control observations and the agent's actual view review."""
    review=json.loads(review_path.read_text())
    decisions={(row['log_id'],row['source_id']):row for row in review['reviews']}
    graph=dict(nodes=[],edges=[],decisions=[])
    task='task:city_controls'
    graph['nodes'].append(dict(id=task,kind='simulation_task'))
    for path in sorted(root.glob('*/controls.json')):
        packet=json.loads(path.read_text());log=packet['log_id']
        for source in packet['controls']:
            decision=decisions.get((log,source['source_id']))
            if decision is None:continue
            key=source['source_id'];evidence='evidence:'+source['track_uuid']
            pending=['verified_face_orientation','confirmed_controlled_approach','native_realization']
            if decision['visual_status'] not in ('readable_stop_face','readable_other_face'):pending.append('readable_front_face')
            if decision.get('source_category_conflict'):pending.append('resolve_category_and_observation_association')
            graph['nodes'].extend([
                dict(id=key,kind='observed_control',category=source['category'],
                    observed_face_kind=decision.get('observed_face_kind'),supplementary_rule=decision.get('supplementary_rule'),
                    track_uuid=source['track_uuid'],
                    first_timestamp_ns=source['first_timestamp_ns'],last_timestamp_ns=source['last_timestamp_ns'],
                    median_position_local_xyz_m=source['median_position_local_xyz_m'],
                    median_dimensions_lwh_m=source['dimensions_lwh_m'],
                    content=dict(observation=source,visual_review=decision,source_log=log),
                    task_status={task:dict(missing=pending)}),
                dict(id=evidence,kind='source_evidence',source=str(path),views=decision.get('reviewed_views',source['views']),source_log=log)])
            graph['edges'].extend([dict(source=key,relation='supported_by',target=evidence),
                                   dict(source=task,relation='depends_on',target=key)])
            graph['decisions'].append(dict(source_id=key,task=task,choice=decision['decision'],
                agent=review['agent'],reason=decision['note'],required_properties=decision['required_properties'],
                evidence_ids=[evidence]))
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "review", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--junction-review", type=Path)
    parser.add_argument("--access-review", type=Path)
    parser.add_argument("--geometry-review", type=Path, help="Current source-access geometry findings")
    parser.add_argument('--controls-root',type=Path)
    parser.add_argument('--control-review',type=Path)
    parser.add_argument('--coarse-geometry-review',type=Path)
    parser.add_argument('--continuation-review',type=Path)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    reviewed = json.loads(args.review.read_text())
    seed=control_graph(args.controls_root,args.control_review) if args.controls_root else dict(nodes=[],edges=[],decisions=[])
    graph = enrich(seed, scene, args.scene.resolve())
    lanes = scene["fusion_map"]["lanes"]
    shapes = [LineString(np.asarray(lane["centerline"])[:, :2]) for lane in lanes]
    tree = STRtree(shapes)
    findings = {row["area_id"]:row for row in reviewed["areas"]}
    tasks = {}
    for area in scene["areas"]:
        task_id = "task:"+area["id"]
        area_id = "region:"+area["id"]
        tasks[area["id"]] = task_id
        graph["nodes"].extend([dict(id=task_id, kind="simulation_task", area_id=area["id"]),
                               {**area, "id":area_id, "kind":"source_area"}])
        graph["edges"].append(dict(source=task_id, relation="depends_on", target=area_id))
        region = box(*area["influence_bbox_local_xy"])
        for node in seed['nodes']:
            if node['kind']!='observed_control':continue
            x,y=node['median_position_local_xyz_m'][:2]
            xmin,ymin,xmax,ymax=area['influence_bbox_local_xy']
            if xmin<=x<=xmax and ymin<=y<=ymax:
                graph['edges'].append(dict(source=area_id,relation='depends_on',target=node['id']))
        for i in sorted(tree.query(region, predicate="intersects")):
            graph["edges"].append(dict(source=area_id, relation="depends_on", target=lanes[i]["id"],
                                       scope="Road inside the Area's declared influence region"))
        observation = findings[area["id"]]
        evidence_id = "evidence:"+area["id"]+":source_views"
        graph["nodes"].append(dict(id=evidence_id, kind="source_evidence", views=observation["views"],
                                   source_log=observation["log_id"], reviewed_by=reviewed["agent"]))
        graph["edges"].append(dict(source=area_id, relation="supported_by", target=evidence_id))
        for index, finding in enumerate(observation["findings"]):
            key = f'observed:{area["id"]}:{index}'
            graph["nodes"].append(dict(id=key, kind="reviewed_visual_observation", content=finding,
                task_status={task_id:{"missing":["source_pose_and_extent", "native_realization"]}}))
            graph["edges"].extend([dict(source=key, relation="supported_by", target=evidence_id),
                                   dict(source=area_id, relation="depends_on", target=key)])
            graph["decisions"].append(dict(source_id=key, task=task_id, choice="retain",
                agent=reviewed["agent"], reason=finding["reason"], required_properties=finding["retain"],
                evidence_ids=[evidence_id]))
        graph["decisions"].append(dict(source_id=area_id, task=task_id, choice="enhance",
            agent=reviewed["agent"], reason=observation["construction_focus"],
            required_properties=["legal_connectivity", "smooth_road_and_curb", "source_style", "controllable_actors"],
            evidence_ids=[evidence_id]))
    for index, port in enumerate(scene["fusion_map"]["unresolved_boundaries"]):
        key = f'feedback:fusion_port:{index}'
        graph["nodes"].append(dict(id=key, kind="static_review_failure", content=port,
                                   source=str(args.scene.resolve()), resolution="pending"))
        for lane_id in port["lane_ids"]:
            graph["edges"].append(dict(source=lane_id, relation="requires_review", target=key))
    junctions = json.loads(args.junction_review.read_text())["uncovered_groups"] if args.junction_review else []
    for index, group in enumerate(junctions):
        key = f'feedback:native_junction:{index}'
        graph["nodes"].append(dict(id=key, kind="static_review_failure", content=group,
                                   source=str(args.junction_review), resolution="pending"))
        for lane_id in group["uncovered_internal_lane_ids"]:
            graph["edges"].append(dict(source=lane_id, relation="requires_review", target=key))
    if args.geometry_review:
        geometry=json.loads(args.geometry_review.read_text())
        for finding in geometry['findings']:
            if not finding['needs_review']:continue
            key='feedback:geometry:'+finding['seam_id']
            graph['nodes'].append(dict(id=key,kind='static_review_failure',content=finding,
                source=str(args.geometry_review),resolution='pending'))
            graph['edges'].append(dict(source=finding['seam_id'],relation='requires_review',target=key))
    if args.coarse_geometry_review:
        repair=json.loads(args.coarse_geometry_review.read_text())
        key='construction:coarse_endpoint_order';task='task:city_road_geometry';evidence='evidence:coarse_order_solve'
        graph['nodes'].extend([
            dict(id=task,kind='simulation_task'),
            dict(id=evidence,kind='source_evidence',source=str(args.coarse_geometry_review)),
            dict(id=key,kind='geometry_construction',content=repair,
                 task_status={task:dict(missing=['native_geometry_and_driving'])})])
        graph['edges'].extend([dict(source=task,relation='depends_on',target=key),
                               dict(source=key,relation='supported_by',target=evidence)])
        changed={row['lane_id'] for row in repair['changed_lanes']}
        for lane in lanes:
            if lane.get('source_lane_id') in changed:
                graph['edges'].append(dict(source=lane['id'],relation='depends_on',target=key))
        graph['decisions'].append(dict(source_id=key,task=task,choice='enhance',agent=reviewed['agent'],
            reason='Repair reversed inferred straight connectors through shared endpoint ordering while preserving the source maneuver graph, permissions and AV2 observations.',
            required_properties=['forward_geometry','source_topology_and_access_preserved','native_geometry_and_driving'],evidence_ids=[evidence]))
    if args.continuation_review:
        repair=json.loads(args.continuation_review.read_text())
        task='task:city_source_continuity';evidence='evidence:retained_source_intervals'
        graph['nodes'].extend([dict(id=task,kind='simulation_task'),
            dict(id=evidence,kind='source_evidence',source=str(args.continuation_review),content=repair)])
        source_ids={part[0] for row in repair['proposals'] for part in row['intervals']}
        graph_nodes={node['id']:node for node in graph['nodes']}
        for lane in lanes:
            if lane.get('source_lane_id') not in source_ids:continue
            graph_nodes[lane['id']].setdefault('task_status',{})[task]=dict(missing=['native_continuity'])
            graph['edges'].extend([dict(source=task,relation='depends_on',target=lane['id']),
                dict(source=lane['id'],relation='supported_by',target=evidence)])
            graph['decisions'].append(dict(source_id=lane['id'],task=task,choice='retain',agent=repair['agent'],
                reason=repair['reason'],required_properties=repair['required_properties'],evidence_ids=[evidence]))
    for source_id,override in scene['fusion_map'].get('coarse_lane_overrides',{}).items():
        targets=override.get('ownership_fine_lane_ids',[]) or sorted({k for path in override.get('replacement_fine_paths',[]) for k in path})
        if not targets:continue
        key='correspondence:'+source_id;task='task:city_source_ownership'
        if not any(n['id']==task for n in graph['nodes']):graph['nodes'].append(dict(id=task,kind='simulation_task'))
        graph['nodes'].append(dict(id=key,kind='reviewed_source_correspondence',content=override,
            task_status={task:dict(missing=['native_continuity'])}))
        graph['edges'].append(dict(source=task,relation='depends_on',target=key))
        for fine_id in targets:
            graph['edges'].append(dict(source=key,relation='corresponds_to',target=fine_id))
        graph['decisions'].append(dict(source_id=key,task=task,choice='retain',agent=reviewed['agent'],
            reason=override['evidence'],required_properties=['observed_fine_geometry','source_directed_path','native_continuity'],evidence_ids=[]))
    # A source frame is shared by its roads, camera calibration and traffic.
    # Keep this dependency visible when an Area uses a registered observation.
    assembly=scene['source_info'].get('assembly_review',{})
    for row in assembly.get('log_registrations',[]):
        key='source_frame:'+row['log_id'];task='task:city_source_alignment'
        graph['nodes'].append(dict(id=key,kind='reviewed_source_frame',content=row,
            task_status={task:{'missing':[]}}))
        graph['decisions'].append(dict(source_id=key,task=task,choice='retain',
            agent=reviewed['agent'],reason='Apply the same reviewed rigid frame to source roads, cameras and actors; retain observation differences.',
            required_properties=['shared_road_camera_actor_frame','source_difference_provenance'],evidence_ids=[]))
        graph['edges'].append(dict(source=task,relation='depends_on',target=key))
        for area in scene['areas']:
            if row['log_id'] in {v['log_id'] for v in area.get('source_observations',[])}:
                graph['edges'].append(dict(source='region:'+area['id'],relation='depends_on',target=key))
    if assembly.get('log_registrations'):
        graph['nodes'].append(dict(id='task:city_source_alignment',kind='simulation_task'))
    conflicts = json.loads(args.access_review.read_text())["candidates"] if args.access_review else []
    for index, conflict in enumerate(conflicts):
        key = f'feedback:source_state:{index}'
        graph["nodes"].append(dict(id=key, kind="source_state_candidate", content=conflict,
                                   source=str(args.access_review), resolution="retrieve_history_and_views"))
        for source_id in (conflict["lane_id"],conflict["road_id"]):
            graph["edges"].append(dict(source=source_id, relation="requires_review", target=key))
    graph["city_review"] = {"review":str(args.review), "agent":reviewed["agent"],
        "source_ownership":reviewed["source_ownership"],
        "scope":"Reviewed camera observations and actual fusion failures. Control-to-lane bindings, source registration, native assets and driving remain separate review tasks."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"graph.json").write_text(json.dumps(graph, separators=(",", ":"))+"\n")
    conditions = compile_conditions(graph, scene)
    (args.output/"conditions.json").write_text(json.dumps(conditions, separators=(",", ":"))+"\n")
    summary = dict(nodes=len(graph["nodes"]), edges=len(graph["edges"]),
                   reviewed_areas=len(tasks), decisions=len(graph["decisions"]),
                   fusion_failures=len(scene["fusion_map"]["unresolved_boundaries"]),
                   junction_path_preflight_failures=len(junctions), source_state_candidates=len(conflicts),
                   reviewed_control_observations=sum(n['kind']=='observed_control' for n in graph['nodes']),
                   confirmed_control_relations=len(conditions["confirmed_control_relations"]),
                   pending_realizations=len(conditions["unresolved"]))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(summary)


if __name__ == "__main__":
    main()
