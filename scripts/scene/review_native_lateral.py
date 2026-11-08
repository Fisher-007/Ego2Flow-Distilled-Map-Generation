"""Check native lateral relations for source-observed lane-change candidates.

Candidates are same-direction AV2 vehicle lanes sharing a dashed-white boundary.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString


def source_candidates(scene, minimum_length=8., neighbor_end_margin=2., dashed_only=True, query_fraction=.5):
    lanes = {l['id']: l for l in scene['fusion_map']['lanes']}
    children = defaultdict(list)
    for lane in lanes.values():
        children[lane.get('source_parent_lane_id', lane['id'])].append(lane)
    rows = []
    for key, lane in lanes.items():
        if lane['source'] != 'av2' or lane.get('is_intersection') or lane.get('lane_type') != 'VEHICLE':
            continue
        line = LineString(np.asarray(lane['centerline'])[:, :2])
        if line.length < minimum_length:
            continue
        for side, opposite in [('left', 'right'), ('right', 'left')]:
            neighbor = lane.get(side + '_neighbor')
            if not neighbor or (dashed_only and lane.get(side + '_mark_type') != 'DASHED_WHITE'):
                continue
            candidates = []
            for target in children.get(neighbor, []):
                if (target.get('lane_type') != 'VEHICLE' or target.get('is_intersection')
                        or (dashed_only and target.get(opposite + '_mark_type') != 'DASHED_WHITE')):
                    continue
                if target.get(opposite + '_neighbor') != lane.get('source_parent_lane_id', key):
                    continue
                other = LineString(np.asarray(target['centerline'])[:, :2])
                midpoint = line.interpolate(query_fraction, normalized=True)
                station = other.project(midpoint)
                if min(station, other.length - station) < neighbor_end_margin:
                    continue
                a = np.asarray(line.interpolate(min(line.length, line.length * query_fraction + 1)).coords[0]) - np.asarray(line.interpolate(max(0, line.length * query_fraction - 1)).coords[0])
                b = np.asarray(other.interpolate(min(other.length, station + 1)).coords[0]) - np.asarray(other.interpolate(max(0., station - 1)).coords[0])
                cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                distance = midpoint.distance(other)
                if cosine > .98 and 1.5 < distance < 7:
                    candidates.append((distance, target, cosine))
            if not candidates:
                continue
            _, target, cosine = min(candidates, key=lambda r: r[0])
            rows.append(dict(source_lane=key,source_neighbor=target['id'],side=side,
                source_marking=lane[side+'_mark_type'],source_heading_cosine=cosine))
    return rows


def review(scene, metadata, native, source_scene=None):
    from map_generation.road_builder import fit_lane
    from map_generation.lane_sections import reference_station
    spans=metadata['lane_to_road_intervals'];rows=[]
    source_scene=source_scene or scene
    source_lanes={l['id']:l for l in source_scene['fusion_map']['lanes']}
    compiled={l['id']:l for l in scene['fusion_map']['lanes']};families=defaultdict(list);lines={};geometries={}
    for key,lane in compiled.items():families[lane.get('source_parent_lane_id',key)].append(key)
    for candidate in source_candidates(source_scene):
        key=candidate['source_lane'];target=candidate['source_neighbor'];side=candidate['side']
        original=source_lanes[key];point=LineString(original['centerline']).interpolate(.5,normalized=True)
        children=families[original.get('source_parent_lane_id',key)]
        def distance(child):
            if child not in lines:lines[child]=LineString(compiled[child]['centerline'])
            line=lines[child];s=line.project(point)
            return line.distance(point),-min(s,line.length-s)
        child=min(children,key=distance)
        if child not in geometries:geometries[child]=fit_lane(compiled[child],1.)[0]
        source_station=reference_station(geometries[child],point.coords[0])
        choices=[s for s in spans[child] if s['source_s_start_m']-1e-6<=source_station<=s['source_s_end_m']+1e-6]
        external=[s for s in choices if s['junction_id']is None]
        span=(external or choices)[0]
        if 'source_stations_m' in span:
            station=float(np.interp(source_station,span['source_stations_m'],span['road_stations_m']))
        else:station=span['road_s_start_m']+source_station-span['source_s_start_m']
        requested_station = station
        waypoint = native.get_waypoint_xodr(int(span['road_id']), int(span['lane_id']), station)
        if waypoint is None and abs(station - span['road_s_end_m']) <= 1e-6:
            # Query the preceding float32 station at native endpoints and record its displacement.
            station = float(np.nextafter(np.float32(station), np.float32(-np.inf)))
            waypoint = native.get_waypoint_xodr(int(span['road_id']), int(span['lane_id']), station)
        if waypoint is None:
            raise ValueError('Source midpoint has no native sample: ' + json.dumps(dict(
                source=key, compiler_child=child, source_station_m=source_station,
                native_station_m=station, selected_span=span, candidate_spans=choices)))
        adjacent = getattr(waypoint, 'get_' + side + '_lane')()
        target_children=families[source_lanes[target].get('source_parent_lane_id',target)]
        expected = {(int(s['road_id']), int(s['lane_id'])) for c in target_children for s in spans[c]}
        actual = (adjacent.road_id, adjacent.lane_id) if adjacent else None
        marking = getattr(waypoint, side + '_lane_marking')
        rows.append(dict(**candidate,
            compiled_source_lane=child,source_query_position_enu_m=list(point.coords[0]),
            compiled_source_station_m=source_station,
            road_id=int(span['road_id']), lane_id=int(span['lane_id']), s_m=station,
            requested_native_station_m=requested_station,
            native_endpoint_adjustment_m=station-requested_station,
            source_has_external_interval=bool(external),native_road_kind=metadata['roads'][str(span['road_id'])]['kind'],
            native_neighbor=actual, expected_native_neighbors=sorted(expected),
            native_marking=str(marking.type) if marking else None,
            native_lane_change=str(waypoint.lane_change),
            native_position_enu_m=[waypoint.transform.location.x, -waypoint.transform.location.y, waypoint.transform.location.z],
            neighbor_preserved=actual in expected))
    return dict(candidate_count=len(rows), preserved_count=sum(r['neighbor_preserved'] for r in rows),
        missing_count=sum(not r['neighbor_preserved'] for r in rows), candidates=rows,
        source_relations_represented_only_in_junctions=sum(not r['source_has_external_interval'] for r in rows),
        scope='Reciprocal, same-direction AV2 vehicle neighbors queried at the original source midpoint; compiler subdivisions keep the source denominator. Native lateral access, not a traffic maneuver success claim.')


def attach_feedback(graph, result, report_path, reason):
    task = 'task:native_lateral_navigation'
    nodes = {n['id']: n for n in graph['nodes']}
    if task not in nodes:
        graph['nodes'].append(dict(id=task, kind='simulation_task'))
    for row in result['candidates']:
        key = 'native_lateral:' + row['source_lane'] + ':' + row['side']
        content = dict(id=key, kind='native_lane_relation', content=row,
            task_status={task: dict(complete=False, missing=(
                ['native_marking_and_driving_review'] if row['neighbor_preserved'] else
                ['native_neighbor', 'native_marking_and_driving_review']))})
        if key in nodes:
            nodes[key].update(content)
        else:
            graph['nodes'].append(content)
        for source, relation, target in [(task, 'depends_on', key),
                (key, 'from_source_lane', row['source_lane']),
                (key, 'to_source_lane', row['source_neighbor'])]:
            edge=dict(source=source, relation=relation, target=target)
            if edge not in graph['edges']:
                graph['edges'].append(edge)
        decision=dict(source_id=key, task=task, choice='enhance',
            agent='construction_agent', reason=reason,
            evidence_ids=[row['source_lane'], row['source_neighbor']],
            review_source=str(report_path),
            required_properties=['source_geometry_and_permissions', 'native_neighbor',
                                 'native_marking', 'native_driving_review'])
        graph['decisions']=[d for d in graph['decisions']
                            if (d.get('source_id'), d.get('task')) != (key, task)]
        graph['decisions'].append(decision)
    return graph


def main():
    import carla
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['scene', 'roads', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--graph', type=Path, help='Attach native failures to the existing evidence graph')
    p.add_argument('--source-scene',type=Path,help='Use a fixed earlier source scene for candidate locations and denominator while compiled lanes are subdivided')
    p.add_argument('--reason', help='Agent interpretation and intended repair of the observed failures')
    a = p.parse_args()
    if a.graph and not a.reason:
        p.error('--graph requires the Agent reason for the proposed repair')
    scene = json.loads(a.scene.read_text())
    metadata = json.loads((a.roads / 'road_lane_ids.json').read_text())
    native = carla.Map(scene['map_id'], (a.roads / 'roads.xodr').read_text())
    result = review(scene, metadata, native,json.loads(a.source_scene.read_text())if a.source_scene else None)
    result.update(source_scene=str(a.scene.resolve()), compiled_roads=str(a.roads.resolve()))
    if a.source_scene:result['candidate_source_scene']=str(a.source_scene.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    if a.graph:
        graph = attach_feedback(json.loads(a.graph.read_text()), result, a.output, a.reason)
        a.output.with_name(a.output.stem + '_graph.json').write_text(json.dumps(graph, separators=(',', ':')) + '\n')
    print({k: v for k, v in result.items() if k != 'candidates'}, flush=True)


if __name__ == '__main__':
    main()
