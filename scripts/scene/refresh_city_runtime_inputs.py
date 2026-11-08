"""Refresh source-time city handoffs after native regrouping or elevation changes."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')



def source_waypoint_transfer(old_mapping, current_mapping, source_lanes=None, source_tolerance=.003):
    """Resolve retained native samples through their exact source-lane stations."""
    from collections import defaultdict
    from scripts.scene.transfer_native_signals import _station
    from shapely.geometry import LineString, Point, Polygon
    envelopes = {}
    index = defaultdict(list)
    for source_id, spans in old_mapping['lane_to_road_intervals'].items():
        for span in spans:
            index[(int(span['road_id']), int(span['lane_id']))].append((source_id, span))

    def compatible(meta, route_ids):
        if meta.get('kind') == 'lane_group':
            return any(key in route_ids for key in meta['source_lane_ids'])
        path = meta.get('source_lane_path') or ([meta['scene_lane_id']] if meta.get('scene_lane_id')
                else [meta.get('from_lane_id'), meta.get('to_lane_id')])
        return any(all(route_ids[i] == path[i-offset]
                       for i in range(max(0, offset), min(len(route_ids), offset+len(path))))
                   for offset in range(1-len(path), len(route_ids)))

    def transfer(record, native, route_ids):
        candidates = []
        for source_id, old in index[(record['road_id'], record['lane_id'])]:
            if source_id not in route_ids or not (
                    old['road_s_start_m']-1e-6 <= record['s_m'] <= old['road_s_end_m']+1e-6):
                continue
            source_s = _station(old, record['s_m'])
            for target in current_mapping['lane_to_road_intervals'][source_id]:
                if not target['source_s_start_m']-1e-6 <= source_s <= target['source_s_end_m']+1e-6:
                    continue
                rid, lane = int(target['road_id']), int(target['lane_id'])
                if not compatible(current_mapping['roads'][str(rid)], route_ids):
                    continue
                road_s = _station(target, source_s, True)
                wp = native.get_waypoint_xodr(rid, lane, road_s)
                if wp is None:
                    continue
                q = wp.transform.location
                error = float(np.linalg.norm(np.array([q.x,q.y])-record['position_carla_m'][:2]))
                candidates.append((error, rid != record['road_id'], lane != record['lane_id'],
                    rid, lane, wp, dict(source_lane_id=source_id, source_s_m=source_s,
                    previous_road_id=record['road_id'], previous_lane_id=record['lane_id'],
                    previous_s_m=record['s_m'], road_id=rid, lane_id=lane, s_m=road_s)))
        if not candidates:
            # Inferred bridge intervals are compiled movement geometry, not
            # AV2 lane intervals. Retain only the same declared movement and XY.
            meta = old_mapping['roads'].get(str(record['road_id']))
            current = current_mapping['roads'].get(str(record['road_id']))
            if meta and meta == current and compatible(meta, route_ids):
                station = 0.
                for key, lo, hi in meta.get('source_intervals', []):
                    length = hi-lo
                    if key.startswith('bridge:') and station-1e-6 <= record['s_m'] <= station+length+1e-6:
                        wp = native.get_waypoint_xodr(record['road_id'], record['lane_id'], record['s_m'])
                        if wp is not None:
                            q = wp.transform.location
                            error = float(np.linalg.norm(np.array([q.x,q.y])-record['position_carla_m'][:2]))
                            if error <= source_tolerance:
                                return wp, dict(source_lane_id=key, source_s_m=lo+record['s_m']-station,
                                    previous_road_id=record['road_id'], previous_lane_id=record['lane_id'],
                                    previous_s_m=record['s_m'], road_id=wp.road_id, lane_id=wp.lane_id, s_m=wp.s,
                                    source_geometry_support='retained_compiled_bridge', source_geometry_supported=None,
                                    current_source_centerline_error_m=None, current_source_envelope_outside_m=None,
                                    source_lane_path=meta['source_lane_path'], retained_native_xy=True,
                                    bridge_scope='Same explicit inferred movement interval and directed source path; not an observed AV2 lane.')
                    station += length
            raise ValueError('Retained waypoint has no compatible current source interval: '+str(record))
        selected = min(candidates, key=lambda row: row[:5])
        wp, binding = selected[-2], selected[-1]
        if source_lanes is not None:
            key = binding['source_lane_id']
            if key not in envelopes:
                lane = source_lanes[key]
                centerline = LineString(np.asarray(lane['centerline'])[:, :2])
                left, right = lane.get('left_boundary'), lane.get('right_boundary')
                envelope = (Polygon([p[:2] for p in left] + [p[:2] for p in reversed(right)]).buffer(0)
                            if left and right else centerline.buffer(lane['width_m']/2, cap_style=2))
                envelopes[key] = centerline, envelope
            centerline, envelope = envelopes[key]
            point = Point(wp.transform.location.x, -wp.transform.location.y)
            outside = float(envelope.distance(point))
            progress = centerline.project(point)
            first = np.array(centerline.interpolate(max(0., progress-.01)).coords[0])
            last = np.array(centerline.interpolate(min(centerline.length, progress+.01)).coords[0])
            tangent = last-first
            forward = wp.transform.get_forward_vector()
            direction = float(tangent @ np.array([forward.x,-forward.y]) / np.linalg.norm(tangent))
            center_error = float(centerline.distance(point))
            source_supported = outside <= source_tolerance or center_error <= source_tolerance
            retained_native = selected[0] <= source_tolerance
            binding.update(current_source_centerline_error_m=center_error,
                source_geometry_supported=source_supported,
                retained_native_xy=retained_native,
                source_geometry_support=('boundary_envelope' if outside <= source_tolerance else 'retained_centerline' if center_error <= source_tolerance else 'unchanged_native_candidate'),
                current_source_envelope_outside_m=outside,
                current_source_boundary_clearance_m=float(envelope.boundary.distance(point)),
                source_direction_dot=direction)
            if not (source_supported or retained_native) or direction <= 0:
                raise ValueError('Transferred native sample leaves its retained directed source lane: '+str(binding))
        return wp, binding
    return transfer


def refresh_waypoints(value, native, checks, xy_tolerance, transfer=None, route_ids=None):
    """Update executable samples at retained native or explicit source stations."""
    if isinstance(value, list):
        for item in value:
            refresh_waypoints(item, native, checks, xy_tolerance, transfer, route_ids)
    elif isinstance(value, dict):
        route_ids = value.get('source_lane_ids', route_ids)
        if all(key in value for key in ('road_id', 'lane_id', 's_m', 'position_carla_m')):
            binding = None
            if transfer is None:
                wp = native.get_waypoint_xodr(value['road_id'], value['lane_id'], value['s_m'])
            else:
                wp, binding = transfer(value, native, route_ids)
            if wp is None:
                raise ValueError('Retained source task waypoint no longer exists: ' + str(value))
            q = wp.transform.location
            position = [q.x, q.y, q.z]
            delta = np.asarray(position) - value['position_carla_m']
            xy = float(np.linalg.norm(delta[:2]))
            if transfer is None and xy > xy_tolerance:
                raise ValueError('Retained source task changed in XY by %r m' % xy)
            old_width = value.get('lane_width_m')
            value['position_carla_m'] = position
            if binding is not None:
                value.update(road_id=wp.road_id, lane_id=wp.lane_id, s_m=wp.s)
                if 'is_junction' in value:
                    value['is_junction'] = wp.is_junction
            forward = wp.transform.get_forward_vector()
            if 'lane_center_forward_xy' in value:
                value['lane_center_forward_xy'] = [forward.x, forward.y]
            if 'lane_width_m' in value:
                if transfer is None and abs(value['lane_width_m'] - wp.lane_width) > 1e-6:
                    raise ValueError('Retained source task width changed')
                value['lane_width_m'] = wp.lane_width
            checks.append(dict(xy_change_m=xy, z_change_m=float(abs(delta[2])),
                source_station_transfer=binding, width_change_m=None if old_width is None else float(wp.lane_width-old_width)))
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                child_ids = value.get('navigation_context', {}).get('source_lane_ids', route_ids) if key == 'navigation_waypoints' else route_ids
                refresh_waypoints(child, native, checks, xy_tolerance, transfer, child_ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source-scenarios', 'scene', 'conditions', 'roads', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--map', required=True)
    p.add_argument('--xy-tolerance-m', type=float, default=.003)
    p.add_argument('--previous-roads', type=Path, help='Rebind exact retained source stations when native road/lane identifiers were regrouped')
    p.add_argument('--areas', nargs='+', help='Explicit source Area subset for an affected implementation check')
    p.add_argument('--signals', type=Path, help='Current source-station signal bindings with unchanged source programs')
    a = p.parse_args()
    import carla
    configs = sorted(a.source_scenarios.glob('*/scenario.json'))
    if not configs:
        raise ValueError('No source Area scenarios')
    mapping = read(a.roads / 'road_lane_ids.json')
    source_lanes = {lane['id']:lane for lane in read(a.scene)['fusion_map']['lanes']} if a.previous_roads else None
    transfer = source_waypoint_transfer(read(a.previous_roads / 'road_lane_ids.json'), mapping, source_lanes, a.xy_tolerance_m) if a.previous_roads else None
    if a.areas:
        configs = [path for path in configs if read(path)['area_id'] in a.areas]
        if not configs:raise ValueError('No matching source Area scenarios')
    source_ids = set(mapping['key_to_road_id']) | set(mapping['lane_to_road_intervals'])
    native = carla.Map(a.map, (a.roads / 'roads.xodr').read_text())
    print('Current navigation loaded', flush=True)
    rows = []
    a.output.mkdir(parents=True, exist_ok=True)
    for path in configs:
        config = read(path)
        area_id = config['area_id']
        source = path.parent
        out = a.output / source.name
        if (out / 'scenario.json').exists():
            raise ValueError('Use a new runtime output directory')
        routes = read(source / 'handoff/candidate_routes.json')
        original_routes = copy.deepcopy(routes)
        checks = []
        refresh_waypoints(routes, native, checks, a.xy_tolerance_m, transfer)
        for key, route in routes.items():
            missing = set(route['source_lane_ids']) - source_ids
            if missing:
                raise ValueError('Source task intervals absent from current native metadata: ' + str(sorted(missing)))
            # Keep source reference positions alongside waypoints sampled from the current OpenDRIVE.
            if transfer is not None and 'native_plan' in route.get('navigation_context', {}):
                context = route['navigation_context']
                context['previous_native_plan'] = context.pop('native_plan')
                context['native_plan'] = dict(waypoints=len(route['navigation_waypoints']), source_station_transfer=True,
                    scope='Same source-backed navigation context, native samples rebound through recorded source intervals.')
            route['height_revision'] = dict(map=a.map, navigation=str((a.roads / 'roads.xodr').resolve()),
                source_positions_retained=True,
                scope='Exact source reference positions and branch choices retained. Executable native records have current source-station identities, widths and elevations; physical execution remains pending.')
        info = read(source / 'handoff/map_info.json')
        info['inherited_runtime_evidence'] = info.pop('runtime_evidence', {})
        info['inherited_generation'] = info.pop('generation', {})
        info.update(map_version=a.map.rsplit('/', 1)[-1], carla_map_name=a.map,
            installed_package=str(Path(os.environ['CARLA_ROOT']) / 'Unreal/CarlaUE4/Content' / a.map.removeprefix('/Game/').split('/')[0]),
            runtime_evidence=dict(status='Pending current-map source views, physical context and driving checks.'),
            generation=dict(scene=str(a.scene.resolve()), conditions=str(a.conditions.resolve()),
                roads=str(a.roads.resolve()), retained_source_scenarios=str(a.source_scenarios.resolve()),
                revision='Recorded physical-height feedback; source Area, route branch, traffic timestamp and actor layers retained.'))
        config.update(map=a.map, parked_context=False, native_context_status='pending_current_support_and_spawn')
        config['inherited_scenario'] = str(path.resolve())
        config['current_navigation'] = str((a.roads / 'roads.xodr').resolve())
        signals = read(source / 'handoff/signal_programs.json')
        if a.signals:
            current_signals = read(a.signals)
            if current_signals['source_programs'] != signals['source_programs']:
                raise ValueError('Current binding manifest changed source signal programs')
            signals = dict(signals, bindings=current_signals['bindings'], current_binding_source=str(a.signals),
                scope='Retained source programs and current source-station native approach bindings. Actual runtime checks remain separate.')
        for name, data in [('map_info.json', info), ('candidate_routes.json', routes),
                           ('area_index.json', read(source / 'handoff/area_index.json')),
                           ('signal_programs.json', signals)]:
            save(out / 'handoff' / name, data)
        save(out / 'scenario.json', config)
        sample = read(source / 'source_joint_state.json')
        save(out / 'source_joint_state.json', sample)
        row = dict(area_id=area_id, source_scenario=str(path), map=a.map,
            source_joint_state_retained=read(out / 'source_joint_state.json') == sample,
            initial_distribution_retained=config['initial_distribution_source'] == read(path)['initial_distribution_source'],
            source_route_ids_retained=all(routes[k]['source_lane_ids'] == original_routes[k]['source_lane_ids'] for k in routes),
            source_route_positions_retained=all(routes[k]['positions_carla_m'] == original_routes[k]['positions_carla_m'] for k in routes),
            signal_programs_retained=signals['source_programs'] == read(source / 'handoff/signal_programs.json')['source_programs'],
            current_signal_bindings=str(a.signals) if a.signals else None,
            previous_native_roads=str(a.previous_roads) if a.previous_roads else None,
            source_station_transfers=sum(c['source_station_transfer'] is not None for c in checks),
            changed_native_identities=sum(c['source_station_transfer'] is not None and any(c['source_station_transfer'][k] != c['source_station_transfer']['previous_'+k] for k in ('road_id','lane_id','s_m')) for c in checks),
            max_source_centerline_error_m=max([c['source_station_transfer']['current_source_centerline_error_m'] for c in checks if c['source_station_transfer'] is not None and c['source_station_transfer']['current_source_centerline_error_m'] is not None]+[0.]),
            max_source_envelope_outside_m=max([c['source_station_transfer']['current_source_envelope_outside_m'] for c in checks if c['source_station_transfer'] is not None and c['source_station_transfer']['current_source_envelope_outside_m'] is not None]+[0.]),
            source_geometry_conflict_records=sum(c['source_station_transfer'] is not None and c['source_station_transfer'].get('source_geometry_supported', True) is False for c in checks),
            retained_compiled_bridge_records=sum(c['source_station_transfer'] is not None and c['source_station_transfer'].get('source_geometry_support') == 'retained_compiled_bridge' for c in checks),
            old_native_xy_equality_required=transfer is None,
            max_native_width_change_m=max([abs(c['width_change_m']) for c in checks if c['width_change_m'] is not None]+[0.]),
            waypoint_records=len(checks), max_xy_change_m=max(c['xy_change_m'] for c in checks),
            max_z_change_m=max(c['z_change_m'] for c in checks), native_context_prepared=False)
        save(out / 'native_waypoint_transfer.json', dict(records=checks, previous_roads=str(a.previous_roads) if a.previous_roads else None,
            scope='Same source route and station, current compiled native identity/width/elevation. Native physical execution remains pending.'))
        rows.append(row)
        print(row, flush=True)
        save(a.output / 'review.json', dict(configured_areas=len(configs), prepared_areas=len(rows),
            records=rows, source_time_and_routes_retained=True, native_runtime_passed=False,
            scope='Current source-station navigation handoffs only. Previous native parked fits are not copied. Source joint states and branches remain exact; native execution pending.'))
    if len({row['area_id'] for row in rows}) != len(configs):
        raise ValueError('Repeated source Area identity')


if __name__ == '__main__':
    main()
