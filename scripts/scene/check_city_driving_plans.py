"""Check selected Area traversals against one loaded complete native map."""
import argparse
import json
import os
from pathlib import Path
import sys


def source_routes(directory):
    """Read prepared routes by their data, excluding reports in the same folder."""
    rows = []
    for path in sorted(directory.glob('*.json')):
        source = json.loads(path.read_text())
        if not isinstance(source, dict) or 'area_id' not in source or 'route' not in source:
            continue
        if not source['route'].get('lane_ids') or not source['route'].get('positions'):
            raise ValueError('Empty source route: ' + str(path))
        rows.append((path, source))
    if not rows:
        raise ValueError('No prepared source routes found in ' + str(directory))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'roads', 'routes', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--native-planner', action='store_true')
    a = p.parse_args()
    routes = source_routes(a.routes)
    sys.path.insert(0, str(Path(os.environ['CARLA_ROOT']) / 'PythonAPI/carla'))
    import carla
    import networkx as nx
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from shapely.geometry import LineString
    from scripts.scene.drive_route import plan_route
    scene = json.loads(a.scene.read_text())
    lanes = {r['id']: r for r in scene['fusion_map']['lanes']}
    mapping = json.loads((a.roads / 'road_lane_ids.json').read_text())
    native = carla.Map(scene['map_id'], (a.roads / 'roads.xodr').read_text())
    print('Complete native map loaded', flush=True)
    if a.native_planner:
        from map_generation.native_route_planner import NativeRoutePlanner
        planner = NativeRoutePlanner(native, sampling_resolution=1.)
    else:
        planner = GlobalRoutePlanner(native, sampling_resolution=1.)
    print('Complete native planner ready', flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    for path, source in routes:
        route = source['route']
        try:
            _, plan, _, _, _, detail = plan_route(native, route, lanes, mapping,
                                                 LineString(route['positions']), planner=planner,
                                                 vehicle_class=source.get('mode', 'passenger'))
            points = [dict(road_id=wp.road_id, lane_id=wp.lane_id, s_m=wp.s,
                position_carla_m=[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z],
                road_option=option.name) for wp, option in plan]
            (a.output / path.name).write_text(json.dumps(dict(source=source, native_plan=points,
                review=detail, actual_driving='pending'), separators=(',', ':')) + '\n')
            record = dict(area_id=source['area_id'], passed=True, waypoints=len(plan),
                          max_plan_offset_m=detail['max_plan_offset_m'])
        except (ValueError, RuntimeError, KeyError, nx.NetworkXException) as error:
            record = dict(area_id=source['area_id'], passed=False, error=str(error))
        records.append(record)
        (a.output / 'review.json').write_text(json.dumps(dict(records=records,
            scope='Source-selected routes on the complete native graph. Failed plans remain feedback; successful plans still require physical driving.'), indent=2) + '\n')
        print(record, flush=True)


if __name__ == '__main__':
    main()
