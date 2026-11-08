"""Measure blocking road-collision heights at pavement interior samples."""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely

from map_generation.av2_surfaces import mesh_triangles
from map_generation.junction_surfaces import query_surface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roads', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True, help='Patch build report naming the new mesh IDs')
    p.add_argument('--expected-map', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import carla

    roads = json.loads(a.roads.read_text())
    report = json.loads(a.report.read_text())
    ids = {r['id'] for r in report['patches']}
    client = carla.Client('127.0.0.1', a.port)
    client.set_timeout(30)
    world = client.get_world()
    if world.get_map().name.removeprefix('/Game/') != a.expected_map.removeprefix('/Game/'):
        raise ValueError('Loaded world differs from requested map')
    if list(world.get_actors().filter('vehicle.*')):
        raise ValueError('Run the patch probe before spawning vehicles')
    rows, checked = [], []
    for mesh in roads['meshes']:
        if mesh['id'] not in ids:
            continue
        tri = mesh_triangles(mesh)
        poly = shapely.union_all(shapely.polygons(tri[:, :, :2]))
        xmin, ymin, xmax, ymax = poly.bounds
        grid = np.array([(x, y) for x in np.arange(xmin + .1, xmax, .5)
                         for y in np.arange(ymin + .1, ymax, .5)]).reshape(-1, 2)
        grid = grid[shapely.contains_xy(poly, grid[:, 0], grid[:, 1])]
        # At least one interior sample per exported triangle also covers
        # centimetre-wide omissions missed by a regular XY grid.
        xy = np.unique(np.concatenate((grid, tri[:, :, :2].mean(axis=1))), axis=0)
        heights = query_surface(tri, xy)
        for point, expected in zip(xy, heights):
            hit = world.project_point(carla.Location(x=float(point[0]), y=float(-point[1]), z=float(expected) + 1),
                                      carla.Vector3D(z=-1), 2.)
            row = {'patch': mesh['id'], 'xy_enu_m': point.tolist(), 'expected_z_m': float(expected), 'hit': None}
            if hit:
                row.update(hit={'xyz_carla_m': [hit.location.x, hit.location.y, hit.location.z],
                                'label': str(hit.label)}, error_m=hit.location.z - float(expected))
            row['passed'] = hit is not None and str(hit.label) == 'Roads' and abs(row['error_m']) <= .002
            rows.append(row)
        checked.append(mesh['id'])
    errors = [abs(r['error_m']) for r in rows if 'error_m' in r]
    result = {'map': world.get_map().name, 'patch_ids': checked, 'sample_count': len(rows),
        'failed_count': sum(not r['passed'] for r in rows),
        'max_abs_error_m': max(errors) if errors else None,
        'passed': bool(rows) and set(checked) == ids and all(r['passed'] for r in rows),
        'method': '0.5 m interior grid plus every exported triangle centroid; first blocking SensorTrace hit, Roads label and 2 mm height tolerance.',
        'scope': 'New patch physical surfaces only; not full-region drivability, wheel-contact testing or visual fidelity.',
        'samples': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))
    if not result['passed']:
        raise RuntimeError('Native patch collision check failed; see ' + str(a.output))


if __name__ == '__main__':
    main()
