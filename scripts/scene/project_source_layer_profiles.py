"""Project a selected inferred source chain below observed road support."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import osqp
import shapely
from scipy.sparse import coo_matrix

from map_generation.av2_surfaces import mesh_triangles
from map_generation.junction_surfaces import query_surface


def project(scene, roads, selection):
    lanes = {row['id']: row for row in scene['fusion_map']['lanes']}
    keys = selection['source_lane_ids']
    chosen = [lanes[key] for key in keys]
    if any(row['source'] == 'av2' for row in chosen):
        raise ValueError('Observed AV2 lanes cannot be lowered by inferred layer projection')
    for a, b in zip(chosen[:-1], chosen[1:]):
        if b['id'] not in a['successors']:
            raise ValueError('Selected source chain is not a directed continuation')
    arrays, stations, variables, all_xyz = [], [], [], []
    for index, lane in enumerate(chosen):
        old = np.asarray(lane['centerline'], float)
        old_s = np.r_[0., np.linalg.norm(np.diff(old[:, :2], axis=0), axis=1).cumsum()]
        grid = np.arange(0., old_s[-1], .5)
        # Keep source knots and merge numerically coincident solve knots.
        grid = grid[np.min(abs(grid[:, None] - old_s[None, :]), axis=1) > 1e-6]
        at = np.unique(np.r_[old_s, grid])
        xyz = np.column_stack([np.interp(at, old_s, old[:, j]) for j in range(3)])
        if index and np.linalg.norm(xyz[0, :2] - arrays[-1][-1, :2]) > .002:
            raise ValueError('Selected chain has a noncoincident planar endpoint')
        # Retain all original XY vertices, including coincident source
        # knots, while giving one physical position one height variable.
        ids = []
        for j, point in enumerate(xyz):
            shared = (index and j == 0) or (j > 0 and at[j]-at[j-1] <= 1e-6)
            if shared:
                ids.append(len(all_xyz)-1)
            else:
                ids.append(len(all_xyz))
                all_xyz.append(point.tolist())
        ids = np.asarray(ids, dtype=int)
        arrays.append(xyz); stations.append(at); variables.append(ids)
    original = np.asarray(all_xyz)
    upper_ids = set(selection['overlying_observed_sources'])
    observed = [lane for lane in scene['lanes'] if lane['id'] in upper_ids]
    if len(observed) != len(upper_ids):
        raise ValueError('Current observed upper lane identities are incomplete')
    domain = shapely.union_all([shapely.Polygon(lane['left_boundary'] + lane['right_boundary'][::-1])
                                for lane in observed])
    fine = [m for m in roads['meshes'] if m.get('kind') in
            ('av2_boundary_surface', 'av2_grounded_junction')]
    triangles = np.concatenate([mesh_triangles(m) for m in fine])
    tree = shapely.STRtree(shapely.polygons(triangles[:, :, :2]))
    settings = selection['construction_conditions']
    separation = settings['clear_headroom_m'] + settings['overlying_surface_thickness_allowance_m']
    objective_rows, objective_cols, objective_values, targets = [], [], [], []
    con_rows, con_cols, con_values, lower, upper = [], [], [], [], []
    def objective(pairs, target, weight):
        row = len(targets)
        for col, value in pairs:
            objective_rows.append(row); objective_cols.append(int(col)); objective_values.append(value * weight**.5)
        targets.append(target * weight**.5)
    def constraint(pairs, lo, hi):
        row = len(lower)
        for col, value in pairs:
            con_rows.append(row); con_cols.append(int(col)); con_values.append(value)
        lower.append(lo); upper.append(hi)
    for i, point in enumerate(original):
        objective([(i, 1.)], point[2], .01)
    for xyz, ss, ids in zip(arrays, stations, variables):
        ds = np.diff(ss)
        for a, b, length in zip(ids[:-1], ids[1:], ds):
            bound = settings['maximum_inferred_grade'] * length
            constraint([(a, -1.), (b, 1.)], -bound, bound)
        for a, b, c, left, right in zip(ids[:-2], ids[1:-1], ids[2:], ds[:-1], ds[1:]):
            if min(left, right) > 1e-6:
                objective([(a, 1/left), (b, -1/left-1/right), (c, 1/right)], 0., 10.)
    constraint([(0, 1.)], original[0, 2], original[0, 2])
    constraint([(len(original)-1, 1.)], original[-1, 2], original[-1, 2])
    constraints = []
    for lane, xyz, ids in zip(chosen, arrays, variables):
        tangents = np.gradient(xyz[:, :2], axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        normals = np.c_[-tangents[:, 1], tangents[:, 0]]
        for row, (point, normal, variable) in enumerate(zip(xyz, normals, ids)):
            transverse = point[:2] + np.linspace(-lane['width_m']/2, lane['width_m']/2, 9)[:, None] * normal
            inside = shapely.dwithin(domain, shapely.points(transverse), 1e-7)
            if not inside.any():
                continue
            heights = query_surface(triangles, transverse[inside], tree)
            ceiling = float(heights.min() - separation)
            constraint([(variable, 1.)], -np.inf, ceiling)
            constraints.append(dict(source_id=lane['id'], variable=int(variable), station_m=float(stations[keys.index(lane['id'])][row]),
                                    xy_m=point[:2].tolist(), lowest_upper_surface_m=float(heights.min()),
                                    maximum_lower_road_z_m=ceiling, samples=int(inside.sum())))
    if not constraints:
        raise ValueError('Selected underground chain has no measured upper surface constraints')
    n = len(original)
    design = coo_matrix((objective_values, (objective_rows, objective_cols)), shape=(len(targets), n)).tocsc()
    matrix = coo_matrix((con_values, (con_rows, con_cols)), shape=(len(lower), n)).tocsc()
    solver = osqp.OSQP()
    solver.setup(P=(design.T @ design).tocsc(), q=-design.T @ np.asarray(targets), A=matrix,
                 l=np.asarray(lower), u=np.asarray(upper), eps_abs=1e-7, eps_rel=1e-7,
                 max_iter=100000, polishing=True, verbose=False)
    solution = solver.solve()
    if solution.info.status_val != 1:
        raise RuntimeError('Source layer profile projection: ' + solution.info.status)
    result = deepcopy(scene)
    output_lanes = {lane['id']: lane for lane in result['fusion_map']['lanes']}
    records = []
    for lane, xyz, ss, ids in zip(chosen, arrays, stations, variables):
        changed = xyz.copy(); changed[:, 2] = solution.x[ids]
        output_lanes[lane['id']]['centerline'] = changed.tolist()
        output_lanes[lane['id']]['height_source'] = 'Explicit source layer with bounded local profile; observed outside ports fixed, unmeasured headroom recorded'
        old = np.asarray(lane['centerline'])
        polyline = shapely.LineString(old[:, :2])
        grade = np.abs(np.diff(changed[:, 2])) / np.diff(ss)
        records.append(dict(source_id=lane['id'], samples=len(xyz), max_grade=float(grade.max()),
                            maximum_height_change_m=float(abs(changed[:, 2]-xyz[:, 2]).max()),
                            xy_distance_to_original_polyline_m=float(shapely.distance(shapely.points(changed[:, :2]), polyline).max()),
                            original_endpoints=old[[0, -1]].tolist(), projected_endpoints=changed[[0, -1]].tolist()))
    for row in constraints:
        row['lower_road_z_m'] = float(solution.x[row['variable']])
        row['clear_headroom_m'] = row['lowest_upper_surface_m'] - row['lower_road_z_m'] - settings['overlying_surface_thickness_allowance_m']
    relations = [dict(a=selection['underground_source'], b=key, relation='below',
                      clear_headroom_m=settings['clear_headroom_m'],
                      clearance_status=settings['headroom_status']) for key in selection['overlying_observed_sources']]
    result.setdefault('physical_layer_relations', []).extend(relations)
    report = dict(method=__doc__, solver=solution.info.status, selected_sources=keys,
                  profiles=records, clearance_constraints=constraints,
                  minimum_clear_headroom_m=min(row['clear_headroom_m'] for row in constraints),
                  maximum_grade=max(row['max_grade'] for row in records),
                  original_AV2_geometry_unchanged=result['lanes'] == scene['lanes'],
                  outside_fusion_lanes_unchanged=all(output_lanes[k] == v for k, v in lanes.items() if k not in keys),
                  source_topology_and_permissions_unchanged=all(
                      {k:v for k,v in output_lanes[key].items() if k not in ('centerline','height_source')} ==
                      {k:v for k,v in lanes[key].items() if k not in ('centerline','height_source')} for key in keys),
                  fixed_outer_endpoint_height_error_m=float(max(abs(solution.x[0]-original[0, 2]),
                                                               abs(solution.x[-1]-original[-1, 2]))),
                  headroom_inference=settings['headroom_status'],
                  scope='Source-conditioned local height projection. Native elevations, physical meshes, terrain approaches and driving require dependent reconstruction.')
    report['passed'] = (report['minimum_clear_headroom_m'] >= settings['clear_headroom_m']-1e-5 and
                        report['maximum_grade'] <= settings['maximum_inferred_grade']+1e-5 and
                        report['fixed_outer_endpoint_height_error_m'] < 1e-5 and
                        report['original_AV2_geometry_unchanged'] and report['outside_fusion_lanes_unchanged'] and
                        report['source_topology_and_permissions_unchanged'])
    result['source_layer_profile_projection'] = report
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'roads', 'conditions', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    conditions = read(args.conditions)
    selected = next(row['native_construction']['settings']['selection'] for row in conditions['native_objects']
                    if row.get('native_construction', {}).get('kind') == 'local_source_layer_profile')
    result, report = project(read(args.scene), read(args.roads), selected)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'scene.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (args.output/'review.json').write_text(json.dumps(report, indent=2)+'\n')
    print({key:report[key] for key in ['passed','solver','selected_sources','minimum_clear_headroom_m',
                                     'maximum_grade','fixed_outer_endpoint_height_error_m']}, flush=True)
    if not report['passed']:
        raise SystemExit('Local source-layer projection requires review')


if __name__ == '__main__':
    main()
