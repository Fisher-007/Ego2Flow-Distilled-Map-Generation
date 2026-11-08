"""Project source-selected bridge profiles above retained lower road surfaces."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import shapely

from map_generation.av2_surfaces import mesh_triangles
from map_generation.junction_surfaces import query_surface
from scripts.scene.smooth_city_road_profiles import smooth


def source_ids(mesh, by_id):
    ids = {mesh.get('source_lane_id')} | set(mesh.get('source_lane_ids', []))
    if mesh.get('kind') == 'connection_closure':
        ids.update(mesh[side + '_source_lane_id'] if side + '_source_lane_id' in mesh
                   else by_id[mesh[side + '_mesh_id']].get('source_lane_id')
                   for side in ('from', 'to'))
    return ids



def record_layer_relations(scene, selection):
    lanes = {lane['id']:lane for lane in scene['fusion_map']['lanes']}
    relations = scene.setdefault('physical_layer_relations', [])
    known = {(r['a'],r['b']) for r in relations}
    added = []
    for upper in selection['bridge_source_lane_ids']:
        upper_line = shapely.LineString(np.asarray(lanes[upper]['centerline'])[:, :2])
        for lower in selection['lower_source_lane_ids']:
            lower_line = shapely.LineString(np.asarray(lanes[lower]['centerline'])[:, :2])
            if not upper_line.intersects(lower_line) or (upper,lower) in known:
                continue
            relation = dict(a=upper, b=lower, relation='above',
                clear_headroom_m=selection['clear_headroom_m'],
                clearance_status=selection['headroom_status'],
                terrain_carving_excluded_source_ids=selection['clearance_source_lane_ids'],
                at_grade_source_ids=selection['lower_source_lane_ids'],
                elevated_continuation_evidence=selection['bridge_continuation_chains'],
                surface_role_evidence=selection['evidence'])
            relations.append(relation);known.add((upper,lower));added.append([upper,lower])
    return added


def project(scene, roads, selection):
    original = {l['id']: l for l in scene['fusion_map']['lanes']}
    editable = set(selection['source_lane_ids'])
    if any(original[k]['source'] == 'av2' for k in editable):
        raise ValueError('Measured AV2 geometry is not an inferred bridge variable')
    lower_sources = set(selection['lower_source_lane_ids'])
    if editable & lower_sources:
        raise ValueError('The retained lower road cannot be a bridge projection variable')
    neighbors = set()
    for key, lane in original.items():
        if key in editable:
            neighbors.update(t for t in lane['successors'] if t in original)
        elif editable.intersection(lane['successors']):
            neighbors.add(key)
    local_ids = editable | neighbors
    local = [deepcopy(original[k]) for k in sorted(local_ids)]
    for lane in local:
        lane['successors'] = [k for k in lane['successors'] if k in local_ids]
    local_scene = dict(coarse_map=dict(lanes=local))
    reference = deepcopy(local_scene)
    by_mesh = {m['id']: m for m in roads['meshes']}
    lower_meshes = [m for m in roads['meshes']
                    if source_ids(m, by_mesh) & lower_sources
                    and m.get('kind') != 'inferred_shared_road_ground']
    triangles = np.concatenate([mesh_triangles(m) for m in lower_meshes])
    polygons = shapely.polygons(triangles[:, :, :2])
    tree = shapely.STRtree(polygons)
    domain = shapely.union_all(polygons)
    separation = selection['clear_headroom_m'] + selection['surface_thickness_allowance_m']
    bounds = []
    for key in selection['clearance_source_lane_ids']:
        lane = original[key]
        p = np.asarray(lane['centerline'])
        ss = np.r_[0., np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).cumsum()]
        at = np.unique(np.r_[ss, np.arange(0, ss[-1], selection['knot_spacing_m'])])
        for station in at:
            xy = np.array([np.interp(station, ss, p[:, j]) for j in (0, 1)])
            lo, hi = max(0., station-.25), min(ss[-1], station+.25)
            tangent = np.array([np.interp(hi, ss, p[:, j])-np.interp(lo, ss, p[:, j])
                                for j in (0, 1)])
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            query = xy + np.linspace(-lane['width_m']/2, lane['width_m']/2, 9)[:, None]*normal
            inside = shapely.dwithin(domain, shapely.points(query), 1e-7)
            if not inside.any():
                continue
            z = query_surface(triangles, query[inside], tree)
            floor = float(z.max())
            bounds.append(dict(lane_id=key, station_m=float(station),
                minimum_z_m=floor+separation, retained_lower_surface_z_m=floor,
                transverse_samples=int(inside.sum()), xy_m=xy.tolist()))
    if not bounds:
        raise ValueError('Selected bridge has no retained lower surface crossing')
    anchors = []
    for lane in local:
        if lane['id'] not in editable:
            p = np.asarray(lane['centerline'])
            length = float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())
            anchors.extend([dict(coarse_lane_id=lane['id'], station_m=0., target_z_m=float(p[0, 2])),
                            dict(coarse_lane_id=lane['id'], station_m=length, target_z_m=float(p[-1, 2]))])
    fit = smooth(local_scene, anchors, spacing=selection['knot_spacing_m'],
        max_grade=selection['maximum_inferred_grade'], editable=editable,
        reference=reference, shared_carriageway_profile=True,
        carriageway_groups=selection['carriageway_groups'],
        height_bounds=bounds, include_solve_knots=True,
        source_prior_weights=selection.get('source_prior_weights'))
    changed = {l['id']: l for l in local_scene['coarse_map']['lanes']}
    result = deepcopy(scene)
    output = {l['id']: l for l in result['fusion_map']['lanes']}
    profiles = []
    for key in sorted(editable):
        p = np.asarray(changed[key]['centerline'])
        old = np.asarray(original[key]['centerline'])
        line = shapely.LineString(old[:, :2])
        station = np.r_[0., np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).cumsum()]
        ds = np.diff(station)
        valid = ds > 1e-6
        output[key]['centerline'] = p.tolist()
        output[key]['height_source'] = 'Source-supported upper bridge relation with explicit inferred clearance; retained outer profiles fixed'
        profiles.append(dict(source_id=key, samples=len(p),
            max_grade=float((abs(np.diff(p[:, 2]))[valid]/ds[valid]).max()),
            source_xy_distance_m=float(shapely.distance(shapely.points(p[:, :2]), line).max())))
    for row in bounds:
        p = np.asarray(output[row['lane_id']]['centerline'])
        ss = np.r_[0., np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).cumsum()]
        row['projected_upper_z_m'] = float(np.interp(row['station_m'], ss, p[:, 2]))
        row['clear_headroom_m'] = row['projected_upper_z_m']-row['retained_lower_surface_z_m']-selection['surface_thickness_allowance_m']
    unchanged = all(output[k] == v for k, v in original.items() if k not in editable)
    permissions = all({k:v for k,v in output[key].items() if k not in ('centerline', 'height_source')}
                      == {k:v for k,v in original[key].items() if k not in ('centerline', 'height_source')}
                      for key in editable)
    report = dict(method=__doc__, selection=selection, solver=fit,
        lower_physical_mesh_ids=[m['id'] for m in lower_meshes],
        profiles=profiles, clearance_constraints=bounds,
        minimum_clear_headroom_m=min(r['clear_headroom_m'] for r in bounds),
        maximum_grade=max(r['max_grade'] for r in profiles),
        maximum_source_xy_distance_m=max(r['source_xy_distance_m'] for r in profiles),
        original_AV2_geometry_unchanged=result['lanes'] == scene['lanes'],
        outside_source_lanes_exact=unchanged,
        source_topology_and_permissions_unchanged=permissions,
        scope='Local inferred source-profile projection. Physical meshes, native elevation, shared approach floors and actual driving remain dependent checks.')
    report['passed'] = (report['minimum_clear_headroom_m'] >= selection['clear_headroom_m']-1e-4
        and report['maximum_grade'] <= selection['maximum_inferred_grade']+1e-4
        and report['maximum_source_xy_distance_m'] < 1e-6
        and report['original_AV2_geometry_unchanged'] and unchanged and permissions)
    report['recorded_layer_crossings'] = record_layer_relations(result, selection)
    result['source_bridge_profile_projection'] = report
    return result, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'roads', 'conditions', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    c = read(a.conditions)
    selection = next(r['native_construction']['settings'] for r in c['native_objects']
                     if r.get('native_construction', {}).get('kind') == 'source_bridge_profile')
    result, report = project(read(a.scene), read(a.roads), selection)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (a.output/'review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed', 'minimum_clear_headroom_m',
        'maximum_grade', 'maximum_source_xy_distance_m', 'original_AV2_geometry_unchanged',
        'outside_source_lanes_exact', 'source_topology_and_permissions_unchanged']}, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit('Source bridge profile requires feedback')


if __name__ == '__main__':
    main()
