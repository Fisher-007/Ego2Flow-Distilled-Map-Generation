"""Reconstruct selected shared source-road heights while preserving physical contacts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import shapely
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from map_generation.av2_surfaces import mesh_triangles
from map_generation.junction_surfaces import query_surface
from map_generation.shared_road_ground import profile_height_conditions, surface_laplacian


def reconstruct(scene, geometry, conditions):
    lanes = {l['id']: l for l in scene['lanes'] + scene.get('av2_context', []) + scene['fusion_map']['lanes']}
    result = deepcopy(geometry)
    meshes = {m['id']: m for m in result['meshes']}
    components = {c['id']: c for c in result['components']}
    records = []
    for selection in conditions['selections']:
        key = selection['mesh_id']
        mesh, component = meshes[key], components[key]
        vertices = np.asarray(mesh['vertices'], float)
        faces = np.asarray(mesh['faces'], int)
        xy, before = vertices[:, :2], vertices[:, 2].copy()
        profiles = [dict(source_id=sid, side=side, points_enu_m=lanes[sid][side])
                    for sid in mesh['source_lane_ids'] for side in ('left_boundary', 'right_boundary')]
        support, target = profile_height_conditions(xy, profiles, selection['profile_radius_m'])
        stiffness, mass = surface_laplacian(xy, faces)
        anchor = mass * (support > 0) / selection['profile_anchor_length_m']**2
        system = stiffness + diags(anchor)
        fixed = np.zeros(len(xy), bool)
        points = shapely.points(xy)
        for row in component['interfaces']:
            fixed |= shapely.dwithin(points, shapely.LineString(np.asarray(row['cap_xyz_m'])[:, :2]), 1e-6)
        free = ~fixed
        rhs = anchor[free] * target[free] - system[free][:, fixed] @ before[fixed]
        vertices[free, 2] = spsolve(system[free][:, free].tocsc(), rhs)
        if not np.isfinite(vertices).all():
            raise RuntimeError('Source-road height system is not finite: ' + key)
        mesh['vertices'] = vertices.tolist()
        mesh['physical_role'] = 'One reconstructed physical source-road floor; raw AV2 boundary observations retained separately'
        mesh['road_height_reconstruction'] = selection
        tri = vertices[faces]
        normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        slopes = np.linalg.norm(normal[:, :2], axis=1) / abs(normal[:, 2])
        area = abs(normal[:, 2])/2
        order = np.argsort(slopes)
        cdf = area[order].cumsum()/area.sum()
        interfaces = []
        for row in component['interfaces']:
            cap = np.asarray(row['cap_xyz_m'])
            t = np.linspace(0, 1, int(np.ceil(np.linalg.norm(cap[1, :2]-cap[0, :2])/.1))+1)
            sample = cap[0] + t[:, None]*(cap[1]-cap[0])
            old_z = query_surface(np.asarray(geometry['meshes'][list(meshes).index(key)]['vertices'])[faces], sample[:, :2], allow_missing=True)
            new_z = query_surface(tri, sample[:, :2], allow_missing=True)
            valid = np.isfinite(old_z) & np.isfinite(new_z)
            interfaces.append(dict(source_lane_id=row['source_lane_id'], samples=len(t),
                                   finite_samples=int(valid.sum()),
                                   max_contact_change_m=float(abs(new_z[valid]-old_z[valid]).max(initial=0)),
                                   max_retained_contact_gap_m=float(abs(new_z[valid]-sample[valid, 2]).max(initial=0))))
        source_residuals = []
        domain = shapely.union_all(shapely.polygons(tri[:, :, :2]))
        tree = shapely.STRtree(shapely.polygons(tri[:, :, :2]))
        for sid in mesh['source_lane_ids']:
            p = np.asarray(lanes[sid]['centerline'])
            ss = np.r_[0, np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).cumsum()]
            at = np.unique(np.r_[ss, np.arange(0, ss[-1], .2)])
            sample = np.column_stack([np.interp(at, ss, p[:, j]) for j in range(3)])
            sample = sample[shapely.covers(domain, shapely.points(sample[:, :2]))]
            if not len(sample):
                continue
            actual = query_surface(tri, sample[:, :2], tree)
            residual = actual-sample[:, 2]
            source_residuals.append(dict(source_id=sid, samples=len(sample),
                abs_physical_minus_virtual_source_z_quantiles_m=np.quantile(abs(residual), [.5,.95,1]).tolist()))
        record = dict(mesh_id=key, selection=selection, profiles=len(profiles), vertices=len(vertices),
            triangles=len(faces), fixed_contact_vertices=int(fixed.sum()),
            unsupported_soft_profile_vertices=int((support==0).sum()),
            maximum_height_change_m=float(abs(vertices[:, 2]-before).max()),
            original_height_quantiles_m=np.quantile(before,[0,.05,.5,.95,1]).tolist(),
            reconstructed_height_quantiles_m=np.quantile(vertices[:,2],[0,.05,.5,.95,1]).tolist(),
            area_weighted_surface_slope_quantiles=np.interp([.5,.95,.99,1],cdf,slopes[order]).tolist(),
            maximum_fixed_vertex_height_change_m=float(abs(vertices[fixed,2]-before[fixed]).max(initial=0)),
            interfaces=interfaces, source_center_residuals=source_residuals,
            method='Linear triangle FEM with area-weighted soft current AV2 boundary profiles and fixed actual retained road contacts. Original mesh XY and faces unchanged.',
            uncertainty='Shared physical reconstruction resolves incompatible source layers; it is not new surveyed ground.')
        component['road_height_reconstruction'] = record
        records.append(record)
        print(dict(mesh_id=key,max_height_change_m=record['maximum_height_change_m'],
                   fixed_contact_vertices=int(fixed.sum()),
                   area_weighted_slopes=record['area_weighted_surface_slope_quantiles'],
                   max_contact_change_m=max((x['max_contact_change_m'] for x in interfaces),default=0)),flush=True)
    selected = {r['mesh_id'] for r in conditions['selections']}
    outside_unchanged = all(a==b for a,b in zip(geometry['meshes'],result['meshes']) if a['id'] not in selected)
    xy_faces_unchanged = all(a['faces']==b['faces'] and
        np.array_equal(np.asarray(a['vertices'])[:,:2],np.asarray(b['vertices'])[:,:2])
        for a,b in zip(geometry['meshes'],result['meshes']))
    report = dict(method=__doc__, selections=conditions, components=records,
                  outside_components_unchanged=outside_unchanged, all_xy_and_faces_unchanged=xy_faces_unchanged,
                  source_scene_unchanged=True, native_driving_height_projection='pending dependent physical assembly')
    report['reconstruction_checks_passed'] = outside_unchanged and xy_faces_unchanged and all(
        r['maximum_fixed_vertex_height_change_m']==0 and
        max((q['max_contact_change_m'] for q in r['interfaces']),default=0)<1e-6 for r in records)
    result['source_road_height_reconstruction'] = report
    return result, report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','geometry','conditions','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    read=lambda path:json.loads(path.read_text())
    geometry,report=reconstruct(read(a.scene),read(a.geometry),read(a.conditions))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'ground_geometry.json').write_text(json.dumps(geometry,separators=(',',':'))+'\n')
    (a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(dict(reconstruction_checks_passed=report['reconstruction_checks_passed'],
               selected_components=len(report['components'])),flush=True)
    if not report['reconstruction_checks_passed']:
        raise SystemExit('Source-road physical reconstruction requires review')


if __name__=='__main__':
    main()
