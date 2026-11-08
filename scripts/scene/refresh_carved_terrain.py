"""Refresh a completed XY terrain carve only where road ownership changed."""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely

from map_generation.mesh_geometry import carve_terrain, spatial_union
from scripts.scene.build_assets import BOOLEAN_GRID_M, plane_heights
from scripts.scene.prepare_city_ground import at_grade_roads


def lifted_triangles(parts, parents, source):
    pieces, owner = shapely.get_parts(
        shapely.constrained_delaunay_triangles(parts), return_index=True)
    positive = shapely.area(pieces) > 1e-12
    pieces, owner = pieces[positive], owner[positive]
    xy = shapely.get_coordinates(pieces).reshape(-1, 4, 2)[:, :3]
    tri = source[parents[owner]]
    z = plane_heights(np.repeat(tri, 3, axis=0), xy.reshape(-1, 2)).reshape(-1, 3)
    result = np.concatenate([xy, z[:, :, None]], axis=2)
    reverse = np.cross(result[:, 1] - result[:, 0],
                       result[:, 2] - result[:, 0])[:, 2] < 0
    result[reverse] = result[reverse, ::-1]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'source-terrain-scene', 'roads', 'carved', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--separate-level-buffer-m', type=float, default=3.)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    scene, roads = read(a.scene), read(a.roads)
    ground, _, _, _ = at_grade_roads(scene, roads, a.separate_level_buffer_m)
    triangles = np.concatenate([np.asarray(m['vertices'])[m['faces']]
                                for m in ground['meshes']])
    footprint = spatial_union(shapely.polygons(triangles[:, :, :2]),
                              grid_size=BOOLEAN_GRID_M)
    footprint = shapely.set_precision(shapely.set_precision(
        footprint, BOOLEAN_GRID_M), 0)
    previous = shapely.from_wkt(a.carved.with_name('road_footprint.wkt').read_text())
    added, removed = footprint.difference(previous), previous.difference(footprint)
    changed = added.union(removed)
    # The millimetre collar creates a common retained/source DEM boundary.
    # It affects only which original DEM triangles are reused, not road XY.
    region = changed.buffer(.001)
    print(json.dumps(dict(added_road_area_m2=added.area,
                          removed_road_area_m2=removed.area,
                          terrain_refresh_area_m2=region.area)), flush=True)
    del triangles, ground, roads
    stored = np.load(a.carved)
    vertices, faces = stored['vertices'], stored['faces']
    old_tri = vertices[faces]
    old_polys = shapely.polygons(old_tri[:, :, :2])
    print('Clipping retained terrain around changed road ownership', flush=True)
    cut = carve_terrain(old_polys, region)
    print('Retained terrain clipped', flush=True)
    clipped = cut['clipped']
    xy = shapely.get_coordinates(clipped).reshape(-1, 4, 2)[:, :3]
    z = plane_heights(np.repeat(old_tri[cut['parents']], 3, axis=0),
                      xy.reshape(-1, 2)).reshape(-1, 3)
    outside = np.concatenate([xy, z[:, :, None]], axis=2)
    reverse = np.cross(outside[:, 1] - outside[:, 0],
                       outside[:, 2] - outside[:, 0])[:, 2] < 0
    outside[reverse] = outside[reverse, ::-1]
    del old_polys, old_tri
    source_scene = read(a.source_terrain_scene)
    if source_scene['coordinates']['origin_city_xyz'] != scene['coordinates']['origin_city_xyz']:
        raise ValueError('Original DEM and current road scene use different origins')
    source = source_scene['terrain']
    canonical = np.asarray(source['vertices'])[source['faces']]
    del source_scene, source, scene
    projected = np.cross(canonical[:, 1] - canonical[:, 0],
                         canonical[:, 2] - canonical[:, 0])[:, 2]
    canonical = canonical[abs(projected) > 1e-10]
    polys = shapely.polygons(canonical[:, :, :2])
    ids = shapely.STRtree(polys).query(region, predicate='intersects')
    print('Restoring canonical DEM planes', len(ids), flush=True)
    parts = shapely.intersection(polys[ids], region)
    parts = shapely.make_valid(parts, method='structure', keep_collapsed=False)
    positive = shapely.area(parts) > 1e-12
    parts, ids = parts[positive], ids[positive]
    # Use tiled subtraction and retain canonical DEM parents for height interpolation.
    print('Subtracting current roads from local canonical DEM parts', len(parts), flush=True)
    source_cut = carve_terrain(parts, footprint)
    retained = ~source_cut['affected']
    restored_parts = np.concatenate([parts[retained], source_cut['clipped']])
    restored_parents = np.concatenate([ids[retained], ids[source_cut['parents']]])
    restored = lifted_triangles(restored_parts, restored_parents, canonical)
    desired_area = float(shapely.area(restored_parts).sum())
    actual_area = float(shapely.area(shapely.polygons(restored[:, :, :2])).sum())
    # Compute road overlap from each triangle's area minus its retained remnant.
    print('Checking restored DEM against local road partitions', len(restored), flush=True)
    restored_polys = shapely.polygons(restored[:, :, :2])
    road_cut = carve_terrain(restored_polys, footprint)
    overlap = max(0., float(shapely.area(restored_polys[road_cut['affected']]).sum()
                           - shapely.area(road_cut['clipped']).sum()))
    print('Restored DEM road overlap', overlap, flush=True)
    if abs(actual_area - desired_area) > 1e-6 or overlap > 1e-6:
        raise ValueError('Incremental source terrain changes requested area or overlaps roads')
    appended = np.concatenate([outside, restored])
    appended_faces = np.arange(appended.size // 3).reshape(-1, 3) + len(vertices)
    new_vertices = np.concatenate([vertices, appended.reshape(-1, 3)])
    new_faces = np.concatenate([faces[~cut['affected']], appended_faces])
    used, indices = np.unique(new_faces, return_inverse=True)
    new_vertices, new_faces = new_vertices[used], indices.reshape(-1, 3)
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez(a.output / 'carved_top.npz', vertices=new_vertices, faces=new_faces)
    (a.output / 'road_footprint.wkt').write_text(
        shapely.to_wkt(footprint, rounding_precision=-1))
    report = dict(previous_carve=str(a.carved), input_scene=str(a.scene), current_roads=str(a.roads),
        canonical_terrain_scene=str(a.source_terrain_scene),
        added_road_area_m2=added.area, removed_road_area_m2=removed.area,
        terrain_refresh_area_m2=region.area,
        unchanged_terrain_faces=int((~cut['affected']).sum()),
        replaced_terrain_faces=int(cut['affected'].sum()),
        terrain_result_triangles=len(new_faces), terrain_result_vertices=len(new_vertices),
        terrain_road_overlap_m2=overlap,
        terrain_clip_area_error_m2=abs(actual_area - desired_area),
        retained_boundary_clip_area_error_m2=cut['area_error_m2'],
        restored_source_clip_area_error_m2=source_cut['area_error_m2'],
        restored_source_local_overlap_m2=source_cut['overlap_m2'],
        road_footprint_area_m2=footprint.area,
        scope='Updated XY carve. Unaffected saved DEM faces are exact; changed regions use original DEM planes. Ground heights, closures and foundations still require fitting.')
    (a.output / 'carved_top_review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
