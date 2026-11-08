"""Exact polygon unions grouped spatially for large physical city meshes."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely


def compact_vertices(mesh):
    """Share identical XYZ vertices without changing triangle coordinates."""
    vertices = np.asarray(mesh['vertices'])
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    mesh['vertices'] = unique.tolist()
    mesh['faces'] = inverse[np.asarray(mesh['faces'])].tolist()
    return dict(input_vertices=len(vertices), shared_vertices=len(unique),
                method='Exact XYZ equality; original faces and coordinates retained')


def boundary_segments(boundary):
    coords, ids = shapely.get_coordinates(shapely.get_parts(boundary), return_index=True)
    return shapely.linestrings(np.stack((coords[:-1], coords[1:]), axis=1)[ids[:-1] == ids[1:]])


def spatial_union(polygons, tile_m=100., workers=8, grid_size=None):
    """Union complete polygons per tile, then union tile results without clipping."""
    if grid_size is not None:
        # Normalize query footprints on the Boolean grid before noding coincident boundaries.
        polygons = shapely.set_precision(polygons, grid_size)
        polygons = polygons[~shapely.is_empty(polygons)]
    if len(polygons) < 250000:
        return shapely.union_all(polygons, grid_size=grid_size)
    cells = np.floor(shapely.bounds(polygons)[:, :2] / tile_m).astype(np.int64)
    cells -= cells.min(axis=0)
    keys = cells[:, 0] * (cells[:, 1].max() + 1) + cells[:, 1]
    order = np.argsort(keys)
    splits = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(order)]
    def unite(index):
        return shapely.union_all(polygons[order[splits[index]:splits[index+1]]], grid_size=grid_size)
    # Partition intact triangles into tiles; workers inherit the caller's CPU affinity.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(unite, range(len(splits) - 1)))
    return shapely.union_all(parts, grid_size=grid_size)


def carve_terrain(polygons, footprint, tile_m=100., workers=8):
    """Exact local booleans against a partition of the complete road outline."""
    parts = []
    def divide(geometry):
        if geometry.is_empty:
            return
        x0, y0, x1, y1 = geometry.bounds
        if max(x1-x0, y1-y0) <= tile_m:
            parts.append(geometry)
            return
        if x1-x0 >= y1-y0:
            middle = (x0+x1)/2
            boxes = [shapely.box(x0, y0, middle, y1), shapely.box(middle, y0, x1, y1)]
        else:
            middle = (y0+y1)/2
            boxes = [shapely.box(x0, y0, x1, middle), shapely.box(x0, middle, x1, y1)]
        for box in boxes:
            divide(geometry.intersection(box))
    divide(footprint)
    tree = shapely.STRtree(parts)
    bounds = shapely.bounds(polygons)
    cells = np.floor(bounds[:, :2] / tile_m).astype(np.int64)
    cells -= cells.min(axis=0)
    keys = cells[:, 0] * (cells[:, 1].max() + 1) + cells[:, 1]
    order = np.argsort(keys)
    cuts = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(order)]
    def cut(index):
        ids = order[cuts[index]:cuts[index+1]]
        box = shapely.box(*bounds[ids, :2].min(axis=0), *bounds[ids, 2:].max(axis=0))
        local = shapely.union_all(tree.geometries[tree.query(box, predicate='intersects')])
        shapely.prepare(local)
        affected = shapely.intersects(local, polygons[ids])
        removed = shapely.covers(local, polygons[ids])
        boundary = ids[affected & ~removed]
        remnants = shapely.difference(polygons[boundary], local)
        clipped, parents = shapely.get_parts(shapely.constrained_delaunay_triangles(remnants), return_index=True)
        # Normalize query-ring winding for consistent shared-edge overlap tests.
        overlap = float(shapely.area(shapely.intersection(
            shapely.orient_polygons(clipped), local)).sum())
        delta = float(shapely.area(clipped).sum() - shapely.area(remnants).sum())
        return ids[affected], int(removed.sum()), boundary[parents], clipped, len(boundary), overlap, delta
    affected = np.zeros(len(polygons), dtype=bool)
    removed_count = boundary_count = 0
    parents, triangles, overlap, delta = [], [], 0., 0.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for touched, removed, source, clipped, count, area, error in pool.map(cut, range(len(cuts)-1)):
            affected[touched] = True
            removed_count += removed
            boundary_count += count
            parents.append(source); triangles.append(clipped)
            overlap += area; delta += error
    return dict(affected=affected, removed_count=removed_count, boundary_count=boundary_count,
                parents=np.concatenate(parents), clipped=np.concatenate(triangles),
                overlap_m2=overlap, area_error_m2=abs(delta), footprint_parts=len(parts))
