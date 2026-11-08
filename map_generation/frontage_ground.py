"""Fit urban ground with a screened graph Laplacian.

Road-edge heights and the retained terrain boundary provide Dirichlet constraints.
"""
import numpy as np
import shapely
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
from scripts.scene.build_assets import surface_heights, BOOLEAN_GRID_M
from scripts.scene.repair_support_surface import close_surface_edges
from .mesh_geometry import spatial_union, boundary_segments


def smooth_ground(terrain, roads, selected, protected, settings, observed_z=None, road_footprint=None, close_edges=True):
    original = np.asarray(terrain['vertices'], dtype=float)
    # Carving duplicates endpoints. Solve one height per shared XY, then return
    # to the existing indices so UVs, topology and the observation border stay.
    xy, first, inverse = np.unique(np.round(original[:, :2], 7), axis=0,
                                   return_index=True, return_inverse=True)
    z = original[first, 2].copy()
    faces = inverse[np.asarray(terrain['faces'])]
    points = shapely.points(xy)
    in_protection = shapely.covered_by(points, protected)
    active = shapely.covered_by(points, selected)
    if not settings.get('infer_dem_in_separate_level_buffers', False):
        active &= ~in_protection
    used = np.zeros(len(xy), dtype=bool); used[faces.ravel()] = True
    active &= used
    triangles = np.concatenate([np.asarray(m['vertices'])[m['faces']] for m in roads['meshes']])
    polygons = shapely.polygons(triangles[:, :, :2])
    tree = shapely.STRtree(polygons)
    if road_footprint is None:
        road_footprint = spatial_union(polygons, grid_size=BOOLEAN_GRID_M)
    boundary = road_footprint.boundary
    # City outlines contain hundreds of thousands of edges. Query their
    # spatial index while retaining the same exact nearest-boundary geometry.
    segments = boundary_segments(boundary)
    segment_tree = shapely.STRtree(segments)
    nearest_segments = segments[segment_tree.nearest(points)]
    distance = shapely.distance(points, nearest_segments)
    active &= distance < settings['ground_distance_from_road_m']
    observed_values = np.asarray(observed_z)[first] if observed_z is not None else None
    observed_candidates = (np.isfinite(observed_values) & used & shapely.covered_by(points, selected)
                           if observed_values is not None else np.zeros(len(xy), dtype=bool))
    # Apply observed ground heights only to matching terrain layers.
    ids = np.flatnonzero(active | observed_candidates)
    closest = shapely.get_coordinates(shapely.get_point(shapely.shortest_line(points[ids], nearest_segments[ids]), -1))
    road_z = surface_heights(triangles, tree, closest, 'continuous frontage road edge')
    target = z.copy()
    curb = active & (distance <= settings.get('level_width_m', 1.6))
    offsets = np.full(len(ids), settings.get('height_m', .12), dtype=float)
    role_fixed = np.zeros(len(xy), dtype=bool)
    role_reports = []
    for condition in settings.get('road_relative_height_regions', []):
        polygon = shapely.Polygon(condition['polygon_xy_m'])
        distance_to_region = shapely.distance(points[ids], polygon)
        blend = float(condition.get('blend_distance_m', 0.))
        weight = (np.clip(1. - distance_to_region / blend, 0., 1.)
                  if blend > 0 else (distance_to_region <= 1e-7).astype(float))
        affected = active[ids] & (weight > 0)
        offsets[affected] += weight[affected] * (condition['height_m'] - offsets[affected])
        role_fixed[ids[affected]] = True
        role_reports.append(dict(id=condition['id'], active_vertices=int(affected.sum()),
                                 height_m=condition['height_m'], blend_distance_m=blend))
    target[ids] = road_z + offsets
    # Keep retained at-grade road edges as fixed height constraints.
    fixed = ~active | curb | role_fixed
    observed = np.zeros(len(xy), dtype=bool)
    support_conflicts = np.zeros(len(xy), dtype=bool)
    support_examples = []
    if observed_z is not None:
        observed = observed_candidates.copy()
        quality = settings.get('observed_ground_quality', {}).get('final_road_support')
        if quality:
            # Check source-ground continuity against the final road boundary.
            residual = observed_values[ids] - road_z
            tolerance = quality['base_height_tolerance_m'] + quality['maximum_context_grade'] * distance[ids]
            conflicts = observed[ids] & (abs(residual) > tolerance)
            support_conflicts[ids[conflicts]] = True
            examples = np.flatnonzero(conflicts)
            examples = examples[np.argsort(-abs(residual[examples]))[:20]]
            support_examples = [dict(xy_m=xy[ids[i]].tolist(), raster_z_m=float(observed_values[ids[i]]),
                road_edge_z_m=float(road_z[i]), distance_to_road_edge_m=float(distance[ids[i]]),
                residual_m=float(residual[i]), tolerance_m=float(tolerance[i])) for i in examples]
            observed &= ~support_conflicts
        fixed |= observed
    free = np.flatnonzero(~fixed)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    length = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    weights = 1. / np.maximum(length, .1)
    n = len(xy)
    graph = coo_matrix((np.tile(weights, 2), (np.r_[edges[:, 0], edges[:, 1]],
                        np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    screen = settings['ground_screening_weight']
    laplacian = diags(degree * (1 + screen)) - graph
    known = z.copy(); known[curb | role_fixed] = target[curb | role_fixed]
    if observed_z is not None:
        known[observed] = observed_values[observed]
    rhs = degree[free] * screen * z[free] - laplacian[free][:, fixed] @ known[fixed]
    system = laplacian[free][:, free]
    solution, status = cg(system, rhs, x0=z[free], rtol=1e-8, atol=1e-8,
                          maxiter=2000, M=diags(1 / system.diagonal()))
    if status:
        raise RuntimeError('Frontage ground solve did not converge: ' + str(status))
    known[free] = solution
    result = original.copy()
    mutable = (active | observed)[inverse]
    result[mutable, 2] = known[inverse[mutable]]
    terrain['vertices'] = result.tolist()
    if close_edges:
        close_surface_edges(terrain, roads, road_footprint)
    eligible = active[edges].all(axis=1) & (length > .25)
    def grades(values):
        g = abs(values[edges[:, 0]] - values[edges[:, 1]]) / np.maximum(length, 1e-8)
        return np.quantile(g[eligible], [.5, .95, 1]).tolist()
    report = {'method': 'Screened graph Laplacian on shared carved-terrain vertices, source-owned AV2 ground pins, road-relative strip and retained separate-level/outer boundaries',
              'active_vertices': int(active.sum()), 'solved_vertices': len(free),
              'fixed_observed_vertices': int(observed.sum()),
              'raster_final_road_support_conflicts': int(support_conflicts.sum()),
              'raster_final_road_support_examples': support_examples,
              'max_height_change_m': float(abs(result[:, 2] - original[:, 2]).max()),
              'terrain_edge_grade_median_p95_max_before': grades(z),
              'terrain_edge_grade_median_p95_max_after': grades(known),
              'unmodified_context_vertices_exactly_unchanged': bool(np.array_equal(result[~mutable], original[~mutable])),
              'protected_qualified_observed_vertices': int((in_protection & observed).sum()),
              'protected_unobserved_vertices_exactly_unchanged': bool(np.array_equal(
                  result[(in_protection & ~observed)[inverse]], original[(in_protection & ~observed)[inverse]])),
              'inferred_dem_in_separate_level_buffers': settings.get('infer_dem_in_separate_level_buffers', False),
              'curb_height_m': settings.get('height_m', .12),
              'road_relative_height_regions': role_reports,
              'scope': 'Qualified source raster cells are fixed context support. Conflicting raster values remain in source files and review. Qualified observed ground supersedes inferred DEM even inside planar separate-level buffers. Remaining eligible context is inferred; road geometry, actor poses and terrain XY stay unchanged. A selected buffer-inference policy treats unobserved DEM under separate roads as context, not a measured fixed height; separate road meshes still stay out of terrain carving.'}
    terrain['report']['continuous_frontage'] = report
    return report
