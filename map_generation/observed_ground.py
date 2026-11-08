"""Source-owned AV2 ground observations for context terrain support."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from av2.map.map_api import ArgoverseStaticMap

from .coordinates import LocalFrame, source_log_registration


def context_ground(scene, xy, data_root, radius_m=30., quality=None):
    frame = LocalFrame(scene['coordinates']['city'], scene['coordinates']['origin_city_xyz'])
    by_map = defaultdict(list)
    boundary_by_map = defaultdict(list)
    for lane in scene['lanes']:
        polygon = shapely.Polygon(lane['left_boundary'] + lane['right_boundary'][::-1])
        observations = lane.get('source_observations')
        if observations is None:
            # Read scene-level provenance for single-log raster support.
            observations = [{'source': scene['source_info']['av2_map']}]
        for observation in observations:
            directory = str((Path(data_root) / observation['source']).parent)
            by_map[directory].append(polygon)
            for side in ('left_boundary', 'right_boundary'):
                edge = np.asarray(lane[side])
                boundary_by_map[directory].extend(np.stack((edge[:-1], edge[1:]), axis=1))
    points = shapely.points(xy)
    values = np.full(len(xy), np.nan)
    distances = np.full(len(xy), np.inf)
    owners = np.full(len(xy), -1, dtype=int)
    records = []
    for map_index, (directory, polygons) in enumerate(sorted(by_map.items())):
        tree = shapely.STRtree(polygons)
        possible = np.unique(tree.query(points, predicate='dwithin', distance=radius_m)[0])
        nearest = tree.nearest(points[possible])
        distance = shapely.distance(points[possible], tree.geometries[nearest])
        possible, distance = possible[distance < distances[possible]], distance[distance < distances[possible]]
        ground = ArgoverseStaticMap.from_map_dir(Path(directory), build_raster=True).raster_ground_height_layer
        registration = source_log_registration(scene, Path(directory).parent.name)
        position = np.asarray(xy)[possible]
        residual_xy = np.zeros(len(possible))
        if registration is None:
            z = ground.get_ground_height_at_xy(frame.local_to_city(position)) - frame.origin[2]
        else:
            basis = np.asarray(registration['rotation_matrix'])
            translation = np.asarray(registration['translation_local_m'])
            # Invert the registered height field along the map-Z ray and record the XY residual.
            z = np.full(len(possible), np.median([p[2] for lane in scene['lanes'] for p in lane['centerline']]))
            valid = np.ones(len(possible), dtype=bool)
            for _ in range(4):
                source = (np.c_[position, z] - translation) @ basis
                ground_z = ground.get_ground_height_at_xy(frame.local_to_city(source[:, :2])) - frame.origin[2]
                valid &= np.isfinite(ground_z)
                # Keep a finite ray for unsupported cells; these observations
                # stay missing in the returned support field.
                source[valid, 2] = ground_z[valid]
                target = source @ basis.T + translation
                z[valid] = target[valid, 2]
                residual_xy[valid] = np.linalg.norm(target[valid, :2] - position[valid], axis=1)
            z[~valid] = np.nan
        finite = np.isfinite(z)
        quality_review = None
        if quality is not None:
            # Require local continuity when selecting source-ground support across layers.
            segments = np.asarray(boundary_by_map[directory])
            d = segments[:, 1, :2] - segments[:, 0, :2]
            length2 = (d * d).sum(axis=1)
            segments, d, length2 = segments[length2 > 0], d[length2 > 0], length2[length2 > 0]
            segment_tree = shapely.STRtree(shapely.linestrings(segments[:, :, :2]))
            nearest = segment_tree.nearest(points[possible])
            t = np.clip(((position - segments[nearest, 0, :2]) * d[nearest]).sum(axis=1) / length2[nearest], 0, 1)
            support = segments[nearest, 0] + t[:, None] * (segments[nearest, 1] - segments[nearest, 0])
            distance_to_source = np.linalg.norm(position - support[:, :2], axis=1)
            tolerance = quality['base_height_tolerance_m'] + quality['maximum_context_grade'] * distance_to_source
            residual = z - support[:, 2]
            conflict = finite & (abs(residual) > tolerance)
            examples = np.flatnonzero(conflict)
            examples = examples[np.argsort(-abs(residual[examples]))[:8]]
            quality_review = dict(settings=quality, finite_before=int(finite.sum()),
                conflicting_vertices=int(conflict.sum()),
                examples=[dict(xy_m=position[i].tolist(), raster_z_m=float(z[i]),
                    nearest_source_boundary_xyz_m=support[i].tolist(), residual_m=float(residual[i]),
                    tolerance_m=float(tolerance[i])) for i in examples],
                interpretation='Continuity eligibility for single-level urban context; conflicting source values stay unresolved and do not become fixed ground constraints.')
            finite &= ~conflict
        ids = possible[finite]
        values[ids], distances[ids], owners[ids] = z[finite], distance[finite], map_index
        records.append(dict(map_dir=directory, source_log_registration=registration,
                            finite_candidate_vertices=int(finite.sum()),
                            ground_constraint_quality=quality_review,
                            registered_query_max_xy_residual_m=float(residual_xy[finite].max(initial=0.))))
        print('Observed context ground', Path(directory).parent.name, int(finite.sum()), flush=True)
    for i, record in enumerate(records):
        record['selected_vertices'] = int((owners == i).sum())
    return values, dict(method='Finite official AV2 ground cells, source ownership by nearest retained source-lane polygon; identical reviewed log registration, map XY retained.',
                        observation_radius_m=radius_m, observed_vertices=int(np.isfinite(values).sum()), source_maps=records,
                        scope='Source-provided ground raster in the observed vicinity; finite values alone do not establish valid physical support. Optional road-continuity review determines fixed context constraints; missing or conflicting support remains inferred/pending evidence.')
