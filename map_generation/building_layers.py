"""Construct non-overlapping solid cells from source-tagged building levels.

Each cell records its contributing source masses and vertical extent.
"""
from copy import deepcopy
from collections import defaultdict
import numpy as np
from shapely import set_precision
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree
from .buildings import _polygons
from .building_passages import tagged_lower_envelope, realize as realize_passages


def union_solids(buildings, grid_m=1e-5):
    """Partition the union of vertical prisms; retain gaps between their levels."""
    if not buildings:
        return [], dict(input_count=0, output_count=0)
    shapes = [set_precision(Polygon(b['exterior'], b['holes']), grid_m) for b in buildings]
    tree = STRtree(shapes)
    cells = list(polygonize(unary_union([p.boundary for p in shapes])))
    groups = defaultdict(list)
    source_volume = sum(p.area * b['height_m'] for p, b in zip(shapes, buildings))
    union_volume = 0.
    for cell in cells:
        indices = sorted(map(int, tree.query(cell.representative_point(), predicate='intersects')))
        intervals = sorted((buildings[i]['base_z_m'], buildings[i]['roof_z_m'], i) for i in indices)
        merged = []
        for low, high, i in intervals:
            if high <= low:
                raise ValueError(f"Nonpositive source solid {buildings[i]['id']}")
            if merged and low <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], high)
                merged[-1][2].append(i)
            else:
                merged.append([low, high, [i]])
        for low, high, ids in merged:
            # Ownership selects metadata only; the complete union is unchanged.
            owner = min(ids, key=lambda i: (shapes[i].area, buildings[i]['id']))
            key = (float(low), float(high), owner, tuple(sorted(ids)))
            groups[key].append(cell)
            union_volume += cell.area * (high - low)
    output = []
    for (low, high, owner, ids), cells in groups.items():
        for polygon in _polygons(unary_union(cells)):
            record = deepcopy(buildings[owner])
            record.update(id=f"{buildings[owner]['id']}:solid:{len(output)}",
                exterior=list(map(list, polygon.exterior.coords)),
                holes=[list(map(list, r.coords)) for r in polygon.interiors],
                base_z_m=low, roof_z_m=high, height_m=high-low,
                volume_source_ids=[buildings[i]['id'] for i in ids],
                volume_source_features=sorted({buildings[i]['source_id'] for i in ids}),
                volume_partition='Exact union of source-conditioned vertical prisms on recorded coordinate grid')
            output.append(record)
    emitted_volume = sum(Polygon(b['exterior'], b['holes']).area * b['height_m'] for b in output)
    return output, dict(input_count=len(buildings), output_count=len(output), grid_m=grid_m,
        input_summed_volume_m3=source_volume, union_volume_m3=union_volume,
        duplicate_volume_removed_m3=source_volume-union_volume,
        emitted_volume_m3=emitted_volume, union_volume_error_m3=emitted_volume-union_volume,
        coordinate_precision_area_change_m2=sum(
            Polygon(b['exterior'], b['holes']).symmetric_difference(p).area for b,p in zip(buildings,shapes)),
        semantics='Union preserves every physical source-conditioned solid and all vertical gaps; source ownership and original tags remain available.')


def realize(scene, map_key=None, default_clearance_m=4.5, lateral_margin_m=.3,
            vehicle_length_m=4.8, vehicle_width_m=2.2):
    map_key = map_key or ('fusion_map' if 'fusion_map' in scene else 'coarse_map')
    canonical = deepcopy(scene.get('building_layer_source_buildings', scene['buildings']))
    prepared, elevated, underground = [], [], []
    for source in canonical:
        b = deepcopy(source)
        if b['nominal_height_m'] <= 0:
            if float(b['tags'].get('building:levels', '-1')) == 0 and float(
                    b['tags'].get('building:levels:underground', '0')) > 0:
                underground.append(dict(id=b['id'], source_id=b['source_id'], tags=b['tags'],
                    reason='Zero above-ground levels: retain source as underground evidence without inventing an above-ground solid.'))
                continue
            raise ValueError(f"Building height requires source review: {b['id']}")
        lower = tagged_lower_envelope(b)
        if lower:
            if lower['bottom_z_m'] >= b['roof_z_m']:
                raise ValueError(f"Source lower envelope reaches roof: {b['id']}")
            b.update(base_z_m=lower['bottom_z_m'], height_m=b['roof_z_m']-lower['bottom_z_m'],
                     source_lower_envelope=lower)
            elevated.append(dict(id=b['id'], **lower))
        if b['tags'].get('building') == 'roof' and not lower:
            b.update(base_z_m=b['roof_z_m']-.35, height_m=.35,
                     roof_only_prior='Open canopy roof with assumed 0.35 m slab; support posts require separate source evidence.')
        prepared.append(b)
    working = dict(scene, buildings=prepared, fusion_map=scene[map_key],
                   building_passage_source_buildings=[])
    working, passages = realize_passages(working, default_clearance_m, lateral_margin_m,
                                         vehicle_length_m, vehicle_width_m)
    solids, union_report = union_solids(working['buildings'])
    output = dict(scene, buildings=solids, building_layer_source_buildings=canonical,
        building_passage_source_buildings=working['building_passage_source_buildings'],
        building_passage_review=passages)
    report = dict(elevated_source_parts=elevated, underground_source_records=underground,
                  solid_union=union_report, passages=passages,
                  road_map_key=map_key,
                  scope='Coarse source-conditioned building geometry. Assumed storey heights and missing passage headroom are explicit; native appearance, physical support and driving still require review.')
    output['building_layer_review'] = report
    return output, report


def measure_solids(buildings, route_points=None, vehicle_width_m=2.2, vehicle_height_m=2.2, routes=None):
    """Measure true overlap and swept route obstruction, including elevated solids."""
    shapes = [Polygon(b['exterior'], b['holes']) for b in buildings]
    tree = STRtree(shapes)
    overlap, collisions = [], []
    for i, p in enumerate(shapes):
        a = buildings[i]
        for j in tree.query(p, predicate='intersects'):
            j = int(j)
            if j <= i:
                continue
            b = buildings[j]
            dz = min(a['roof_z_m'], b['roof_z_m']) - max(a['base_z_m'], b['base_z_m'])
            if dz <= 1e-7:
                continue
            area = p.intersection(shapes[j]).area
            if area * dz > 1e-7:
                overlap.append(dict(a=a['id'], b=b['id'], volume_m3=area*dz))
    route_items = list(routes.items()) if routes is not None else (
        [('route', route_points)] if route_points is not None else [])
    for route_id, points in route_items:
        points = np.asarray(points)
        for n, (a, b) in enumerate(zip(points[:-1], points[1:])):
            if np.linalg.norm(b[:2]-a[:2]) < 1e-7:
                continue
            line = LineString([a[:2], b[:2]])
            envelope = line.buffer(vehicle_width_m/2, cap_style='flat')
            for j in tree.query(envelope, predicate='intersects'):
                j = int(j); building = buildings[j]
                if building['base_z_m'] >= max(a[2], b[2])+vehicle_height_m or building['roof_z_m'] <= min(a[2], b[2]):
                    continue
                area = envelope.intersection(shapes[j]).area
                if area > 1e-7:
                    collisions.append(dict(route_id=route_id, segment=n, building_id=building['id'], area_m2=area,
                        source_ids=building.get('volume_source_features',[building['source_id']])))
    valid = all(p.is_valid and p.area > 0 for p in shapes)
    positive = all(np.isfinite([b['base_z_m'],b['roof_z_m']]).all() and
                   b['roof_z_m'] > b['base_z_m'] for b in buildings)
    return dict(valid_polygons=valid, positive_finite_solids=positive,
        overlapping_pairs=overlap, overlap_volume_m3=sum(r['volume_m3'] for r in overlap),
        route_body_collisions=collisions, route_body_collision_count=len(collisions),
        route_vehicle_width_m=vehicle_width_m, route_vehicle_height_m=vehicle_height_m,
        measured_route_count=len(route_items), route_clearance_measured=bool(route_items),
        passed=valid and positive and not overlap and not collisions,
        scope='Segment-wise conservative body-width and height-envelope test; native steering and full rigid-body motion remain required.')
