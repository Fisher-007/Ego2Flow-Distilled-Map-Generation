"""Construct roadside parking pavement from source use and side attributes."""
import numpy as np
import shapely
from scripts.scene.build_assets import surface_heights


def parallel_sides(tags):
    sides = []
    for side in ('left', 'right'):
        legacy = tags.get('parking:lane:'+side, tags.get('parking:lane:both'))
        modern = tags.get('parking:'+side+':orientation', tags.get('parking:both:orientation'))
        location = tags.get('parking:'+side, tags.get('parking:both'))
        if legacy == 'parallel' or (modern == 'parallel' and location == 'lane'):
            sides.append(side)
    return sides


def add_pavement(scene, roads, protected, visible, width=2.7):
    original = list(roads['meshes'])
    triangles = np.concatenate([np.asarray(m['vertices'])[m['faces']] for m in original])
    polygons = shapely.polygons(triangles[:, :, :2]); tree = shapely.STRtree(polygons)
    footprint = shapely.union_all(polygons)
    lanes = {l['id']: l for l in scene['fusion_map']['lanes']}
    intersections = shapely.union_all([shapely.union_all(shapely.polygons(np.asarray(m['vertices'])[m['faces']][:, :, :2]))
        for m in original if str(m.get('source_lane_id', '')).startswith('sumo::')]).buffer(1.)
    buildings = shapely.union_all([shapely.Polygon(b['exterior'], b.get('holes', [])) for b in scene['buildings']])
    records = []; occupied = footprint
    for source in scene['roads']:
        sides = parallel_sides(source['tags'])
        if not sides or source['tags'].get('bridge') in ('yes', 'viaduct'):
            continue
        line = shapely.LineString(np.asarray(source['reference_line'])[:, :2])
        if not line.intersects(visible):
            continue
        way = str(source['id']).split(':')[-1]
        lane_ids={key for key,lane in lanes.items() if lane.get('edge_id','').lstrip('-').split('#')[0]==way}
        # Extend source parking attributes through confirmed coarse/fine connections.
        connectors={key for key,lane in lanes.items() if key.startswith('seam:') and
                    (set(lane.get('successors',[])) & lane_ids or
                     any(key in lanes[k].get('successors',[]) for k in lane_ids))}
        members = [m for m in original if m.get('source_lane_id') in lane_ids|connectors]
        if not members:
            continue
        road_area = shapely.union_all([shapely.union_all(shapely.polygons(np.asarray(m['vertices'])[m['faces']][:, :, :2])) for m in members])
        region = road_area.buffer(width, join_style='mitre').difference(occupied)
        side_regions = [line.buffer(30 if s == 'left' else -30, single_sided=True) for s in sides]
        region = region.intersection(shapely.union_all(side_regions)).intersection(visible)
        region = region.difference(shapely.union_all([protected, intersections, buildings]))
        if region.area < 1.:
            continue
        region = shapely.union_all([p for p in shapely.get_parts(region) if p.geom_type == 'Polygon' and p.area > .1])
        parts = shapely.get_parts(shapely.constrained_delaunay_triangles(shapely.segmentize(region, .5)))
        xy = shapely.get_coordinates(parts).reshape(-1, 4, 2)[:, :3]
        flat = xy.reshape(-1, 2)
        nearest = shapely.get_coordinates(shapely.get_point(shapely.shortest_line(shapely.points(flat), footprint), -1))
        z = surface_heights(triangles, tree, nearest, 'parking edge road support')
        vertices = np.c_[flat, z]; faces = np.arange(len(vertices)).reshape(-1, 3)
        tri = vertices[faces]
        reverse = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])[:, 2] < 0
        faces[reverse] = faces[reverse, ::-1]
        key = 'parking_surface:'+way
        roads['meshes'].append({'id': key, 'source_lane_id': None, 'xodr_road_id': None,
            'vertices': vertices.tolist(), 'faces': faces.tolist(), 'surface_role': 'source_tagged_parking',
            'source_way': source['id'], 'navigation_lane_added': False})
        records.append({'id': key, 'source_way': source['id'], 'source_tags': source['tags'],
            'sides': sides, 'inferred_width_m': width, 'area_m2': region.area,
            'source_connected_seam_ids':sorted(connectors),
            'geometry': shapely.to_geojson(region),
            'scope': 'Physical parking pavement outside retained observation/access regions; side/use from OSM, width inferred. No new route or temporal parking regulation.'})
        occupied = occupied.union(region)
    return {'regions': records, 'source_vehicle_states_changed': False, 'navigation_lanes_added': False}


def footprint_review(xy_corners, parking, pavement=None):
    """Measure native vehicle-body occupancy over pavement footprints."""
    body = shapely.Polygon(xy_corners)
    regions = [(r['id'], shapely.from_geojson(r['geometry'])) for r in parking['regions']]
    matches = [(key, body.intersection(poly).area/body.area) for key, poly in regions if body.intersects(poly)]
    if not matches:
        return {'parking_region': None, 'scope': 'Outside the newly reconstructed tagged parking; original evidence still governs.'}
    key, fraction = max(matches, key=lambda x: x[1])
    result={'parking_region': key, 'body_fraction_on_parking': fraction,
            'body_outside_parking_m2': body.area*(1-fraction),
            'contained_in_tagged_parking': fraction > .995}
    if pavement is not None:
        supported=body.intersection(pavement).area/body.area
        result.update(body_fraction_on_pavement=supported,body_outside_pavement_m2=body.area*(1-supported),
            source_actor_clear_of_sidewalk=supported>.995,
            scope='Retain observed vehicle states and report pavement support and motor-lane overlap.')
    return result


def junction_pavement(scene, roads, parking, protected, visible):
    """Separate parking keep-clear zones from physical junction pavement.

    A parking exclusion at an existing junction must not create a raised
    sidewalk tongue inside its inferred paved envelope. This supplies ground
    support only; the parking polygons and legal navigation remain unchanged.
    """
    meshes=roads['meshes'];lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    shape=lambda m:shapely.union_all(shapely.polygons(np.asarray(m['vertices'])[m['faces']][:,:,:2]))
    pavement=shapely.union_all([shape(m) for m in meshes])
    intersections=shapely.union_all([shape(m) for m in meshes if str(m.get('source_lane_id','')).startswith('sumo::')]).buffer(1.)
    buildings=shapely.union_all([shapely.Polygon(b['exterior'],b.get('holes',[])) for b in scene['buildings']])
    sources={r['id']:r for r in scene['roads']};parts=[]
    for record in parking['regions']:
        source=sources[record['source_way']];way=source['id'].split(':')[-1]
        ids={key for key,lane in lanes.items() if lane.get('edge_id','').lstrip('-').split('#')[0]==way}
        ids.update(record.get('source_connected_seam_ids',[]))
        area=shapely.union_all([shape(m) for m in meshes if m.get('source_lane_id') in ids])
        line=shapely.LineString(np.asarray(source['reference_line'])[:,:2])
        sides=shapely.union_all([line.buffer(30 if s=='left' else -30,single_sided=True) for s in record['sides']])
        proposed=area.buffer(record['inferred_width_m'],join_style='mitre').intersection(sides)
        parts.append(proposed.intersection(intersections).intersection(visible).difference(pavement.union(protected).union(buildings)))
    return shapely.union_all(parts)
