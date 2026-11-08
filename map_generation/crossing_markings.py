"""Realize graph-selected AV2 crossing edges on their physical road support."""
import numpy as np
import shapely

from map_generation.road_markings import RoadSurface, source_surface_layers


def zebra_paint(edges, spec):
    """Keep the source extent while allowing a measured common stripe axis."""
    a, b = edges
    if np.linalg.norm(a[0]-b[0])+np.linalg.norm(a[-1]-b[-1]) > np.linalg.norm(a[0]-b[-1])+np.linalg.norm(a[-1]-b[0]):
        b = b[::-1]
    extent = shapely.make_valid(shapely.Polygon(np.concatenate([a, b[::-1]])))
    width, gap = spec['stripe_width_m'], spec['stripe_gap_m']
    if 'bar_direction_xy' in spec:
        direction = np.asarray(spec['bar_direction_xy'], dtype=float)
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        vertices = np.concatenate([a, b])
        along, across = vertices @ direction, vertices @ normal
        anchor = np.asarray(spec.get('bar_anchor_xy', vertices.mean(axis=0)))
        phase = float(anchor @ normal)
        period = width + gap
        first = int(np.floor((across.min()-width/2-phase)/period))
        last = int(np.ceil((across.max()+width/2-phase)/period))
        stripes = [shapely.LineString([
            direction*(along.min()-width)+normal*(phase+i*period),
            direction*(along.max()+width)+normal*(phase+i*period)
        ]).buffer(width/2, cap_style='flat', join_style='mitre')
            for i in range(first, last+1)]
    else:
        # Keep the display condition until a source axis is selected.
        lines = [shapely.LineString(v) for v in (a, b)]
        length = sum(v.length for v in lines)/2
        stations = np.arange(width/2, length, width+gap)
        stripes = [shapely.LineString([
            v.interpolate(float(station/length), normalized=True).coords[0] for v in lines
        ]).buffer(width/2, cap_style='flat', join_style='mitre') for station in stations]
    return shapely.union_all(stripes).intersection(extent)


def build_crossings(scene, roads, conditions):
    assets = {a['id']: a for a in scene['assets']}
    selected = [r for r in conditions['native_objects'] if r['choice'] in ('retain', 'enhance')
                and r.get('native_construction', {}).get('kind') == 'pedestrian_crossing_paint']
    shapes = []
    for row in selected:
        source = assets[row['source_id']]
        spec = row['native_construction']['settings']
        edges = [np.asarray(source[k], dtype=float)[:, :2] for k in ('edge1', 'edge2')]
        if spec['pattern'] == 'parallel_boundaries':
            paint = shapely.union_all([shapely.LineString(edge).buffer(
                spec['width_m']/2, cap_style='flat', join_style='mitre') for edge in edges])
        elif spec['pattern'] == 'zebra':
            paint = zebra_paint(edges, spec)
        else:
            raise ValueError('Crossing pattern has no construction: '+spec['pattern'])
        shapes.append(paint)
    _, supports = source_surface_layers(scene, roads)
    # Source crossing geometry uses the retained ground layer, excluding overpasses.
    surface = RoadSurface(dict(meshes=supports[0.]), shapely.union_all(shapes))
    meshes, records = [], []
    for row, paint in zip(selected, shapes):
        spec = row['native_construction']['settings']
        clipped = paint.intersection(surface.footprint)
        triangles, check = surface.cut_and_drape(clipped, spec['normal_offset_m'])
        name = spec.get('mesh_id', 'crossing_%04d' % len(meshes))
        records.append(dict(source_id=row['source_id'], mesh_id=name, settings=spec,
            requested_area_m2=paint.area, realized_area_m2=clipped.area,
            outside_road_area_m2=max(0., paint.area-clipped.area), projection=check,
            native_visibility='pending'))
        meshes.append(dict(id=name, color='white', source_id=row['source_id'],
            vertices=triangles.reshape(-1, 3).tolist(),
            faces=np.arange(triangles.size//3).reshape(-1, 3).tolist(),
            semantic_class='RoadLines', semantic_id=24, ue_asset_category='RoadLine',
            collision_required='NoCollision', cast_shadow_required=False))
    report = dict(source_crossings=len(selected), records=records,
        triangles=sum(len(m['faces']) for m in meshes),
        scope='Source crossing extents and explicit display pattern; native visual review follows.')
    report['passed'] = bool(records) and all(r['realized_area_m2'] > 0 and
        r['projection']['max_interpolation_gap_m'] <= .001 for r in records)
    return dict(frame='SceneIR/Blender ENU metres', meshes=meshes, report=report)
