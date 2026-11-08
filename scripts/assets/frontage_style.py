"""Source-tag and footprint defaults for graph-reviewed building style."""
import shapely


def choose_style(building, families, index):
    tags = building['tags']; polygon = shapely.Polygon(building['exterior'], building.get('holes', []))
    name = tags.get('name', '').lower()
    if (tags.get('shop') or tags.get('amenity') or tags.get('leisure')
            or tags.get('building') in ('commercial', 'retail', 'office', 'industrial', 'warehouse', 'school', 'university', 'hospital')
            or any(word in name for word in ('professional', 'hospital'))):
        kind = 'commercial'
    elif tags.get('building') in ('apartments', 'residential') or any(word in name for word in ('apartment', 'arms', 'court')) or polygon.area > 450:
        kind = 'apartment'
    elif polygon.area < 45:
        kind = 'ancillary'
    else:
        kind = 'house'
    explicit = any(key in tags for key in ('height', 'building:levels'))
    floors = max(1, int(round(float(tags.get('building:levels', 2 if kind == 'apartment' else 1)))))
    height = building['nominal_height_m'] if explicit else {'commercial': 4.0, 'apartment': 6.4, 'house': 3.8, 'ancillary': 2.8}[kind]
    roof = 'flat' if kind in ('commercial', 'apartment') or building.get('holes') else 'low_hip'
    if tags.get('roof:shape') in ('flat', 'hipped', 'gabled'):
        roof = 'flat' if tags['roof:shape'] == 'flat' else 'low_hip'
    return {'kind': kind, 'family': families[index % len(families)], 'floors': floors,
            'nominal_height_m': height, 'roof': roof, 'roof_rise_m': .75 if roof != 'flat' else .18,
            'window_width_m': 1.25 if kind != 'commercial' else 1.7,
            'window_height_m': 1.25 if kind != 'commercial' else 2.,
            'bay_spacing_m': 3.6 if kind != 'commercial' else 2.8,
            'source_height_retained': explicit,
            'inference': 'OSM use/explicit levels when supplied; otherwise footprint-scale type and reviewed AV2 low-rise style. Window count and unspecified roofs are inferred.'}

