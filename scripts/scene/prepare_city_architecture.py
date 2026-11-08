"""Apply reviewed appearance families to complete, supported city buildings."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import shapely

from map_generation.information_graph import add_node, add_edge, compile_conditions
from scripts.assets.frontage_style import choose_style



def style_rule_matches(building, rule):
    tags = building['tags']
    sources = set(building.get('volume_source_features', [])) | {building.get('source_id'), building['id']}
    if rule.get('source_ids') and not sources.intersection(rule['source_ids']):
        return False
    height = building['nominal_height_m']
    if height < rule.get('minimum_height_m', 0) or height > rule.get('maximum_height_m', float('inf')):
        return False
    if rule.get('tags_any') and not any(tags.get(k) in values for k, values in rule['tags_any'].items()):
        return False
    if rule.get('tags_exclude') and any(tags.get(k) in values for k, values in rule['tags_exclude'].items()):
        return False
    if 'minimum_footprint_area_m2' in rule or 'maximum_footprint_area_m2' in rule:
        area = shapely.Polygon(building['exterior'], building.get('holes', [])).area
        if not rule.get('minimum_footprint_area_m2', 0) <= area <= rule.get('maximum_footprint_area_m2', float('inf')):
            return False
    return True


def conditioned_style(building, settings, index, reference_area=None):
    selection = settings.get('building_style')
    profile = settings.get('area_style_profiles', {}).get(reference_area, {})
    style = (deepcopy(selection['default']) if selection else
             choose_style(building, settings['residential_materials'], index))
    style.update(deepcopy(profile.get('default', {})))
    matched = []
    # Apply source and tag constraints after neighborhood style defaults.
    for scope, rules in [('area', profile.get('rules', [])),
                         ('source', selection.get('rules', []) if selection else [])]:
        for rule in rules:
            if style_rule_matches(building, rule):
                style.update(deepcopy(rule['style']))
                matched.append(scope + ':' + rule['id'])
    tags = building['tags']
    storey = style.get('inferred_storey_height_m', profile.get('inferred_storey_height_m',
        selection.get('inferred_storey_height_m', 3.2) if selection else 3.2))
    height = building['roof_z_m'] - building['entrance_z_m']
    if 'building:levels' in tags:
        floors = max(1, int(round(float(tags['building:levels']))))
        floor_source = 'explicit building:levels'
    elif 'inferred_floors' in style:
        floors = max(1, int(style['inferred_floors']))
        floor_source = 'reviewed regional rule; inferred'
    else:
        floors = max(1, int(round(height / storey)))
        floor_source = 'source height / conditioned storey spacing; inferred'
    style.update(floors=floors, floor_count_source=floor_source,
        inferred_storey_height_m=storey, nominal_height_m=building['nominal_height_m'],
        source_height_retained=True, condition_rules=matched)
    # Reconsider inherited building-height estimates using source silhouettes.
    style['source_height_retained_semantics'] = 'Retain the input height geometry and its source attribution during appearance generation.'
    style['height_provenance'] = deepcopy(building.get('height_provenance', {
        'description': building.get('height_source', 'Unrecorded input height source'),
        'explicit_height_tags': {k: tags[k] for k in ('height', 'building:height', 'building:levels') if k in tags},
        'default_estimate': 'assumed default' in building.get('height_source', '').lower(),
        'evidence_status': 'not independently verified by appearance preparation',
        'conflict_policy': 'Reliable observed main silhouette may reopen this property; retained input is not a measurement.'}))
    style['inference'] = profile.get('inference', selection.get('inference', style.get('inference')) if selection else style.get('inference'))
    if profile:
        style['area_style_profile'] = reference_area
        style['profile_evidence'] = deepcopy(profile.get('evidence', []))
    return style


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'graph', 'selection', 'visual-review', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    read = lambda f: json.loads(f.read_text())
    scene, graph, spec, views = map(read, (a.scene, a.graph, a.selection, a.visual_review))
    task = spec['task']; key = 'appearance:city_source_frontage'
    add_node(graph, dict(id=task, kind='simulation_task'))
    add_node(graph, dict(id=key, kind='native_appearance_condition', content=spec,
        native_construction=dict(kind='city_source_frontage', settings=spec),
        task_status={task: dict(complete=False, missing=['native_render_and_driving'])}))
    add_edge(graph, task, 'depends_on', key)
    observations = {row['area_id']: row for row in views['areas']}
    areas = scene['areas']
    centers = np.array([np.mean(np.array(area['bbox_local_xy']).reshape(2, 2), axis=0)
                        for area in areas])
    evidence = []
    for area in areas:
        source = observations[area['id']]; identity = 'appearance_reference:' + area['id']
        add_node(graph, dict(id=identity, kind='reviewed_source_appearance', content=source))
        add_edge(graph, key, 'supported_by', identity); evidence.append(identity)
    add_edge(graph, key, 'depends_on', spec.get('ground_support_condition','ground_support:city_road_foundations'))
    graph['decisions'].append(dict(source_id=key, task=task, choice='enhance',
        agent=spec['agent'], reason=spec['reason'], evidence_ids=evidence,
        required_properties=spec['required_properties']))
    conditions = compile_conditions(graph, scene, task)
    selected = next(row for row in conditions['native_objects'] if row['source_id'] == key)
    settings = selected['native_construction']['settings']
    rng = np.random.default_rng(settings['seed'])
    plan = dict(buildings=[], retained_buildings=[], graph_condition=selected,
                scope=settings['scope'], source_scene=str(a.scene),
                replace_complete_building_set=bool(settings.get('replace_complete_building_set', False)),
                material_reuse_paths=settings.get('material_reuse_paths', {}))
    excluded = set(settings.get('excluded_building_ids', []))
    for i, b in enumerate(sorted(scene['buildings'], key=lambda b: b['id'])):
        if b['id'] in excluded:
            plan['retained_buildings'].append(dict(id=b['id'], reason='Explicit retained source/native building identity'))
            continue
        # Passage solids already encode collision headroom and upper masses.
        # Their openings must also survive later facade/detail construction.
        retain_solid = (b.get('passage_parent_id') or b['tags'].get('building') in ('roof', 'carport', 'grandstand', 'greenhouse')
                        or b['base_z_m']>b.get('entrance_z_m',b['base_z_m']))
        if retain_solid:
            plan['retained_buildings'].append(dict(id=b['id'], reason='source-supported passage, elevated solid or open/special structure'))
            if not plan['replace_complete_building_set']:
                continue
        if 'final_support_range_m' not in b:
            raise ValueError('Fit final city foundations before facade construction: ' + b['id'])
        polygon = shapely.Polygon(b['exterior'], b.get('holes', []))
        xy = np.array(polygon.centroid.coords[0])
        distances = np.linalg.norm(centers - xy, axis=1)
        nearest = np.argsort(distances)[:3]
        weights = np.exp(-(distances[nearest] - distances[nearest[0]]) / settings['reference_transition_m'])
        weights /= weights.sum()
        selected_area = areas[int(rng.choice(nearest, p=weights))]['id']
        reference = observations[selected_area]
        style = conditioned_style(b, settings, i, selected_area)
        profile = settings.get('area_style_profiles', {}).get(selected_area, {})
        material_sets = profile.get('materials_by_kind', {})
        families = material_sets.get(style['kind'], material_sets.get(
            'commercial' if style['kind'] == 'commercial' else 'residential'))
        if families is None:
            families = settings.get('area_materials_by_id',{}).get(selected_area)
        if families is None:
            families = (settings['area_materials'].get(reference.get('style_family',''), settings['commercial_materials'])
                        if style['kind'] == 'commercial' else settings['residential_materials'])
        # Keep the selected building heights during detail generation.
        style['nominal_height_m'] = b['nominal_height_m']
        style['family'] = str(rng.choice(families))
        style['source_height_retained'] = True
        style['roof_rise_m'] = min(style['roof_rise_m'], max(.0, b['roof_z_m'] - b['entrance_z_m'] - 2.6))
        row = deepcopy(b); row['style'] = style
        overrides = deepcopy(profile.get('material_overrides', {}))
        overrides.update(style.get('material_overrides', {}))
        if overrides:
            row['material_overrides'] = {**row.get('material_overrides', {}), **overrides}
        if retain_solid:
            row['solid_parts'] = [deepcopy(b)]
            row['detail_policy'] = 'Retain this exact source-conditioned solid; add no panels across access openings.'
        row['appearance_evidence'] = dict(reference_area=selected_area,
            material_inferred=True, eligible_references=[dict(area_id=areas[int(j)]['id'], distance_m=float(distances[j]), weight=float(w))
                for j, w in zip(nearest, weights)],
            source_building_type=b['tags'].get('building'), construction_family=style['kind'], material_family=style['family'],
            geometry_profile=style.get('area_style_profile'), profile_evidence=profile.get('evidence', []),
            matched_geometry_rules=style['condition_rules'], floor_count_source=style['floor_count_source'], height_provenance=deepcopy(style['height_provenance']))
        plan['buildings'].append(row)
    a.output.mkdir(parents=True, exist_ok=True)
    summary = dict(buildings=len(plan['buildings']), retained_passage_solids=len(plan['retained_buildings']),
        types=dict(Counter(b['style']['kind'] for b in plan['buildings'])),
        materials=dict(Counter(b['style']['family'] for b in plan['buildings'])),
        roofs=dict(Counter(b['style']['roof'] for b in plan['buildings'])),
        area_geometry_profiles=dict(Counter(b['style'].get('area_style_profile', 'legacy') for b in plan['buildings'])),
        source_footprints_and_heights_retained=True, native_render_complete=False)
    for name, data in [('frontage_plan.json', plan), ('graph.json', graph), ('conditions.json', conditions), ('review.json', summary)]:
        (a.output / name).write_text(json.dumps(data, separators=(',', ':')) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
