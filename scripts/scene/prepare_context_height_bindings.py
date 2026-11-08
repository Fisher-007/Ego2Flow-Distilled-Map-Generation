"""Prepare source-identified context height bindings after physical ground feedback."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np
import shapely

from scripts.scene.build_assets import surface_heights
from scripts.scene.prepare_city_ground import at_grade_roads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('previous-scene', 'scene', 'terrain', 'roads', 'source-assets',
                'architecture-import', 'canopy', 'conditions', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--building-replacement', type=Path,
        help='Complete current architecture export; prepare canopy/control support while building geometry is installed separately')
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    scene, terrain, roads = map(read, (a.scene, a.terrain, a.roads))
    old = read(a.previous_scene) if a.building_replacement is None else {'buildings': []}
    conditions, canopy = map(read, (a.conditions, a.canopy))
    previous = {b['id']: b for b in old['buildings']}
    current = {b['id']: b for b in scene['buildings']}
    assets = read(a.source_assets)
    source = {b['source_id']: b for b in assets['objects'] if b['kind'] == 'building'}
    omitted = {b['source_id']: b for b in assets.get('omitted_degenerate_geometry', []) if b.get('kind') == 'building'}
    reuse = conditions.get('context_height_binding_options', {})
    permitted = set(reuse.get('retain_native_footprint_source_ids', []))
    detail = {b['source_id']: b['new_mesh'] for b in read(a.architecture_import)['buildings']}
    buildings, unresolved, retained_omissions = [], [], []
    replacement_dependency = None
    if a.building_replacement:
        replacement = read(a.building_replacement)
        plan = read(Path(replacement['plan']))
        exported = {b['source_id'] for b in replacement['buildings']}
        omissions = {b['source_id'] for b in replacement.get('omitted_degenerate_geometry', [])}
        if (not plan.get('replace_complete_building_set')
                or Path(plan['source_scene']).resolve() != a.scene.resolve()
                or exported | omissions != set(current)):
            raise ValueError('Building replacement must account for every current source solid from this scene')
        replacement_dependency = dict(export=str(a.building_replacement),
            exported_source_solids=len(exported), recorded_omissions=len(omissions),
            current_source_solids=len(current), source_scene=str(a.scene),
            scope='Replace the complete building geometry using its reconstructed placement.',
            native_installation='Requires separate current building import and saved/runtime verification')
    for key in ([] if replacement_dependency else sorted(set(previous) | set(current))):
        before, after = previous.get(key), current.get(key)
        if before is None or after is None:
            unresolved.append(dict(source_id=key, reason='Source solid identity changed; needs geometry replacement'))
            continue
        polygon = lambda b: shapely.Polygon(b['exterior'], b.get('holes', []))
        difference = polygon(before).symmetric_difference(polygon(after)).area
        displacement = polygon(before).hausdorff_distance(polygon(after))
        if key not in source:
            if (key in omitted and difference == 0
                    and polygon(after).area == omitted[key]['footprint_area_m2']):
                retained_omissions.append(dict(omitted[key], current_footprint_exact=True,
                    evidence=str(a.source_assets)))
            else:
                unresolved.append(dict(source_id=key, reason='No native source asset or exact recorded omission'))
            continue
        bounded_reuse = key in permitted and displacement <= reuse['max_current_partition_hausdorff_m']
        if difference > 1e-6 and not bounded_reuse:
            unresolved.append(dict(source_id=key, reason='Source footprint changed; needs geometry replacement',
                                   footprint_difference_m2=difference, footprint_hausdorff_m=displacement))
            continue
        scale = ((after['roof_z_m'] - after['base_z_m']) /
                 (before['roof_z_m'] - before['base_z_m']))
        offset = after['base_z_m'] - scale * before['base_z_m']
        name = re.sub(r'[^A-Za-z0-9_]', '_', source[key]['name'])
        buildings.append(dict(source_id=key, mesh_path=detail.get(key),
            mesh_names=[] if key in detail else [name, 'map_assets_' + name],
            previous_base_roof_z_m=[before['base_z_m'], before['roof_z_m']],
            current_base_roof_z_m=[after['base_z_m'], after['roof_z_m']],
            z_scale=scale, z_offset_m=offset,
            footprint_difference_m2=difference,
            footprint_hausdorff_m=displacement,
            explicit_native_footprint_reuse=bounded_reuse,
            nominal_height_unchanged=before['nominal_height_m'] == after['nominal_height_m']))
    ground, _, _, _ = at_grade_roads(scene, roads, 3.)
    triangles = np.concatenate([np.asarray(terrain['vertices'])[terrain['faces']],
        *[np.asarray(m['vertices'])[m['faces']] for m in ground['meshes'] if m['faces']]])
    tree = shapely.STRtree(shapely.polygons(triangles[:, :, :2]))
    def heights(xy, role):
        return surface_heights(triangles, tree, np.asarray(xy), role)
    candidates = canopy['canopy_candidates']
    support = heights([x['position_enu_m'][:2] for x in candidates], 'current canopy support')
    trees = []
    for index, (item, z) in enumerate(zip(candidates, support)):
        inferred = item.get('context_kind') == 'inferred_outer'
        top = float(z + item['target_height_m']) if inferred else float(item['observed_top_z_m'])
        if top <= z:
            unresolved.append(dict(candidate=index, reason='Physical ground reaches observed crown top'))
            continue
        trees.append(dict(candidate=index,
            actor_label=('MAP_InferredCanopy_' if inferred else 'MAP_ObservedCanopy_') + str(index),
            context_kind=item.get('context_kind', 'lidar_supported'),
            support_xy_m=item['position_enu_m'][:2],
            target_bottom_z_m=float(z), target_top_z_m=top,
            observed_top_retained=not inferred,
            inferred_height_retained=inferred))
    bound = deepcopy(conditions)
    controls = []
    for row in bound['native_objects']:
        native = row.get('native_construction', {})
        spec = native.get('settings', {})
        if native.get('kind') == 'source_sign_face':
            center, normal = spec['center_local_xyz_m'], spec['front_normal_xyz']
            xy = [center[i] - .055 * normal[i] for i in (0, 1)]
            target = spec
        elif native.get('kind') == 'observed_signal_heads':
            target = spec['mast_support']
            xy = target['join_local_xyz_m'][:2]
        else:
            continue
        before = target.get('ground_z_m')
        z = float(heights([xy], 'current source control support')[0])
        target['ground_z_m'] = z
        target['physical_support_binding'] = dict(scene=str(a.scene), terrain=str(a.terrain),
            roads=str(a.roads), support_xy_m=xy, previous_ground_z_m=before,
            method='Current at-grade road or terrain triangle plane at the actual support base')
        controls.append(dict(source_id=row['source_id'], support_xy_m=xy,
                             previous_ground_z_m=before, ground_z_m=z))
    result = dict(previous_scene=str(a.previous_scene), current_scene=str(a.scene),
        terrain=str(a.terrain), roads=str(a.roads), source_assets=str(a.source_assets),
        architecture_import=str(a.architecture_import), source_canopy=str(a.canopy),
        buildings=buildings, canopy=trees, controls=controls, unresolved=unresolved,
        building_replacement_dependency=replacement_dependency,
        retained_source_omissions=retained_omissions,
        context_partition_reuse={} if replacement_dependency else reuse,
        passed=not unresolved and all(x['nominal_height_unchanged'] for x in buildings),
        scope='Bind only inferred vertical context support. Retain source XY, source identities, existing detail/materials, observed canopy tops and observed control face/head poses. Actual native application and visual/driving review remain separate.')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'bindings.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.output / 'conditions.json').write_text(json.dumps(bound, indent=2) + '\n')
    print(dict(buildings=len(buildings), canopy=len(trees), controls=len(controls),
               unresolved=unresolved, passed=result['passed']), flush=True)


if __name__ == '__main__':
    main()
