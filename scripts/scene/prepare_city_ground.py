"""Realize graph-selected road, terrain and foundation height consistency."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import shapely

from map_generation.frontage_ground import smooth_ground
from map_generation.av2_surfaces import remove_zero_area_faces
from map_generation.mesh_geometry import spatial_union, compact_vertices
from scripts.scene.build_assets import prepare_terrain, surface_heights, BOOLEAN_GRID_M


def at_grade_roads(scene, roads, buffer_m, propagate_internal=False, building_passage_at_grade=False):
    lanes = {lane['id']: lane for lane in scene['fusion_map']['lanes']}
    separate = set()
    for key, lane in lanes.items():
        tags = lane.get('source_tags', {})
        exempt_tunnel = building_passage_at_grade and tags.get('tunnel') == 'building_passage'
        if (tags.get('bridge') not in (None, 'no', 'false', '0') or
            (not exempt_tunnel and tags.get('tunnel') not in (None, 'no', 'false', '0')) or str(tags.get('layer', '0')) != '0'):
            separate.add(key)
    # Carry source layer relations through height fitting into terrain construction.
    for relation in scene.get('physical_layer_relations',[]):
        excluded=relation.get('terrain_carving_excluded_source_ids',[relation['a'],relation['b']])
        separate.update(key for key in excluded if key in lanes)
    if propagate_internal:
        neighbors={k:set(l['successors']) for k,l in lanes.items()}
        for key,lane in lanes.items():
            for target in lane['successors']:
                if target in neighbors:neighbors[target].add(key)
        pending=list(separate)
        while pending:
            key=pending.pop()
            for target in neighbors[key]:
                if target not in separate and target in lanes and lanes[target].get('internal'):
                    separate.add(target);pending.append(target)
    by_id = {m['id']: m for m in roads['meshes']}
    ground, excluded = [], []
    for mesh in roads['meshes']:
        keys = {mesh.get('source_lane_id')} | set(mesh.get('source_lane_ids',[]))
        if mesh.get('kind') == 'connection_closure':
            keys.update(mesh[side + '_source_lane_id'] if side + '_source_lane_id' in mesh
                        else by_id[mesh[side + '_mesh_id']].get('source_lane_id') for side in ('from', 'to'))
        # Classify shared ground by its realized domain rather than nearby source aliases.
        is_shared_ground=mesh.get('kind')=='inferred_shared_road_ground'
        (excluded if keys & separate and not is_shared_ground else ground).append(mesh)
    protection = shapely.union_all([shapely.LineString(lanes[key]['centerline']).buffer(
        lanes[key]['width_m'] / 2 + buffer_m) for key in separate])
    return dict(roads, meshes=ground), protection, sorted(separate), [m['id'] for m in excluded]


def fit_foundations(scene, terrain, roads):
    # Fit foundations before reconstructing elevated parts and passage openings.
    layered = scene.get('building_layer_source_buildings', [])
    if layered:
        scene['buildings'] = deepcopy(layered)
    canonical = [] if layered else scene.get('building_passage_source_buildings', [])
    if canonical:
        parents = {b['id'] for b in canonical}
        scene['buildings'] = [b for b in scene['buildings']
            if b.get('passage_parent_id', b['id']) not in parents] + canonical
    ground_triangles = np.asarray(terrain['vertices'])[terrain['faces']]
    ground_tree = shapely.STRtree(shapely.polygons(ground_triangles[:, :, :2]))
    road_triangles = np.concatenate([np.asarray(m['vertices'])[m['faces']] for m in roads['meshes']])
    road_tree = shapely.STRtree(shapely.polygons(road_triangles[:, :, :2]))
    records = []
    for building in scene['buildings']:
        polygon = shapely.Polygon(building['exterior'], building.get('holes', []))
        xy = shapely.get_coordinates(shapely.line_interpolate_point(polygon.exterior,
             np.linspace(0, polygon.length, max(8, int(np.ceil(polygon.length))))))
        covered = np.zeros(len(xy), dtype=bool)
        covered[ground_tree.query(shapely.points(xy), predicate='dwithin', distance=1e-7)[0]] = True
        z = np.empty(len(xy))
        if covered.any():
            z[covered] = surface_heights(ground_triangles, ground_tree, xy[covered], 'city foundation terrain')
        if (~covered).any():
            z[~covered] = surface_heights(road_triangles, road_tree, xy[~covered], 'existing building road overlap')
        old = [building['base_z_m'], building['roof_z_m']]
        building.update(base_z_m=float(z.min() - .12), roof_z_m=float(z.max() + building['nominal_height_m']),
                        entrance_z_m=float(np.median(z)), final_support_range_m=[float(z.min()), float(z.max())])
        building['height_m'] = building['roof_z_m'] - building['base_z_m']
        records.append(dict(source_id=building['id'], old_base_roof_z_m=old,
                            final_base_roof_z_m=[building['base_z_m'], building['roof_z_m']],
                            support_range_m=building['final_support_range_m'], road_overlap_samples=int((~covered).sum())))
    if layered:
        from map_generation.building_layers import realize
        scene['building_layer_source_buildings'] = deepcopy(scene['buildings'])
        previous = scene['building_layer_review']['passages']
        scene.update(realize(scene, default_clearance_m=previous['default_clearance_m'],
            lateral_margin_m=previous['lateral_margin_m'],
            vehicle_length_m=previous.get('vehicle_body_length_m',4.8),
            vehicle_width_m=previous.get('vehicle_body_width_m',2.2))[0])
    elif canonical:
        from map_generation.building_passages import realize
        previous = scene['building_passage_review']
        scene.update(realize(scene, previous['default_clearance_m'], previous['lateral_margin_m'],
            previous.get('vehicle_body_length_m',4.8),previous.get('vehicle_body_width_m',2.2))[0])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'roads', 'conditions', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--source-terrain-scene', type=Path,
        help='Canonical source scene whose uncarved terrain is used when rebuilding changed road footprints')
    parser.add_argument('--resume-carved', type=Path)
    parser.add_argument('--resume-ground', type=Path,
                        help='Completed terrain/support report directory for foundation-only feedback')
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    scene, roads, conditions = read(args.scene), read(args.roads), read(args.conditions)
    if args.source_terrain_scene:
        source=read(args.source_terrain_scene)
        if source['coordinates']['origin_city_xyz']!=scene['coordinates']['origin_city_xyz']:
            raise ValueError('Canonical terrain and current road scene use different coordinate origins')
        scene['terrain']=deepcopy(source['terrain'])
        del source
    removed = remove_zero_area_faces(roads)
    choice = next(r for r in conditions['native_objects'] if r.get('native_construction', {}).get('kind') == 'city_ground_support' and r['choice'] == 'enhance')
    settings = choice['native_construction']['settings']
    ground_roads, protected, separate, excluded = at_grade_roads(scene, roads, settings['separate_level_buffer_m'])
    args.output.mkdir(parents=True, exist_ok=True)
    selection = dict(condition=choice, input_scene=str(args.scene), input_roads=str(args.roads),
                     separate_level_sources=separate, excluded_terrain_carving_mesh_ids=excluded,
                     method='Road geometry stays intact. Tagged bridge/tunnel/separate-level surfaces stay separate from terrain; at-grade road edges constrain inferred ground and existing building foundations.')
    selection['source_terrain_scene']=str(args.source_terrain_scene)if args.source_terrain_scene else str(args.scene)
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    if args.resume_ground:
        terrain = read(args.resume_ground / 'terrain_mesh.json')
        review = terrain['report']['continuous_frontage']
        selection['reused_final_ground'] = str(args.resume_ground)
    elif args.resume_carved:
        before = shapely.from_wkt((args.resume_carved.parent / 'road_footprint.wkt').read_text())
        carve_review = read(args.resume_carved.with_name('carved_top_review.json'))
        if 'current_roads' in carve_review:
            # Read the current-input metadata attached to incremental DEM carves.
            source_selection = dict(input_scene=carve_review.get('input_scene'),
                                    input_roads=carve_review['current_roads'])
        else:
            source_selection = read(args.resume_carved.parent.parent / 'selection.json')
        if all(source_selection[k] == selection[k] for k in ('input_scene', 'input_roads')):
            # Reuse matching source files; compare footprints when the paths differ.
            difference = 0.
            selection['carving_reuse'] = 'Same explicitly retained input scene and road files'
        else:
            triangles = np.concatenate([np.asarray(m['vertices'])[m['faces']] for m in ground_roads['meshes']])
            current = shapely.set_precision(shapely.set_precision(
                spatial_union(shapely.polygons(triangles[:, :, :2]), grid_size=BOOLEAN_GRID_M), BOOLEAN_GRID_M), 0)
            difference = before.symmetric_difference(current).area
        selection['reused_carving_footprint_difference_m2'] = difference
        if difference > 1e-6:
            raise ValueError(f'Road footprint changed by {difference} m2; rebuild terrain carving for the revised roads')
        if args.resume_carved.suffix == '.npz':
            stored = np.load(args.resume_carved)
            terrain = dict(vertices=stored['vertices'].tolist(), faces=stored['faces'].tolist(),
                seam_vertices=[], seam_faces=[], report=read(args.resume_carved.with_name('carved_top_review.json')))
        else:
            terrain = read(args.resume_carved)
    else:
        carved = args.output / 'carved'; carved.mkdir(exist_ok=True)
        # Defer closure generation until the ground fit completes.
        prepare_terrain(scene, ground_roads, carved, close_edges=False)
        terrain = read(carved / 'terrain_mesh.json')
    if not args.resume_ground:
        review = fit_city_ground(scene, terrain, ground_roads, protected, settings, args, selection)
    (args.output / 'terrain_mesh.json').write_text(json.dumps(terrain, separators=(',', ':')) + '\n')
    print('Fitting building foundations', flush=True)
    foundations = fit_foundations(scene, terrain, roads)
    scene['terrain'] = dict(scene['terrain'], vertices=terrain['vertices'], faces=terrain['faces'])
    scene['city_ground_support'] = dict(selection_file=str(args.output / 'selection.json'), condition_id=choice['source_id'])
    (args.output / 'scene.json').write_text(json.dumps(scene, separators=(',', ':')) + '\n')
    (args.output / 'road_meshes.json').write_text(json.dumps(roads, separators=(',', ':')) + '\n')
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    report = dict(ground=review, foundations=foundations, separate_level_sources=separate,
                  building_footprints_retained=True, nominal_building_heights_retained=True,
                  source_supported_passages=scene.get('building_passage_review'),
                  source_road_geometry_and_rules_retained=True,
                  zero_area_faces_removed=removed,
                  scope='Inferred context support around retained roads. Observed curbs, accesses, bridge decks and driving require their separate source-dependent construction and review.')
    (args.output / 'support_review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(ground=review, buildings=len(foundations), separate_level_sources=len(separate))), flush=True)


def fit_city_ground(scene, terrain, ground_roads, protected, settings, args, selection):
    print('Sharing duplicated carved-terrain vertices', flush=True)
    selection['terrain_vertex_compaction'] = compact_vertices(terrain)
    # Construct closure skirts from the solved terrain.
    terrain['seam_vertices'], terrain['seam_faces'] = [], []
    vertices = np.asarray(terrain['vertices'])
    selected = shapely.box(*vertices[:, :2].min(0), *vertices[:, :2].max(0))
    observed_z = None
    if args.data_root:
        from map_generation.observed_ground import context_ground
        observed_z, observed_review = context_ground(scene, vertices[:, :2], args.data_root,
            settings['observed_ground_radius_m'], settings.get('observed_ground_quality'))
        (args.output / 'observed_ground_review.json').write_text(json.dumps(observed_review, indent=2) + '\n')
    print('Fitting continuous city ground', flush=True)
    footprint_path = (args.resume_carved.parent if args.resume_carved else args.output / 'carved') / 'road_footprint.wkt'
    footprint = shapely.from_wkt(footprint_path.read_text())
    review = smooth_ground(terrain, ground_roads, selected, protected, settings, observed_z, footprint)
    return review


if __name__ == '__main__':
    main()
