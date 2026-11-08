"""Export final OpenDRIVE road meshes, carved DEM and building masses to FBX.

Run with the map Python environment; this script clips terrain using Shapely,
then invokes Blender for the blend/FBX export. All input geometry is ENU meters.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Use a shared 0.01 mm Boolean grid before extruding terrain closures.
BOOLEAN_GRID_M = 1e-5


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def plane_heights(triangles, xy):
    """Interpolate original triangle planes at new boundary vertices."""
    import numpy as np

    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    delta = xy - triangles[:, 0, :2]
    return triangles[:, 0, 2] - (normals[:, :2] * delta).sum(axis=1) / normals[:, 2]


def surface_heights(triangles, tree, xy, label, coverage_tolerance_m=2 * BOOLEAN_GRID_M):
    import numpy as np
    import shapely

    # Boundary coordinates can differ from their source faces by GEOS roundoff.
    pairs = tree.query(shapely.points(xy), predicate="dwithin", distance=coverage_tolerance_m)
    heights = np.full(len(xy), -np.inf)
    np.maximum.at(heights, pairs[0], plane_heights(triangles[pairs[1]], xy[pairs[0]]))
    if not np.isfinite(heights).all():
        missing = xy[~np.isfinite(heights)]
        near = tree.nearest(shapely.points(missing))
        distances = shapely.distance(shapely.points(missing), tree.geometries[near])
        raise ValueError("Road boundary extends outside %s coverage: max distance %.9g m, first XY %s" %
                         (label, distances.max(), missing[:3].tolist()))
    return heights


def prepare_terrain(scene, roads, output, close_edges=True):
    import numpy as np
    import shapely
    from map_generation.mesh_geometry import spatial_union, carve_terrain

    road_triangles = np.concatenate([
        np.asarray(mesh["vertices"], dtype=float)[np.asarray(mesh["faces"], dtype=int)]
        for mesh in roads["meshes"]
    ])
    if road_triangles.shape[1:] != (3, 3):
        raise ValueError("Road builder must supply triangle faces")
    normals = np.cross(road_triangles[:, 1] - road_triangles[:, 0],
                       road_triangles[:, 2] - road_triangles[:, 0])
    if not np.isfinite(road_triangles).all() or (normals[:, 2] <= 0).any():
        raise ValueError("Road triangles must be finite with upward ENU winding")
    road_polygons = shapely.polygons(road_triangles[:, :, :2])
    original_footprint = spatial_union(road_polygons, grid_size=BOOLEAN_GRID_M)
    # Snap the outline once, then intersect the DEM at floating precision.
    footprint = shapely.orient_polygons(shapely.set_precision(
        shapely.set_precision(original_footprint, BOOLEAN_GRID_M), 0), exterior_cw=True)
    (output / 'road_footprint.wkt').write_text(shapely.to_wkt(footprint, rounding_precision=-1))
    print("Road footprint ready: %.1f m2" % footprint.area, flush=True)

    terrain = scene["terrain"]
    vertices = np.asarray(terrain["vertices"], dtype=float)
    faces = np.asarray(terrain["faces"], dtype=int)
    triangles = vertices[faces]
    # An already assembled terrain may contain vertical closure walls. They
    # cannot supply a height field for XY clipping and are rebuilt below.
    projected_area = np.cross(triangles[:, 1] - triangles[:, 0],
                              triangles[:, 2] - triangles[:, 0])[:, 2]
    top = np.abs(projected_area) > 1e-10
    faces, triangles = faces[top], triangles[top]
    polygons = shapely.polygons(triangles[:, :, :2])
    cut = carve_terrain(polygons, footprint)
    affected, clipped, parents = cut['affected'], cut['clipped'], cut['parents']
    xy = shapely.get_coordinates(clipped).reshape(-1, 4, 2)[:, :3]
    source_triangles = triangles[parents]
    z = plane_heights(np.repeat(source_triangles, 3, axis=0), xy.reshape(-1, 2)).reshape(-1, 3)
    added = np.concatenate((xy, z[:, :, None]), axis=2)
    added_faces = np.arange(added.size // 3).reshape(-1, 3) + len(vertices)
    winding = np.cross(added[:, 1] - added[:, 0], added[:, 2] - added[:, 0])[:, 2]
    added_faces[winding < 0] = added_faces[winding < 0, ::-1]
    carved_vertices = np.concatenate((vertices, added.reshape(-1, 3)))
    carved_faces = np.concatenate((faces[~affected], added_faces))
    # Compact away source DEM vertices wholly inside roads.
    used, indices = np.unique(carved_faces, return_inverse=True)
    carved_vertices, carved_faces = carved_vertices[used], indices.reshape(-1, 3)

    overlap, area_error = cut['overlap_m2'], cut['area_error_m2']
    if overlap > 1e-6 or area_error > 1e-6:
        raise ValueError("Terrain triangulation overlaps roads or changes clipped area: "
                         "overlap=%.9g m2, area_error=%.9g m2" % (overlap, area_error))
    print("Terrain clipped: %d triangles" % len(carved_faces), flush=True)
    np.savez(output / 'carved_top.npz', vertices=carved_vertices, faces=carved_faces)
    write_json(output / 'carved_top_review.json', dict(
        road_meshes=len(roads['meshes']), road_triangles=len(road_triangles),
        terrain_source_triangles=len(faces), terrain_removed_triangles=cut['removed_count'],
        terrain_clipped_triangles=cut['boundary_count'], road_footprint_computation_parts=cut['footprint_parts'],
        terrain_result_triangles=len(carved_faces), terrain_result_vertices=len(carved_vertices),
        terrain_road_overlap_m2=overlap, terrain_clip_area_error_m2=area_error,
        scope='Completed XY carve before ground fitting and edge closure; same input geometry required for reuse.'))

    # Build vertical skirts along the road union boundary, including internal holes.
    outlines = (shapely.get_parts(shapely.segmentize(shapely.boundary(footprint), 1.0))
                if close_edges else [])
    road_tree, terrain_tree = shapely.STRtree(road_polygons), shapely.STRtree(polygons)
    skirt_vertices, skirt_faces, deltas = [], [], []
    for outline in outlines:
        points = shapely.get_coordinates(outline)
        road_z = surface_heights(road_triangles, road_tree, points, "road mesh")
        ground_z = surface_heights(triangles, terrain_tree, points, "terrain")
        deltas.extend((road_z - ground_z).tolist())
        # Each segment gets its own vertices; skip zero-height triangles.
        for i in range(len(points) - 1):
            a = np.asarray([*points[i], road_z[i], ground_z[i]])
            b = np.asarray([*points[i + 1], road_z[i + 1], ground_z[i + 1]])
            da, db = a[2] - a[3], b[2] - b[3]
            sections = [a, b]
            # At a cut/fill transition, split at zero height to avoid a folded
            # wall quad where road and DEM exchange their vertical order.
            if da * db < 0:
                sections.insert(1, a + (b - a) * da / (da - db))
            for start, end in zip(sections, sections[1:]):
                quad = np.asarray([start[:3], end[:3],
                                   [*end[:2], end[3]], [*start[:2], start[3]]])
                base = len(skirt_vertices)
                skirt_vertices.extend(quad.tolist())
                for face in ((0, 1, 2), (0, 2, 3)):
                    tri = quad[list(face)]
                    if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-10:
                        skirt_faces.append([base + index for index in face])

    report = {
        "method": "Road-triangle XY union cleaned on a 0.01 mm grid, then subtracted from DEM triangles at floating precision; GEOS constrained Delaunay triangulation of boundary remnants",
        "seam": "Vertical sidewalls on the road union boundary only; road and original DEM plane heights sampled at at most 1 m spacing",
        "assumptions": ["Road geometry and elevation are used unchanged, without display offsets.",
                        "Sidewalls connect road boundaries to the terrain as inferred retaining edges.",
                        "Where road faces overlap, the highest face supplies the boundary wall height.",
                        "Terrain outside the road footprint keeps its original piecewise planar DEM height."],
        "road_meshes": len(roads["meshes"]),
        "connection_closure": roads.get("connection_closure"),
        "av2_surface_restoration": roads.get("av2_surface_restoration"),
        "road_triangles": len(road_triangles),
        "road_footprint_area_m2": float(footprint.area),
        "boolean_grid_m": BOOLEAN_GRID_M,
        "road_boolean_rounding_note": "The 0.01 mm grid also applies during road-outline union to resolve the actual non-noded microscopic intersections. Source road triangles remain unchanged. Terrain overlap and clipping area error are measured against that declared outline.",
        "terrain_source_triangles": len(faces),
        "terrain_removed_triangles": cut['removed_count'],
        "terrain_clipped_triangles": cut['boundary_count'],
        "road_footprint_computation_parts": cut['footprint_parts'],
        "terrain_result_triangles": len(carved_faces),
        "terrain_result_vertices": len(carved_vertices),
        "terrain_road_overlap_m2": overlap,
        "terrain_clip_area_error_m2": area_error,
        "seam_triangles": len(skirt_faces),
        "road_minus_dem_min_m": min(deltas) if deltas else None,
        "road_minus_dem_max_m": max(deltas) if deltas else None,
        "edge_closure_deferred_to_final_ground": not close_edges,
        "shapely_version": shapely.__version__,
        "geos_version": shapely.geos_version_string,
    }
    write_json(output / "terrain_mesh.json", {
        "vertices": carved_vertices.tolist(), "faces": carved_faces.tolist(),
        "seam_vertices": skirt_vertices, "seam_faces": skirt_faces, "report": report,
    })
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('connection_closure', 'av2_surface_restoration')}, indent=2), flush=True)


def export_blender(scene, roads, output, scene_path, roads_path, surfaces_only=False, tile_size_m=0., surface_kinds=None, center_tile_origins=False, coordinate_condition=None, road_patch=None):
    import bpy
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from map_generation.export_geometry import footprint_area, resolved_skirt_faces, resolved_source_faces, native_surface_bounds

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preview_coarse import building_mesh, material, mesh_object

    kinds = surface_kinds or ['road', 'terrain', 'road_edge']
    terrain = (json.loads((output / "terrain_mesh.json").read_text())
               if any(kind in kinds for kind in ('terrain', 'road_edge')) else None)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1
    bpy.context.scene.unit_settings.length_unit = "METERS"
    collection = bpy.data.collections.new("Final road and context meshes in ENU meters")
    bpy.context.scene.collection.children.link(collection)
    materials = {key: material(key, color) for key, color in {
        "road": (.22, .23, .24), "terrain": (.50, .57, .43),
        "building": (.67, .56, .44), "road_edge": (.37, .36, .33),
    }.items()}
    records = []
    omitted_geometry=[]

    def record(obj, kind, source_id, origin=None):
        xyz = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get('co', xyz)
        xyz = xyz.reshape(-1, 3)
        records.append({"name": obj.name, "kind": kind, "source_id": source_id,
                        "vertices": len(obj.data.vertices), "faces": len(obj.data.polygons),
                        "bounds_m": {"min": xyz.min(0).tolist(), "max": xyz.max(0).tolist()}})
        if origin is not None:
            records[-1]['asset_origin_enu_m']=origin.tolist()
            world=xyz.astype(float)+origin
            records[-1]['world_bounds_m']={'min':world.min(0).tolist(),'max':world.max(0).tolist()}
        if kind in ('road', 'terrain', 'road_edge'):
            faces = np.empty(len(obj.data.loops), dtype=np.int32)
            obj.data.loops.foreach_get('vertex_index', faces)
            records[-1]['native_precision'] = native_surface_bounds(xyz, faces.reshape(-1, 3))

    def surface_objects(kind, vertices, faces, source_id, pavement_face_indices=()):
        if not len(faces):
            return 0.
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        pavement = np.zeros(len(faces), dtype=bool)
        pavement[np.asarray(pavement_face_indices, dtype=int)] = True
        if kind in ('terrain', 'road_edge'):
            keep, diagnosis = resolved_source_faces(vertices, faces)
            if diagnosis['removed_triangles']:
                omitted_geometry.append(dict(kind=kind, stage='source_before_float',
                                             **diagnosis))
            faces = faces[keep]
            pavement = pavement[keep]
            if not len(faces):
                return 0.
        if tile_size_m:
            # Assign complete triangles to spatial assets.
            cells = np.floor(vertices[faces, :2].mean(1) / tile_size_m).astype(np.int64)
            _, group = np.unique(cells, axis=0, return_inverse=True)
            order = np.argsort(group, kind='stable')
            breaks = np.r_[0, np.flatnonzero(np.diff(group[order])) + 1, len(order)]
            parts = [order[start:min(start + 250000, high)] for low, high in zip(breaks[:-1], breaks[1:])
                     for start in range(low, high, 250000)]
        else:
            parts = [np.arange(len(faces))]
        error = 0.
        for index, ids in enumerate(parts):
            offset=(road_patch.get('surface_name_offsets',{}).get(kind,
                    road_patch.get('surface_name_offset',0) if kind=='road' else 0)
                    if road_patch else 0)
            name = kind + ('_' + str(index+offset) if tile_size_m else '')
            if kind=='road_edge':
                # Native import rejects zero-height closure sheets. Inspect
                # the same float coordinates Blender will actually export.
                used,local=np.unique(faces[ids],return_inverse=True)
                keep,diagnosis=resolved_skirt_faces(vertices[used],local.reshape(-1,3))
                if diagnosis['removed_triangles']:
                    omitted_geometry.append(dict(name=name,kind=kind,**diagnosis))
                ids=ids[keep]
                if not len(ids):continue
            used, local = np.unique(faces[ids], return_inverse=True)
            expected = vertices[used]
            origin=None
            if center_tile_origins:
                # Store tile-local vertices with city coordinates in the component placement.
                origin=np.r_[np.round((expected[:,:2].min(0)+expected[:,:2].max(0))/2),0.]
            exported=expected if origin is None else expected-origin
            obj = mesh_object(name, exported.tolist(), local.reshape(-1, 3).tolist(), materials[kind], collection)
            if pavement[ids].any():
                obj.data.materials.append(materials['road'])
                obj.data.polygons.foreach_set('material_index', pavement[ids].astype(np.int32))
            actual = np.empty(expected.size, dtype=np.float32)
            obj.data.vertices.foreach_get('co', actual)
            restored=actual.reshape(-1,3).astype(float)+(origin if origin is not None else 0.)
            error = max(error, float(np.max(np.abs(restored - expected))))
            obj['source_mesh_file'] = str(roads_path) if kind == 'road' else str(output / 'terrain_mesh.json')
            record(obj, kind, source_id, origin)
            if pavement[ids].any():
                records[-1]['material_kinds'] = [kind, 'road']
                records[-1]['material_face_counts'] = {kind:int((~pavement[ids]).sum()), 'road':int(pavement[ids].sum())}
            if tile_size_m:records[-1]['surface_tile_cell_xy']=cells[ids[0]].tolist()
        if error > .001:
            raise ValueError('Blender surface coordinate error exceeds 1 mm: ' + str(error))
        return error

    # Concatenate unchanged pieces into one road asset, avoiding thousands of
    # small UE assets/draw calls. Their provenance remains in road_meshes.json.
    kinds=surface_kinds or ['road','terrain','road_edge']
    road_coordinate_error=terrain_coordinate_error=edge_coordinate_error=None
    if 'road' in kinds:
        road_vertices, road_faces = [], []
        for mesh in roads["meshes"]:
            offset = len(road_vertices)
            road_vertices.extend(mesh["vertices"])
            road_faces.extend([[index + offset for index in face] for face in mesh["faces"]])
        road_coordinate_error = surface_objects('road', road_vertices, road_faces,
            str(roads_path) if tile_size_m else [mesh['id'] for mesh in roads['meshes']])
        del road_vertices, road_faces
    if 'terrain' in kinds:terrain_coordinate_error = surface_objects('terrain', terrain['vertices'], terrain['faces'], None, terrain.get('pavement_face_indices', ()))
    if 'road_edge' in kinds:edge_coordinate_error = surface_objects('road_edge', terrain['seam_vertices'], terrain['seam_faces'], None)
    for building in ([] if surfaces_only else scene["buildings"]):
        area=footprint_area(building)
        if area<=1e-8:
            omitted_geometry.append(dict(source_id=building['id'],kind='building',footprint_area_m2=area,
                reason='Numerically zero source footprint; source record retained'))
            continue
        from scripts.scene.export_source_asset_subset import triangulate_object
        obj=building_mesh(building, materials["building"], collection)
        triangles=triangulate_object(obj)
        record(obj, "building", building["id"])
        records[-1]["explicit_triangulation"]=triangles

    fbx_settings = {
        "global_scale": 1.0, "apply_unit_scale": True,
        "apply_scale_options": "FBX_SCALE_UNITS", "axis_forward": "-Y", "axis_up": "Z",
        "use_space_transform": True, "bake_space_transform": False, "bake_anim": False,
    }
    # The blend is inspectable in world space. FBX keeps local vertices at
    # zero transform; the native installer applies the recorded origins.
    for entry in records:
        if 'asset_origin_enu_m' in entry:bpy.data.objects[entry['name']].location=entry['asset_origin_enu_m']
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "map_assets.blend"))
    for entry in records:
        if 'asset_origin_enu_m' in entry:bpy.data.objects[entry['name']].location=(0.,0.,0.)
    bpy.ops.export_scene.fbx(filepath=str(output / "map_assets.fbx"),
                             use_selection=False, object_types={"MESH"}, **fbx_settings)
    report = {
        "scene": str(scene_path), "roads": str(roads_path), "blender_version": bpy.app.version_string,
        "surfaces_only": surfaces_only,
        "surface_kinds": kinds,
        "frame": "x east, y north, z up; right-handed; meters",
        "expected_ue_frame": "x east, y south, z up; left-handed; centimeters",
        "blend": str(output / "map_assets.blend"), "fbx": str(output / "map_assets.fbx"),
        "fbx_settings": fbx_settings, "objects": records,"omitted_degenerate_geometry":omitted_geometry,
        "terrain": terrain["report"] if terrain is not None else None,
        "building_assumption": "Flat mass geometry; base=min DEM, roof=max DEM plus OSM-tagged/assumed nominal height; holes preserved",
        "road_source_vertices_used_without_offsets": True,
        "road_connection_closure": roads.get("connection_closure"),
        "av2_surface_restoration": roads.get("av2_surface_restoration"),
        "road_blender_float_error_m": road_coordinate_error,
        "terrain_blender_float_error_m": terrain_coordinate_error,
        "edge_blender_float_error_m": edge_coordinate_error,
        "surface_tile_size_m": tile_size_m,
        "centered_surface_origins": center_tile_origins,
        "native_coordinate_condition": coordinate_condition,
        "surface_partition": "Whole unchanged triangles assigned by centroid; at most 250000 triangles per tiled asset; no geometry simplification",
        "collision": "Triangles exported for UE complex-as-simple collision; actual import and driving checks are separate",
    }
    if road_patch:
        report['replace_surface_names']=road_patch['replace_surface_names']
        report['road_patch']=road_patch
    (output / "assets.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Asset export complete: %s" % (output / "assets.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    default_blender = Path(__file__).resolve().parents[3] / "apps/blender-4.5.14-linux-x64/blender"
    parser.add_argument("--prepared-terrain", type=Path, help="Already carved, source-constrained terrain matching the supplied road surfaces")
    parser.add_argument("--blender", type=Path, default=default_blender)
    parser.add_argument("--blender-stage", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--surfaces-only", action="store_true", help="Export road, terrain and edges for an existing map; retain its building assets")
    parser.add_argument('--tile-size-m', type=float, default=0., help='Spatial surface assets for complete city export; zero retains one asset per surface')
    parser.add_argument('--surface-kinds', nargs='+', choices=['road','terrain','road_edge'], help='Export selected surface kinds for a partial native replacement')
    parser.add_argument('--center-tile-origins',action='store_true',help='Store surface vertices near their tile origin; install using the recorded world placement')
    parser.add_argument('--coordinate-conditions',type=Path,help='Graph-selected native surface coordinate convention')
    parser.add_argument('--road-patch',type=Path,help='Prepared complete-cell road replacement metadata')
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    coordinate_condition=None
    if args.coordinate_conditions:
        conditions=json.loads(args.coordinate_conditions.read_text())
        coordinate_condition=next(r for r in conditions['native_objects']if r['choice']=='enhance'
            and r.get('native_construction',{}).get('kind')=='native_surface_coordinates')
        settings=coordinate_condition['native_construction']['settings']
        args.center_tile_origins=settings['center_tile_origins']
        args.tile_size_m=settings['tile_size_m']
    scene_path, roads_path, output = args.scene.resolve(), args.roads.resolve(), args.output_dir.resolve()
    scene, roads = json.loads(scene_path.read_text()), json.loads(roads_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if args.blender_stage:
        patch=json.loads(args.road_patch.read_text()) if args.road_patch else None
        export_blender(scene, roads, output, scene_path, roads_path, args.surfaces_only, args.tile_size_m, args.surface_kinds, args.center_tile_origins, coordinate_condition,patch)
    else:
        needs_terrain = any(kind in (args.surface_kinds or ['road', 'terrain', 'road_edge'])
                            for kind in ('terrain', 'road_edge'))
        if needs_terrain:
            if args.prepared_terrain:
                import shutil
                shutil.copyfile(args.prepared_terrain, output / "terrain_mesh.json")
                prepared = json.loads((output / "terrain_mesh.json").read_text())
                if prepared.get('pavement_region_wkt'):
                    import numpy as np
                    import shapely
                    triangles = np.asarray(prepared['vertices'])[prepared['faces']]
                    paving = shapely.from_wkt(prepared['pavement_region_wkt'])
                    selected = shapely.covers(paving, shapely.polygons(triangles[:, :, :2]))
                    prepared['pavement_face_indices'] = np.flatnonzero(selected).tolist()
                    (output / "terrain_mesh.json").write_text(json.dumps(prepared))
                    print('Reviewed inferred paving faces: %d' % selected.sum(), flush=True)
            else:
                prepare_terrain(scene, roads, output)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        subprocess.run([str(args.blender.resolve()), "--background", "--threads", "6",
                        "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--",
                        "--scene", str(scene_path), "--roads", str(roads_path),
                        "--output-dir", str(output), "--blender-stage", '--tile-size-m', str(args.tile_size_m)] +
                       (["--surfaces-only"] if args.surfaces_only else [])+
                       (["--center-tile-origins"] if args.center_tile_origins else [])+
                       (["--coordinate-conditions",str(args.coordinate_conditions.resolve())] if args.coordinate_conditions else [])+
                       (["--road-patch",str(args.road_patch.resolve())] if args.road_patch else [])+
                       (["--surface-kinds"]+args.surface_kinds if args.surface_kinds else []), env=env, check=True)


if __name__ == "__main__":
    main()
