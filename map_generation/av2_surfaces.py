"""Restore physical surfaces from AV2 lane boundaries.

Transverse heights are represented in the mesh while OpenDRIVE retains its
centerline, topology, and lane identities.
"""

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points


def plane_z(triangles, xy):
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    return triangles[:, 0, 2] - (
        (xy - triangles[:, 0, :2]) * normals[:, :2]).sum(axis=1) / normals[:, 2]


def mesh_triangles(mesh):
    return np.asarray(mesh["vertices"], dtype=float)[np.asarray(mesh["faces"],dtype=int).reshape(-1,3)]


def remove_zero_area_faces(roads):
    """Remove exactly degenerate stored faces; retain all positive-area geometry."""
    removed = []
    for mesh in roads['meshes']:
        triangles = mesh_triangles(mesh)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        zero = (normals == 0).all(axis=1)
        if zero.any():
            removed.append(dict(mesh_id=mesh['id'], face_indices=np.flatnonzero(zero).tolist(),
                                removed_3d_area_m2=0.))
            mesh['faces'] = np.asarray(mesh['faces'])[~zero].tolist()
    roads['degenerate_face_removal'] = removed
    return removed


def native_mesh(lane, old_meshes):
    """GEOS triangulates the original polygon; its original XYZ knots are kept."""
    vertices = np.concatenate((lane["left_boundary"], lane["right_boundary"][::-1]))
    polygon = Polygon(vertices[:, :2])
    if not polygon.is_valid:
        raise ValueError(f"Invalid native polygon {lane['id']}: {shapely.is_valid_reason(polygon)}")
    index_by_xy = {}
    for i, xyz in enumerate(vertices):
        key = tuple(xyz[:2])
        if key in index_by_xy and vertices[index_by_xy[key], 2] != xyz[2]:
            raise ValueError(f"Conflicting original boundary heights at {lane['id']}: {key}")
        index_by_xy[key] = i
    parts = shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
    faces = [[index_by_xy[tuple(xy)] for xy in np.asarray(part.exterior.coords)[:3]]
             for part in parts]
    for face in faces:
        tri = vertices[face]
        if np.cross(tri[1] - tri[0], tri[2] - tri[0])[2] < 0:
            face.reverse()
    return {
        "id": "av2_boundary_surface_" + str(lane["source_lane_id"]),
        "kind": "av2_boundary_surface", "source_lane_id": lane["id"],
        "native_source_lane_id": lane["source_lane_id"],
        "is_intersection": lane["is_intersection"],
        "source_mesh_ids": [m["id"] for m in old_meshes],
        "source_xodr_road_ids": sorted({m["xodr_road_id"] for m in old_meshes}),
        "source_interval_m": None, "xodr_road_id": None,
        "xodr_lane_id": -1, "junction_id": None,
        "vertices": vertices.tolist(), "faces": faces,
        "left_boundary": lane["left_boundary"], "right_boundary": lane["right_boundary"],
    }, polygon


def shared_boundaries(lanes):
    pairs = sorted({tuple(sorted((key, lane[side]))) for key, lane in lanes.items()
                    for side in ("left_neighbor", "right_neighbor") if lane.get(side) in lanes})
    rows = []
    for a, b in pairs:
        options = []
        for sa in ("left_boundary", "right_boundary"):
            for sb in ("left_boundary", "right_boundary"):
                la, lb = LineString(lanes[a][sa]), LineString(lanes[b][sb])
                common = LineString(np.asarray(la.coords)[:, :2]).intersection(
                    LineString(np.asarray(lb.coords)[:, :2]))
                options.append((common.length, sa, sb, la, lb, common))
        length, sa, sb, la, lb, common = max(options, key=lambda row: row[0])
        errors = []
        for part in shapely.get_parts(common):
            if part.geom_type == "LineString":
                for xy in shapely.get_coordinates(shapely.segmentize(part, .5)):
                    p = Point(xy)
                    errors.append(abs(la.interpolate(la.project(p)).z - lb.interpolate(lb.project(p)).z))
        rows.append({"source_lane_ids": [a, b], "sides": [sa, sb],
                     "common_xy_length_m": length,
                     "max_shared_z_difference_m": max(errors) if errors else None})
    return rows


def center_residuals(native, polygons, original, physical_meshes=None):
    if physical_meshes is None:
        surfaces = {key: mesh_triangles(mesh) for key, mesh in native.items()}
        trees = {key: shapely.STRtree(shapely.polygons(tri[:, :, :2])) for key, tri in surfaces.items()}
    else:
        # Query every ground component belonging to the source lane.
        triangles = np.concatenate([mesh_triangles(m) for m in physical_meshes])
        tree = shapely.STRtree(shapely.polygons(triangles[:, :, :2]))
        surfaces = {key: triangles for key in native}
        trees = {key: tree for key in native}
    seen, rows = set(), []
    for mesh in original:
        key = mesh.get("source_lane_id")
        if key not in native:
            continue
        centers = (np.asarray(mesh["left_boundary"]) + np.asarray(mesh["right_boundary"])) / 2
        selected = []
        for xyz, rounded in zip(centers, np.round(centers, 5)):
            token = (key, *rounded)
            if token not in seen:
                seen.add(token)
                selected.append(xyz)
        if not selected:
            continue
        centers = np.asarray(selected)
        points = shapely.points(centers[:, :2])
        gaps = shapely.distance(polygons[key], points)
        xy = centers[:, :2].copy()
        # Preserve the same source-polygon projection used by the scalar
        # measurement, then query all points on this piece in one tree call.
        for index in np.flatnonzero(gaps):
            xy[index] = np.asarray(nearest_points(polygons[key], points[index])[0].coords)[0]
        pairs = trees[key].query(shapely.points(xy), predicate="dwithin", distance=1e-7)
        z = np.full(len(xy), -np.inf)
        np.maximum.at(z, pairs[0], plane_z(surfaces[key][pairs[1]], xy[pairs[0]]))
        if not np.isfinite(z).all():
            missing = xy[~np.isfinite(z)]
            raise ValueError(f"Actual physical layer misses source center {key} at {missing[:5].tolist()}")
        rows.extend({"source_lane_id": key, "is_intersection": native[key]["is_intersection"],
                     "center_xyz_m": xyz.tolist(), "xy_distance_to_surface_m": float(gap),
                     "surface_z_m": float(height), "surface_minus_native_z_m": float(height - xyz[2])}
                    for xyz, gap, height in zip(centers, gaps, z))
    summary = {}
    for name, subset in (("all", rows), ("non_intersection", [r for r in rows if not r["is_intersection"]]),
                         ("intersection", [r for r in rows if r["is_intersection"]])):
        if not subset:
            summary[name] = {"samples": 0, "abs_z_quantiles_0_50_90_99_100_m": [], "max_xy_distance_m": None, "worst_z_sample": None}
            continue
        errors = [abs(r["surface_minus_native_z_m"]) for r in subset]
        summary[name] = {"samples": len(subset),
                         "abs_z_quantiles_0_50_90_99_100_m": np.quantile(errors, [0, .5, .9, .99, 1]).tolist(),
                         "max_xy_distance_m": max(r["xy_distance_to_surface_m"] for r in subset),
                         "worst_z_sample": max(subset, key=lambda r: abs(r["surface_minus_native_z_m"]))}
    return summary, rows


def overlap_report(native, other=None):
    """Measure height differences at vertices of positive-area triangle intersections."""
    first = list(native.values())
    second = first if other is None else other
    triangles = np.concatenate([mesh_triangles(m) for m in second]) if second else np.empty((0,3,3))
    owners = np.concatenate([np.full(len(m["faces"]), i, dtype=int) for i, m in enumerate(second)]) if second else np.empty(0,dtype=int)
    polys = shapely.polygons(triangles[:, :, :2])
    tree = shapely.STRtree(polys)
    records = {}
    for i, mesh in enumerate(first):
        source = mesh_triangles(mesh)
        source_polys = shapely.polygons(source[:, :, :2])
        pairs = tree.query(source_polys, predicate="intersects")
        if other is None:
            pairs = pairs[:, owners[pairs[1]] > i]
        intersections = shapely.intersection(source_polys[pairs[0]], polys[pairs[1]])
        for n in np.flatnonzero(shapely.area(intersections) > 1e-10):
            ai, bi = pairs[:, n]
            partner = second[owners[bi]]
            key = (mesh["source_lane_id"], partner.get("source_lane_id") or partner["id"])
            part = intersections[n]
            xy = shapely.get_coordinates(part)
            za = plane_z(np.repeat(source[ai:ai+1], len(xy), axis=0), xy)
            zb = plane_z(np.repeat(triangles[bi:bi+1], len(xy), axis=0), xy)
            worst = int(np.argmax(abs(za - zb)))
            row = records.setdefault(key, {
                "source_lane_ids": list(key), "other_kind": partner.get("kind", "xodr_surface"),
                "involves_native_intersection": mesh["is_intersection"] or partner.get("is_intersection", False),
                "pairwise_overlap_area_m2": 0., "max_height_difference_m": -1.,
            })
            row["pairwise_overlap_area_m2"] += part.area
            if abs(za[worst] - zb[worst]) > row["max_height_difference_m"]:
                row.update(max_height_difference_m=float(abs(za[worst] - zb[worst])),
                           worst_xy_m=xy[worst].tolist(), surface_z_m=[float(za[worst]), float(zb[worst])])
    rows = sorted(records.values(), key=lambda r: r["max_height_difference_m"], reverse=True)
    return {"method": "Positive-area triangle overlap; exact plane-difference extrema at intersection vertices. Areas sum per surface pair, not road-union area.",
            "pairs": len(rows), "max_height_difference_m": max((r["max_height_difference_m"] for r in rows), default=0),
            "pairs_above_1cm": sum(r["max_height_difference_m"] > .01 for r in rows),
            "pairs_above_10cm": sum(r["max_height_difference_m"] > .1 for r in rows),
            "records": rows}


def seam_report(scene, lanes):
    fusion = {lane["id"]: lane for lane in scene["fusion_map"]["lanes"]}
    rows = []
    historical = []
    for match in scene["fusion_map"]["boundary_matches"]:
        entry = match["direction"] == "entry"
        seam = fusion.get(match["seam_id"])
        if seam is None:
            historical.append(dict(match=match, status="no_current_direct_seam_id",
                reason="Source assembly retained the historical match; physical closure reconstruction uses current successor IDs."))
            continue
        fine_end = np.asarray(seam["centerline"][-1 if entry else 0])
        for key in match["fine_lane_ids"]:
            lane = lanes.get(key) or fusion.get(key)
            if lane is None:
                historical.append(dict(match=match, fine_lane_id=key, status="no_current_direct_fine_id"))
                continue
            cap = np.asarray([lane["left_boundary"][0 if entry else -1],
                              lane["right_boundary"][0 if entry else -1]])
            rows.append({"seam_id": seam["id"], "fine_lane_id": key,
                         "physical_surface_restored": key in lanes,
                         "coarse_lane_id": match["coarse_lane_id"], "direction": match["direction"],
                         "fine_endpoint_xyz_m": fine_end.tolist(), "original_boundary_cap_xyz_m": cap.tolist(),
                         "cap_edge_z_minus_flat_seam_center_m": (cap[:, 2] - fine_end[2]).tolist(),
                         "coarse_endpoint_xyz_m": seam["centerline"][0 if entry else -1],
                         "existing_connector_length_xy_m": LineString(seam["centerline"]).length})
    return {"scope": "Existing coarse and generated connector meshes are unchanged; their flat fine-end caps can disagree with restored transverse heights.",
            "historical_matches_without_current_direct_lane": historical,
            "interfaces": len(rows), "connectors": len(scene["fusion_map"]["boundary_matches"]),
            "max_abs_cap_edge_height_difference_m": max((abs(z) for r in rows for z in r["cap_edge_z_minus_flat_seam_center_m"]),default=0.),
            "records": rows}


def restore_surfaces(scene, roads, lane_types=("VEHICLE", "BUS"), source_scope="observed"):
    from map_generation.road_builder import close_surface_connections

    # Group compiler pieces by source lane before restoring its physical boundary.
    sources={lane['id']:lane for lane in scene['lanes']+scene.get('av2_context',[])}
    aliases={lane['id']:lane.get('source_parent_lane_id',lane['id'])
             for lane in scene['fusion_map']['lanes'] if lane['source']=='av2'}
    aliases={key:value for key,value in aliases.items() if value in sources}
    if source_scope == "current_intervals":
        # Restore each retained source interval once.
        sources={lane["id"]:lane for lane in scene["fusion_map"]["lanes"] if lane["source"]=="av2"}
        aliases={key:key for key in sources}
    original=[dict(mesh,source_lane_id=aliases[mesh['source_lane_id']],compiler_source_lane_id=mesh['source_lane_id'])
              if mesh.get('source_lane_id') in aliases and aliases[mesh['source_lane_id']]!=mesh['source_lane_id'] else mesh
              for mesh in roads['meshes']]
    by_id = {mesh["id"]: mesh for mesh in original}
    by_source = {}
    for mesh in original:
        by_source.setdefault(mesh.get("source_lane_id"), []).append(mesh)
    lanes = {lane["id"]: lane for lane in sources.values()
             if lane["lane_type"] in lane_types and lane["id"] in by_source}
    native, polygons = {}, {}
    for key, lane in lanes.items():
        native[key], polygons[key] = native_mesh(lane, by_source[key])
        if source_scope == "current_intervals":
            native[key]["id"] = "av2_boundary_surface_" + key
            native[key]["source_parent_lane_id"] = lane.get("source_parent_lane_id")
            native[key]["source_parent_interval_m"] = lane.get("source_parent_interval_m")
    fine_union = shapely.union_all(list(polygons.values()))
    related = {m["id"] for m in original if m.get("kind") == "connection_closure" and
               (by_id[m["from_mesh_id"]]["source_lane_id"] in lanes or
                by_id[m["to_mesh_id"]]["source_lane_id"] in lanes)}
    kept = [m for m in original if m.get("source_lane_id") not in lanes and m["id"] not in related]
    meshes = kept + list(native.values())
    source_indices = {}
    for i, mesh in enumerate(meshes):
        if mesh.get("source_lane_id"):
            source_indices.setdefault(mesh["source_lane_id"], []).append(i)
    for indices in source_indices.values():
        indices.sort(key=lambda i: meshes[i].get("source_interval_m") or [0])
    links = {}
    for lane in scene["fusion_map"]["lanes"]:
        a = aliases.get(lane['id'],lane['id'])
        for child in lane["successors"]:
            b=aliases.get(child,child)
            if a==b:continue
            if (a in lanes or b in lanes) and a in source_indices and b in source_indices:
                links.setdefault(source_indices[a][-1], []).append(source_indices[b][0])
    # Use native lane caps to close uncovered parts of the road graph.
    prior_count = len(meshes)
    closure = close_surface_connections(meshes, {i: [i] for i in range(len(meshes))}, links)
    for record, mesh in zip(closure["patches"], meshes[prior_count:]):
        record["id"] = mesh["id"] = "av2_restored_" + mesh["id"]
    print(f"Restored {len(native)} native lanes; rebuilt {len(meshes) - prior_count} closure patches", flush=True)

    native_triangles = np.concatenate([mesh_triangles(m) for m in native.values()])
    restored_union = shapely.union_all(shapely.polygons(native_triangles[:, :, :2]))
    old_triangles = np.concatenate([mesh_triangles(m) for m in original])
    # Limit the coverage union to triangles intersecting the measured footprint.
    coverage_unions = []
    def local_coverage(triangles, label):
        polygons = shapely.polygons(triangles[:, :, :2])
        pairs = shapely.STRtree(polygons).query(shapely.get_parts(fine_union), predicate="intersects")
        coverage = shapely.union_all(polygons[np.unique(pairs[1])])
        record = dict(label=label, initial_validity=shapely.is_valid_reason(coverage),
                      initial_area_m2=coverage.area, method="triangle_union")
        if not coverage.is_valid:
            # Union nested polygon shells before measuring their area.
            coverage = shapely.union_all(shapely.get_parts(coverage))
            record["method"] = "union_invalid_coverage_parts"
            if not coverage.is_valid:
                raise ValueError("Invalid coverage union: " + shapely.is_valid_reason(coverage))
        record.update(final_validity=shapely.is_valid_reason(coverage),
                      area_delta_m2=coverage.area-record["initial_area_m2"])
        coverage_unions.append(record)
        return coverage
    old_union = local_coverage(old_triangles, "before")
    print("Measured original coverage within AV2 footprint", flush=True)
    new_triangles = np.concatenate([mesh_triangles(m) for m in meshes])
    new_union = local_coverage(new_triangles, "after")
    print("Measured restored coverage within AV2 footprint", flush=True)
    coverage = []
    for key, mesh in native.items():
        actual = shapely.union_all(shapely.polygons(mesh_triangles(mesh)[:, :, :2]))
        coverage.append({"source_lane_id": key, "source_polygon_area_m2": polygons[key].area,
                         "symmetric_difference_area_m2": actual.symmetric_difference(polygons[key]).area})
    centers, samples = center_residuals(native, polygons, original)
    shared = shared_boundaries(lanes)
    native_overlaps = overlap_report(native)
    retained_overlaps = overlap_report(native, kept)
    old_kept_closures = [r for r in roads["connection_closure"]["patches"] if r["id"] not in related]
    combined_closure = {
        "method": "Unaffected original closure patches retained; every closure touching replaced AV2 removed; only actual fusion successor cap gaps rebuilt against the new road footprint.",
        "patch_count": len(old_kept_closures) + closure["patch_count"],
        "patches": old_kept_closures + closure["patches"], "restored_fine_connections": closure,
    }
    report = {
        "method": "GEOS constrained Delaunay triangulation of each original AV2 left/right boundary polygon; original XYZ knots retained, interior Z linearly interpolated per triangle.",
        "compiler_children_to_source": {key:value for key,value in aliases.items() if key!=value},
        "source_scope": source_scope,
        "native_lane_count": len(native), "native_triangles": len(native_triangles),
        "output_mesh_count": len(meshes), "retained_mesh_count": len(kept),
        "retained_meshes_exactly_unchanged": all(by_id[m["id"]] == m for m in kept),
        "replaced_source_mesh_count": sum(m.get("source_lane_id") in lanes for m in original),
        "removed_related_closure_ids": sorted(related), "rebuilt_connections": closure,
        "coverage": {"source_union_area_m2": fine_union.area,
                     "union_diagnostics": coverage_unions,
                     "native_mesh_union_symmetric_difference_m2": restored_union.symmetric_difference(fine_union).area,
                     "fine_area_without_any_road_before_m2": fine_union.difference(old_union).area,
                     "fine_area_without_any_road_after_m2": fine_union.difference(new_union).area,
                     "per_lane": coverage},
        "native_centerline_to_physical_surface": centers,
        "shared_boundaries": {"pairs": len(shared), "max_z_difference_m": max(r["max_shared_z_difference_m"] or 0 for r in shared), "records": shared},
        "native_surface_overlaps": native_overlaps, "native_vs_retained_surface_overlaps": retained_overlaps,
        "fine_coarse_interfaces": seam_report(scene, lanes),
        "limitations": [
            "This file changes physical/render surfaces only. No OpenDRIVE file, centerline, topology, lane ID, native lane width or waypoint roll is changed.",
            "Native waypoint roll is zero; the report measures reference-height residuals against the physical mesh.",
            "Overlapping native junction surfaces and fine/coarse transverse mismatches are measured, not averaged or flattened.",
            "Interior surface heights are explicit interpolation of boundary measurements, not surveyed interior samples.",
            "Coarse and generated connector meshes are kept exactly; terrain carving, asset export and physical driving are separate steps.",
        ],
    }
    output = {**roads, "meshes": meshes, "connection_closure": combined_closure,
              "surface_scope": "AV2 physical surfaces restored from original boundary XYZ; other road surfaces retained. Native OpenDRIVE is unchanged and lacks transverse roll.",
              "av2_surface_restoration": report}
    return output, report, samples
