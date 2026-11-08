"""Construct selected source-epoch road contacts with fixed neighboring support."""
import argparse
import json
from pathlib import Path

import numpy as np
import orjson
import shapely

from map_generation.av2_surfaces import mesh_triangles, native_mesh
from map_generation.shared_road_ground import clip_outside, realize_shared_ground


def realize(scene, roads, selection):
    index = {m["id"]: m for m in roads["meshes"]}
    lanes = {l["id"]: l for l in scene["fusion_map"]["lanes"]}
    owner = lanes[selection["lane_id"]]
    observation, = [o for o in owner["source_observations"]
                    if o["source"] == selection["observation_source"]
                    and o["source_lane_id"] == selection["observation_lane_id"]]
    origin = np.asarray(scene["coordinates"]["origin_city_xyz"])
    lane = dict(id=owner["id"], source_lane_id=observation["source_lane_id"],
                is_intersection=owner["is_intersection"])
    for side in ("left", "right"):
        points = observation["definition_differences"][side + "_lane_boundary"]
        lane[side + "_boundary"] = (
            np.asarray([[p[a] for a in ("x", "y", "z")] for p in points]) - origin).tolist()
    source_mesh, footprint = native_mesh(lane, [])
    retired = set(selection["retire_physical_mesh_ids"])
    local_ids = set(selection["mesh_ids"])
    local = [m for m in roads["meshes"] if m["id"] in local_ids]
    if set(m["id"] for m in local) != local_ids or not retired <= local_ids:
        raise ValueError("Selected local or retired physical owners are missing")
    covered = shapely.union_all([footprint] + [
        shapely.union_all(shapely.polygons(mesh_triangles(index[k])[:, :, :2]))
        for k in selection["coverage_support_mesh_ids"]])
    retirement = []
    for key in retired:
        surface = shapely.union_all(shapely.polygons(mesh_triangles(index[key])[:, :, :2]))
        uncovered = surface.difference(covered).area
        if uncovered > 1e-6:
            raise ValueError("Selected source pair does not cover retired physical alias: " + key)
        retirement.append(dict(mesh_id=key, area_m2=surface.area,
                               outside_selected_source_coverage_m2=uncovered))
    line = shapely.LineString(np.asarray(lane[selection["contact_side"] + "_boundary"])[:, :2])
    strip = footprint.intersection(line.buffer(selection["contact_width_m"], cap_style="flat"))
    source_mesh = clip_outside(source_mesh, footprint.difference(strip))
    source_mesh["support_domain_wkt"] = shapely.union_all(
        shapely.polygons(mesh_triangles(source_mesh)[:, :, :2])).wkt
    source_mesh.update(id=selection["id"] + "_observed",
                       source_lane_id=selection["lane_id"],
                       source_xodr_road_ids=sorted({r for k in retired
                           for r in index[k].get("source_xodr_road_ids", [])}),
                       source_observation=observation,
                       source_epoch_contact_selection=selection["id"])
    mutable = set(selection["inferred_mesh_ids"])
    modified = []
    for mesh in local:
        if mesh["id"] in retired:
            key = selection["id"] + "_transition_" + str(len(modified))
            modified.append(dict(mesh, id=key, source_lane_id=key,
                kind="inferred_source_epoch_transition",
                retired_physical_source_id=mesh["id"],
                source_epoch_contact_selection=selection["id"],
                inference_scope="Connector between the selected current-source contact and retained neighboring epochs; not a measured AV2 surface."))
            mutable.add(key)
        else:
            modified.append(mesh)
    modified.append(source_mesh)
    fixed = [m["id"] for m in modified if m["id"] not in mutable]
    domain = footprint.buffer(selection["transition_margin_m"], join_style="mitre")
    condition = dict(id=selection["id"],
        polygon_xy_m=shapely.get_coordinates(domain.exterior).tolist(),
        mesh_ids=[m["id"] for m in modified], fixed_support_mesh_ids=fixed,
        boundary_condition="entire_perimeter", height_energy="triangle_fem",
        height_solver="direct", contact_gap_m=0, reason=selection["reason"])
    out, reviews = realize_shared_ground(dict(roads, meshes=modified), [condition],
                                         selection["mesh_spacing_m"])
    after = {m["id"]: m for m in out["meshes"]}
    if any(after[m["id"]] != m for m in modified if m["id"] in fixed):
        raise ValueError("Source-epoch realization altered a retained fixed support")
    patch_id = reviews[0]["id"]
    patch = after[patch_id]
    patch["id"] = selection["id"] + "_floor"
    if patch.get("source_lane_id") == patch_id:
        patch["source_lane_id"] = patch["id"]
    patch["retired_physical_source_ids"] = sorted(retired)
    patch["source_epoch_contact_selection"] = selection["id"]
    outside = [m for m in roads["meshes"] if m["id"] not in local_ids]
    existing = {m["id"] for m in outside}
    for i, mesh in enumerate(out["meshes"]):
        if mesh["id"] in existing:
            previous = mesh["id"]
            mesh["id"] = selection["id"] + "_part_" + str(i)
            if mesh.get("source_lane_id") == previous:
                mesh["source_lane_id"] = mesh["id"]
    reviews[0]["realized_mesh_id"] = patch["id"]
    result = dict(roads, meshes=outside + out["meshes"])
    tri = mesh_triangles(patch)
    normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area = abs(normal[:, 2])/2
    grade = np.linalg.norm(normal[:, :2], axis=1)/np.maximum(abs(normal[:, 2]), 1e-15)
    before_domain = shapely.union_all([shapely.union_all(
        shapely.polygons(mesh_triangles(m)[:, :, :2])) for m in local])
    after_domain = shapely.union_all([shapely.union_all(
        shapely.polygons(mesh_triangles(m)[:, :, :2])) for m in out["meshes"] if m["faces"]])
    report = dict(selection=selection, retired_physical_aliases=retirement,
        observed_contact_strip_area_m2=strip.area, local_mesh_ids=sorted(local_ids),
        updated_mesh_ids=[m["id"] for m in out["meshes"]],
        unchanged_outside_meshes=len(outside), outside_mesh_objects_unchanged=True,
        fixed_supports_exact=True, raw_source_and_navigation_modified=False,
        source_xy_coverage_difference_m2=before_domain.symmetric_difference(after_domain).area,
        max_transition_grade=float(grade.max()),
        transition_area_grade_over20_m2=float(area[grade>.2].sum()),
        transition_area_grade_over40_m2=float(area[grade>.4].sum()),
        shared_ground_review=reviews, native_and_driving_status="pending",
        scope="Explicit source-period physical ownership and finite inferred connector. Exact source stripe and retained fixed supports; not all original physical representations unchanged.")
    result.setdefault("source_epoch_contact_reviews", []).append(report)
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scene", "roads", "conditions", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    scene = orjson.loads(args.scene.read_bytes())
    roads = orjson.loads(args.roads.read_bytes())
    conditions = json.loads(args.conditions.read_text())
    result, report = realize(scene, roads, conditions)
    args.output.mkdir(parents=True, exist_ok=True)
    flags = orjson.OPT_SERIALIZE_NUMPY
    (args.output/"road_meshes.json").write_bytes(orjson.dumps(result, option=flags))
    (args.output/"review.json").write_bytes(orjson.dumps(report, option=flags|orjson.OPT_INDENT_2))
    print(orjson.dumps(report, option=flags).decode(), flush=True)


if __name__ == "__main__":
    main()
