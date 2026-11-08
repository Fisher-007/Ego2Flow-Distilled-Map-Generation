"""Create city configuration and review areas from selected source candidates."""

import argparse
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(selection, catalog, candidates, data_root, output):
    intent = json.loads(selection.read_text())
    inventory = json.loads(catalog.read_text())
    available = json.loads(candidates.read_text())
    city = intent["city"]
    if inventory["city"] != city or available["city"] != city:
        raise ValueError("Selection and source inventories must name the same city")
    by_candidate = {v["candidate_id"]: v for v in available["candidates"]}
    by_log = {v["log_id"]: v for v in inventory["candidates"]}
    identities = intent["selected_candidate_ids"]
    if len(set(identities)) != len(identities):
        raise ValueError("An Area candidate cannot be selected twice")
    bounds = np.asarray(intent["bbox_city_xy"], dtype=float)
    areas, primary_logs = [], []
    for index, identity in enumerate(identities, 1):
        candidate = by_candidate[identity]
        observations = candidate["sources"]
        primary = observations[0]
        if primary["log_id"] not in primary_logs:
            primary_logs.append(primary["log_id"])
        core = np.asarray(candidate["junction_bbox_city_xy"], dtype=float)
        for extra in intent.get("core_companion_candidates", {}).get(identity, []):
            other = np.asarray(by_candidate[extra]["junction_bbox_city_xy"])
            core = np.r_[np.minimum(core[:2], other[:2]), np.maximum(core[2:], other[2:])]
        core += np.asarray([-30, -30, 30, 30])
        influence = core + np.asarray([-120, -120, 120, 120])
        if np.any(influence[:2] < bounds[:2]) or np.any(influence[2:] > bounds[2:]):
            raise ValueError("Declared rectangle must contain selected Area influence: " + identity)
        areas.append({
            "id": f"{city.lower()}_{index:02d}", "candidate_id": identity,
            "center_city_xy": candidate["center_city_xy"],
            "core_bbox_city_xy": core.tolist(), "influence_bbox_city_xy": influence.tolist(),
            "source_observations": observations, "task_family": "junction_and_approach",
            "selection_status": "source_geometry_selected; source_image_and_native_review_pending",
        })
    center = (bounds[:2] + bounds[2:]) / 2
    origin_options = []
    for identity in primary_logs:
        source = by_log[identity]
        vector = data_root / "av2" / source["vector_map"]
        if not vector.is_file():
            raise FileNotFoundError(vector)
        pose = data_root / "av2/sensor" / source["split"] / identity / "city_SE3_egovehicle.feather"
        if not pose.is_file():
            raise FileNotFoundError(pose)
        lanes = json.loads(vector.read_text())["lane_segments"].values()
        for lane in lanes:
            if lane["lane_type"] != "VEHICLE":
                continue
            for side in ("left_lane_boundary", "right_lane_boundary"):
                for vertex_index, point in enumerate(lane[side]):
                    xyz = [point[k] for k in ("x", "y", "z")]
                    if np.isfinite(xyz).all():
                        origin_options.append((float(np.linalg.norm(np.asarray(xyz[:2])-center)),
                                               identity, lane["id"], side, vertex_index, xyz))
    _, main_log, main_lane, side, vertex_index, origin = min(origin_options)
    source = by_log[main_log]
    log_paths = ["av2/sensor/" + by_log[i]["split"] + "/" + i for i in primary_logs]
    terrain_bounds = (bounds + [-450, -450, 450, 450] - np.tile(origin[:2], 2)).tolist()
    config = {
        "prototype_id": city.lower() + "_city",
        "purpose": f"Complete {city} declared road rectangle with {len(areas)} distinct AV2-supported Areas",
        "development_only": True, "city": city, "log_id": main_log, "split": source["split"],
        "coordinates": {
            "frame": "AV2 city", "units": "m", "bbox_order": "xmin,ymin,xmax,ymax",
            "origin_city_xyz": origin,
            "origin_z_source": "Exact finite AV2 vehicle-lane boundary vertex nearest the declared rectangle center; source XYZ retained.",
            "origin_evidence": {"kind": "av2_boundary_vertex", "log_id": main_log,
                                "vector_map": source["vector_map"], "lane_id": main_lane,
                                "side": side, "vertex_index": vertex_index},
        },
        "bbox_city_xy": bounds.tolist(),
        "sources": {
            "city_catalog": str(catalog.resolve()),
            "av2_log": "av2/sensor/" + source["split"] + "/" + main_log,
            "av2_logs": log_paths, "osm": "osm/" + city.lower() + "_region.osm",
            "terrain_dem": "terrain/" + city.lower() + "_region/usgs_3dep.tif",
        },
        "landmarks": [],
        "coordinate_check_route": {
            "id": "av2_coordinate_probe", "lane_ids": [main_lane],
            "scope": "One measured vehicle lane for frame validation only; not a driving route."
        },
        "coarse": {
            "buffer_m": 150, "default_lane_width_m": 3.2,
            "terrain_spacing_m": 4, "default_building_height_m": 6,
            "building_level_height_m": 3, "terrain_bounds_local_xy": terrain_bounds,
            "turnarounds_require_dedicated_lane": True, "join_nearby_junctions": False,
            "check_route": {
                "start_city_xy": areas[0]["center_city_xy"],
                "end_city_xy": areas[-1]["center_city_xy"],
                "status": "Select actual passenger-permitted lane IDs after OSM conversion",
            },
            "assumptions": [
                "Default lane width is 3.2 m and building height is 6 m where source measurements or tags are unavailable.",
                "USGS raster identity, vertical alignment and source epoch will be selected from actual regional evidence.",
                "Do not treat bare-earth height as an elevated road deck or flatten AV2 lane Z.",
                "Retain source permissions and narrow non-vehicle paths; no inferred default U-turn without a dedicated lane.",
                "Current OSM and AV2 recorded-state differences require explicit fusion review.",
            ],
        },
        "fusion": {
            "transition_margin_m": 12, "max_seam_distance_m": 45, "anchor_distance_m": 6,
            "context_bbox_city_xy": bounds.tolist(), "context_lane_ids": [],
            "context_reason": f"Use actual {city} source catalog and explicit ownership review; no other city source decisions inherited.",
            "default_vehicle_speed_mps": 11.176, "default_bicycle_speed_mps": 5.56,
            "speed_assumption": "Configured operating speed caps.",
            "ownership_evidence_margin_m": 2,
            "status": "source_registration_and_ownership_review_pending",
        },
        "development_context": {"source_recordings": len(primary_logs), "selection": intent["selection_rationale"]},
        "areas": areas,
        "scope": "All roads inside the declared rectangle, including the unobserved connecting roads; selected Areas do not clip the full map.",
        "reference_state": {"kind": "per_area_recorded_joint_state", "source": str(selection.resolve()),
                            "note": "Each Area retains its own log/timestamp; these are not one simultaneous citywide observation."},
    }
    write_json(output / "city_config.json", config)
    distances = [float(np.linalg.norm(np.asarray(a["center_city_xy"])-b["center_city_xy"]))
                 for i, a in enumerate(areas) for b in areas[i+1:]]
    record = {
        **intent, "areas": areas, "city_config": str((output/"city_config.json").resolve()),
        "source_catalog": str(catalog.resolve()), "candidate_catalog": str(candidates.resolve()),
        "primary_source_logs": primary_logs, "coordinates": config["coordinates"],
        "minimum_area_center_separation_m": (min(distances) if distances else None),
        "primary_observation_maximum_ego_distance_m": max(a["source_observations"][0]["ego_distance_m"] for a in areas),
        "raw_sources_verified": True, "review_status": "source_images_and_region_connectivity_pending",
    }
    write_json(output / "area_selection.json", record)
    print(json.dumps({"city": city, "areas": len(areas), "primary_logs": len(primary_logs),
                      "origin_city_xyz": origin, "minimum_center_separation_m": (min(distances) if distances else None),
                      "config": str(output/"city_config.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "catalog", "candidates", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.selection, args.catalog, args.candidates, args.data_root, args.output)


if __name__ == "__main__":
    main()
