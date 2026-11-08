"""Inspect actual coarse-map geometry, source retention and the cross-junction route."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from map_generation.coarse import route_check
from map_generation.scene import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    config = json.loads(args.config.read_text())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    coarse = scene["coarse_map"]
    source = build_scene(config, args.data_root)
    route = route_check(scene, config)
    coordinates = np.concatenate([*[lane["centerline"] for lane in coarse["lanes"]], scene["terrain"]["vertices"]])
    polygons = [Polygon(b["exterior"], b["holes"]) for b in scene["buildings"]]
    route_xy = LineString(np.asarray(route["positions"])[:, :2])
    building_union = unary_union(polygons)
    building_overlap = max(0., sum(p.area for p in polygons) - building_union.area)
    route_building_intersection = route_xy.intersection(building_union).length
    route_points = np.asarray(route["positions"])
    delta = np.diff(route_points, axis=0)
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    grades = np.abs(delta[horizontal > .01, 2]) / horizontal[horizontal > .01]
    source_retained = all(scene[key] == source[key] for key in ("roads", "lanes", "controls", "assets", "protected"))
    bus_lanes = [lane for lane in coarse["lanes"]
                 if lane["edge_id"].split("#")[0] in config["coarse"].get("forward_bus_osm_way_ids", [])
                 and lane["id"].endswith("_0")]
    checks = {
        "original_av2_and_osm_layers_unchanged": source_retained,
        "road_and_terrain_xyz_finite": bool(np.isfinite(coordinates).all()),
        "junction_surfaces_have_valid_area": all(Polygon(j["boundary"]).is_valid and Polygon(j["boundary"]).area > 0 for j in coarse["junctions"]),
        "all_building_polygons_valid": all(p.is_valid and p.area > 0 for p in polygons),
        "prototype_building_footprints_do_not_overlap": building_overlap < .01,
        "all_building_roofs_above_sampled_ground": all(b["base_z_m"]+b["height_m"] > b["ground_range_m"][1] for b in scene["buildings"]),
        "configured_bus_lanes_restricted": (not config["coarse"].get("forward_bus_osm_way_ids", []) or bool(bus_lanes)) and all("bus" in l["allowed_classes"] and "passenger" not in l["allowed_classes"] for l in bus_lanes),
        "configured_passenger_route": route["passed"],
        "route_centerline_clear_of_buildings": route_building_intersection == 0,
        "terrain_covers_full_mesh": scene["terrain"]["coverage"]["dem_fraction"] == 1,
    }
    report = {"checks": checks, "passed": all(checks.values()),
              "counts": {"road_edges": sum(not e["internal"] for e in coarse["edges"]),
                         "road_lanes": sum(not l["internal"] for l in coarse["lanes"]),
                         "internal_lanes": sum(l["internal"] for l in coarse["lanes"]),
                         "junctions": len(coarse["junctions"]), "signal_programs": len(coarse["signal_programs"]),
                         "buildings": len(polygons), "buildings_with_holes": sum(bool(b["holes"]) for b in scene["buildings"]),
                         "terrain_vertices": len(scene["terrain"]["vertices"]), "terrain_triangles": len(scene["terrain"]["faces"])},
              "route": {k: v for k, v in route.items() if k != "positions"},
              "building_height_sources": dict(Counter(b["height_source"] for b in scene["buildings"])),
              "building_overlap_area_m2": building_overlap,
              "route_centerline_inside_buildings_m": route_building_intersection,
              "route_absolute_grade_p50_p95_max": np.percentile(grades, [50, 95, 100]).tolist(),
              "terrain_alignment": scene["terrain"]["alignment"],
              "terrain_coverage": scene["terrain"]["coverage"],
              "restricted_bus_lane_ids": [l["id"] for l in bus_lanes],
              "bus_lane_check": "checked" if config["coarse"].get("forward_bus_osm_way_ids", []) else "no_configured_bus_lane_patch",
              "scope": "Static coarse-map inspection; signal timing, final road surface, vehicle behavior and CARLA driving remain unvalidated."}
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    fig, ax = plt.subplots(figsize=(12, 10))
    for b in scene["buildings"]:
        xy = np.asarray(b["exterior"])
        ax.fill(xy[:, 0], xy[:, 1], color="#ccc4b7", lw=.3, edgecolor="#97886f")
        for hole in b["holes"]:
            xy = np.asarray(hole)
            ax.fill(xy[:, 0], xy[:, 1], color="white")
    for lane in coarse["lanes"]:
        xy = np.asarray(lane["centerline"])
        ax.plot(xy[:, 0], xy[:, 1], color="#bb7544", alpha=.6, lw=.6)
    for lane in scene["lanes"]:
        xy = np.asarray(lane["centerline"])
        ax.plot(xy[:, 0], xy[:, 1], color="#176ca1", lw=.7)
    xy = np.asarray(route["positions"])
    ax.plot(xy[:, 0], xy[:, 1], color="#d7191c", lw=2, label="Checked passenger route")
    for landmark in scene["areas"][0].get("landmarks", []):
        x, y = landmark["position"][:2]
        ax.scatter([x], [y], s=80, facecolors="none", edgecolors="black")
        ax.annotate(landmark["label"], (x, y), xytext=(6, 15), textcoords="offset points", fontsize=8)
    ax.plot([], [], color="#176ca1", label="AV2 source lanes (not fused)")
    ax.plot([], [], color="#bb7544", label="OSM / SUMO coarse lanes")
    ax.set(xlabel="Local east (m)", ylabel="Local north (m)", aspect="equal",
           title=f"{config['city']} coarse scene: source layers and continuous lane route")
    ax.legend(loc="upper right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "plan.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Coarse geometry check failed")


if __name__ == "__main__":
    main()
