"""Expose distinct observed road junctions for the city-building Agent to select."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    junctions, sources = [], []
    for source in catalog["candidates"]:
        data = json.loads((args.data_root / source["vector_map"]).read_text())
        lanes = data["lane_segments"].values()
        polygons = []
        for lane in lanes:
            if lane["lane_type"] != "VEHICLE" or not lane["is_intersection"]:
                continue
            left, right = [[[p["x"], p["y"]] for p in lane[side]]
                           for side in ("left_lane_boundary", "right_lane_boundary")]
            polygon = Polygon(left+right[::-1]).buffer(0)
            polygons.append(polygon)
            junctions.append(polygon)
        pose_file = args.data_root / "sensor" / source["split"] / source["log_id"] / "city_SE3_egovehicle.feather"
        poses = pd.read_feather(pose_file).sort_values("timestamp_ns")
        xy = poses[["tx_m", "ty_m"]].to_numpy()
        sources.append((source, unary_union(polygons), LineString(xy), poses))
    merged = unary_union([polygon.buffer(2) for polygon in junctions])
    components = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
    rows = []
    for polygon in components:
        center = polygon.centroid
        related = []
        for source, region, ego, poses in sources:
            if region.intersection(polygon).area < 5:
                continue
            distance = ego.distance(center)
            xy = poses[["tx_m", "ty_m"]].to_numpy()
            nearest = int(np.argmin(np.linalg.norm(xy-np.asarray(center.coords)[0], axis=1)))
            related.append({"log_id": source["log_id"], "split": source["split"],
                            "ego_distance_m": distance,
                            "nearest_timestamp_ns": int(poses["timestamp_ns"].iloc[nearest]),
                            "junction_coverage_fraction": region.intersection(polygon).area/max(1.,polygon.area)})
        if not related:
            continue
        rows.append({"center_city_xy": list(center.coords)[0], "junction_bbox_city_xy": list(polygon.bounds),
                     "junction_area_m2": polygon.area, "sources": sorted(related,key=lambda r:r["ego_distance_m"]),
                     "minimum_ego_distance_m": min(r["ego_distance_m"] for r in related)})
    rows.sort(key=lambda row: (row["center_city_xy"][0], row["center_city_xy"][1]))
    for index, row in enumerate(rows, 1):
        row["candidate_id"] = f"{catalog['city'].lower()}_junction_{index:02d}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"city": catalog["city"], "candidates": rows,
        "scope": "Distinct geometric junction candidates for source-view and topology review; candidates for the requested map scope.",
        "method": "Union observed vehicle intersection polygons across logs with a 2m grouping margin; retain source timestamps and ego proximity."},indent=2)+"\n")
    print({"junction_candidates": len(rows), "within_60m_of_recorded_ego": sum(r['minimum_ego_distance_m']<=60 for r in rows)})


if __name__ == "__main__":
    main()
