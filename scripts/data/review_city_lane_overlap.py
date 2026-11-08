"""Find co-located source lanes across AV2 logs before city assembly."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from av2.map.map_api import ArgoverseStaticMap
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree

from scripts.data.source_lane_geometry import source_centerline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    observations, instances = {}, defaultdict(list)
    for source in catalog["candidates"]:
        static = ArgoverseStaticMap.from_json(args.data_root / source["vector_map"])
        lanes = static.vector_lane_segments
        for identity, lane in lanes.items():
            instances[identity].append(source["log_id"])
            if identity not in observations:
                center = source_centerline(static, lane)
                polygon = Polygon(np.r_[lane.left_lane_boundary.xyz[:, :2], lane.right_lane_boundary.xyz[::-1, :2]]).buffer(0)
                observations[identity] = {"lane": lane, "center": center, "shape": polygon}
    identities = list(observations)
    shapes = [observations[key]["shape"] for key in identities]
    tree = STRtree(shapes)
    pairs = []
    for i, key in enumerate(identities):
        first = observations[key]
        for j in tree.query(shapes[i], predicate="intersects"):
            if j <= i:
                continue
            other = identities[j]
            if set(instances[key]) & set(instances[other]):
                continue
            second = observations[other]
            if first["lane"].lane_type != second["lane"].lane_type:
                continue
            area = shapes[i].intersection(shapes[j]).area
            fraction = area / max(1e-8, min(shapes[i].area, shapes[j].area))
            if fraction < .6:
                continue
            a, b = first["center"], second["center"]
            direction_a, direction_b = a[-1, :2]-a[0, :2], b[-1, :2]-b[0, :2]
            cosine = float(direction_a@direction_b/max(1e-8, np.linalg.norm(direction_a)*np.linalg.norm(direction_b)))
            if cosine < .8:
                continue
            distance, nearest = cKDTree(a[:, :2]).query(b[:, :2])
            common = distance < 1.
            finite = common & np.isfinite(b[:, 2]) & np.isfinite(a[nearest, 2])
            dz = b[finite, 2]-a[nearest[finite], 2]
            endpoints = lambda points: [[float(v) if np.isfinite(v) else None for v in p] for p in points[[0,-1]]]
            pairs.append({"first_lane": key, "second_lane": other,
                          "first_logs": instances[key], "second_logs": instances[other],
                          "overlap_area_m2": area, "smaller_polygon_overlap_fraction": fraction,
                          "heading_cosine": cosine, "center_hausdorff_xy_m": LineString(a[:, :2]).hausdorff_distance(LineString(b[:, :2])),
                          "nearest_within_1m": int(common.sum()),
                          "median_second_minus_first_z_m": float(np.median(dz)) if len(dz) else None,
                          "p95_abs_z_difference_m": float(np.percentile(abs(dz),95)) if len(dz) else None,
                          "first_length_m": LineString(a[:, :2]).length, "second_length_m": LineString(b[:, :2]).length,
                          "first_start_end": endpoints(a), "second_start_end": endpoints(b)})
    report = {"city": catalog["city"], "unique_source_lane_ids": len(identities),
              "shared_id_count": sum(len(v)>1 for v in instances.values()),
              "lanes_with_missing_source_height": [key for key,v in observations.items() if not np.isfinite(v["center"][:,2]).all()],
              "co_located_distinct_id_candidates": pairs,
              "scope": "Cross-log geometry candidates for Agent source ownership review; no automatic lane identity or topology change."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({"lanes": len(identities), "co_located_candidates": len(pairs), "report": str(args.output)})


if __name__ == "__main__":
    main()
