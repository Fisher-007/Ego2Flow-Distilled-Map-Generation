"""Summarize actual AV2 city coverage, duplicate lanes and ego trajectories."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    records, footprints, lanes_by_id, paths = [], {}, defaultdict(list), {}
    for index, candidate in enumerate(catalog["candidates"], 1):
        identity = candidate["log_id"]
        source = json.loads((args.data_root / candidate["vector_map"]).read_text())
        segments = list(source["lane_segments"].values())
        lines = []
        for lane in segments:
            left, right = [np.array([[p[k] for k in ("x", "y", "z")] for p in lane[side]])
                           for side in ("left_lane_boundary", "right_lane_boundary")]
            lanes_by_id[lane["id"]].append((identity, lane))
            lines.extend([left[:, :2], right[:, :2]])
        footprints[identity] = unary_union([LineString(line).buffer(2) for line in lines])
        poses = pd.read_feather(args.data_root / "sensor" / candidate["split"] / identity / "city_SE3_egovehicle.feather")
        xyz = poses[["tx_m", "ty_m", "tz_m"]].to_numpy()
        # Pose records are read in timestamp order, including stopped windows.
        xyz = xyz[np.argsort(poses["timestamp_ns"].to_numpy())]
        paths[identity] = xyz
        records.append(dict(candidate, plot_index=index,
                            driven_distance_m=float(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum()),
                            ego_bbox_city_xy=np.r_[xyz[:, :2].min(0), xyz[:, :2].max(0)].tolist(),
                            ego_start_city_xyz=xyz[0].tolist(), ego_end_city_xyz=xyz[-1].tolist()))
    graph = nx.Graph()
    graph.add_nodes_from(footprints)
    identities = list(footprints)
    overlaps = []
    for i, first in enumerate(identities):
        for second in identities[i+1:]:
            area = footprints[first].intersection(footprints[second]).area
            if area > 10:
                graph.add_edge(first, second)
                overlaps.append({"first": first, "second": second, "lane_boundary_overlap_m2": area})
    repeated = {str(key): values for key, values in lanes_by_id.items() if len(values) > 1}
    conflicts = []
    for key, values in repeated.items():
        first_id, first = values[0]
        for identity, value in values[1:]:
            changed = [field for field in first if first[field] != value.get(field)]
            if changed:
                conflicts.append({"source_lane_id": key, "first_log": first_id,
                                  "other_log": identity, "changed_fields": changed})
    all_xy = np.concatenate([p[:, :2] for p in paths.values()]) if paths else np.empty((0, 2))
    result = {"city": catalog["city"], "source_catalog": str(args.catalog),
              "source_index_complete": not catalog["failures"],
              "logs": records, "overlapping_log_pairs": overlaps,
              "source_coverage_components": [sorted(c) for c in nx.connected_components(graph)],
              "unique_numeric_lane_ids": len(lanes_by_id), "repeated_numeric_lane_ids": len(repeated),
              "lane_definition_conflicts": conflicts,
              "ego_bbox_city_xy": np.r_[all_xy.min(0), all_xy.max(0)].tolist() if len(all_xy) else None,
              "interpretation": "Components describe observed source overlap, not OSM connectivity. Logs and repeated recordings do not define distinct Areas."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    figure, axis = plt.subplots(figsize=(12, 10))
    for record in records:
        identity = record["log_id"]
        source = json.loads((args.data_root / record["vector_map"]).read_text())
        lines = [np.array([[p["x"], p["y"]] for p in lane[side]])
                 for lane in source["lane_segments"].values()
                 for side in ("left_lane_boundary", "right_lane_boundary")]
        axis.add_collection(LineCollection(lines, colors="#bcc6d4", linewidths=.35))
        path = paths[identity]
        axis.plot(path[:, 0], path[:, 1], linewidth=1.6)
        axis.annotate(str(record["plot_index"]), path[len(path)//2, :2], fontsize=8)
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set(xlabel="AV2 city x (m)", ylabel="AV2 city y (m)",
             title=f'{catalog["city"]}: {len(records)} source recordings\nLane boundaries and recorded ego paths')
    axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(args.output / "source_coverage.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"logs": len(records), "coverage_components": len(result["source_coverage_components"]),
                      "shared_lane_ids": len(repeated), "lane_conflicts": len(conflicts)}, indent=2))


if __name__ == "__main__":
    main()
