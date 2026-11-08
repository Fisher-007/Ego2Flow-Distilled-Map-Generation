"""Build fusion-map roads and shared OpenDRIVE/Blender geometry, CPU only."""

import argparse
import json
import os
from pathlib import Path

from map_generation.road_builder import build_roads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trim", type=float, default=0.5, help="Maximum junction cut per lane end, metres")
    parser.add_argument("--fit-step", type=float, default=1.0, help="Maximum source chord sample interval, metres")
    parser.add_argument("--sample-step", type=float, default=0.5, help="Maximum road mesh sample interval, metres")
    parser.add_argument("--workers", type=int, default=1, help="Fitting workers; inherit the launcher's CPU affinity")
    parser.add_argument("--reuse-xodr", action="store_true", help="Resume surface sampling/checks from existing roads.xodr and road_lane_ids.json for this same source scene")
    parser.add_argument("--carla-root", type=Path, default=os.environ.get("CARLA_ROOT"),
                        help="Optional CARLA source root for its official GlobalRoutePlanner check")
    parser.add_argument('--reuse-samples',action='store_true',help='Reuse native sampling cache from this same compiled map after a downstream surface check failure')
    parser.add_argument("--native-planner", action="store_true",
                        help="Use the CARLA planner adapter with native endpoint identities and lane-bounded sampling")
    args = parser.parse_args()
    if min(args.trim, args.fit_step, args.sample_step) <= 0:
        parser.error("Distances must be positive")
    report = build_roads(json.loads(args.scene.read_text()), args.output, args.trim, args.fit_step, args.sample_step, args.carla_root, args.workers, args.reuse_xodr, args.reuse_samples, args.native_planner)
    print(json.dumps({k: report[k] for k in ("passed", "source_lane_count", "road_count", "junction_count",
                                           "source_successor_edges", "routing_issues", "carla_parameter_check", "mesh_count")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
