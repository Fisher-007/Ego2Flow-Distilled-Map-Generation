"""Fuse the currently assembled fine sources into the same complete city network."""

import argparse
import json
from pathlib import Path

from map_generation.scene import build_scene
from map_generation.fusion import fuse_scene


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","coarse-scene","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--assembly",required=True,help="AV2 assembly relative to data root")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    config["sources"]["av2_assembly"]=args.assembly
    source=build_scene(config,args.data_root)
    scene=json.loads(args.coarse_scene.read_text())
    for key in ("lanes","assets","protected","source_info","areas","map_region_bbox_local_xy","frame_validation","reference_state"):
        scene[key]=source[key]
    print({"stage":"sources_loaded","fine_lanes":len(scene["lanes"]),"declared_areas":len(scene["areas"])},flush=True)
    result=fuse_scene(scene,config)
    args.output.mkdir(parents=True,exist_ok=True)
    fusion=result["fusion_map"]
    report={"assembled_source_maps":source["source_info"]["av2_maps"],"fine_lanes":len(source["lanes"]),
            "fused_lanes":len(fusion["lanes"]),"matched_boundaries":len(fusion["boundary_matches"]),
            "unresolved_boundaries":fusion["unresolved_boundaries"],
            "scope":"Whole declared coarse region, with the current explicit fine-source assembly; native validation pending."}
    (args.output/"fusion_review.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.output/"scene.json").open("w") as stream:
        json.dump(result,stream,separators=(",",":"),allow_nan=False)
    print({k:v for k,v in report.items() if k not in ("assembled_source_maps","unresolved_boundaries")},flush=True)
    print({"unresolved_boundaries":len(report["unresolved_boundaries"])},flush=True)


if __name__=="__main__":main()
