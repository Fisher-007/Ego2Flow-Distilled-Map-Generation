"""Find vehicle corridors overlapping pedestrian areas or source access restrictions."""

import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree


def inspect(scene, distance=6., overlap_fraction=.6):
    # Use source permissions to distinguish vehicle-access conflicts from adjacent paths.
    roads = [r for r in scene["roads"] if r["tags"].get("highway") == "pedestrian"
             or r["tags"].get("motor_vehicle") == "no" or r["tags"].get("motorcar") == "no"]
    shapes = [LineString(np.asarray(r["reference_line"], dtype=object)[:, :2].astype(float)) for r in roads]
    index = STRtree(shapes)
    findings = []
    for lane in scene["lanes"]:
        if lane["lane_type"] != "VEHICLE" or lane["is_intersection"]:
            continue
        p = np.asarray(lane["centerline"])
        line = LineString(p[:, :2]);direction = p[-1, :2]-p[0, :2]
        if np.linalg.norm(direction) < .1:
            continue
        for i in index.query(line.buffer(distance)):
            source = shapes[i];q = np.asarray(source.coords)
            heading = q[-1]-q[0]
            cosine = abs(float(direction@heading/max(1e-8,np.linalg.norm(direction)*np.linalg.norm(heading))))
            covered = line.intersection(source.buffer(distance)).length
            if cosine < .8 or covered < 3 or covered/max(line.length,1e-8) < overlap_fraction:
                continue
            findings.append(dict(lane_id=lane["id"], road_id=roads[i]["id"],
                lane_source_observations=lane.get("source_observations",[]), osm_tags=roads[i]["tags"],
                centerline_distance_m=line.distance(source), overlap_length_m=covered,
                overlap_fraction=covered/line.length, heading_abs_cosine=cosine,
                status="source_state_candidate_requires_review",
                question="Check source imagery, access evidence and road history; select the intended map state before generating a connection."))
    return dict(candidates=findings, criteria=dict(distance_m=distance,overlap_fraction=overlap_fraction),
        scope=__doc__, source_geometry_modified=False, permissions_modified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=inspect(json.loads(args.scene.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(dict(candidates=len(result["candidates"]),source_ways=len({r["road_id"] for r in result["candidates"]})))


if __name__=="__main__":
    main()
