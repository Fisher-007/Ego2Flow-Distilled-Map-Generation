"""Test whether co-located AV2 lane differences admit one rigid source registration."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from av2.map.map_api import ArgoverseStaticMap
from scipy.spatial.transform import Rotation

from scripts.data.source_lane_geometry import source_centerline


def distribution(values):
    return {"median":float(np.median(values)),"p95":float(np.percentile(values,95)),"max":float(np.max(values))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("catalog","overlaps","data-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    catalog=json.loads(args.catalog.read_text())
    maps={row["log_id"]:ArgoverseStaticMap.from_json(args.data_root/row["vector_map"]) for row in catalog["candidates"]}
    groups=defaultdict(list)
    for pair in json.loads(args.overlaps.read_text())["co_located_distinct_id_candidates"]:
        first,second=pair["first_logs"][0],pair["second_logs"][0]
        a=maps[first].vector_lane_segments[pair["first_lane"]]
        b=maps[second].vector_lane_segments[pair["second_lane"]]
        if a.is_intersection or b.is_intersection or pair["center_hausdorff_xy_m"]>2:
            continue
        if abs(pair["first_length_m"]-pair["second_length_m"])>2:
            continue
        groups[(first,second)].append(pair)
    result=[]
    for (first,second),pairs in groups.items():
        # Match whole-lane candidates by minimum shape discrepancy.
        used_a,used_b,selected=set(),set(),[]
        for pair in sorted(pairs,key=lambda p:p["center_hausdorff_xy_m"]):
            a,b=pair["first_lane"],pair["second_lane"]
            if a in used_a or b in used_b:continue
            used_a.add(a);used_b.add(b);selected.append(pair)
        points_a,points_b=[],[]
        accepted=[]
        for pair in selected:
            a=source_centerline(maps[first],maps[first].vector_lane_segments[pair["first_lane"]])
            b=source_centerline(maps[second],maps[second].vector_lane_segments[pair["second_lane"]])
            if not np.isfinite(a).all() or not np.isfinite(b).all():continue
            points_a.append(a[::5]);points_b.append(b[::5]);accepted.append(pair)
        if len(accepted)<4:continue
        a,b=np.asarray(points_a),np.asarray(points_b)
        fit=np.arange(len(a))%2==0
        x,y=a[fit].reshape(-1,3),b[fit].reshape(-1,3)
        rotation,_=Rotation.align_vectors(y-y.mean(0),x-x.mean(0))
        translation=y.mean(0)-rotation.apply(x.mean(0))
        xa,yb=a[~fit].reshape(-1,3),b[~fit].reshape(-1,3)
        before=xa-yb;after=rotation.apply(xa)+translation-yb
        result.append({"first_log":first,"reference_log":second,"matched_lanes":len(accepted),
            "source_lane_pairs":[[p["first_lane"],p["second_lane"]] for p in accepted],
            "rotation_matrix":rotation.as_matrix().tolist(),"translation_city_m":translation.tolist(),
            "rotation_xyz_deg":rotation.as_euler("xyz",degrees=True).tolist(),
            "check_before_xyz_m":distribution(np.linalg.norm(before,axis=1)),
            "check_after_xyz_m":distribution(np.linalg.norm(after,axis=1)),
            "check_after_xy_m":distribution(np.linalg.norm(after[:,:2],axis=1)),
            "check_after_abs_z_m":distribution(abs(after[:,2])),
            "applied":False})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"cases":result,
        "scope":"Rigid fit diagnostic using whole ordinary-lane candidates and alternate-lane checking. Physical identities still need source review; no source geometry, camera or actor was moved."},indent=2)+"\n")
    for row in result:
        print(row["first_log"][:8],row["reference_log"][:8],row["matched_lanes"],
              "before",row["check_before_xyz_m"],"after",row["check_after_xyz_m"],flush=True)


if __name__=="__main__":main()
