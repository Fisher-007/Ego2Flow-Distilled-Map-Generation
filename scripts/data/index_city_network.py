#!/usr/bin/env python3
"""Locate a city's AV2 training maps before choosing distinct connected Areas.

Reuse public S3 listings and the existing map-overlap cache. Download vector
maps and ego poses only; image and lidar acquisition follows Area selection.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from urllib.request import urlopen

from scripts.data.av2_source import S3, listing, map_location, write_json


def fetch(url, path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        contents = response.read()
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(contents)
    temporary.replace(path)


def inspect_log(log_id, city, root, split="train"):
    cached = root / "metadata/map_overlap" / log_id / "source.json"
    if cached.exists():
        source = json.loads(cached.read_text())
    else:
        source = map_location(log_id, split=split)
        cached.parent.mkdir(parents=True, exist_ok=True)
        write_json(cached, source)
    if source["city_code"] != city:
        return source, None
    folder = root / "sensor" / split / log_id
    path = folder / "map" / source["vector_map_url"].rsplit("/", 1)[-1]
    fetch(source["vector_map_url"], path)
    fetch(S3 + f"datasets/av2/sensor/{split}/" + log_id + "/city_SE3_egovehicle.feather",
          folder / "city_SE3_egovehicle.feather")
    data = json.loads(path.read_text())
    lanes = list(data["lane_segments"].values())
    points = [p for lane in lanes for side in ("left_lane_boundary", "right_lane_boundary")
              for p in lane[side]]
    bounds = [min(p[axis] for p in points) for axis in ("x", "y")]
    bounds += [max(p[axis] for p in points) for axis in ("x", "y")]
    result = dict(source, vector_map=str(path.relative_to(root)),
                  bbox_city_xy=bounds, lane_count=len(lanes),
                  intersection_lane_count=sum(lane["is_intersection"] for lane in lanes),
                  center_city_xy=[(bounds[0] + bounds[2])/2, (bounds[1] + bounds[3])/2])
    return source, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="AV2 directory")
    parser.add_argument("--city", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity_file = args.data_root / f"metadata/sensor_{args.split}_log_ids.json"
    if not identity_file.exists():
        _, prefixes = listing(f"datasets/av2/sensor/{args.split}/", delimiter="/")
        identity_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(identity_file, [p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes])
    identities = json.loads(identity_file.read_text())
    sources, candidates, failures = [], [], []

    def attempt(identity):
        for retry in range(3):
            try:
                return inspect_log(identity, args.city, args.data_root, args.split)
            except Exception:
                if retry == 2:
                    raise
                time.sleep(2 * (retry + 1))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(attempt, identity): identity for identity in identities}
        for count, future in enumerate(as_completed(pending), 1):
            try:
                source, candidate = future.result()
                sources.append(source)
                if candidate:
                    candidates.append(candidate)
            except Exception as error:
                failures.append({"log_id": pending[future], "error": str(error)})
            if count % 25 == 0 or count == len(pending):
                print(f"Indexed {count}/{len(pending)}; {args.city} maps {len(candidates)}; "
                      f"failed {len(failures)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "catalog.json", {
        "city": args.city, "split": args.split, "listed_logs": len(identities),
        "indexed_logs": len(sources), "failures": failures,
        "candidates": sorted(candidates, key=lambda entry: entry["log_id"]),
        "scope": "Source inventory; overlapping recordings are not distinct Areas."
    })
    write_json(args.output / "all_cities.json", sorted(sources, key=lambda entry: entry["log_id"]))
    if failures:
        raise SystemExit("Incomplete source index; see catalog.json failures and rerun cached lookup.")


if __name__ == "__main__":
    main()
