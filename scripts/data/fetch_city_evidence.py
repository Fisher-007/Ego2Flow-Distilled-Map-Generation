"""Fetch sparse ring views, lidar and complete labels for selected city sources."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from urllib.request import urlopen

from scripts.data.av2_source import S3, listing, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=2.)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-existing-cameras", nargs="+", help="Fetch only the nearest LiDAR for already downloaded images in these cameras")
    parser.add_argument("--logs", nargs="+", help="Selected source log IDs")
    parser.add_argument("--area-anchors", action="store_true", help="Fetch each ring camera nearest the selected Area source timestamps, with paired LiDAR")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    anchors_by_log = {}
    if args.area_anchors:
        for area in config["areas"]:
            source = area["source_observations"][0]
            anchors_by_log.setdefault(source["log_id"], []).append(source["nearest_timestamp_ns"])
    reports = []
    for relative in config["sources"]["av2_logs"]:
        folder = args.data_root / relative
        split, identity = Path(relative).parts[-2:]
        if args.logs and identity not in args.logs:
            continue
        prefix = f"datasets/av2/sensor/{split}/{identity}/"
        cache = args.data_root / "av2/metadata" / f"{identity}_objects.json"
        if cache.exists():
            objects = json.loads(cache.read_text())
        else:
            objects, _ = listing(prefix)
            write_json(cache, objects)
        selected, streams = [], {}
        for obj in objects:
            name = obj["key"].removeprefix(prefix)
            parts = Path(name).parts
            if name in ("annotations.feather", "city_SE3_egovehicle.feather") or parts[0] in ("calibration", "map"):
                selected.append(obj)
            elif name.startswith("sensors/cameras/ring_") and name.endswith(".jpg"):
                streams.setdefault(parts[-2], []).append(obj)
            elif name.startswith("sensors/lidar/") and name.endswith(".feather"):
                streams.setdefault("lidar", []).append(obj)
        for stream, objects_in_stream in streams.items():
            if args.area_anchors:
                for anchor in anchors_by_log.get(identity, []):
                    selected.append(min(objects_in_stream, key=lambda obj: abs(int(Path(obj["key"]).stem)-anchor)))
                continue
            last = None
            for obj in sorted(objects_in_stream, key=lambda obj: int(Path(obj["key"]).stem)):
                timestamp = int(Path(obj["key"]).stem)
                if last is None or timestamp-last >= args.interval_s*1e9:
                    selected.append(obj)
                    last = timestamp

        # Independent sparse sampling of the streams drifts out of alignment.
        # Pair each actual image with its nearest source sweep before extraction.
        images = ([dict(key=prefix+str(path.relative_to(folder)))
                   for camera in args.pair_existing_cameras
                   for path in sorted((folder/'sensors/cameras'/camera).glob('*.jpg'))]
                  if args.pair_existing_cameras else
                  [obj for obj in selected if obj['key'].endswith('.jpg')])
        if args.pair_existing_cameras:
            selected = []
        lidar = streams['lidar']; pairs = []
        for image in images:
            timestamp = int(Path(image['key']).stem)
            sweep = min(lidar, key=lambda obj: abs(int(Path(obj['key']).stem)-timestamp))
            age = (int(Path(sweep['key']).stem)-timestamp)/1e9
            pairs.append(dict(image=image['key'], lidar=sweep['key'], difference_s=age))
            selected.append(sweep)
        selected = list({obj['key']:obj for obj in selected}.values())

        def fetch(obj):
            path = folder / obj["key"].removeprefix(prefix)
            if path.is_file() and path.stat().st_size == obj["size"]:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(3):
                try:
                    with urlopen(S3+obj["key"], timeout=90) as response:
                        payload = response.read()
                    if len(payload) != obj["size"]:
                        raise IOError("Incomplete source object: "+obj["key"])
                    temporary = path.with_suffix(path.suffix+".partial")
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2*(attempt+1))

        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(fetch, obj): obj for obj in selected}
            for future in as_completed(pending):
                try:
                    future.result()
                except Exception as error:
                    failures.append({"key": pending[future]["key"], "error": str(error)})
        report = {"log_id": identity, "split": split, "interval_s": args.interval_s,
                  "acquisition_strategy": "selected_area_anchors" if args.area_anchors else "sparse_interval",
                  "area_anchor_timestamps_ns": anchors_by_log.get(identity, []),
                  "objects": selected, "bytes": sum(obj["size"] for obj in selected),
                  "image_lidar_pairs": pairs,
                  "maximum_pair_difference_s": max((abs(row['difference_s']) for row in pairs), default=None),
                  "failures": failures, "scope": "Static construction evidence and labels; sparse imagery does not provide synchronized planner histories."}
        write_json(args.output / (identity+".json"), report)
        reports.append({k:v for k,v in report.items() if k!="objects"})
        write_json(args.output / "summary.json", reports)
        print(f"Evidence {len(reports)}/{len(config['sources']['av2_logs'])}: {identity} "
              f"files {len(selected)}, bytes {report['bytes']}, failed {len(failures)}", flush=True)
    if any(row["failures"] for row in reports):
        raise SystemExit("Some evidence objects require a cached retry; see summary.json.")


if __name__ == "__main__":
    main()
