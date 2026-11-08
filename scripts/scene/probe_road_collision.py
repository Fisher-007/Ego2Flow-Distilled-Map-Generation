"""Compare downward collision queries in CARLA with exported road surfaces."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import shapely

from map_generation.av2_surfaces import mesh_triangles, plane_z


_query_world = None


def initialize_queries(port):
    import carla
    global _query_world
    client = carla.Client("127.0.0.1", port)
    client.set_timeout(60)
    _query_world = client.get_world()


def query_collision(item):
    import carla
    index, xy, top, distance = item
    hit = _query_world.project_point(carla.Location(x=float(xy[0]), y=float(-xy[1]), z=top + 1),
                                     carla.Vector3D(z=-1), distance)
    positions = [] if hit is None else [{"enu_m": [hit.location.x, -hit.location.y, hit.location.z],
                                        "label": str(hit.label)}]
    return index, positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--roads", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-map", required=True)
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--tolerance-m", type=float, default=.002)
    parser.add_argument("--all-roads", action="store_true", help="Include retained AV2, OSM and generated roads for the selected classes")
    parser.add_argument("--classes", nargs="+", default=["passenger"], help="Source vehicle classes selected by --all-roads")
    parser.add_argument("--source-lane-ids", type=Path, help="Optional JSON list of source lane IDs limiting the existing sample domain")
    parser.add_argument("--spacing-m", type=float, default=20., help="Maximum longitudinal sample spacing for --all-roads")
    parser.add_argument("--query-workers", type=int, default=1, help="Independent native RPC connections within the assigned CPU set")
    args = parser.parse_args()
    import carla

    scene = json.loads(args.scene.read_text())
    selected_ids = set(json.loads(args.source_lane_ids.read_text())) if args.source_lane_ids else None
    classes = set(args.classes)
    roads = json.loads(args.roads.read_text())
    triangles = np.concatenate([mesh_triangles(mesh) for mesh in roads["meshes"] if mesh['faces']])
    tree = shapely.STRtree(shapely.polygons(triangles[:, :, :2]))
    fine_ids = {lane["id"] for lane in scene["fusion_map"]["lanes"]}
    samples = [(lane["id"], xyz)
               for lane in scene["lanes"] + scene.get("av2_context", [])
               if lane["id"] in fine_ids and lane["lane_type"] in ("VEHICLE", "BUS")
               for xyz in lane["centerline"][1:-1]]
    if args.all_roads:
        from shapely.geometry import LineString
        samples=[]
        for lane in scene['fusion_map']['lanes']:
            if selected_ids is not None and lane['id'] not in selected_ids:continue
            if not classes.intersection(lane['allowed_classes']):continue
            line=LineString(lane['centerline']);count=max(1,int(np.ceil(line.length/args.spacing_m)))
            samples.extend((lane['id'],list(line.interpolate((i+.5)/count,normalized=True).coords[0])) for i in range(count))
    if selected_ids is not None:
        available_ids = {lane["id"] for lane in scene["fusion_map"]["lanes"]}
        if selected_ids - available_ids:
            raise ValueError("Unknown source lane IDs: " + str(sorted(selected_ids - available_ids)))
        samples = [(lane_id, xyz) for lane_id, xyz in samples if lane_id in selected_ids]
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(600)
    world = client.get_world()
    actual_map = world.get_map().name.removeprefix("/Game/").removesuffix(".umap")
    if actual_map != args.expected_map.removeprefix("/Game/"):
        raise ValueError(f"Loaded {actual_map}, expected {args.expected_map}")
    if len(world.get_actors().filter("vehicle.*")):
        raise ValueError("Run the surface query before spawning vehicles")
    rows, queries = [], []
    for index,(lane_id, xyz) in enumerate(samples):
        xy = np.asarray(xyz[:2])
        ids = tree.query(shapely.Point(xy), predicate="dwithin", distance=1e-7)
        row = {"source_lane_id": lane_id, "source_center_enu_m": xyz}
        if not len(ids):
            row.update(status="no_exported_road", hits=[])
        else:
            heights = plane_z(triangles[ids], np.tile(xy, (len(ids), 1)))
            # Probe the selected source layer and record competing intersections.
            top = float(heights[np.argmin(abs(heights-xyz[2]))]) if args.all_roads else float(heights.max())
            # project_point uses a single blocking SensorTrace query. cast_ray
            # instead reports OverlapChannel volumes, including initial overlaps
            # with invisible signal/route triggers at the ray's start position.
            queries.append((index, xy.tolist(), top, 2. if args.all_roads else top - float(heights.min()) + 2))
            row.update(expected_road_z_m=top, exported_surface_heights_m=heights.tolist(),
                       physical_minus_source_center_z_m=top - xyz[2])
        rows.append(row)
    import multiprocessing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Queue read-only RPC queries through independent clients.
    with multiprocessing.get_context("spawn").Pool(args.query_workers, initializer=initialize_queries,
                                                    initargs=(args.port,)) as pool:
        for completed, (index, positions) in enumerate(pool.imap(query_collision, queries, chunksize=16), 1):
            row = rows[index]
            row["hits"] = positions
            if positions:
                error = positions[0]["enu_m"][2] - row["expected_road_z_m"]
                row.update(first_hit_minus_road_z_m=error,
                           status="matched" if abs(error) <= args.tolerance_m else "height_mismatch")
            else:
                row["status"] = "no_collision_hit"
            if completed % 10000 == 0:
                progress = dict(completed=completed, total=len(queries), query_workers=args.query_workers)
                args.output.with_suffix(".progress.json").write_text(json.dumps(progress) + "\n")
                print(progress, flush=True)
    errors = [abs(row["first_hit_minus_road_z_m"]) for row in rows if "first_hit_minus_road_z_m" in row]
    report = {
        "map": actual_map, "scene": str(args.scene), "road_meshes": str(args.roads),
        "method": ("Downward CARLA project_point at equally spaced interior source-class-selected road samples; exported triangle height nearest the source vertical layer."
                   if args.all_roads else "Downward CARLA project_point at interior source centerline knots; expected top exported triangle plane at identical XY."),
        "query_workers": args.query_workers,
        "sampling": dict(all_roads=args.all_roads,
                         all_passenger_roads=args.all_roads and classes == {"passenger"} and selected_ids is None,
                         classes=sorted(classes) if args.all_roads else None,
                         source_lane_ids=sorted(selected_ids) if selected_ids is not None else None,
                         sampled_source_lane_ids=sorted({r["source_lane_id"] for r in rows}),
                         maximum_longitudinal_spacing_m=args.spacing_m if args.all_roads else None),
        "scope": "First blocking hit on CARLA SensorTrace with the imported collision setup. Overlap-only trigger volumes are excluded by this channel; vehicle wheel contact and driving remain separate checks.",
        "sample_count": len(rows), "tolerance_m": args.tolerance_m,
        "status_counts": {name: sum(row["status"] == name for row in rows)
                          for name in ("matched", "no_exported_road", "no_collision_hit", "height_mismatch")},
        "first_hit_label_counts": dict(Counter(row["hits"][0]["label"] for row in rows if row["hits"])),
        "max_abs_first_hit_error_m": max(errors) if errors else None,
        "passed": bool(rows) and all(row["status"] == "matched" for row in rows), "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"Collision surface check failed: {args.output}")


if __name__ == "__main__":
    main()
