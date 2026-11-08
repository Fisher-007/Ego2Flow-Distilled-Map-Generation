"""Report physical-parent and native source-interval dependencies."""
import argparse
import json
from pathlib import Path


def source_ids(mesh):
    return set(mesh.get("source_lane_ids", [])) | {
        mesh[k] for k in ("source_lane_id", "from_source_lane_id", "to_source_lane_id")
        if mesh.get(k)
    }


def inspect(meshes, mapping, selected_mesh_ids, selected_source_ids=(), road_ids=None):
    selected = set(selected_mesh_ids)
    by_id = {m["id"]: m for m in meshes}
    sources = set(selected_source_ids)
    lineage = set(selected)
    for key in selected:
        mesh = by_id[key]
        sources.update(source_ids(mesh))
        lineage.update(mesh.get("selection", {}).get("mesh_ids", []))
    by_source = {}
    for mesh in meshes:
        for sid in source_ids(mesh):
            by_source.setdefault(sid, []).append(mesh["id"])

    def identity(mesh):
        keys = ("id", "kind", "source_lane_id", "source_lane_ids",
                "source_interval_m", "xodr_road_id", "xodr_lane_id",
                "native_interval_m", "from_mesh_id", "to_mesh_id",
                "from_source_lane_id", "to_source_lane_id")
        return {k: mesh[k] for k in keys if k in mesh}

    same_source = [identity(m) for m in meshes
                   if m["id"] not in selected and source_ids(m) & sources]
    closures = []
    for mesh in meshes:
        if mesh.get("kind") != "connection_closure" or mesh["id"] in selected:
            continue
        parents = {mesh.get("from_mesh_id"), mesh.get("to_mesh_id")} - {None}
        matched = parents & lineage
        shared = source_ids(mesh) & sources
        if matched or shared:
            closures.append(dict(**identity(mesh),
                                 selected_parent_lineage=sorted(matched),
                                 selected_source_ids=sorted(shared),
                                 missing_current_parent_ids=sorted(parents - by_id.keys())))
    adjacency = []
    requested = None if road_ids is None else {int(v) for v in road_ids}
    for rid, row in mapping["roads"].items():
        if requested is not None and int(rid) not in requested:
            continue
        intervals = row.get("source_intervals", [])
        for a, b in zip(intervals, intervals[1:]):
            left, right = str(a[0]), str(b[0])
            if left == right or (left in sources) == (right in sources):
                continue
            outside = right if left in sources else left
            adjacency.append(dict(
                road_id=int(rid), left_interval=a, right_interval=b,
                outside_source_id=outside,
                outside_physical_meshes=[identity(by_id[k]) for k in by_source.get(outside, [])],
                source_lane_path=row.get("source_lane_path", []),
                allowed_classes=row.get("allowed_classes", [])))
    return dict(selected_mesh_ids=sorted(selected),
                selected_source_ids=sorted(sources),
                requested_native_road_ids=None if requested is None else sorted(requested),
                same_source_unselected_meshes=same_source,
                derived_closure_dependencies=closures,
                adjacent_native_source_intervals=adjacency,
                scope="Explicit parent/source lineage and consecutive mapped intervals only; no XY ownership inference or map modification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("roads", "mapping", "selection", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    import orjson
    roads = orjson.loads(args.roads.read_bytes())
    mapping = orjson.loads(args.mapping.read_bytes())
    selection = json.loads(args.selection.read_text())
    result = inspect(roads["meshes"], mapping, selection["selected_mesh_ids"],
                     selection.get("selected_source_ids", []),
                     selection.get("native_road_ids"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(dict(same_source=len(result["same_source_unselected_meshes"]),
               derived_closures=len(result["derived_closure_dependencies"]),
               adjacent_intervals=len(result["adjacent_native_source_intervals"])), flush=True)


if __name__ == "__main__":
    main()
