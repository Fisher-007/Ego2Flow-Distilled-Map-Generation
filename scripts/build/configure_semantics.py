"""Place meshes in CARLA semantic folders and rebind map components.

CARLA Tagger uses the fourth asset-path segment for persistent classification.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.build.configure_open_surfaces import MAP, validate_map_asset


LABELS = {"Road": 1, "Terrain": 10, "Building": 3, "RoadLine": 24}


def classify_mesh_name(name):
    """Names emitted by build_assets.py and the separate markings.fbx export."""
    if re.fullmatch(r"map_assets_road(?:_\d+)?", name):
        return "Road"
    if re.fullmatch(r"map_assets_(?:terrain|road_edge)(?:_\d+)?", name):
        return "Terrain"
    if name.startswith("map_assets_building_"):
        return "Building"
    if re.fullmatch(r"(?:map_assets|markings)_road_markings(?:_(?:white|yellow))?(?:_\d+)?", name):
        return "RoadLine"
    if re.fullmatch(r"markings_crossing_\d+", name):
        return "RoadLine"
    return None


def configure_in_editor(output, map_asset=MAP):
    import unreal

    map_name = validate_map_asset(map_asset)
    output = Path(output)
    source_root = map_asset.rsplit("/", 1)[0]
    package_root = "/".join(map_asset.split("/")[:3])
    static_root = package_root + "/Static/"
    assets, meshes_api = unreal.EditorAssetLibrary, unreal.EditorStaticMeshLibrary
    if not unreal.EditorLevelLibrary.load_level(map_asset):
        raise RuntimeError("Could not load project map: " + map_asset)

    def vector(v):
        return [v.x, v.y, v.z]

    def transform(t):
        return {"location_cm": vector(t.translation),
                "rotation_xyzw": [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
                "scale": vector(t.scale3d)}

    def path(obj):
        return obj.get_path_name() if obj else None

    def mesh_values(mesh):
        bounds = mesh.get_bounds()
        lods = meshes_api.get_lod_count(mesh)
        return {
            "class": mesh.get_class().get_name(),
            "bounds": {"origin_cm": vector(bounds.origin), "extent_cm": vector(bounds.box_extent),
                       "sphere_radius_cm": bounds.sphere_radius},
            "lods": [{"vertices": meshes_api.get_number_verts(mesh, i),
                      "sections": mesh.get_num_sections(i)} for i in range(lods)],
            "materials": [{"material": path(slot.get_editor_property("material_interface")),
                           "slot_name": str(slot.get_editor_property("material_slot_name"))}
                          for slot in mesh.get_editor_property("static_materials")],
            "collision_complexity": str(meshes_api.get_collision_complexity(mesh)),
            "simple_collision_count": meshes_api.get_simple_collision_count(mesh),
            "convex_collision_count": meshes_api.get_convex_collision_count(mesh),
            "lod_for_collision": mesh.get_editor_property("LODForCollision"),
            "generate_mesh_distance_field": mesh.get_editor_property("bGenerateMeshDistanceField"),
            "distance_field_self_shadow_bias": mesh.get_editor_property("DistanceFieldSelfShadowBias"),
        }

    def snapshot():
        actors, values, components = {}, {}, {}
        for actor in unreal.EditorLevelLibrary.get_all_level_actors():
            actors[actor.get_name()] = {"class": actor.get_class().get_name(),
                                       "transform": transform(actor.get_actor_transform())}
            for component in actor.get_components_by_class(unreal.StaticMeshComponent):
                key = component.get_path_name()
                components[key] = component
                values[key] = {
                    "actor": actor.get_name(), "mesh": path(component.get_editor_property("static_mesh")),
                    "transform": transform(component.get_world_transform()),
                    "mobility": str(component.get_editor_property("Mobility")),
                    "visible": component.is_visible(), "hidden_in_game": component.get_editor_property("bHiddenInGame"),
                    "materials": [path(m) for m in component.get_materials()],
                    "collision_enabled": str(component.get_collision_enabled()),
                    "collision_profile": str(component.get_collision_profile_name()),
                    "cast_shadow": component.get_editor_property("CastShadow"),
                    "affect_distance_field_lighting": component.get_editor_property("bAffectDistanceFieldLighting"),
                }
        return {"actors": actors, "mesh_components": values,
                "distance_field_ao": unreal.SystemLibrary.get_console_variable_int_value("r.DistanceFieldAO")}, components

    before, components = snapshot()
    targets, unknown = {}, []
    for key, value in before["mesh_components"].items():
        source = value["mesh"]
        if not source:
            continue
        parts = source.split("/")
        current_map_copy = source.startswith(static_root) and len(parts) > 6 and parts[5] == map_name
        if not (source.startswith(source_root + "/") or current_map_copy):
            continue
        mesh = components[key].get_editor_property("static_mesh")
        category = classify_mesh_name(mesh.get_name())
        basis = "exported_mesh_name"
        # Identify imported buildings through their material when FBX strips OSM namespace prefixes.
        if category is None and value["materials"] == [source_root + "/building.building"]:
            category, basis = "Building", "dedicated_generated_building_material"
        if category is None:
            unknown.append(source)
        else:
            targets[key] = {"source": source, "category": category, "mesh": mesh, "classification_basis": basis}
    report = {"map": map_asset, "passed": False, "saved": False, "before": before,
              "unknown_project_meshes": unknown, "copies": [], "targets": [],
              "label_ids": LABELS, "runtime_labels_verified": False,
              "scope": "Copy and rebind only the selected map's generated meshes. No source asset rename/deletion, other-map rebinding, engine change, or stencil assignment.",
              "marking_settings": "Only RoadLine components use NoCollision, CastShadow=False and bAffectDistanceFieldLighting=False. Existing Road/Terrain/Building settings are preserved.",
              "road_edge_classification": "Terrain: road-to-DEM closure walls",
              "validation_limit": "API-readable asset counts, bounds, materials and collision settings are compared. CARLA startup tagging and semantic sensor/raycast labels require runtime verification."}

    def write_report():
        output.write_text(json.dumps(report, indent=2) + "\n")

    write_report()
    if unknown or not targets:
        raise RuntimeError("Unclassified or absent project meshes; inspect " + str(output))
    copied, sources = {}, {}
    expected = copy.deepcopy(before)
    # Check serialized collision settings on the duplicated mesh.
    for key, item in targets.items():
        source, category, mesh = item["source"], item["category"], item["mesh"]
        destination_folder = static_root + category + "/" + map_name + "/"
        if source not in copied:
            sources[source] = mesh_values(mesh)
            if source.startswith(destination_folder):
                duplicate, operation = mesh, "already_classified"
            else:
                destination = destination_folder + mesh.get_name()
                # Allocate a new asset name for unbound copies; reuse bound copies.
                destination, _ = unreal.AssetToolsHelpers.get_asset_tools().create_unique_asset_name(destination, "")
                duplicate = assets.duplicate_loaded_asset(mesh, destination)
                operation = "duplicated"
                if not isinstance(duplicate, unreal.StaticMesh):
                    raise RuntimeError("Could not duplicate project StaticMesh: " + source)
            actual = mesh_values(duplicate)
            row = {"source": source, "destination": path(duplicate), "operation": operation,
                   "source_values": sources[source], "destination_values": actual,
                   "api_values_match": actual == sources[source], "saved": False}
            report["copies"].append(row)
            # Write asset snapshots in batches.
            if len(report["copies"]) % 256 == 0:
                write_report()
                print('Semantic assets copied:', len(report['copies']), flush=True)
            if not row["api_values_match"]:
                write_report()
                raise RuntimeError("Duplicate changed mesh geometry/material/collision settings: " + source)
            if operation == "duplicated":
                row["saved"] = assets.save_loaded_asset(duplicate, only_if_is_dirty=False)
                if not row["saved"]:
                    write_report()
                    raise RuntimeError("Could not save semantic mesh copy: " + path(duplicate))
            copied[source] = duplicate
        duplicate = copied[source]
        item["destination"] = duplicate
        expected["mesh_components"][key]["mesh"] = path(duplicate)
        if category == "RoadLine":
            expected["mesh_components"][key].update(
                collision_enabled=str(unreal.CollisionEnabled.NO_COLLISION), collision_profile="NoCollision",
                cast_shadow=False, affect_distance_field_lighting=False)

    for key, item in targets.items():
        component = components[key]
        component.modify()
        duplicate = item["destination"]
        if component.get_editor_property("static_mesh") != duplicate and not component.set_static_mesh(duplicate):
            raise RuntimeError("Could not rebind map component: " + key)
        if item["category"] == "RoadLine":
            component.set_collision_profile_name("NoCollision")
            component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
            component.set_editor_property("CastShadow", False)
            component.set_editor_property("bAffectDistanceFieldLighting", False)
    after, _ = snapshot()
    source_unchanged = all(mesh_values(assets.load_asset(source)) == values for source, values in sources.items())
    for key, item in sorted(targets.items()):
        destination = after["mesh_components"][key]["mesh"]
        folder = destination.split("/")[4]
        report["targets"].append({"component": key, "category": item["category"],
                                  "classification_basis": item["classification_basis"],
                                  "expected_label_id": LABELS[item["category"]],
                                  "actual_path_category": folder, "path_category_matches": folder == item["category"],
                                  "before": before["mesh_components"][key], "after": after["mesh_components"][key]})
    report.update(after=after, checks={
        "only_bindings_and_explicit_marking_settings_changed": after == expected,
        "source_asset_api_values_unchanged": source_unchanged,
        "all_path_categories_match": all(row["path_category_matches"] for row in report["targets"]),
        "all_copy_api_values_match": all(row["api_values_match"] for row in report["copies"]),
    })
    write_report()
    if not all(report["checks"].values()):
        raise RuntimeError("Unexpected semantic setup change; inspect " + str(output))
    if not unreal.EditorLevelLibrary.save_current_level():
        raise RuntimeError("Could not save semantic bindings for " + map_asset)
    report.update(passed=True, saved=True)
    write_report()


def run_semantic_setup(carla_root, engine_root, output, cpu_set, env=None, map_asset=MAP):
    validate_map_asset(map_asset)
    carla_root, engine_root, output = (Path(p).expanduser().resolve() for p in (carla_root, engine_root, output))
    output.mkdir(parents=True, exist_ok=True)
    result = output / "semantics.json"
    result.unlink(missing_ok=True)
    script = output / "configure_semantics_ue.py"
    script.write_text(
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('mapbuild_semantics', %r)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.configure_in_editor(%r, map_asset=%r)\n"
        % (str(Path(__file__).resolve()), str(result), map_asset))
    environment = dict(os.environ if env is None else env, CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    for category in ("CONFIG", "CACHE", "DATA"):
        directory = output / "runtime" / category.lower()
        directory.mkdir(parents=True, exist_ok=True)
        environment["XDG_" + category + "_HOME"] = str(directory)
    command = ["taskset", "-c", cpu_set, str(engine_root / "Engine/Binaries/Linux/UE4Editor"),
               str(carla_root / "Unreal/CarlaUE4/CarlaUE4.uproject"), "-run=pythonscript",
               "-script=" + str(script), "-nullrhi", "-unattended", "-nosourcecontrol", "-nosound",
               "-nosplash", "-nocrashreports", "-abslog=" + str(output / "semantics.log")]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "semantics.stdout.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not result.is_file() or not json.loads(result.read_text())["passed"]:
        raise RuntimeError("Semantic setup did not complete; inspect " + str(output / "semantics.stdout.log"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--carla-root", type=Path, default=os.environ.get("CARLA_ROOT"))
    parser.add_argument("--engine-root", type=Path, default=os.environ.get("UE4_ROOT"))
    parser.add_argument("--cpu-set", default=os.environ.get("MAP_CPUSET"))
    parser.add_argument("--map-asset", required=True,
                        help="/Game/MAP<map>/Maps/<map>/<map>; map begins with a letter and uses letters, digits or underscores")
    args = parser.parse_args()
    try:
        validate_map_asset(args.map_asset)
    except ValueError as exc:
        parser.error(str(exc))
    if not all((args.carla_root, args.engine_root, args.cpu_set)):
        parser.error("Set CARLA_ROOT, UE4_ROOT and MAP_CPUSET, or supply their command-line options")
    print(run_semantic_setup(args.carla_root, args.engine_root, args.output, args.cpu_set, map_asset=args.map_asset))


if __name__ == "__main__":
    main()
