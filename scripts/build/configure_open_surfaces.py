"""Exclude open road, terrain, and edge sheets from distance-field occlusion."""

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess


MAP = None


def validate_map_asset(map_asset):
    match = re.fullmatch(r"/Game/MAP([A-Za-z][A-Za-z0-9_]*)/Maps/\1/\1", map_asset)
    if not match:
        raise ValueError("Expected /Game/MAP<map>/Maps/<map>/<map> for a generated project map")
    return match.group(1)


def configure_in_editor(output, map_asset=MAP):
    import unreal

    name = validate_map_asset(map_asset)
    asset_root = map_asset.rsplit("/", 1)[0]
    target_meshes = {asset_root + "/map_assets_" + name + ".map_assets_" + name
                     for name in ("road", "terrain", "road_edge")}
    # A semantic setup may already have copied and rebound these same meshes.
    package = "/Game/MAP" + name
    target_meshes.update(package + "/Static/" + category + "/" + name + "/map_assets_" + kind + ".map_assets_" + kind
                         for kind, category in (("road", "Road"), ("terrain", "Terrain"), ("road_edge", "Terrain")))
    if not unreal.EditorLevelLibrary.load_level(map_asset):
        raise RuntimeError("Could not load project map: " + map_asset)

    def transform_values(transform):
        location, rotation, scale = transform.translation, transform.rotation, transform.scale3d
        return {"location_cm": [location.x, location.y, location.z],
                "rotation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
                "scale": [scale.x, scale.y, scale.z]}

    def property_value(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (unreal.Color, unreal.LinearColor)):
            return [value.r, value.g, value.b, value.a]
        if isinstance(value, unreal.Object):
            return value.get_path_name()
        return str(value)  # Enum names, not temporary struct representations.

    def snapshot():
        result = {"actors": {}, "mesh_components": {}, "lights": {},
                  "distance_field_ao": unreal.SystemLibrary.get_console_variable_int_value("r.DistanceFieldAO")}
        components = {}
        for actor in unreal.EditorLevelLibrary.get_all_level_actors():
            result["actors"][actor.get_name()] = {
                "class": actor.get_class().get_name(), "transform": transform_values(actor.get_actor_transform())}
            for component in actor.get_components_by_class(unreal.StaticMeshComponent):
                mesh = component.get_editor_property("static_mesh")
                key = component.get_path_name()
                components[key] = component
                result["mesh_components"][key] = {
                    "actor": actor.get_name(), "mesh": mesh.get_path_name() if mesh else None,
                    "transform": transform_values(component.get_world_transform()),
                    "visible": component.is_visible(), "hidden_in_game": component.get_editor_property("bHiddenInGame"),
                    "cast_shadow": component.get_editor_property("CastShadow"),
                    "affect_distance_field_lighting": component.get_editor_property("bAffectDistanceFieldLighting"),
                    "collision_enabled": str(component.get_collision_enabled()),
                    "collision_profile": str(component.get_collision_profile_name()),
                    "materials": [m.get_path_name() if m else None for m in component.get_materials()],
                }
            for component in actor.get_components_by_class(unreal.LightComponentBase):
                values = {name: property_value(component.get_editor_property(name)) for name in (
                    "Mobility", "Intensity", "LightColor", "bAffectsWorld", "CastShadows", "IndirectLightingIntensity")}
                if isinstance(component, unreal.SkyLightComponent):
                    values.update({name: property_value(component.get_editor_property(name)) for name in (
                        "SourceType", "Cubemap", "bRealTimeCapture", "bCaptureEmissiveOnly",
                        "bLowerHemisphereIsBlack", "LowerHemisphereColor")})
                result["lights"][component.get_path_name()] = values
        return result, components

    before, components = snapshot()
    def selected_mesh(path):
        return bool(path and path.startswith(package + '/') and
                    re.fullmatch(r'map_assets_(?:road|terrain|road_edge)(?:_\d+)?', path.rsplit('.', 1)[-1]))
    targets = {key: item for key, item in before["mesh_components"].items() if selected_mesh(item['mesh'])}
    names = [item["mesh"].rsplit(".", 1)[-1] for item in targets.values()]
    # Configure the exported road, terrain, and closure components.
    if not any(re.fullmatch(r'map_assets_road(?:_\d+)?', n) for n in names) or len(names) != len(set(names)):
        raise RuntimeError("Expected exported road surfaces with one component per asset")
    for key in targets:
        # Set distance-field occluder membership through the public EditAnywhere property.
        components[key].set_editor_property("bAffectDistanceFieldLighting", False)
    after, _ = snapshot()
    expected = copy.deepcopy(before)
    for key in targets:
        expected["mesh_components"][key]["affect_distance_field_lighting"] = False
    checks = {"only_target_distance_field_flags_changed": after == expected,
              "global_distance_field_ao_unchanged": before["distance_field_ao"] == after["distance_field_ao"]}
    report = {
        "map": map_asset, "passed": False, "saved": False, "checks": checks,
        "targets": [{"component": key, "before": before["mesh_components"][key],
                     "after": after["mesh_components"][key]} for key in sorted(targets)],
        "project_mesh_component_count": sum(bool(item["mesh"] and item["mesh"].startswith(package + "/"))
                                             for item in after["mesh_components"].values()),
        "other_mesh_component_count": len(after["mesh_components"]) - len(targets),
        "before": before, "after": after,
        "assumption": "Open thin ground sheets are excluded as volumetric distance-field occluders; geometry and collision stay intact",
        "scope": "NullRHI verifies component settings and that global DFAO is unchanged; Epic rendering separately checks appearance. This operation leaves building component flags intact; their mesh distance fields depend on the asset build settings. Ground sheets cease casting distance-field shadows, while CastShadow remains unchanged.",
    }
    # Save diagnostics before validation and map saving.
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    if not all(checks.values()):
        raise RuntimeError("Unexpected map settings change; inspect before/after in " + str(output))
    if not unreal.EditorLevelLibrary.save_current_level():
        raise RuntimeError("Could not save project map after open-surface settings")
    report.update(passed=True, saved=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")


def run_surface_setup(carla_root, engine_root, output, cpu_set, env=None, map_asset=MAP):
    carla_root, engine_root, output = (Path(p).expanduser().resolve() for p in (carla_root, engine_root, output))
    output.mkdir(parents=True, exist_ok=True)
    result = output / "open_surfaces.json"
    result.unlink(missing_ok=True)
    script = output / "configure_open_surfaces_ue.py"
    script.write_text(
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('mapbuild_surfaces', %r)\n"
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
               "-nosplash", "-nocrashreports", "-abslog=" + str(output / "open_surfaces.log")]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "open_surfaces.stdout.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not result.is_file() or not json.loads(result.read_text())["passed"]:
        raise RuntimeError("Open-surface setup did not complete; inspect " + str(output / "open_surfaces.stdout.log"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--carla-root", type=Path, default=os.environ.get("CARLA_ROOT"))
    parser.add_argument("--engine-root", type=Path, default=os.environ.get("UE4_ROOT"))
    parser.add_argument("--cpu-set", default=os.environ.get("MAP_CPUSET"))
    parser.add_argument("--map-asset", required=True)
    args = parser.parse_args()
    if not all((args.carla_root, args.engine_root, args.cpu_set)):
        parser.error("Set CARLA_ROOT, UE4_ROOT and MAP_CPUSET, or supply their command-line options")
    print(run_surface_setup(args.carla_root, args.engine_root, args.output, args.cpu_set, map_asset=args.map_asset))


if __name__ == "__main__":
    main()
