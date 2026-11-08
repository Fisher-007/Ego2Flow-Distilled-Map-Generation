"""Place centered export tiles during initial map import using their manifest."""
import json
import os
from pathlib import Path
import subprocess


def configure_in_editor(output, map_asset, assets_file):
    import unreal
    from scripts.build.configure_open_surfaces import validate_map_asset
    name = validate_map_asset(map_asset)
    assets = json.loads(Path(assets_file).read_text())
    selected = {"map_assets_" + row["name"]: row for row in assets["objects"]
                if row.get("asset_origin_enu_m") is not None}
    if not unreal.EditorLevelLibrary.load_level(map_asset):
        raise RuntimeError("Could not load new project map")
    records, seen = [], set()
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh = component.get_editor_property("static_mesh")
            if mesh is None or not mesh.get_path_name().startswith("/Game/MAP" + name + "/"):
                continue
            key = mesh.get_name()
            if key not in selected:
                continue
            if key in seen:
                raise ValueError("Expected one native component per centered export tile: " + key)
            seen.add(key)
            entry = selected[key]
            origin = entry["asset_origin_enu_m"]
            expected = [origin[0] * 100, -origin[1] * 100, origin[2] * 100]
            component.set_world_location(unreal.Vector(*expected), False, False)
            transform = component.get_world_transform()
            actual = [getattr(transform.translation, axis) for axis in "xyz"]
            rotation = [getattr(transform.rotation, axis) for axis in "xyzw"]
            scale = [getattr(transform.scale3d, axis) for axis in "xyz"]
            if rotation != [0., 0., 0., 1.] or scale != [1., 1., 1.]:
                raise ValueError("Centered tile must retain exported unit scale and axis frame: " + key)
            error = max(abs(a - b) for a, b in zip(actual, expected))
            if error > .1:
                raise ValueError("Centered tile placement differs by more than 1 mm: " + key)
            bounds = mesh.get_bounds()
            low, high = entry["bounds_m"]["min"], entry["bounds_m"]["max"]
            expected_bounds = [(low[i] + high[i]) * 50 * (1 if i != 1 else -1) for i in range(3)]
            expected_bounds += [(high[i] - low[i]) * 50 for i in range(3)]
            actual_bounds = [getattr(bounds.origin, a) for a in "xyz"] + [getattr(bounds.box_extent, a) for a in "xyz"]
            bounds_error = max(abs(a - b) for a, b in zip(actual_bounds, expected_bounds))
            actual_low = [actual_bounds[i]-actual_bounds[i+3] for i in range(3)]
            actual_high = [actual_bounds[i]+actual_bounds[i+3] for i in range(3)]
            complete_low = [expected_bounds[i]-expected_bounds[i+3] for i in range(3)]
            complete_high = [expected_bounds[i]+expected_bounds[i+3] for i in range(3)]
            precision = entry.get("native_precision")
            resolved = (precision or {}).get("bounds_cm") or dict(min=complete_low,max=complete_high)
            coverage_error = max([0.] + [actual_low[i]-resolved["min"][i] for i in range(3)]
                                 + [resolved["max"][i]-actual_high[i] for i in range(3)])
            envelope_error = max([0.] + [complete_low[i]-actual_low[i] for i in range(3)]
                                 + [actual_high[i]-complete_high[i] for i in range(3)])
            if coverage_error > .1 or envelope_error > .1:
                raise ValueError("Imported tile loses resolved source bounds or exceeds source envelope by more than 1 mm: " + key)
            records.append(dict(mesh=mesh.get_path_name(), component=component.get_path_name(),
                                asset_origin_enu_m=origin, location_cm=actual,
                                placement_error_cm=error, complete_bounds_difference_cm=bounds_error,
                                resolved_coverage_error_cm=coverage_error, source_envelope_error_cm=envelope_error,
                                source_precision=precision))
    missing = sorted(selected.keys() - seen)
    report = dict(map=map_asset, assets=assets_file, placed=records, missing=missing,
                  passed=not missing, scope="Manifest placement and imported local bounds; runtime collision and rendering remain separate.")
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    if missing:
        raise ValueError("Missing centered export tiles: " + str(missing))
    if not unreal.EditorLevelLibrary.save_current_level():
        raise RuntimeError("Could not save centered tile placement")


def run_asset_placement(carla, engine, output, cpu_set, env, map_asset, assets_file):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result, script = output / "asset_origins.json", output / "place_assets_ue.py"
    script.write_text("import sys\nsys.path.insert(0,%r)\nfrom scripts.build.configure_asset_origins import configure_in_editor\nconfigure_in_editor(%r,%r,%r)\n" %
                      (str(Path(__file__).resolve().parents[2]), str(result), map_asset, str(Path(assets_file).resolve())))
    command = ["taskset", "-c", cpu_set, str(Path(engine) / "Engine/Binaries/Linux/UE4Editor"),
               str(Path(carla) / "Unreal/CarlaUE4/CarlaUE4.uproject"), "-run=pythonscript",
               "-script=" + str(script), "-nullrhi", "-unattended", "-nosourcecontrol",
               "-nosound", "-nocrashreports"]
    with (output / "editor.log").open("w") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode != 0 or not result.exists() or not json.loads(result.read_text())["passed"]:
        raise RuntimeError("Centered tile placement failed; inspect editor.log")
    return result
