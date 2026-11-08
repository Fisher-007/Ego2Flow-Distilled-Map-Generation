"""Set matte inputs on the selected map's surface materials."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


ASSET_ROOT = None
MATERIAL_NAMES = ("road", "terrain", "building", "road_edge")


def configure_in_editor(output, asset_root=ASSET_ROOT):
    """Update material inputs through the Unreal Python API."""
    import unreal

    library = unreal.MaterialEditingLibrary

    def inputs(material, stage):
        # Read scalar overrides through UMaterial's public text exporter.
        exported = Path(output).parent / (material.get_name() + "_" + stage + ".t3d")
        task = unreal.AssetExportTask()
        task.object, task.filename = material, str(exported)
        task.exporter = unreal.ObjectExporterT3D()
        task.automated, task.prompt, task.replace_identical = True, False, True
        if not unreal.Exporter.run_asset_export_task(task):
            raise RuntimeError("Could not export material input properties: " + str(exported))
        text = exported.read_text(encoding="utf-8-sig")
        result = {}
        for name in ("base_color", "roughness", "metallic", "specular"):
            prop = getattr(unreal.MaterialProperty, "MP_" + name.upper())
            node = library.get_material_property_input_node(material, prop)
            entry = {"node": node.get_name() if node else None,
                     "class": node.get_class().get_name() if node else None}
            if isinstance(node, unreal.MaterialExpressionConstant):
                entry["node_value"] = node.get_editor_property("r")
            elif isinstance(node, unreal.MaterialExpressionVectorParameter):
                value = node.get_editor_property("default_value")
                entry["node_value"] = [value.r, value.g, value.b, value.a]
            if name != "base_color":
                line = re.search(r"^\s*" + name.title() + r"=\((.*)\)\s*$", text, re.MULTILINE)
                properties = line.group(1) if line else ""
                override = re.search(r"(?:^|,)UseConstant=(True|False)(?:,|$)", properties)
                constant = re.search(r"(?:^|,)Constant=([^,)]+)", properties)
                default = 0.0 if name == "metallic" else 0.5
                entry["exported_properties"] = properties or None
                entry["use_constant"] = override.group(1) == "True" if override else False
                entry["use_constant_source"] = ("T3D override" if override else
                                                "No T3D override; UE4 UMaterial class default False")
                entry["input_constant"] = float(constant.group(1)) if constant else default
                entry["input_constant_source"] = "T3D override" if constant else "UE4 UMaterial class default"
                if entry["use_constant"]:
                    entry["effective_value"] = entry["input_constant"]
                elif node is not None:
                    entry["effective_value"] = entry.get("node_value")
                else:
                    # Defaults for unconnected material inputs.
                    entry["effective_value"] = default
                    entry["value_source"] = "UE4.26 unconnected material input default"
            result[name] = entry
        return result

    records, absent = [], []
    for name in MATERIAL_NAMES:
        path = asset_root + "/" + name
        if not unreal.EditorAssetLibrary.does_asset_exist(path):
            absent.append(name)
            continue
        material = unreal.EditorAssetLibrary.load_asset(path)
        if not isinstance(material, unreal.Material):
            raise RuntimeError("Expected generated Material: " + path)
        before = inputs(material, "before")
        # UseConstant overrides connected material nodes.
        if any(before[key]["use_constant"] for key in ("roughness", "metallic")):
            raise RuntimeError("Material has a constant input override; no change made: " + path)
        for row, (property_name, value) in enumerate((("roughness", 1.0), ("metallic", 0.0))):
            prop = getattr(unreal.MaterialProperty, "MP_" + property_name.upper())
            node = library.get_material_property_input_node(material, prop)
            if not isinstance(node, unreal.MaterialExpressionConstant):
                node = library.create_material_expression(material, unreal.MaterialExpressionConstant, -200, 150 * row)
            node.set_editor_property("r", value)
            if not library.connect_material_property(node, "", prop):
                raise RuntimeError("Could not connect " + path + ":" + property_name)
        library.recompile_material(material)
        after = inputs(material, "after")
        passed = (after["roughness"]["effective_value"] == 1.0
                  and after["metallic"]["effective_value"] == 0.0
                  and after["specular"]["effective_value"] == 0.5
                  and after["specular"] == before["specular"]
                  and after["base_color"] == before["base_color"])
        if not passed:
            raise RuntimeError("Material inputs did not match the matte surface settings: " + path)
        saved = unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
        if not saved:
            raise RuntimeError("Material save failed: " + path)
        records.append({"asset": path, "before": before, "after": after, "saved": saved, "passed": passed})
    if not records:
        raise RuntimeError("No generated surface materials found in " + asset_root)
    report = {"asset_root": asset_root, "materials": records, "absent_material_names": absent, "passed": True,
              "reason": "UE4 FBX import preserves the diffuse color but drops Blender's untextured roughness scalar",
              "scope": "Saved material graph inputs; Linux shader prewarm and rendered appearance require separate verification"}
    Path(output).write_text(json.dumps(report, indent=2) + "\n")


def run_material_setup(carla_root, engine_root, output, cpu_set, env=None, asset_root=ASSET_ROOT):
    """Run the scoped editor operation; also used between import and assembly."""
    carla_root, engine_root, output = (Path(p).expanduser().resolve() for p in (carla_root, engine_root, output))
    output.mkdir(parents=True, exist_ok=True)
    result = output / "materials.json"
    result.unlink(missing_ok=True)
    script = output / "configure_materials_ue.py"
    script.write_text(
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('mapbuild_materials', %r)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.configure_in_editor(%r, %r)\n" % (str(Path(__file__).resolve()), str(result), asset_root))
    environment = dict(os.environ if env is None else env, CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    for category in ("CONFIG", "CACHE", "DATA"):
        directory = output / "runtime" / category.lower()
        directory.mkdir(parents=True, exist_ok=True)
        environment["XDG_" + category + "_HOME"] = str(directory)
    command = ["taskset", "-c", cpu_set, str(engine_root / "Engine/Binaries/Linux/UE4Editor"),
               str(carla_root / "Unreal/CarlaUE4/CarlaUE4.uproject"), "-run=pythonscript",
               "-script=" + str(script), "-nullrhi", "-unattended", "-nosourcecontrol", "-nosound",
               "-nosplash", "-nocrashreports", "-abslog=" + str(output / "materials.log")]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "materials.stdout.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    # UE Python commandlets may exit zero even after a Python exception.
    if not result.is_file() or not json.loads(result.read_text())["passed"]:
        raise RuntimeError("Material setup did not complete; inspect " + str(output / "materials.stdout.log"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--carla-root", type=Path, default=os.environ.get("CARLA_ROOT"))
    parser.add_argument("--engine-root", type=Path, default=os.environ.get("UE4_ROOT"))
    parser.add_argument("--cpu-set", default=os.environ.get("MAP_CPUSET"))
    parser.add_argument("--asset-root", required=True, help="Material folder of the selected generated map")
    args = parser.parse_args()
    if not all((args.carla_root, args.engine_root, args.cpu_set)):
        parser.error("Set CARLA_ROOT, UE4_ROOT and MAP_CPUSET, or supply their command-line options")
    print(run_material_setup(args.carla_root, args.engine_root, args.output, args.cpu_set, asset_root=args.asset_root))


if __name__ == "__main__":
    main()
