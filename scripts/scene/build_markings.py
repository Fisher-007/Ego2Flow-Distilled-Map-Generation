"""Export AV2 paint as a separate FBX on the supplied physical roads.

Run in the map environment. Import RoadLine components with NoCollision
and CastShadow=false.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def export_blender(output, dsc_root):
    import bpy

    spec = importlib.util.spec_from_file_location("dsc_material_helpers", dsc_root / "addon/helpers.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    markings = json.loads((output / "markings.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.context.scene.unit_settings.length_unit = "METERS"
    objects, max_error = [], 0.0
    for source in markings["meshes"]:
        origin=source.get('asset_origin_enu_m',[0.,0.,0.])
        local=[[p[i]-origin[i]for i in range(3)]for p in source['vertices']]
        mesh = bpy.data.meshes.new(source["id"])
        mesh.from_pydata(local, [], source["faces"])
        mesh.update()
        obj = bpy.data.objects.new(source["id"], mesh)
        bpy.context.collection.objects.link(obj)
        if 'material_rgba' in source:
            material = bpy.data.materials.new('observed_paint_' + source['id'])
            material.use_nodes = True
            principled = material.node_tree.nodes.get('Principled BSDF')
            principled.inputs['Base Color'].default_value = source['material_rgba']
            principled.inputs['Roughness'].default_value = .85
            principled.inputs['Specular IOR Level'].default_value = .1
            material.diffuse_color = source['material_rgba']
        else:
            helpers.assign_materials(obj)
            index = helpers.get_material_index(obj, "road_mark_" + source["color"])
            material = obj.data.materials[index]
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for face in mesh.polygons:
            face.material_index = 0
        obj["semantic_class"] = "RoadLines"
        obj["ue_asset_category"] = "RoadLine"
        obj["collision_required"] = "NoCollision"
        obj["cast_shadow_required"] = False
        for vertex, expected in zip(mesh.vertices, source["vertices"]):
            max_error = max(max_error, *(abs(vertex.co[i]+origin[i] - expected[i]) for i in range(3)))
        principled = material.node_tree.nodes.get("Principled BSDF")
        objects.append({"name": obj.name, "vertices": len(mesh.vertices), "triangles": len(mesh.polygons),
                        "asset_origin_enu_m":origin,
                        "material": material.name, "material_slots": len(obj.data.materials), "semantic_id": 24,
                        "material_inputs": {key: list(principled.inputs[key].default_value)
                                            if key == "Base Color" else float(principled.inputs[key].default_value)
                                            for key in ("Base Color", "Roughness", "Metallic", "Specular IOR Level")}})
    settings = {"global_scale": 1.0, "apply_unit_scale": True, "apply_scale_options": "FBX_SCALE_UNITS",
                "axis_forward": "-Y", "axis_up": "Z", "use_space_transform": True,
                "bake_space_transform": False, "bake_anim": False}
    for entry in objects:bpy.data.objects[entry['name']].location=entry['asset_origin_enu_m']
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "markings.blend"))
    for entry in objects:bpy.data.objects[entry['name']].location=(0.,0.,0.)
    bpy.ops.export_scene.fbx(filepath=str(output / "markings.fbx"), use_selection=False,
                             object_types={"MESH"}, **settings)
    report = {"fbx": str(output / "markings.fbx"), "blend": str(output / "markings.blend"),
              "fbx_settings": settings, "blender_version": bpy.app.version_string,
              "objects": objects, "max_blender_coordinate_error_m": max_error,
              "material_source": str(dsc_root / "addon/helpers.py") + ":assign_materials, or explicit source-reviewed material_rgba",
              "required_ue_settings": {"asset_folder": "/Game/<project>/Static/RoadLine",
                                       "collision": "NoCollision", "cast_shadow": False},
              # Allow 1 mm for Blender float32 world-coordinate export.
              "coordinate_tolerance_m": 0.001,
              "passed": max_error <= 0.001}
    (output / "fbx_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError("Blender coordinate conversion exceeds 1 mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    apps = Path(__file__).resolve().parents[3] / "apps"
    parser.add_argument("--blender", type=Path, default=apps / "blender-4.5.14-linux-x64/blender")
    parser.add_argument("--dsc-root", type=Path, default=apps / "blender-driving-scenario-creator")
    parser.add_argument("--conditions", type=Path, help="Recorded source-paint ambiguity and display choices")
    parser.add_argument("--width", type=float, default=0.12)
    parser.add_argument("--dash", type=float, default=3.0)
    parser.add_argument("--gap", type=float, default=6.0)
    parser.add_argument("--offset", type=float, default=0.002)
    parser.add_argument("--blender-stage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.blender_stage:
        export_blender(output, args.dsc_root.resolve())
        return
    from map_generation.road_markings import build_markings
    scene_text, roads_text = args.scene.read_text(), args.roads.read_text()
    result = build_markings(json.loads(scene_text), json.loads(roads_text), args.width, args.dash, args.gap, args.offset,
                            diagnostics=output, conditions=json.loads(args.conditions.read_text()) if args.conditions else None)
    result["report"].update(scene=str(args.scene.resolve()), roads=str(args.roads.resolve()))
    result["report"]["source_files_unchanged"] = args.scene.read_text() == scene_text and args.roads.read_text() == roads_text
    (output / "markings.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    (output / "report.json").write_text(json.dumps(result["report"], indent=2) + "\n")
    if not result["report"]["passed"]:
        raise ValueError("Marking projection failed; inspect report.json")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    subprocess.run([str(args.blender.resolve()), "--background", "--threads", "4", "--python-exit-code", "1",
                    "--python", str(Path(__file__).resolve()), "--", "--scene", str(args.scene.resolve()),
                    "--roads", str(args.roads.resolve()), "--output", str(output), "--dsc-root", str(args.dsc_root.resolve()),
                    "--blender-stage"], env=env, check=True)
    print(json.dumps({key: result["report"][key] for key in ("passed", "source_mark_type_counts", "mesh_count",
                     "triangles", "max_surface_interpolation_gap_m", "source_files_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
