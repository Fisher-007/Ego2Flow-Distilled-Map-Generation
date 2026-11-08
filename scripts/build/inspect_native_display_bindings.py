"""Read saved native material assignments and selected source control displays."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess


def inspect(map_path, output, label_prefixes, comparison_blueprint=None):
    import unreal
    level, library = unreal.EditorLevelLibrary, unreal.EditorAssetLibrary
    if not level.load_level(map_path):
        raise RuntimeError("Cannot load native map")
    actors = list(level.get_all_level_actors())
    temporary = None
    if comparison_blueprint:
        temporary = level.spawn_actor_from_object(library.load_asset(comparison_blueprint), unreal.Vector(0, 0, 0))
        temporary.set_actor_label("MAP_DisplayComparison_Unsaved")
        actors.append(temporary)
    records, summary, examples = [], Counter(), {}
    path = lambda obj: obj.get_path_name() if obj is not None else None
    for actor in actors:
        selected = actor is temporary or any(actor.get_actor_label().startswith(p) for p in label_prefixes)
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh = component.get_editor_property("static_mesh")
            if mesh is None:
                continue
            materials = [path(component.get_material(i)) for i in range(component.get_num_materials())]
            group = str(mesh.get_path_name()).split("/Static/")[-1].split("/")[0]
            key = json.dumps([group, materials])
            summary[key] += 1
            examples.setdefault(key, [])
            if len(examples[key]) < 2:
                examples[key].append(dict(actor=actor.get_actor_label(), mesh=path(mesh)))
            if not selected:
                continue
            tr = component.get_world_transform()
            rot = tr.rotation.rotator()
            parent = component.get_attach_parent()
            row = dict(actor=actor.get_actor_label(), component=component.get_name(),
                mesh=path(mesh), materials=materials,
                default_materials=[path(mesh.get_material(i)) for i in range(component.get_num_materials())],
                visible=component.is_visible(), hidden_in_game=component.get_editor_property("hidden_in_game"),
                parent=parent.get_name() if parent else None,
                position_carla_cm=[tr.translation.x, tr.translation.y, tr.translation.z],
                rotation_carla_deg=[rot.pitch, rot.yaw, rot.roll],
                scale=[tr.scale3d.x, tr.scale3d.y, tr.scale3d.z])
            records.append(row)
    material_rows = []
    material_paths = sorted({m for row in records for m in row["materials"] if m})
    for material_path in material_paths:
        material = library.load_asset(material_path)
        editing = unreal.MaterialEditingLibrary
        row = dict(path=material_path, class_name=material.get_class().get_name(),
            scalar_names=[str(x) for x in editing.get_scalar_parameter_names(material)],
            vector_names=[str(x) for x in editing.get_vector_parameter_names(material)],
            texture_names=[str(x) for x in editing.get_texture_parameter_names(material)])
        if isinstance(material, unreal.MaterialInstanceConstant):
            row["parent"] = path(material.get_editor_property("parent"))
            row["scalars"] = {name: editing.get_material_instance_scalar_parameter_value(material, name) for name in row["scalar_names"]}
            row["vectors"] = {}
            for name in row["vector_names"]:
                value = editing.get_material_instance_vector_parameter_value(material, name)
                row["vectors"][name] = [value.r, value.g, value.b, value.a]
            row["textures"] = {name: path(editing.get_material_instance_texture_parameter_value(material, name)) for name in row["texture_names"]}
        material_rows.append(row)
    report = dict(map=map_path, selected_components=records, selected_materials=material_rows,
        material_counts=[dict(group=json.loads(k)[0], materials=json.loads(k)[1], component_count=n,
                              examples=examples[k]) for k,n in summary.most_common()],
        comparison_blueprint=comparison_blueprint,
        scope="Read saved editor bindings plus an optional unsaved stock actor. No level or asset save; runtime emission remains a separate image check.")
    if temporary is not None:
        level.destroy_actor(temporary)
    Path(output).write_text(json.dumps(report, indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label-prefix", action="append", default=[])
    p.add_argument("--comparison-blueprint")
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    script=out/"inspect_ue.py";report=out/"display_bindings.json"
    script.write_text("import sys\nsys.path.insert(0,%r)\nfrom scripts.build.inspect_native_display_bindings import inspect\ninspect(%r,%r,%r,%r)\n" % (
        str(Path.cwd()),a.map,str(report),a.label_prefix,a.comparison_blueprint))
    command=["taskset","-c",os.environ["MAP_CPUSET"],str(Path(os.environ["UE4_ROOT"])/"Engine/Binaries/Linux/UE4Editor"),
        str(Path(os.environ["CARLA_ROOT"])/"Unreal/CarlaUE4/CarlaUE4.uproject"),"-run=pythonscript","-script="+str(script),
        "-nullrhi","-unattended","-nosourcecontrol","-nosound","-nocrashreports"]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
    with (out/"editor.log").open("w") as log:
        run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    print(dict(editor_returncode=run.returncode, report=str(report), report_exists=report.exists()),flush=True)
    if run.returncode or not report.exists():
        raise RuntimeError("Native display inspection failed; inspect editor.log")


if __name__=="__main__":
    main()
