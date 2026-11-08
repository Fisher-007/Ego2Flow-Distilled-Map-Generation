"""Import FBX meshes and OpenDRIVE into a dedicated CARLA map package.

Set CARLA_ROOT, UE4_ROOT, and MAP_CPUSET, or use their command-line options.
Multiple FBX inputs retain separate meshes and exported transforms. Re-running
replaces the generated MAP<map-name> package.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fbx", type=Path, nargs="+", required=True,
                        help="One or more FBX files, all imported into the same map")
    parser.add_argument("--xodr", type=Path, required=True, help="OpenDRIVE file for the assembled map")
    parser.add_argument("--assets-manifest", type=Path, help="Centered tile export manifest for exact native world placement")
    parser.add_argument("--package", required=True,
                        help="Dedicated generated package, named MAP<map-name>; existing contents are replaced")
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Import settings, command logs and result JSON")
    parser.add_argument("--carla-root", type=Path, default=os.environ.get("CARLA_ROOT"),
                        help="Source-built CARLA checkout; defaults to CARLA_ROOT from sourced .env")
    parser.add_argument("--engine-root", type=Path, default=os.environ.get("UE4_ROOT"),
                        help="Compiled Unreal checkout; defaults to UE4_ROOT from sourced .env")
    parser.add_argument("--cpu-set", default=os.environ.get("MAP_CPUSET"),
                        help="Allocated CPU affinity passed to taskset; defaults to MAP_CPUSET")
    parser.add_argument('--replacement-plan',type=Path,
                        help='Existing frontage plan: defer distance fields, excluding open surfaces and building masses that this plan replaces')
    parser.add_argument('--resume-after-import',type=Path,
                        help='Explicit reviewed asset inventory for a completed import with diagnosed empty meshes; keep prior logs and saved assets')
    args = parser.parse_args()
    if args.assets_manifest:args.assets_manifest=args.assets_manifest.resolve()
    if args.replacement_plan:args.replacement_plan=args.replacement_plan.expanduser().resolve()
    if args.resume_after_import:args.resume_after_import=args.resume_after_import.expanduser().resolve()
    if not all((args.carla_root, args.engine_root, args.cpu_set)):
        parser.error("Set CARLA_ROOT, UE4_ROOT and MAP_CPUSET by sourcing .env, or provide their CLI options")
    carla, engine = args.carla_root.expanduser().resolve(), args.engine_root.expanduser().resolve()
    project = carla/"Unreal/CarlaUE4"
    editor = engine/"Engine/Binaries/Linux/UE4Editor"
    helper = carla/"Util/BuildTools/Import.py"
    fbx = [path.expanduser().resolve() for path in args.fbx]
    xodr, output = args.xodr.expanduser().resolve(), args.output.expanduser().resolve()
    package, name = args.package, args.map_name
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or package != "MAP" + name:
        parser.error("Use an alphanumeric/underscore map name starting with a letter, and package MAP<map-name>")
    package_dir = project/"Content"/package
    for path in [*fbx, xodr, editor, helper, project/"CarlaUE4.uproject"]:
        if not path.is_file() or not path.stat().st_size:
            parser.error(f"Required input is missing or empty: {path}")
    if any(path.is_relative_to(package_dir) for path in [*fbx, xodr, output]):
        parser.error("Inputs and output must be outside the generated package that will be replaced")

    # Use CARLA's import defaults, including separate static meshes and collision
    # settings. A single map ImportGroup can contain files from different dirs.
    spec = importlib.util.spec_from_file_location("carla_map_import", helper)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    material_spec = importlib.util.spec_from_file_location(
        "mapbuild_material_setup", Path(__file__).resolve().with_name("configure_materials.py"))
    material_setup = importlib.util.module_from_spec(material_spec)
    material_spec.loader.exec_module(material_setup)
    surface_spec = importlib.util.spec_from_file_location(
        "mapbuild_surface_setup", Path(__file__).resolve().with_name("configure_open_surfaces.py"))
    surface_setup = importlib.util.module_from_spec(surface_spec)
    surface_spec.loader.exec_module(surface_setup)
    semantic_spec = importlib.util.spec_from_file_location(
        "mapbuild_semantic_setup", Path(__file__).resolve().with_name("configure_semantics.py"))
    semantic_setup = importlib.util.module_from_spec(semantic_spec)
    semantic_spec.loader.exec_module(semantic_setup)
    output.mkdir(parents=True, exist_ok=True)
    (output/"map_build.json").unlink(missing_ok=True)
    os.chdir(output)
    maps = [{"name": name, "source": str(fbx[0]), "use_carla_materials": False}]
    settings = Path(upstream.generate_import_setting_file(package, str(output), [], maps, 0, 0))
    groups = json.loads(settings.read_text())
    groups["ImportGroups"][0]["FileNames"] = [str(path) for path in fbx]
    settings.write_text(json.dumps(groups, indent=2)+"\n")

    # CARLA's SavePackage skips existing maps without failing the commandlet.
    # The MAP<map-name> convention scopes replacement to this generated map.
    recovery=None
    if args.resume_after_import:
        recovery=json.loads(args.resume_after_import.read_text())
        manifest=json.loads(Path(recovery['mesh_manifest']).read_text())
        if Path(manifest['fbx']).resolve()!=fbx[0] or recovery['package']!=package:
            raise ValueError('Recovery inventory belongs to another input or package')
        excluded={row['name'] for row in recovery['excluded_assets']}
        missing=[row['name'] for row in manifest['objects']
            if row['name'] not in excluded and not (package_dir/'Maps'/name/('map_assets_'+row['name']+'.uasset')).exists()]
        if missing:raise ValueError('Unresolved missing imported meshes: '+str(missing))
    if package_dir.exists() and recovery is None:
        shutil.rmtree(package_dir)
    upstream.generate_package_file(package, [], maps)
    destination = package_dir/"Maps"/name
    opendrive = destination/"OpenDrive"/f"{name}.xodr"
    opendrive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(xodr, opendrive)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for category in ("CONFIG", "CACHE", "DATA"):
        path = output/"runtime"/category.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[f"XDG_{category}_HOME"] = str(path)
    base = ["taskset", "-c", args.cpu_set, str(editor), str(project/"CarlaUE4.uproject")]
    stages = [
        ("import", ["-run=ImportAssets", f"-importSettings={settings}", "-replaceexisting"]),
        ("assemble", ["-run=PrepareAssetsForCooking", f"-PackageName={package}", "-OnlyPrepareMaps=1"]),
    ]
    commands = {stage: base+options+["-nullrhi", "-unattended", "-nosourcecontrol", "-nosound",
                                    f"-abslog={output/(stage+'.log')}"] for stage, options in stages}
    if args.replacement_plan:
        plan=args.replacement_plan.expanduser().resolve()
        replacement=json.loads(plan.read_text())
        omitted={'map_assets_building_'+re.sub(r'[^A-Za-z0-9_]','_',row['id']) for row in replacement['buildings']}
        # This override belongs only to the initial import. Before any ordinary
        # editor/runtime load, persist zero resolution on the unused fields.
        defer='-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False'
        commands['import'].append(defer)
        script=output/'distance_fields_ue.py'
        script.write_text("import unreal,json,re\nfrom pathlib import Path\n"
            "lib=unreal.EditorAssetLibrary\nomitted=set(%r)\nrows=[]\n"%sorted(omitted)+
            "for path in lib.list_assets(%r,recursive=True,include_folder=False):\n"%(f'/Game/{package}/Maps/{name}')+
            "    asset_name=path.rsplit('/',1)[-1].split('.')[0]\n"
            "    if asset_name not in omitted and not re.fullmatch(r'map_assets_(?:road|terrain|road_edge)(?:_\\d+)?|(?:markings_)?road_markings_(?:white|yellow)',asset_name):continue\n"
            "    mesh=lib.load_asset(path)\n"
            "    if not isinstance(mesh,unreal.StaticMesh):continue\n"
            "    settings=unreal.EditorStaticMeshLibrary.get_lod_build_settings(mesh,0)\n"
            "    previous=settings.distance_field_resolution_scale\n"
            "    settings.distance_field_resolution_scale=0.\n"
            "    unreal.EditorStaticMeshLibrary.set_lod_build_settings(mesh,0,settings)\n"
            "    lib.save_loaded_asset(mesh,only_if_is_dirty=False)\n"
            "    rows.append(dict(asset=path,previous_scale=previous,scale=0.,reason='replaced building mass' if asset_name in omitted else 'open ground or paint'))\n"
            "Path(%r).write_text(json.dumps(dict(replacement_plan=%r,assets=rows,scope='Selected unused volumetric lighting fields only; mesh geometry, materials, collision and ordinary runtime lighting settings remain.'),indent=2)+'\\n')\n"%(str(output/'distance_fields.json'),str(plan)))
        commands['distance_fields']=base+['-run=pythonscript','-script='+str(script),defer,
            '-nullrhi','-unattended','-nosourcecontrol','-nosound','-abslog='+str(output/'distance_fields.log')]
    (output/"commands.json").write_text(json.dumps(commands, indent=2)+"\n")
    for stage, _ in stages:
        if stage=='import' and recovery is not None:
            print('Reusing saved import after explicit geometry/inventory review: '+str(args.resume_after_import),flush=True)
        else:
            print(f"Running {stage}; logs: {output/(stage+'.log')}", flush=True)
            with (output/f"{stage}.stdout.log").open("w") as log:
                subprocess.run(commands[stage], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if stage == "import" and not any(destination.rglob("*.uasset")):
            raise RuntimeError(f"Import produced no map assets; inspect {output/'import.log'}")
        if stage == "import":
            if args.replacement_plan:
                print('Configuring lighting fields for open surfaces and replaced building masses',flush=True)
                with (output/'distance_fields.stdout.log').open('w') as log:
                    subprocess.run(commands['distance_fields'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if not (output/'distance_fields.json').exists():raise RuntimeError('Selective distance-field setup failed')
            print("Restoring generated matte material inputs before map assembly", flush=True)
            material_report = material_setup.run_material_setup(
                carla, engine, output/"materials", args.cpu_set, env, asset_root=f"/Game/{package}/Maps/{name}")

    umap = destination/f"{name}.umap"
    if not umap.is_file() or not umap.stat().st_size:
        raise RuntimeError(f"Map assembly produced no map asset; inspect {output/'assemble.log'}")
    origin_report = None
    if args.assets_manifest:
        from scripts.build.configure_asset_origins import run_asset_placement
        origin_report = run_asset_placement(carla, engine, output/"asset_origins", args.cpu_set, env,
            map_asset=f"/Game/{package}/Maps/{name}/{name}", assets_file=args.assets_manifest)
    print("Configuring the three open ground surfaces as non-distance-field occluders", flush=True)
    surface_report = surface_setup.run_surface_setup(
        carla, engine, output/"open_surfaces", args.cpu_set, env, map_asset=f"/Game/{package}/Maps/{name}/{name}")
    print("Classifying generated meshes using CARLA's semantic asset folders", flush=True)
    semantic_report = semantic_setup.run_semantic_setup(
        carla, engine, output/"semantics", args.cpu_set, env, map_asset=f"/Game/{package}/Maps/{name}/{name}")
    assets = sorted(package_dir.rglob("*.uasset"))
    report = {"package": package, "map_name": name, "map": f"/Game/{package}/Maps/{name}/{name}",
              "fbx": [{"path": str(path), "bytes": path.stat().st_size} for path in fbx],
              "opendrive": str(opendrive), "opendrive_bytes": opendrive.stat().st_size,
              "umap": str(umap), "umap_bytes": umap.stat().st_size,
              "asset_count": len(assets),
              "import_recovery": recovery,
              "asset_origin_report": str(origin_report) if origin_report else None,
              "material_report": str(material_report),
              "open_surface_report": str(surface_report),
              "semantic_report": str(semantic_report),
              "assets": [str(path.relative_to(package_dir)) for path in assets],
              "scope": "Editor import and map assembly; runtime loading and driving are separate checks."}
    (output/"map_build.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
