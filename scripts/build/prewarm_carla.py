"""Populate a map's Unreal derived-data cache before rendering.

Uses CARLA_ROOT, UE4_ROOT, and MAP_CPUSET. An optional cache directory stores
the Boot and filesystem caches separately.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def resolve_map(value, content):
    if value.startswith("/Game/"):
        path = content / value[len("/Game/"):]
    else:
        path = Path(value).expanduser()
        if not path.is_absolute():
            matches = sorted(content.rglob(path.with_suffix(".umap").name))
            if len(matches) != 1:
                raise ValueError(
                    f"Map {value!r} matched {len(matches)} files; use /Game/..."
                )
            path = matches[0]
    path = path.with_suffix(".umap").resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    path.relative_to(content)
    return path


def isolated_cache_config(project, cache):
    """Write a DefaultEngine.ini copy with a separate derived-data cache graph.

    The DEFENGINEINI file preserves commas inside backend tuples. Use this file
    and graph when launching the simulator.
    """
    cache.mkdir(parents=True, exist_ok=True)
    config = cache / "DefaultEngine.ini"
    original = (project / "Config/DefaultEngine.ini").read_text()
    graph = f"""

[MAPPrewarm]
Root=(Type=KeyLength, Length=120, Inner=AsyncPut)
AsyncPut=(Type=AsyncPut, Inner=Hierarchy)
Hierarchy=(Type=Hierarchical, Inner=Boot, Inner=Local)
Boot=(Type=Boot, Filename="{cache / 'Boot.ddc'}", MaxCacheSize=512)
Local=(Type=FileSystem, ReadOnly=false, Clean=false, Flush=false, PurgeTransient=true, DeleteUnused=false, Path="{cache / 'Local'}")
"""
    config.write_text(original + graph)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map", action="append", required=True, dest="maps",
        help="/Game/... package, absolute .umap path, or unique map name; repeatable. "
             "Also includes files under the map's sibling Sublevels/<map name>/.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Log/report directory")
    parser.add_argument("--target-platform", choices=("Linux", "LinuxNoEditor"), default="Linux",
                        help="Linux matches UE4Editor -game; LinuxNoEditor prepares cooked builds")
    parser.add_argument(
        "--cache-dir", type=Path, default=os.environ.get("MAP_DDC_DIR"),
        help="Directory for the persistent Boot/Local cache. "
             "Defaults to MAP_DDC_DIR when set. Use a new directory for a cold-cache check.",
    )
    args = parser.parse_args()
    for name in ("CARLA_ROOT", "UE4_ROOT", "MAP_CPUSET"):
        if not os.environ.get(name):
            parser.error(f"Set {name} before running this command")

    carla = Path(os.environ["CARLA_ROOT"]).expanduser().resolve()
    engine = Path(os.environ["UE4_ROOT"]).expanduser().resolve()
    project = carla / "Unreal/CarlaUE4"
    content = project / "Content"
    requested = list(dict.fromkeys(resolve_map(value, content) for value in args.maps))
    maps = list(requested)
    for path in requested:
        # Include the map's optional sublevels.
        sublevels = path.parent / "Sublevels" / path.stem
        maps.extend(sorted(sublevels.rglob("*.umap")))
    maps = list(dict.fromkeys(maps))

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    editor_log = output / "editor.log"
    command = [
        "taskset", "-c", os.environ["MAP_CPUSET"],
        str(engine / "Engine/Binaries/Linux/UE4Editor"),
        str(project / "CarlaUE4.uproject"),
        "-run=DerivedDataCache", "-fill", "-MapsOnly", "-ProjectOnly",
        "-Map=" + "+".join(str(path.with_suffix("")) for path in maps),
        f"-TargetPlatform={args.target_platform}", "-nullrhi", "-unattended",
        "-nosplash", "-nosound", "-nosourcecontrol", "-nocrashreports",
        "-stdout", f"-abslog={editor_log}",
    ]
    if cache:
        config = isolated_cache_config(project, cache)
        command.extend([f"-DefEngineIni={config}", "-DDC=MAPPrewarm"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for category in ("CONFIG", "CACHE", "DATA"):
        runtime = output / "runtime" / category.lower()
        runtime.mkdir(parents=True, exist_ok=True)
        env[f"XDG_{category}_HOME"] = str(runtime)

    report = {
        "command": command,
        "requested_maps": [str(path) for path in requested],
        "maps": [str(path) for path in maps],
        "cache_dir": str(cache) if cache else None,
    }
    report_path = output / "prewarm.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prewarming {len(maps)} map(s); log: {editor_log}", flush=True)
    stdout_log = output / "commandlet.stdout.log"
    with stdout_log.open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)

    # Read stdout from this invocation.
    text = stdout_log.read_text(errors="replace")
    loaded = re.findall(
        r"LogDerivedDataCacheCommandlet: (?:Display: )?Loading \(\d+\) ([^\r\n]+)", text
    )
    missing = [str(path) for path in requested if str(path) not in loaded]
    # This message follows BlockUntilAllBuildsComplete for distance
    # fields. A successful commandlet exit also waits for DDC disk writes.
    finished = "Done waiting for shaders to finish." in text
    cache_selected = cache is None or (
        f"Using Local data cache path {cache / 'Local'}: Writable" in text
        and "Reverting to the default graph" not in text
    )
    load_errors = re.findall(r"LogDerivedDataCacheCommandlet: Error: [^\r\n]+", text)
    report.update(
        returncode=result.returncode, loaded_maps=loaded, missing_maps=missing,
        completed_wait=finished, cache_selected=cache_selected, load_errors=load_errors,
    )
    report["success"] = (
        result.returncode == 0 and bool(loaded) and not missing
        and finished and cache_selected and not load_errors
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["success"]:
        raise RuntimeError(f"Prewarm did not complete; inspect {report_path} and {stdout_log}")
    print(f"Prewarm complete: {report_path}")


if __name__ == "__main__":
    main()
