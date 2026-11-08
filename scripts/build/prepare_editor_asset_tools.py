"""Install and compile the small wrapper around Unreal's native asset copier."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-actions", type=int, default=4)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / "native/MapBuilderEditorTools"
    project = (Path(os.environ["CARLA_ROOT"]) / "Unreal/CarlaUE4").resolve()
    destination = project / "Plugins/MapBuilderEditorTools"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    command = ["taskset", "-c", os.environ["MAP_CPUSET"], str((Path(os.environ["UE4_ROOT"]) / "Engine/Build/BatchFiles/Linux/Build.sh").resolve()),
               "CarlaUE4Editor", "Linux", "Development", str(project / "CarlaUE4.uproject"),
               "-NoHotReloadFromIDE",
               "-MaxParallelActions=" + str(args.parallel_actions)]
    with (args.output / "build.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report = dict(source=str(source), installed_plugin=str(destination), command=command,
                  returncode=result.returncode,
                  scope="Editor-only wrapper around the installed engine asset copier. City assets and map state are unchanged by plugin preparation.")
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
