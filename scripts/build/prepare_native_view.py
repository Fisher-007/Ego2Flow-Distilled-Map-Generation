"""Create a separate CARLA asset project using shared engine and stock resources."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-carla', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source, target = a.source_carla.resolve(), a.output.absolute()
    target.mkdir(parents=True, exist_ok=False)
    project = target/'Unreal/CarlaUE4'
    original = source/'Unreal/CarlaUE4'
    project.mkdir(parents=True)
    shutil.copy2(original/'CarlaUE4.uproject', project/'CarlaUE4.uproject')
    shutil.copytree(original/'Config', project/'Config')
    for name in ('Binaries', 'Plugins', 'Source'):
        (project/name).symlink_to(original/name, target_is_directory=True)
    (project/'Content').mkdir()
    (project/'Content/Carla').symlink_to(original/'Content/Carla', target_is_directory=True)
    for name in ('Content/Python', 'Saved', 'Intermediate', 'DerivedDataCache'):
        (project/name).mkdir(parents=True, exist_ok=True)
    (target/'PythonAPI').symlink_to(source/'PythonAPI', target_is_directory=True)
    (target/'Import').mkdir()
    # Import.py resolves the project root from its own location. Copy the
    # helper scripts so package manifests are written to the target project.
    helpers = target/'Util/BuildTools'
    helpers.mkdir(parents=True)
    for path in (source/'Util/BuildTools').glob('*.py'):
        shutil.copy2(path, helpers/path.name)
    report = dict(source_carla=str(source), carla_root=str(target),
                  private=['Config', 'Content/generated packages', 'Saved', 'Intermediate', 'Import'],
                  read_only_shared=['Content/Carla', 'Binaries', 'Plugins', 'Source', 'PythonAPI'],
                  scope='Import and render own new packages using unchanged installed engine/stock; no module rebuild or stock mutation.')
    (target/'native_view.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
