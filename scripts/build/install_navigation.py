"""Attach reviewed native navigation to a copy of an existing physical map."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-map', required=True)
    p.add_argument('--name', required=True)
    for name in ('xodr', 'review', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    review = json.loads(a.review.read_text())
    if not review['passed']:
        raise ValueError('Resolve the native navigation review before installation')
    from scripts.scene.merge_native_controls import control_reference_review
    control_links = control_reference_review(a.xodr)
    if not control_links['passed']:
        raise ValueError('Resolve native light-group references before installation: ' + str(control_links))
    output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    project = Path(os.environ['CARLA_ROOT']) / 'Unreal/CarlaUE4'
    target = '/Game/MAP{0}/Maps/{0}/{0}'.format(a.name)
    destination = project / 'Content' / target.removeprefix('/Game/').rsplit('/', 1)[0]
    xodr = destination / 'OpenDrive' / (a.name + '.xodr')
    if xodr.exists() or (destination / (a.name + '.umap')).exists():
        raise ValueError('Use a new output map name')
    xodr.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.xodr, xodr)
    result = output / 'clone.json'
    script = output / 'clone_ue.py'
    script.write_text(
        'import sys\nsys.path.insert(0,%r)\n'
        'from scripts.build.clone_map import clone_in_editor\n'
        'clone_in_editor(%r,%r,%r)\n' %
        (str(Path(__file__).resolve().parents[2]), a.source_map, target, str(result)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    for name in ('CONFIG', 'CACHE', 'DATA'):
        folder = output / 'runtime' / name.lower(); folder.mkdir(parents=True, exist_ok=True)
        env['XDG_' + name + '_HOME'] = str(folder)
    command = ['taskset', '-c', os.environ['MAP_CPUSET'],
        str(Path(os.environ['UE4_ROOT']) / 'Engine/Binaries/Linux/UE4Editor'),
        str(project / 'CarlaUE4.uproject'), '-run=pythonscript', '-script=' + str(script),
        '-nullrhi', '-unattended', '-nosourcecontrol', '-nosound', '-nocrashreports',
        '-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False',
        '-abslog=' + str(output / 'editor.log')]
    start = time.monotonic()
    with (output / 'editor.stdout.log').open('w') as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not result.exists():
        raise RuntimeError('Map clone was not saved; inspect editor.log')
    report = json.loads(result.read_text())
    report.update(source_navigation=str(a.xodr.resolve()), installed_navigation=str(xodr),
        native_control_links=control_links,
        navigation_review=str(a.review.resolve()), editor_returncode=process.returncode,
        editor_wall_s=time.monotonic()-start,
        scope='Existing physical surfaces and assets retained. Supplied native road and control definitions installed. Runtime navigation and control responses remain to be checked.')
    (output / 'navigation_install.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
