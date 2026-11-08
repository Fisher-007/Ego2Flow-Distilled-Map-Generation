"""Build AV2 crossing paint selected by the graph, using the existing FBX path."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from map_generation.crossing_markings import build_crossings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'roads', 'conditions', 'output', 'blender', 'dsc-root'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    result = build_crossings(*[json.loads(path.read_text()) for path in (a.scene, a.roads, a.conditions)])
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'markings.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (a.output/'report.json').write_text(json.dumps(result['report'], indent=2)+'\n')
    if not result['report']['passed']:
        raise ValueError('Crossing projection failed; inspect report.json')
    subprocess.run([str(a.blender), '--background', '--threads', '4', '--python-exit-code', '1',
        '--python', str(Path(__file__).resolve().with_name('build_markings.py')), '--', '--scene', str(a.scene),
        '--roads', str(a.roads), '--output', str(a.output), '--dsc-root', str(a.dsc_root),
        '--blender-stage'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    print({k: v for k, v in result['report'].items() if k != 'records'}, flush=True)


if __name__ == '__main__':
    main()
