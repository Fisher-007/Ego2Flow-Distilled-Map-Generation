"""Sample physical surfaces with the packed map's actual lane IDs and boundaries."""
import argparse
import json
from pathlib import Path
from map_generation.packed_surfaces import build_surfaces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['scene', 'roads', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--sample-step', type=float, default=.5)
    p.add_argument('--previous-samples',type=Path,help='Reuse unaffected intervals after an explicit height-only native revision')
    p.add_argument('--height-revision',type=Path)
    p.add_argument('--compiler-scene',type=Path,help='Recorded observed compiler partitions omitted from the published scene; used only for sampling geometry')
    a = p.parse_args()
    print(build_surfaces(json.loads(a.scene.read_text()), a.roads, a.output, a.sample_step, a.workers,
                         a.previous_samples,a.height_revision,a.compiler_scene), flush=True)


if __name__ == '__main__':
    main()
