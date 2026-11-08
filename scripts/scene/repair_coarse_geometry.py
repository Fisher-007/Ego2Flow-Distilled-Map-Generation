"""Apply constrained geometry feedback to inferred coarse roads before fusion."""
import argparse
import json
from pathlib import Path
from map_generation.coarse_geometry import repair


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--movement-limit',type=float,default=3.)
    args=parser.parse_args();scene=json.loads(args.scene.read_text())
    report=repair(scene['coarse_map'],args.movement_limit)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'geometry_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print({key:value for key,value in report.items() if key not in ('method','changed_lanes','remaining','conflicting_constraints')},flush=True)
    if report['solver_status']!='solved' or report['remaining']:
        raise RuntimeError('Coarse ordering repair requires further geometry review')
    scene.pop('fusion_map',None)
    with (args.output/'scene.json').open('w') as stream:
        json.dump(scene,stream,separators=(',',':'),allow_nan=False)


if __name__=='__main__':main()
