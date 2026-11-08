"""Apply graph-source access restoration and save its actual construction report."""
import argparse
import json
from pathlib import Path
from map_generation.access_fusion import restore_access

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--alignment-distance',type=float,default=3.)
    p.add_argument('--include-fine-junctions',action='store_true',help='Allow source-supported approaches to join an observed AV2 intersection lane; keep its longitudinal path')
    p.add_argument('--minimum-transition',type=float,default=8.)
    p.add_argument('--maximum-path',type=float,default=60.)
    p.add_argument('--maximum-motor-curvature',type=float,help='Require a feasible endpoint curve before selecting a passenger-lane attachment station')
    p.add_argument('--retry-unresolved',action='store_true',help='Reuse matched source paths and only search the pending source ports')
    p.add_argument('--reuse-candidates',type=Path,help='Rebuild the same reviewed source matches after a geometry correction')
    a=p.parse_args();s=restore_access(json.loads(a.scene.read_text()),{'alignment_distance_m':a.alignment_distance,'include_fine_junctions':a.include_fine_junctions,
        'minimum_transition_m':a.minimum_transition,'max_path_m':a.maximum_path,'retry_unresolved':a.retry_unresolved,
        'maximum_motor_curvature_per_m':a.maximum_motor_curvature},
        json.loads(a.reuse_candidates.read_text()) if a.reuse_candidates else None)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'scene.json').open('w') as stream:json.dump(s,stream,separators=(',',':'),allow_nan=False)
    r=s['fusion_map']['source_access_review'];(a.output/'access_review.json').write_text(json.dumps(r,indent=2)+'\n')
    print(dict(connections=len(r['connections']),unresolved=len(r['unresolved']),split_fine_lanes=len(r['fine_station_splits'])))

if __name__=='__main__':main()
