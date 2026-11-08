"""Recheck existing multi-lane output without regenerating its source geometry."""
import argparse,json
from pathlib import Path
from scripts.scene.pack_lane_groups import review_compiled


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','original','roads']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report=review_compiled(json.loads(a.scene.read_text()),a.original,a.roads)
    print({k:v for k,v in report.items() if k not in ['source_edges','lateral','rebuilt_movement_roads']},flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
