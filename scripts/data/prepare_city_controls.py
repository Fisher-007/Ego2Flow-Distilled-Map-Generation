"""Retrieve existing source-image evidence for annotated controls across a city."""
import argparse
import json
from pathlib import Path
from scripts.assets.inspect_observed_controls import inspect_controls


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','config','data-root','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();scene=json.loads(a.scene.read_text());config=json.loads(a.config.read_text())
    a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for relative in config['sources']['av2_logs']:
        path=a.data_root/relative;out=a.output/path.name
        controls=inspect_controls(scene,path,out)
        row=dict(log_id=path.name,controls=len(controls),views=sum(len(r['views']) for r in controls),
            evidence=str(out/'controls.json'))
        rows.append(row);print(row,flush=True)
        (a.output/'summary.json').write_text(json.dumps(dict(logs=rows,
            scope='Source cuboids and calibrated image crops. Agent face/orientation and approach-binding review follows; duplicate observations may describe the same physical control.'),indent=2)+'\n')


if __name__=='__main__':main()
