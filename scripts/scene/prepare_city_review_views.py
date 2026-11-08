"""Prepare calibrated source view histories for every declared city Area."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,sys
from pathlib import Path

from scripts.scene.prepare_city_actor_layers import group_area_views

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('canopy-root','actor-layers','data-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);areas=[];jobs=[]
    for path in sorted(a.canopy_root.glob('*/extracted/evidence.json')):
        evidence=json.loads(path.read_text());log=Path(evidence['log'])
        for area, source_views in group_area_views(evidence['views']).items():
            views=[]
            areas.append(dict(area_id=area,log_id=log.name,split=log.parent.name,views=views,
                source_evidence=str(path)))
            for source_view in source_views:
                view=dict(source_view)
                relative=Path(area)/log.name/view['camera']/str(view['timestamp_ns'])/'history.json'
                view['history_relative_path']=str(relative);views.append(view)
                folder=a.output/'histories'/relative.parent
                folder.mkdir(parents=True,exist_ok=True)
                args=[sys.executable,'-u','-m','scripts.assets.prepare_replay_clip',
                    '--scene',str(a.actor_layers/area/'scene.json'),'--log-dir',str(log),
                    '--start-ns',str(view['timestamp_ns']),'--duration-s','0','--stride','1',
                    '--camera',view['camera'],'--layers',str(a.actor_layers/area/'actor_layers.json'),
                    '--local-images','--output',str(folder)]
                jobs.append(dict(area_id=area,log_id=log.name,camera=view['camera'],
                    timestamp_ns=view['timestamp_ns'],folder=str(folder),argv=args))
    areas.sort(key=lambda r:r['area_id'])
    (a.output/'original_views.json').write_text(json.dumps(dict(areas=areas,scope='Exact previously reviewed source images; any annotation-span correction is recorded separately.'),indent=2)+'\n')
    (a.output/'commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
    def prepare(job):
        with (Path(job['folder'])/'prepare.log').open('w') as f:
            proc=subprocess.run(job['argv'],stdout=f,stderr=subprocess.STDOUT)
        row=dict(area_id=job['area_id'],log_id=job['log_id'],camera=job['camera'],
            timestamp_ns=job['timestamp_ns'],exit_code=proc.returncode,folder=job['folder'])
        print(row,flush=True);return row
    with ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(prepare,jobs))
    (a.output/'history_review.json').write_text(json.dumps(dict(areas=len(areas),views=rows),indent=2)+'\n')
    if any(r['exit_code'] for r in rows):raise RuntimeError('Source histories failed; retained all selected views in report')
    subprocess.run([sys.executable,'-u','-m','scripts.scene.prepare_city_view_snapshots',
        '--views',str(a.output/'original_views.json'),'--histories',str(a.output/'histories'),
        '--data-root',str(a.data_root),'--output',str(a.output/'aligned')],check=True)

if __name__=='__main__':main()
