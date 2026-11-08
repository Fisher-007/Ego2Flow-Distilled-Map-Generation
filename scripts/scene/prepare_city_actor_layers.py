"""Prepare removable actor layers for each source area and observation time."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter


def group_area_views(views):
    """Recover each declared Area from the canopy cache's per-log view list."""
    areas = {}
    for view in views:
        area = view.get('area_id') or view['id'].split('_ring_')[0]
        areas.setdefault(area, []).append(view)
    return areas


def select_actor_anchor(area, log_id, views, aligned_review=None, review_path=None):
    """Use an explicit paired-view selection, retaining the original epoch."""
    original=next(v for v in views if v['camera']=='ring_front_center')
    view=dict(original)
    alignment=None
    if aligned_review is not None:
        alignment=next(r for r in aligned_review['views']
            if r['area_id']==area and r['camera']=='ring_front_center')
        if (alignment['original_timestamp_ns']!=original['timestamp_ns'] or
                Path(alignment['source_image']).parents[3].name!=log_id):
            raise ValueError('Aligned actor selection does not match the original Area/log epoch: '+area)
        view.update(timestamp_ns=alignment['timestamp_ns'],file=alignment['source_image'])
        if 'area_anchor_difference_s' in view:
            view['area_anchor_difference_s']+=(view['timestamp_ns']-original['timestamp_ns'])/1e9
    epoch=dict(original_timestamp_ns=original['timestamp_ns'],
        actor_reference_timestamp_ns=view['timestamp_ns'],
        changed=view['timestamp_ns']!=original['timestamp_ns'],
        time_shift_s=(view['timestamp_ns']-original['timestamp_ns'])/1e9,
        aligned_review=str(review_path) if review_path is not None else None,
        aligned_selection=alignment)
    return view,epoch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','canopy-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--aligned-review',type=Path,
        help='Existing prepare_city_view_snapshots review.json; use its recorded front-center actor epochs')
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    scene=json.loads(a.scene.read_text());jobs=[]
    aligned_review=json.loads(a.aligned_review.read_text()) if a.aligned_review else None
    for path in sorted(a.canopy_root.glob('*/extracted/evidence.json')):
        evidence=json.loads(path.read_text());log=Path(evidence['log'])
        crop=json.loads(Path(evidence['scene']).read_text())
        # Crop source lanes were extracted with the reviewed log registration.
        # Refresh all nongeometry source control/road context from current city.
        crop.update(roads=scene['roads'],controls=scene['controls'])
        crop['actor_layer_source_scene']=str(a.scene)
        for area, views in group_area_views(evidence['views']).items():
            view,epoch=select_actor_anchor(area,log.name,views,aligned_review,a.aligned_review)
            folder=a.output/area;folder.mkdir(exist_ok=True)
            crop_path=folder/'scene.json'
            crop_path.write_text(json.dumps(crop,separators=(',',':'))+'\n')
            jobs.append((area,log,view,folder,crop_path,str(path),epoch))
    def build(job):
        area,log,view,folder,crop_path,evidence,epoch=job
        argv=[sys.executable,'-u','-m','scripts.assets.prepare_actor_layers',
            '--scene',str(crop_path),'--log-dir',str(log),'--anchor-ns',str(view['timestamp_ns']),
            '--output',str(folder)]
        with (folder/'prepare.log').open('w') as handle:
            result=subprocess.run(argv,stdout=handle,stderr=subprocess.STDOUT)
        row=dict(area=area,log_id=log.name,anchor_view=view,source_evidence=evidence,
            source_epoch_alignment=epoch,output=str(folder),exit_code=result.returncode,command=argv)
        if result.returncode==0:
            layers=json.loads((folder/'actor_layers.json').read_text())
            context=json.loads((folder/'runtime_context.json').read_text())
            row.update(counts=layers['counts'],anchor_parked=len(context['parked_actors']),
                anchor_traffic=len(context['traffic_initial_states']),
                unresolved_controls=len(context['unresolved_static_controls']),
                runtime_context=str(folder/'runtime_context.json'))
        print(row,flush=True);return row
    with ThreadPoolExecutor(max_workers=a.workers) as pool:records=list(pool.map(build,jobs))
    counts=Counter()
    for row in records:counts.update(row.get('counts',{}))
    report=dict(areas=records,configured_areas=len(jobs),completed_areas=sum(r['exit_code']==0 for r in records),
        all_track_role_counts=dict(counts),source_scene=str(a.scene),
        scope='Offline measured role proposals, not parking truth. Optional stationary actors remain removable. Moving and unresolved vehicles remain replaceable runtime participants. Native blueprint, support, color and traffic-control association review are separate.',
        source_epoch_policy='One Area/log at a time for observed replay. Do not merge simultaneous actor snapshots from different source epochs.',
        native_support_review_required=True)
    (a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='areas'},flush=True)
    if report['completed_areas']!=len(jobs):raise SystemExit(1)


if __name__=='__main__':main()
