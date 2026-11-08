"""Execute explicit Agent projection feedback for measured failed regions."""
import argparse,json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from shapely.errors import GEOSException
from scripts.scene.constrain_access_group import project
from scripts.scene.repair_access_heights import repair


def work(batch_root,row):
    directory=Path(batch_root)/row['region'];condition=json.loads((directory/'condition.json').read_text())
    output=directory/row.get('output_name','bounded_feedback_v1')
    if(output/'review.json').exists():
        review=json.loads((output/'review.json').read_text())
    else:
        raw=json.loads((directory/'generation/raw_paths.json').read_text());options=dict(row['options'])
        if row.get('initial_paths'):options['initial_paths']=json.loads(Path(row['initial_paths']).read_text())
        output.mkdir(exist_ok=True)
        options.setdefault('checkpoint_path',str(output/'progress.json'))
        try:
            paths,review=project(condition,raw,**options)
            result=dict(paths=paths,condition=str(directory/'condition.json'),raw=str(directory/'generation/raw_paths.json'),passed=review['passed'])
            (output/'paths.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
        except(ValueError,GEOSException)as error:
            review=dict(passed=False,solver_success=False,message=str(error),requires_construction_feedback=True)
        review.update(agent='decision_agent',reason=row['reason'],feedback_options=row['options'])
        (output/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    selected=output;passed=review['passed']
    if not passed and review.get('solver_success'):
        lanes={l['id']:l for l in condition['lanes']};metrics=review['path_metrics']
        geometry=all(p['simple_xy']and p['valid_width_envelope']and p['maximum_curvature_per_m']<=(.2 if 'passenger'in lanes[p['id']]['allowed_classes']else .5)+.001 for p in metrics)
        if condition.get('physical_clearance'):
            geometry=geometry and all(p.get('minimum_building_clearance_margin_m',-1)>=-1e-4 and p.get('building_envelope_overlap_m2') is not None and p['building_envelope_overlap_m2']<.01 for p in metrics)
        if geometry:
            height=directory/(output.name.replace('_feedback','_height_feedback'))
            if(height/'review.json').exists():hr=json.loads((height/'review.json').read_text())
            else:
                repaired,hr=repair(condition,json.loads((output/'paths.json').read_text()),review)
                hr.update(agent='decision_agent',source_paths=str(output/'paths.json'),reason='Actual corrected XY passes; preserve it while repairing shared-port/source-profile grade.')
                height.mkdir(exist_ok=True);(height/'review.json').write_text(json.dumps(hr,indent=2)+'\n')
                if repaired is not None:(height/'paths.json').write_text(json.dumps(repaired,separators=(',',':'))+'\n')
            if hr['passed']:selected=height;passed=True
    return dict(region=row['region'],passed=passed,projection_passed=review['passed'],solver_success=review.get('solver_success'),
        message=review.get('message'),paths=str((selected/'paths.json').resolve()),review=str((selected/'review.json').resolve()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('batch-root','conditions','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    conditions=json.loads(a.conditions.read_text());records=[];a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers)as pool:
        jobs=[pool.submit(work,a.batch_root,row)for row in conditions['regions']]
        for future in as_completed(jobs):
            result=future.result();records.append(result);print(result,flush=True)
            (a.output/'report.json').write_text(json.dumps(dict(conditions=str(a.conditions),records=records),indent=2)+'\n')
    report=dict(conditions=str(a.conditions),configured_regions=len(conditions['regions']),completed_regions=len(records),
        passed=sum(x['passed']for x in records),failed=sum(not x['passed']for x in records),records=records,
        regions={x['region']:dict(paths=x['paths'],review=x['review'])for x in records if x['passed']})
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items()if k not in ('records','regions')},flush=True)
if __name__=='__main__':main()
