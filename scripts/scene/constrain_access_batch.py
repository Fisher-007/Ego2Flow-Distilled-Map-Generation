"""Apply geometric construction constraints to completed city samples."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
from scripts.scene.constrain_access_group import project


def projection_inputs(condition):
    """Collect geometric projection dependencies and application identity."""
    keys=set(condition['generated_path_ids'])
    keys.update(row['neighbor_id']for row in condition['boundaries']if row['fixed'])
    return dict(physical_clearance=condition.get('physical_clearance'),bbox=condition['region']['bbox_local_xy'],
        generated_path_ids=condition['generated_path_ids'],boundaries=condition['boundaries'],
        paths=condition['paths'],source_parts=condition['source_parts'],
        lanes={lane['id']:lane for lane in condition['lanes']if lane['id']in keys})


def reuse_projection(path,previous):
    """Reuse measured outcomes only for identical source and raw path inputs."""
    path=Path(path);previous=Path(previous)
    if not(previous/'steering/review.json').exists():return None
    condition=json.loads(path.read_text());old=json.loads((previous/'condition.json').read_text())
    if projection_inputs(condition)!=projection_inputs(old):return None
    raw=path.parent/'generation/raw_paths.json';old_raw=previous/'generation/raw_paths.json'
    if json.loads(raw.read_text())['paths']!=json.loads(old_raw.read_text())['paths']:return None
    copied=[]
    candidates=[previous/'steering']+[p.parent for p in previous.glob('*feedback*/review.json')]
    for source in candidates:
        target=path.parent/source.name
        if target.exists():continue
        shutil.copytree(source,target)
        record=dict(source=str(source),condition=str(path),raw=str(raw),
            reason='Exact generated lane rows, fixed-neighbor lane rows, region bounds, path/application interval identities, shared/fixed boundaries and actual sampled path arrays match. Unused context rows and graph bookkeeping may differ; those rows are not consumed by project and any learned influence is checked through the actual raw arrays.')
        (target/'reuse.json').write_text(json.dumps(record,indent=2)+'\n')
        output=target/'paths.json'
        if output.exists():
            data=json.loads(output.read_text());data.update(condition=str(path),raw=str(raw))
            output.write_text(json.dumps(data,separators=(',',':'))+'\n')
        copied.append(str(target))
    return dict(region=condition['region']['id'],source=str(previous),copied=copied)


def work(path):
    path=Path(path);condition=json.loads(path.read_text());output=path.parent/'steering'
    raw_path=path.parent/'generation/raw_paths.json';raw=json.loads(raw_path.read_text());output.mkdir(exist_ok=True)
    try:
        paths,review=project(condition,raw)
        review.update(agent=condition.get('agent','decision_agent'),reason='Keep the sampled source-supported group within fixed measured ports, shared inferred ports and passenger steering bounds; unresolved groups require construction feedback.')
        (output/'review.json').write_text(json.dumps(review,indent=2)+'\n')
        (output/'paths.json').write_text(json.dumps(dict(paths=paths,condition=str(path),raw=str(raw_path),passed=review['passed']),separators=(',',':'))+'\n')
        result=dict(region=condition['region']['id'],passed=review['passed'],iterations=review['iterations'],path_metrics=review['path_metrics'],message=review['message'])
    except ValueError as error:
        result=dict(region=condition['region']['id'],passed=False,message=str(error))
        (output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-root',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--start-region',help='Start at this region ID when continuing a separately submitted range')
    p.add_argument('--reuse-projection-from',type=Path,help='Previous batch with identical actual geometric projection inputs')
    a=p.parse_args();records=json.loads((a.batch_root/'manifest.json').read_text())['records']
    inputs=[r['condition'] for r in records if r['status']=='prepared' and (not a.start_region or r['region']>=a.start_region) and (Path(r['condition']).parent/'generation/latents.pt').exists()]
    reused=[]
    if a.reuse_projection_from:
        for path in inputs:
            if (Path(path).parent/'steering/review.json').exists():continue
            row=reuse_projection(path,a.reuse_projection_from/Path(path).parent.name)
            if row is not None:reused.append(row);print(dict(reused_projection=row['region']),flush=True)
        (a.batch_root/'projection_reuse.json').write_text(json.dumps(dict(records=reused),indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(work,path):path for path in inputs if not (Path(path).parent/'steering/review.json').exists()}
        for future in as_completed(futures):
            result=future.result();results.append(result);print(result,flush=True)
            (a.batch_root/'projection_batch.json').write_text(json.dumps(dict(records=results),indent=2)+'\n')
    reviews=[]
    for row in records:
        if row['status']!='prepared':continue
        path=Path(row['condition']).parent/'steering/review.json'
        if path.exists():
            value=json.loads(path.read_text());reviews.append(dict(region=row['region'],passed=value['passed'],review=str(path),reused=(path.parent/'reuse.json').exists()))
    summary=dict(records=reviews,passed=sum(r['passed'] for r in reviews),failed=sum(not r['passed'] for r in reviews),sampled_at_start=len(inputs),reused_projection_reports=sum(r['reused']for r in reviews),new_projection_reports=len(results))
    (a.batch_root/'projection_batch.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k!='records'},flush=True)


if __name__=='__main__':main()
