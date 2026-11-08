"""Apply height-only refinement to generated groups with grade violations."""
import argparse,json
from collections import Counter
from pathlib import Path
from scripts.scene.repair_access_heights import repair

def process(batch_root,feedback_name='height_feedback_v1'):
    batch_root=Path(batch_root);records=[];selection={}
    manifest=json.loads((batch_root/'manifest.json').read_text())
    for row in manifest['records']:
        if row['status']!='prepared':continue
        condition_path=Path(row['condition']);directory=condition_path.parent
        review_path=directory/'steering/review.json'
        if not review_path.exists():continue
        review=json.loads(review_path.read_text())
        if review['passed']:
            records.append(dict(region=row['region'],status='initial_projection_passed'));continue
        condition=json.loads(condition_path.read_text());lanes={l['id']:l for l in condition['lanes']}
        metrics=review.get('path_metrics',[])
        xy_passed=review.get('solver_success',False) and len(metrics)==len(condition['generated_path_ids']) and all(
            p['simple_xy'] and p['valid_width_envelope'] and p['maximum_curvature_per_m']<=(.2 if 'passenger'in lanes[p['id']]['allowed_classes']else .5)+.001 for p in metrics)
        if not xy_passed:
            records.append(dict(region=row['region'],status='requires_geometry_feedback',source_review=str(review_path)));continue
        destination=directory/feedback_name
        if(destination/'review.json').exists():
            repaired_review=json.loads((destination/'review.json').read_text());reused=True
        else:
            paths_path=directory/'steering/paths.json';paths=json.loads(paths_path.read_text())
            result,repaired_review=repair(condition,paths,review)
            repaired_review.update(agent=condition.get('agent','decision_agent'),source_paths=str(paths_path),
                reason='Actual projection passed XY simplicity, road-width envelope and class-specific curvature but failed grade. Preserve sampled/projected XY and exact outer heights while fitting shared inferred port heights and arc-length profiles.')
            destination.mkdir(exist_ok=True)
            (destination/'review.json').write_text(json.dumps(repaired_review,indent=2)+'\n')
            if result is not None:(destination/'paths.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
            reused=False
        status='height_feedback_passed'if repaired_review['passed']else'requires_height_feedback'
        records.append(dict(region=row['region'],status=status,review=str(destination/'review.json'),reused=reused))
        if repaired_review['passed']:selection[row['region']]=dict(paths=str((destination/'paths.json').resolve()),review=str((destination/'review.json').resolve()))
    report=dict(configured_regions=len(manifest['records']),completed_projection_reports=len(records),statuses=dict(Counter(r['status']for r in records)),records=records,regions=selection)
    (batch_root/'height_feedback_batch.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-root',type=Path,required=True)
    p.add_argument('--feedback-name',default='height_feedback_v1')
    a=p.parse_args();report=process(a.batch_root,a.feedback_name)
    print({k:v for k,v in report.items()if k not in ('records','regions')},flush=True)
if __name__=='__main__':main()
