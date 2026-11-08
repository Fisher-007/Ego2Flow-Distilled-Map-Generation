"""Prepare retained OSM/SUMO parallel lanes for the shared native compiler."""
import argparse
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from map_generation.coarse_lane_sections import prepare_group


def prepare(job):
    try:
        lanes,report=prepare_group(job)
        return lanes,dict(passed=True,**report)
    except ValueError as error:
        return [],dict(passed=False,edge_id=job[0]['edge_id'],source_lane_ids=[l['id']for l in job],reason=str(error))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','ports','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();scene=json.loads(a.scene.read_text());mapping=json.loads(a.ports.read_text())
    ends={};body=set()
    for group in mapping['physical_junctions']:
        body.update(group['source_lane_ids'])
        for attr,endpoint in [('incoming_lane_ids','end'),('outgoing_lane_ids','start')]:
            for key in group[attr]:ends[key,endpoint]=group['id']
    candidates=defaultdict(list)
    for lane in scene['fusion_map']['lanes']:
        if lane['source']=='osm_sumo' and not lane.get('internal') and 'passenger' in lane['allowed_classes']and lane['id']not in body:
            candidates[lane['edge_id']].append(lane)
    jobs=[];reports=[]
    for edge,lanes in candidates.items():
        if len({l['source_lane_id']for l in lanes})<2:continue
        ordered=sorted(lanes,key=lambda l:int(l['source_lane_id'].rsplit('_',1)[1]),reverse=True)
        starts={ends.get((l['id'],'start'))for l in ordered};stops={ends.get((l['id'],'end'))for l in ordered}
        if len(starts)!=1 or len(stops)!=1 or None in starts|stops:
            reports.append(dict(passed=False,edge_id=edge,source_lane_ids=[l['id']for l in ordered],
                reason='Clipped coarse lanes require common source endpoints',start_junctions=list(starts),end_junctions=list(stops)))
        else:jobs.append(ordered)
    replacements={};groups=[]
    with ProcessPoolExecutor(max_workers=a.workers)as pool:
        for count,(lanes,report)in enumerate(pool.map(prepare,jobs,chunksize=4),1):
            reports.append(report)
            if lanes:
                groups.append([[l['id']]for l in lanes]);replacements.update({l['id']:l for l in lanes})
            if count%100==0:print(dict(prepared=count,total=len(jobs)),flush=True)
    output=dict(scene);output['fusion_map']=dict(scene['fusion_map'])
    output['fusion_map']['lanes']=[replacements.get(l['id'],l)for l in scene['fusion_map']['lanes']]
    summary=dict(source_groups=len(reports),prepared_groups=len(groups),prepared_lane_pieces=len(replacements),
        pending_groups=sum(not r['passed']for r in reports),
        pending_reasons=dict(Counter(r['reason'].split(':')[0]for r in reports if not r['passed'])),
        scope='Coarse boundary preparation only; actual native compilation, source continuity and physical driving follow. Pending roads remain in the source map.')
    a.output.mkdir(parents=True,exist_ok=True)
    for name,value in [('scene.json',output),('groups.json',dict(groups=groups)),('review.json',dict(summary=summary,groups=reports))]:
        (a.output/name).write_text(json.dumps(value,separators=(',',':'))+'\n')
    print(summary,flush=True)


if __name__=='__main__':main()
