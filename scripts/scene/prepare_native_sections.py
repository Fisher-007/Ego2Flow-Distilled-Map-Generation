"""Compile independent lane sections in parallel and retain native feedback.

The resulting section cache is consumed by pack_lane_groups; junction tails and
whole-network directed topology are checked during that subsequent assembly.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from scenariogeneration import xodr
from map_generation.lane_chains import concatenate_lanes
from map_generation.lane_sections import compile_section,common_source_interval
from map_generation.road_builder import fit_lane,arc_station_to_chord
from scripts.scene.pack_lane_groups import review_native


def compile_one(job):
    chains,lanes,spans,rid,folder,step,*options=job;output=Path(folder);keys=[k for chain in chains for k in chain]
    row=dict(road_id=rid,source_lane_ids=chains)
    try:
        sources=[concatenate_lanes(chain,lanes)for chain in chains];limits=[]
        for i,(source,chain)in enumerate(zip(sources,chains)):
            source['left_neighbor']=sources[i-1]['id']if i else None
            source['right_neighbor']=sources[i+1]['id']if i+1<len(sources)else None
            low=next(s for s in spans[chain[0]]if s['junction_id']is None)['source_s_start_m']
            high=next(s for s in spans[chain[-1]]if s['junction_id']is None)['source_s_end_m']
            first=fit_lane(lanes[chain[0]],1.)[0];last=fit_lane(lanes[chain[-1]],1.)[0]
            limits.append([arc_station_to_chord(lanes[chain[0]],first,low),source['source_chain'][-1]['chord_start_m']+arc_station_to_chord(lanes[chain[-1]],last,high)])
        interval=common_source_interval(sources,limits,requested_interval=options[0] if options else None)
        road,section=compile_section(sources,road_id=rid,interval=interval,center_mode='boundaries',spacing=step)
        section.update(source_lane_records=sources,native_limits_m=limits)
        network=xodr.OpenDrive('native_section');network.add_road(road);path=output/(str(rid)+'.xodr');network.write_xml(str(path))
        probe=dict(map_id='native_section',fusion_map=dict(lanes=[dict(lanes[k],successors=[])for k in keys]))
        report=review_native(probe,dict(grouped_sections=[section],lane_to_road_intervals={}),path)
        ET.ElementTree(road.get_element()).write(output/(str(rid)+'.xml'),encoding='utf-8',xml_declaration=True)
        (output/(str(rid)+'.json')).write_text(json.dumps(section,separators=(',',':'))+'\n')
        (output/(str(rid)+'.review.json')).write_text(json.dumps(report,indent=2)+'\n')
        row.update(passed=report['passed'],native_lane_count=len(report['lateral']),
            max_boundary_error_m=max(r['max_boundary_xy_error_m']for r in report['lateral']),
            max_center_encoding_error_m=max(r['max_compiled_center_xy_error_m']for r in report['lateral']),
            missing_neighbors=sum(len(r['missing_neighbors'])for r in report['lateral']),
            permission_errors=sum(len(r['permission_errors'])for r in report['lateral']),
            outside_centers=sum(len(r['native_centers_outside_source_polygon'])for r in report['lateral']))
        if not row['passed']:row['reason']='Native section geometry or permissions need refinement'
    except ValueError as error:row.update(passed=False,reason=str(error))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','roads','groups','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--intervals',type=Path,help='Agent-selected source-chain intervals; omitted groups keep their complete common cap interval')
    p.add_argument('--workers',type=int,default=1);p.add_argument('--sample-step',type=float,default=.03)
    a=p.parse_args();scene=json.loads(a.scene.read_text());lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    mapping=json.loads((a.roads/'road_lane_ids.json').read_text());groups=json.loads(a.groups.read_text())['groups']
    selected=json.loads(a.intervals.read_text())['sections'] if a.intervals else []
    intervals={tuple(map(tuple,row['source_lane_ids'])):row['reference_interval_m'] for row in selected}
    a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'sections';cache.mkdir(exist_ok=True)
    jobs=[]
    for supplied in groups:
        chains=[[key] if isinstance(key,str) else key for key in supplied]
        keys=[k for chain in chains for k in chain];rid=min(mapping['key_to_road_id'][k]for k in keys)
        jobs.append((chains,{k:lanes[k]for k in keys},{k:mapping['lane_to_road_intervals'][k]for k in keys},rid,str(cache),a.sample_step,intervals.get(tuple(map(tuple,chains)))))
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers)as pool:
        for i,row in enumerate(pool.map(compile_one,jobs,chunksize=1),1):
            rows.append(row)
            if i%50==0 or not row['passed']:print(dict(completed=i,total=len(jobs),latest=row),flush=True)
    summary=dict(total=len(rows),passed=sum(r['passed']for r in rows),failures=dict(Counter(r['reason']for r in rows if not r['passed'])))
    (a.output/'review.json').write_text(json.dumps(dict(summary=summary,groups=rows),indent=2)+'\n')
    (a.output/'passed_groups.json').write_text(json.dumps(dict(groups=[r['source_lane_ids']for r in rows if r['passed']]),indent=2)+'\n')
    print(summary,flush=True)


if __name__=='__main__':main()
