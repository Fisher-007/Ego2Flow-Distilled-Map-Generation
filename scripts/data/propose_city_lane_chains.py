"""Propose source-equivalent whole lanes versus ordered AV2 segment paths."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from shapely.geometry import LineString,Point
from map_generation.city_registration import register_sources
from scripts.data.propose_city_source_alignment import boundary


def compare(a,b):
    sides=('left_lane_boundary','right_lane_boundary')
    x=[boundary(a,s)for s in sides];y=[boundary(b,s)for s in sides]
    ends=max(np.linalg.norm(u[[0,-1],:2]-v[[0,-1],:2],axis=1).max()for u,v in zip(x,y))
    shape=max(LineString(u[:,:2]).hausdorff_distance(LineString(v[:,:2]))for u,v in zip(x,y))
    return float(ends),float(shape)


def contained(short,long,tolerance,height_tolerance=.3):
    errors=[];heights=[]
    for side in ('left_lane_boundary','right_lane_boundary'):
        x,y=boundary(short,side),boundary(long,side);line=LineString(y[:,:2])
        stations=np.r_[0.,np.linalg.norm(np.diff(y[:,:2],axis=0),axis=1).cumsum()]
        for p in x:
            station=line.project(Point(p[:2]));q=line.interpolate(station)
            errors.append(Point(p[:2]).distance(q));heights.append(abs(p[2]-np.interp(station,stations,y[:,2])))
    return max(errors)<=tolerance and np.percentile(heights,95)<=height_tolerance,max(errors),float(np.percentile(heights,95))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','data-root','registration','pairs','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--maximum-difference',type=float,default=.8)
    p.add_argument('--maximum-height-difference',type=float,default=.3,help='Candidate source-version discrepancy only; Agent must review physical identity.')
    a=p.parse_args();config=json.loads(a.config.read_text());regs=json.loads(a.registration.read_text())['proposed_registrations']
    sources=[]
    for rel in config['sources']['av2_logs']:
        path=next((a.data_root/rel/'map').glob('log_map_archive*.json'));sources.append((str(path.relative_to(a.data_root)),json.loads(path.read_text())))
    sources,_=register_sources(sources,dict(registrations=regs))
    maps={Path(path).parts[-3]:data['lane_segments']for path,data in sources};rows=[];seen=set()
    for case in json.loads(a.pairs.read_text())['cases']:
        first,ref=case['registration']['first_log'],case['registration']['reference_log']
        for long_log,short_log,reverse in [(first,ref,False),(ref,first,True)]:
            candidates=defaultdict(set)
            for c in case['partial_overlap']:
                x,y=(c['reference_lane_id'],c['source_lane_id'])if reverse else(c['source_lane_id'],c['reference_lane_id'])
                long,short=maps[long_log][str(x)],maps[short_log][str(y)]
                xl,yl=boundary(long,'left_lane_boundary'),boundary(short,'left_lane_boundary')
                if LineString(xl[:,:2]).length<LineString(yl[:,:2]).length+1:continue
                good,_,_=contained(short,long,a.maximum_difference,a.maximum_height_difference)
                if good:candidates[x].add(y)
            for ident,possible in candidates.items():
                long=maps[long_log][str(ident)]
                def end_distance(short,at_end):
                    return max(np.linalg.norm(boundary(short,s)[-1 if at_end else 0,:2]-boundary(long,s)[-1 if at_end else 0,:2])for s in ('left_lane_boundary','right_lane_boundary'))
                starts=[k for k in possible if end_distance(maps[short_log][str(k)],False)<=a.maximum_difference]
                for start in starts:
                    pending=[(start,[])];paths=[]
                    while pending:
                        key,path=pending.pop()
                        if key in path:continue
                        path=path+[key];lane=maps[short_log][str(key)]
                        if end_distance(lane,True)<=a.maximum_difference:
                            if len(path)>=2:paths.append(path)
                            continue
                        following=[n for n in lane['successors']if n in possible]
                        # Collapse unbranched paths while retaining intermediate turn and exit ports.
                        internal=[n for n in lane['successors']if str(n)in maps[short_log]]
                        if len(following)==1 and internal==following:
                            target=maps[short_log][str(following[0])]
                            incoming=[n for n in target['predecessors']if str(n)in maps[short_log]]
                            if incoming==[key]:pending.append((following[0],path))
                    for path in paths:
                        signature=(ident,tuple(path))
                        if signature in seen:continue
                        seen.add(signature)
                        assembled=dict(long)
                        for side in ('left_lane_boundary','right_lane_boundary'):
                            assembled[side]=[p for key in path for p in maps[short_log][str(key)][side]]
                        ends,shape=compare(long,assembled)
                        if shape>a.maximum_difference:continue
                        rows.append(dict(long_lane_id=ident,long_log=long_log,source_path=path,path_log=short_log,
                            endpoint_xy_max_m=ends,boundary_hausdorff_xy_m=shape,
                            height_p95_abs_difference_m=max(contained(maps[short_log][str(k)],long,a.maximum_difference,a.maximum_height_difference)[2]for k in path),
                            lane_type=long['lane_type'],is_intersection=long['is_intersection'],
                            source_predecessors=maps[short_log][str(path[0])]['predecessors'],
                            source_successors=maps[short_log][str(path[-1])]['successors'],
                            owner_predecessors=long['predecessors'],owner_successors=long['successors'],
                            rationale='Same directed boundary corridor, matching whole endpoints, and an observed unbranched ordered source path; full known long-lane geometry can represent alternate segmentation without dropping physical road extent.'))
    report=dict(proposals=rows,applied=False,maximum_difference_m=a.maximum_difference,maximum_height_difference_m=a.maximum_height_difference,
                scope='Agent review required. Does not collapse branches, change measured boundary coordinates or accept partial paths that fail to cover both whole-lane endpoints.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
