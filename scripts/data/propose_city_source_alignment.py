"""Measure registered cross-log lane identities for explicit Agent source review."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree


def boundary(lane,side):
    return np.asarray([[p[k] for k in ('x','y','z')] for p in lane[side]],float)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','data-root','diagnostic','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();conf=json.loads(a.config.read_text());sources={}
    for rel in conf['sources']['av2_logs']:
        file=next((a.data_root/rel/'map').glob('log_map_archive*.json'));sources[Path(rel).name]=json.loads(file.read_text())
    reports=[]
    for reg in json.loads(a.diagnostic.read_text())['cases']:
        first,ref=reg['first_log'],reg['reference_log']
        r=np.asarray(reg['rotation_matrix']);t=np.asarray(reg['translation_city_m'])
        # Measure each supplied pair against its explicit reference recording.
        targets=sources[ref]['lane_segments']
        ids=list(targets);lines=[LineString(boundary(targets[k],'left_lane_boundary')[:,:2]) for k in ids];index=STRtree(lines)
        pairs=[];missing=[];unmatched=[];partial=[]
        for key,lane in sources[first]['lane_segments'].items():
            raw=[boundary(lane,side) for side in ('left_lane_boundary','right_lane_boundary')]
            # Use a finite Z for planar lookup and retain missing source elevation in the report.
            finite=np.concatenate([v[np.isfinite(v[:,2]),2] for v in raw]);guess=float(np.median(finite)) if len(finite) else 0.
            registered=[np.column_stack([v[:,:2],np.where(np.isfinite(v[:,2]),v[:,2],guess)])@r.T+t for v in raw]
            if any(not np.isfinite(v).all() for v in raw):missing.append(int(key))
            line=LineString(registered[0][:,:2]);candidates=[]
            for i in index.query(line.buffer(2.)):
                other=targets[ids[i]]
                if lane['lane_type']!=other['lane_type'] or lane['is_intersection']!=other['is_intersection']:continue
                reference=[boundary(other,side) for side in ('left_lane_boundary','right_lane_boundary')]
                ends=max(float(np.linalg.norm(x[[0,-1],:2]-y[[0,-1],:2],axis=1).max()) for x,y in zip(registered,reference))
                shape=max(LineString(x[:,:2]).hausdorff_distance(LineString(y[:,:2])) for x,y in zip(registered,reference))
                covered_source=min(LineString(x[:,:2]).intersection(LineString(y[:,:2]).buffer(1.5)).length/max(LineString(x[:,:2]).length,1e-8) for x,y in zip(registered,reference))
                covered_reference=min(LineString(y[:,:2]).intersection(LineString(x[:,:2]).buffer(1.5)).length/max(LineString(y[:,:2]).length,1e-8) for x,y in zip(registered,reference))
                da=registered[0][-1,:2]-registered[0][0,:2];db=reference[0][-1,:2]-reference[0][0,:2]
                direction=float(da@db/max(np.linalg.norm(da)*np.linalg.norm(db),1e-8))
                if ends<=2. and shape<=2.:
                    candidates.append(dict(source_lane_id=int(key),reference_lane_id=int(ids[i]),endpoint_xy_max_m=ends,boundary_hausdorff_xy_m=shape,
                        source_missing_height=int(key) in missing,reference_logs=[log for log,data in sources.items() if ids[i] in data['lane_segments']]))
                elif max(covered_source,covered_reference)>.6 and direction>.8:
                    partial.append(dict(source_lane_id=int(key),reference_lane_id=int(ids[i]),source_covered_fraction=covered_source,
                        reference_covered_fraction=covered_reference,endpoint_xy_max_m=ends,
                        source_predecessors=lane['predecessors'],source_successors=lane['successors'],
                        reference_predecessors=other['predecessors'],reference_successors=other['successors'],
                        source_length_m=line.length,reference_length_m=lines[i].length,
                        source_start_end=registered[0][[0,-1]].tolist(),reference_start_end=reference[0][[0,-1]].tolist()))
            if candidates:pairs.append(min(candidates,key=lambda v:v['boundary_hausdorff_xy_m']))
            else:unmatched.append(int(key))
        reports.append(dict(registration=reg,pairs=pairs,partial_overlap=partial,unmatched_lane_ids=unmatched,missing_height_lane_ids=missing))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'cases':reports,'applied':False,'scope':'All supplied diagnostic source/reference pairs; no city-specific source IDs or automatic accepted transform.'},indent=2)+'\n')
    for c in reports:print(c['registration']['first_log'][:8],dict(matches=len(c['pairs']),unmatched=len(c['unmatched_lane_ids']),missing_height=c['missing_height_lane_ids']))

if __name__=='__main__':main()
