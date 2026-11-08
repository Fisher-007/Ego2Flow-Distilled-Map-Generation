"""Restore explicit Agent-reviewed coarse source intervals and branch paths.

Only missing intervals of the original directed source network are inserted.
Shared intervals have one owner; existing observations and inferred access
connections are retained. Source-only paths keep their original permissions.
"""
import argparse,json
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from map_generation.access_fusion import section


def restore(scene,conditions):
    out=deepcopy(scene);fused=out['fusion_map'];lanes={l['id']:l for l in fused['lanes']}
    originals={l['id']:l for l in out['coarse_map']['lanes']}
    lengths={k:LineString(np.asarray(l['centerline'])[:,:2]).length for k,l in originals.items()}
    parts=defaultdict(list);requested=defaultdict(list);normalized=[]
    for k,l in lanes.items():
        if l.get('source')=='osm_sumo':parts[l['source_lane_id']].append((l['source_interval_m'][0],l['source_interval_m'][1],k))
    for index,row in enumerate(conditions['connections']):
        if row.get('decision','restore_source_path')!='restore_source_path':continue
        path=row['source_path']if row['incoming']else list(reversed(row['source_path']))
        keys=[p[0]for p in path]
        for a,b in zip(keys,keys[1:]):
            if a!=b and b not in originals[a]['successors']:raise ValueError('Not an original directed source edge: '+a+' -> '+b)
        common=set(row['permissions'])
        for k,a,b in path:
            common&=set(originals[k]['allowed_classes'])
            if abs(a-b)>.001:requested[k].append((min(a,b),max(a,b)))
        if common!=set(row['permissions']):raise ValueError('Requested permissions exceed the original directed path')
        start,end=(row['coarse_lane_id'],row['target'])if row['incoming']else(row['target'],row['coarse_lane_id'])
        normalized.append(dict(index=index,from_lane=start,to_lane=end,path=path,permissions=sorted(common)))
    inserted=[]
    for key,ranges in sorted(requested.items()):
        cuts=sorted({v for a,b in ranges for v in [a,b]}|{v for a,b,k in parts[key]for v in [a,b]})
        for a,b in zip(cuts[:-1],cuts[1:]):
            mid=(a+b)/2
            if b-a<=.001 or not any(lo-1e-6<=mid<=hi+1e-6 for lo,hi in ranges):continue
            if any(lo-1e-6<=mid<=hi+1e-6 for lo,hi,k in parts[key]):continue
            original=originals[key];serial=len(inserted)
            ident='source_continuation:'+key+':'+str(serial)
            while ident in lanes:
                serial+=1;ident='source_continuation:'+key+':'+str(serial)
            lane=deepcopy(original)
            lane.update(id=ident,source='osm_sumo',source_lane_id=key,source_interval_m=[a,b],source_length_m=lengths[key],
                centerline=section(original['centerline'],a,b).tolist(),successors=[],
                source_continuation_status='Agent-reviewed missing original interval; geometry and native realization pending.',
                source_continuation_condition_indices=[n['index']for n in normalized if any(k==key and min(x,y)<=mid<=max(x,y)for k,x,y in n['path'])])
            lanes[ident]=lane;parts[key].append((a,b,ident));inserted.append(ident)
    paths=[];added_edges=[]
    for row in normalized:
        trace=[row['from_lane']]
        for key,a,b in row['path']:
            lo,hi=min(a,b),max(a,b)
            if hi-lo<=.001:continue
            for start,end,ident in sorted(parts[key]):
                if start<hi-1e-6 and end>lo+1e-6 and ident not in [row['from_lane'],row['to_lane']]:
                    if ident!=trace[-1]:trace.append(ident)
        if trace[-1]!=row['to_lane']:trace.append(row['to_lane'])
        for a,b in zip(trace,trace[1:]):
            if b not in lanes[a]['successors']:
                lanes[a]['successors'].append(b);added_edges.append([a,b])
        paths.append(dict(condition_index=row['index'],from_lane=row['from_lane'],to_lane=row['to_lane'],trace=trace,permissions=row['permissions']))
    fused['lanes']=list(lanes.values())
    report=dict(conditions=conditions,inserted_lane_ids=inserted,inserted_intervals=len(inserted),reviewed_paths=len(paths),
        added_edges=added_edges,paths=paths,scope='Original OSM/SUMO directed intervals and permissions restored only where explicitly reviewed; source geometry is not independently observed AV2 and remains subject to constrained realization.')
    if fused.get('source_continuation_review'):report['previous_review']=deepcopy(fused['source_continuation_review'])
    fused['source_continuation_review']=report
    if 'source_access_review'in fused:
        covered={(p['from_lane'],True)for p in paths}|{(p['to_lane'],False)for p in paths}
        old=fused['source_access_review']['unresolved'];fused['source_access_review']['unresolved']=[p for p in old if(p['coarse_lane_id'],p['direction']=='into_fine')not in covered]
        report['previous_unresolved_ports']=len(old);report['remaining_unresolved_ports']=len(fused['source_access_review']['unresolved'])
    return out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['scene','conditions','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();s,r=restore(json.loads(a.scene.read_text()),json.loads(a.conditions.read_text()))
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'scene.json').write_text(json.dumps(s,separators=(',',':'),allow_nan=False)+'\n');(a.output/'continuation_review.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items()if k not in ['conditions','inserted_lane_ids','paths','added_edges']},indent=2),flush=True)
if __name__=='__main__':main()
