"""Select exact complete observed source paths while retaining canonical topology."""
import argparse,json
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import numpy as np
from shapely.geometry import LineString
from av2.geometry.interpolate import compute_midpoint_line
from map_generation.city_registration import register_sources
from scripts.data.propose_city_source_alignment import boundary
from scripts.data.select_city_source_owners import solve_exact_owners

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('scene','config','data-root','review','conditions','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();s=json.loads(a.scene.read_text());cfg=json.loads(a.config.read_text());review=json.loads(a.review.read_text());condition=json.loads(a.conditions.read_text())
    sources=[]
    for rel in cfg['sources']['av2_logs']:
        path=next((a.data_root/rel/'map').glob('log_map_archive*.json'))
        sources.append((str(path.relative_to(a.data_root)),json.loads(path.read_text())))
    sources,_=register_sources(sources,review)
    observed={int(k):v for _,d in sources for k,v in d['lane_segments'].items()}
    aliases={int(k):v for k,v in review['ownership_review']['source_aliases'].items()}
    def root(k):
        while k in aliases:k=aliases[k]
        return k
    lanes={x['id']:x for x in s['lanes']+s.get('av2_context',[])}
    options={k:[k]for k in lanes};geometry={k:deepcopy(v)for k,v in lanes.items()};rejected=[];seen=set()
    for row in condition['alternatives']:
        ids=row['source_path'];roots={root(k)for k in ids}
        if len(roots)!=1:raise ValueError('Path is not an already accepted complete physical identity: '+str(row))
        owner='av2:'+str(roots.pop())
        if owner not in lanes:continue
        if (owner,tuple(ids))in seen:continue
        seen.add((owner,tuple(ids)))
        raw=[observed[k]for k in ids];reference=lanes[owner]
        if any(v['lane_type']!=reference['lane_type']for v in raw):raise ValueError('Complete source alternative changes travel mode')
        shift=boundary(observed[int(owner.split(':')[1])],'left_lane_boundary')[0]-np.asarray(reference['left_boundary'])[0]
        replacement=deepcopy(reference)
        valid=True
        for source_field,field in [('left_lane_boundary','left_boundary'),('right_lane_boundary','right_boundary')]:
            chunks=[boundary(v,source_field)for v in raw]
            if any(np.linalg.norm(x[-1]-y[0])>1e-6 for x,y in zip(chunks[:-1],chunks[1:])):
                rejected.append(dict(alternative=row,reason='Observed boundary path is not exactly continuous'));valid=False;break
            replacement[field]=(np.vstack([chunks[0]]+[x[1:]for x in chunks[1:]])-shift).tolist()
        if not valid:continue
        center,width=compute_midpoint_line(np.asarray(replacement['left_boundary']),np.asarray(replacement['right_boundary']))
        replacement['centerline']=center.tolist();replacement['mean_ladder_width_m']=float(width)
        old=np.asarray(reference['centerline']);delta=np.linalg.norm(center[[0,-1],:2]-old[[0,-1],:2],axis=1)
        ratio=LineString(center[:,:2]).length/max(LineString(old[:,:2]).length,1e-8)
        if max(delta)>condition['maximum_whole_endpoint_difference_m'] or not condition['length_ratio_range'][0]<=ratio<=condition['length_ratio_range'][1]:
            rejected.append(dict(alternative=row,reason='Former whole alternative no longer covers the complete canonical movement',endpoint_difference_m=delta.tolist(),length_ratio=ratio));continue
        key=owner+'|'+','.join(map(str,ids));replacement['source_geometry_selection']=dict(observed_source_path=ids,evidence=row['evidence'],reason=condition['reason'])
        replacement['source_geometry_segment_semantics']=[dict(source_lane_id=v['id'],is_intersection=v['is_intersection'],left_mark_type=v['left_lane_mark_type'],right_mark_type=v['right_lane_mark_type'])for v in raw]
        replacement['left_mark_type']=raw[0]['left_lane_mark_type'];replacement['right_mark_type']=raw[0]['right_lane_mark_type']
        options[owner].append(key);geometry[key]=replacement
    choices={k:v for k,v in options.items()if len(v)>1}
    ends={k:np.asarray(v['centerline'])[[0,-1]]for k,v in geometry.items()}
    edges=sorted({(k,n)for k,v in lanes.items()for n in v['successors']if n in lanes and n!=k})
    selected,fit=solve_exact_owners(choices,edges,ends,float(condition['vertical_squared_weight']),float(condition['owner_switch_cost']))
    def measure(selection):
        rows=[]
        for x,y in edges:
            delta=ends[selection.get(x,x)][1]-ends[selection.get(y,y)][0]
            rows.append(dict(source=x,target=y,xy_m=float(np.linalg.norm(delta[:2])),z_m=float(abs(delta[2]))))
        return dict(joins=rows,xy_max=max(x['xy_m']for x in rows),z_max=max(x['z_m']for x in rows),xy_p95=float(np.percentile([x['xy_m']for x in rows],95)),z_p95=float(np.percentile([x['z_m']for x in rows],95)),joins_above_30cm_z=sum(x['z_m']>.3 for x in rows))
    report=dict(before=measure({}),after=measure(selected),selection=[dict(canonical=k,selected=v,observed_source_path=geometry[v].get('source_geometry_selection',{}).get('observed_source_path',[]))for k,v in selected.items()if k!=v],
        alternatives=len(choices),rejected_alternatives=rejected,solver=dict(success=bool(fit.success),message=fit.message,gap=float(fit.mip_gap),objective=float(fit.fun)),
        scope='Complete Agent-reviewed original source paths selected with unchanged canonical IDs, successor topology, camera/actor registrations and outer city extent. No lane vertices are warped or averaged.')
    s['source_geometry_owner_original_lanes']=list(lanes.values())
    s['lanes']=[geometry[selected.get(k,k)]for k in lanes];s.pop('av2_context',None);s.pop('fusion_map',None)
    s['source_geometry_owner_review']=dict(conditions=condition,review=report)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'scene.json').write_text(json.dumps(s,separators=(',',':'),allow_nan=False)+'\n');(a.output/'owner_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(alternatives=len(choices),changed_owners=len(report['selection']),rejected=len(rejected),solver=report['solver'],before={k:v for k,v in report['before'].items()if k!='joins'},after={k:v for k,v in report['after'].items()if k!='joins'}),indent=2),flush=True)
if __name__=='__main__':main()
