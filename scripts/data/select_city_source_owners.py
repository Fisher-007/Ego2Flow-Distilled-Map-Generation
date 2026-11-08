"""Choose exact geometry owners among Agent-confirmed whole-movement alternatives."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from map_generation.city_registration import register_sources
from map_generation.city_sources import combine_exact_observations
from scripts.data.propose_city_source_alignment import boundary

def solve_exact_owners(choices,edges,ends,zweight,switch):
    def energy(left,right):
        delta=ends[left][1]-ends[right][0]
        return float(delta[:2]@delta[:2]+zweight*delta[2]**2)
    costs=[];integer=[];variables={}
    for node,keys in choices.items():
        for key in keys:
            variables[node,key]=len(costs);costs.append(0. if key==node else switch);integer.append(1)
    equations=[];rhs=[]
    def eq(values,target=0.):
        equations.append(values);rhs.append(target)
    for node,keys in choices.items():eq({variables[node,key]:1. for key in keys},1.)
    for left,right in edges:
        if left not in choices and right not in choices:continue
        if left not in choices:
            for key in choices[right]:costs[variables[right,key]]+=energy(left,key)
        elif right not in choices:
            for key in choices[left]:costs[variables[left,key]]+=energy(key,right)
        else:
            products={}
            for x in choices[left]:
                for y in choices[right]:
                    products[x,y]=len(costs);costs.append(energy(x,y));integer.append(0)
            for x in choices[left]:eq({**{products[x,y]:1. for y in choices[right]},variables[left,x]:-1.})
            for y in choices[right]:eq({**{products[x,y]:1. for x in choices[left]},variables[right,y]:-1.})
    rr=[];cc=[];vv=[]
    for i,entries in enumerate(equations):
        for j,v in entries.items():rr.append(i);cc.append(j);vv.append(v)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(equations),len(costs))).tocsc()
    fit=milp(np.asarray(costs),integrality=np.asarray(integer),bounds=Bounds(0,1),
             constraints=LinearConstraint(matrix,np.asarray(rhs),np.asarray(rhs)),
             options={'time_limit':120,'mip_rel_gap':1e-8})
    if fit.x is None:raise ValueError('No feasible exact source-owner selection: '+fit.message)
    selected={node:max(keys,key=lambda key:fit.x[variables[node,key]])for node,keys in choices.items()}
    return selected,fit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','data-root','review','conditions','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());base=json.loads(a.review.read_text());condition=json.loads(a.conditions.read_text())
    sources=[]
    for rel in cfg['sources']['av2_logs']:
        path=next((a.data_root/rel/'map').glob('log_map_archive*.json'))
        sources.append((str(path.relative_to(a.data_root)),json.loads(path.read_text())))
    sources,_=register_sources(sources,base)
    lanes={int(k):v for _,d in sources for k,v in d['lane_segments'].items()}
    aliases={int(k):v for k,v in base['ownership_review']['source_aliases'].items()}
    def root(k):
        while k in aliases:k=aliases[k]
        return k
    alternatives=defaultdict(set)
    for row in condition['whole_alternatives']:
        keys=row['members'];roots={root(k)for k in keys}
        if len(roots)!=1:raise ValueError('Alternatives must already be one Agent-reviewed physical movement')
        owner=roots.pop();alternatives[owner].update(keys);alternatives[owner].add(owner)
    choices={k:sorted(v)for k,v in alternatives.items()}
    current=combine_exact_observations(sources,ownership_review=base['ownership_review'])
    edges=sorted({(lane['id'],n)for lane in current['lanes']for n in lane['successors']if n in lanes and n!=lane['id']})
    ends={k:np.stack([boundary(v,s)[[0,-1]]for s in ('left_lane_boundary','right_lane_boundary')]).mean(0)for k,v in lanes.items()}
    zweight=float(condition['vertical_squared_weight']);switch=float(condition['owner_switch_cost'])
    def energy(left,right):
        delta=ends[left][1]-ends[right][0]
        return float(delta[:2]@delta[:2]+zweight*delta[2]**2)
    selected,fit=solve_exact_owners(choices,edges,ends,zweight,switch)
    chosen={k:selected.get(root(k),root(k))for k in lanes}
    ownership=dict(base['ownership_review']);ownership['source_aliases']={str(k):v for k,v in chosen.items()if k!=v};ownership['neighbor_choices']={}
    ownership['owner_selection']=dict(conditions=str(a.conditions),policy=condition['reason'],selection=[dict(previous=k,selected=v,alternatives=choices[k])for k,v in selected.items()])
    preview=combine_exact_observations(sources,ownership_review=ownership)
    for row in preview['neighbor_conflicts']:
        key=row['canonical_lane_id'];field=row['field'];original=lanes[key][field]
        canonical=chosen[original]if original in chosen else original
        if original is not None and canonical not in row['candidate_ids']:continue
        ownership['neighbor_choices'].setdefault(str(key),{})[field]=dict(source_neighbor_id=original,reason='Exact selected geometry owner supplies neighbor identity; retain all alternative observations.')
    result=combine_exact_observations(sources,ownership_review=ownership)
    def measure(ls):
        rows=[dict(source=x['id'],target=n,xy_m=float(np.linalg.norm(ends[x['id']][1,:2]-ends[n][0,:2])),z_m=float(abs(ends[x['id']][1,2]-ends[n][0,2])))for x in ls for n in x['successors']if n in ends]
        return dict(joins=rows,xy_max=max(x['xy_m']for x in rows),z_max=max(x['z_m']for x in rows),xy_p95=float(np.percentile([x['xy_m']for x in rows],95)),z_p95=float(np.percentile([x['z_m']for x in rows],95)))
    review=dict(base,ownership_review=ownership)
    report=dict(before=measure(current['lanes']),after=measure(result['lanes']),
        solver=dict(success=bool(fit.success),status=int(fit.status),message=fit.message,objective=float(fit.fun),gap=float(fit.mip_gap)),
        changed_owners=sum(k!=v for k,v in selected.items()),alternatives=len(choices),lane_count=len(result['lanes']),
        source_id_geometry_conflicts=result['source_id_geometry_conflicts'],neighbor_conflicts=result['neighbor_conflicts'],
        scope='Only Agent-confirmed whole-movement source owners are selected. Exact observed vertices and all source topology are retained; remaining nonzero physical interfaces still need construction treatment.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(review,indent=2)+'\n')
    a.output.with_suffix('.owner_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({**{k:v for k,v in report.items()if k not in ('before','after')},'before':{k:v for k,v in report['before'].items()if k!='joins'},'after':{k:v for k,v in report['after'].items()if k!='joins'}},indent=2))
if __name__=='__main__':main()
