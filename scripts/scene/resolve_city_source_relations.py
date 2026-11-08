"""Realize reviewed source relations and duplicate path observations.

Geometry is selected by explicit Agent conditions. Direct successor aliases keep
their existing complete path. A duplicate whole lane may map to an existing
multi-piece path with distinct, explicitly identified intermediate branch ports.
"""
import argparse,json
from copy import deepcopy
from pathlib import Path

def resolve(scene,conditions):
    before={l['id']:l for l in scene['fusion_map']['lanes']}
    selected={};records=[]
    for row in conditions.get('decisions',[]):
        if row['decision']!='realize_direct_relation_through_source_path':continue
        a,b=row['from_lane'],row['to_lane'];path=row['source_path']
        if b not in before[a]['successors'] or path[0]!=a or path[-1]!=b or len(path)<3:
            raise ValueError('Reviewed relation does not match current source graph: '+str((a,b)))
        for x,y in zip(path[:-1],path[1:]):
            if y not in before[x]['successors']:raise ValueError('Missing original alternate source relation: '+str((x,y)))
        required=set(before[a]['allowed_classes'])&set(before[b]['allowed_classes'])
        available=set.intersection(*(set(before[k]['allowed_classes'])for k in path))
        if not required<=available:raise ValueError('Alternate source path loses permitted modes: '+str((a,b)))
        selected[a,b]=list(path)
    def expand(a,b,active=()):
        if(a,b)not in selected:return[a,b]
        if(a,b)in active:raise ValueError('Circular relation realization')
        path=selected[a,b];result=[a]
        for x,y in zip(path[:-1],path[1:]):result.extend(expand(x,y,(*active,(a,b)))[1:])
        return result
    out=deepcopy(scene);lanes={l['id']:l for l in out['fusion_map']['lanes']}
    for a,b in selected:lanes[a]['successors'].remove(b)
    for a,b in selected:
        trace=expand(a,b)
        for x,y in zip(trace[:-1],trace[1:]):
            if y not in lanes[x]['successors']:raise ValueError('Final source path is incomplete')
        records.append(dict(source=a,target=b,realized_path=trace,
            preserved_permissions=sorted(set(before[a]['allowed_classes'])&set(before[b]['allowed_classes']))))
    aliases=[];neighbor_changes={}
    for row in conditions.get('path_aliases',[]):
        alias=row['source_lane'];path=row['canonical_path'];branch=row['outgoing_source_ports']
        source=lanes[alias]
        if alias in path or len(path)<2:raise ValueError('Invalid reviewed canonical path')
        if any(alias in r['realized_path']for r in records):
            raise ValueError('Resolve interacting direct and whole-path aliases in separate reviewed passes')
        for x,y in zip(path[:-1],path[1:]):
            if y not in lanes[x]['successors']:raise ValueError('Canonical alias path is not an existing source continuation')
        if not set(source['allowed_classes'])<=set.intersection(*(set(lanes[k]['allowed_classes'])for k in path)):
            raise ValueError('Canonical alias path loses source modes')
        incoming=[k for k,l in lanes.items()if alias in l['successors']]
        for key in incoming:
            if path[0]not in lanes[key]['successors']:
                raise ValueError('Reviewed alias would create a new incoming physical connection')
        if set(branch)!=set(source['successors']):raise ValueError('Every original alias branch needs a reviewed source port')
        branches=[]
        for target,port in branch.items():
            if port not in path or target not in lanes[port]['successors']:
                raise ValueError('Alias branch is not already realized at its reviewed source port')
            trace=path[:path.index(port)+1]+[target]
            required=set(source['allowed_classes'])&set(lanes[target]['allowed_classes'])
            if not required<=set.intersection(*(set(lanes[k]['allowed_classes'])for k in trace)):
                raise ValueError('Alias branch realization loses source modes')
            branches.append(dict(target=target,source_port=port,realized_path=trace,preserved_permissions=sorted(required)))
        actual_neighbors={(k,f)for k,l in lanes.items()for f in ('left_neighbor','right_neighbor')if l.get(f)==alias}
        chosen={(n['lane'],n['field']):n['canonical_lane']for n in row.get('neighbor_references',[])}
        if set(chosen)!=actual_neighbors:raise ValueError('Every neighbor reference needs an explicit canonical source piece')
        for (key,field),canonical in chosen.items():
            if canonical not in path:raise ValueError('Neighbor piece is outside the canonical source path')
            lanes[key][field]=canonical
            neighbor_changes[key,field]=dict(original=alias,canonical=canonical,canonical_path=path)
        for key in incoming:lanes[key]['successors'].remove(alias)
        aliases.append(dict(source_lane=alias,canonical_path=path,source_observation=deepcopy(before[alias]),
            incoming_relations=[dict(source=k,realized_path=[k,*path])for k in incoming],
            outgoing_relations=branches,neighbor_references=row.get('neighbor_references',[]),reason=row['reason']))
        del lanes[alias]
    out['fusion_map']['lanes']=[l for l in out['fusion_map']['lanes']if l['id']in lanes]
    for key,lane in lanes.items():
        original=deepcopy(before[key]);actual=deepcopy(lane)
        original.pop('successors');actual.pop('successors')
        for field in ('left_neighbor','right_neighbor'):
            if(key,field)in neighbor_changes:original[field]=actual[field]
        if original!=actual:raise ValueError('Retained source geometry or properties unexpectedly changed')
    report=dict(source_conditions=conditions,realized_direct_relations=len(records),paths=records,
        path_aliases=aliases,physical_duplicate_lanes=len(aliases),
        neighbor_path_references=[dict(lane=k,field=f,**v)for(k,f),v in neighbor_changes.items()],
        lane_count_before=len(before),lane_count_after=len(lanes),geometry_changes=0,added_physical_connections=0,
        source_successors={k:before[k]['successors']for k in sorted({a for a,b in selected}|{n['source']for r in aliases for n in r['incoming_relations']})},
        scope='Every reviewed original relation has a permitted path through existing source intervals. All source observations are retained; only explicitly reviewed physical duplicates become path aliases. Canonical geometry, modes and distinct turn destinations remain unchanged. Physical realization requires generation and native review.')
    previous=out['fusion_map'].get('source_successor_realization')
    if previous is not None:
        out['fusion_map'].setdefault('source_successor_realization_history',[]).append(deepcopy(previous))
    out['fusion_map']['source_successor_realization']=report
    return out,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','conditions','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();scene,report=resolve(json.loads(a.scene.read_text()),json.loads(a.conditions.read_text()))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(scene,separators=(',',':'),allow_nan=False)+'\n')
    (a.output/'relation_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items()if k not in ['source_conditions','paths','source_successors','path_aliases','neighbor_path_references']},flush=True)

if __name__=='__main__':main()
