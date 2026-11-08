"""Propose whole-lane source ownership from reviewed coherent source frames."""
import argparse
import json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from shapely.geometry import LineString,Point
from map_generation.city_registration import register_sources
from map_generation.city_sources import combine_exact_observations
from scripts.data.propose_city_source_alignment import boundary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','data-root','registration','pairs','conditions','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text());condition=json.loads(a.conditions.read_text())
    registration=json.loads(a.registration.read_text());regs=registration['proposed_registrations']
    sources=[]
    for rel in config['sources']['av2_logs']:
        path=next((a.data_root/rel/'map').glob('log_map_archive*.json'))
        sources.append((str(path.relative_to(a.data_root)),json.loads(path.read_text())))
    registered,_=register_sources(sources,dict(registrations=regs))
    lanes={};logs=defaultdict(set)
    for path,data in registered:
        log=Path(path).parts[-3]
        for key,lane in data['lane_segments'].items():
            lanes[int(key)]=lane;logs[int(key)].add(log)
    anchors={}
    for area in config['areas']:
        log=next(row['log_id'] for row in area['source_observations'] if any(Path(x).name==row['log_id']for x in config['sources']['av2_logs']))
        anchors.setdefault(log,[]).append(np.asarray(area['center_city_xy']))
    parent={k:k for k in lanes};members={k:{k}for k in lanes};seen_logs={k:set(v)for k,v in logs.items()}
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    measures=[];seen=set()
    for case in json.loads(a.pairs.read_text())['cases']:
        for pair in case['pairs']:
            x,y=pair['source_lane_id'],pair['reference_lane_id']
            if x==y or tuple(sorted([x,y]))in seen:continue
            seen.add(tuple(sorted([x,y])))
            first,second=lanes[x],lanes[y];left=[boundary(first,s)for s in ('left_lane_boundary','right_lane_boundary')];right=[boundary(second,s)for s in ('left_lane_boundary','right_lane_boundary')]
            ends=max(np.linalg.norm(u[[0,-1],:2]-v[[0,-1],:2],axis=1).max()for u,v in zip(left,right))
            shape=max(LineString(u[:,:2]).hausdorff_distance(LineString(v[:,:2]))for u,v in zip(left,right))
            dz=[]
            for u,v in [*zip(left,right),*zip(right,left)]:
                line=LineString(v[:,:2]);station=np.r_[0.,np.linalg.norm(np.diff(v[:,:2],axis=0),axis=1).cumsum()]
                dz.extend(abs(p[2]-np.interp(line.project(Point(p[:2])),station,v[:,2]))for p in u)
            da=left[0][-1,:2]-left[0][0,:2];db=right[0][-1,:2]-right[0][0,:2]
            direction=float(da@db/max(np.linalg.norm(da)*np.linalg.norm(db),1e-8))
            measured=dict(first=x,second=y,endpoint_xy_max_m=float(ends),boundary_hausdorff_xy_m=float(shape),height_p95_abs_difference_m=float(np.percentile(dz,95)),direction_cosine=direction,first_logs=sorted(logs[x]),second_logs=sorted(logs[y]),
                          marking_differences={k:[first[k],second[k]]for k in ('left_lane_mark_type','right_lane_mark_type')if first[k]!=second[k]})
            measured['eligible']=bool(ends<=condition['maximum_endpoint_difference_m'] and shape<=condition['maximum_boundary_difference_m'] and measured['height_p95_abs_difference_m']<=condition['maximum_height_difference_m'] and direction>=condition['minimum_heading_cosine'])
            measures.append(measured)
    accepted=[];ambiguous=[]
    for pair in sorted(measures,key=lambda r:(r['boundary_hausdorff_xy_m'],r['height_p95_abs_difference_m'])):
        if not pair['eligible']:continue
        x,y=root(pair['first']),root(pair['second'])
        if x==y:continue
        if seen_logs[x]&seen_logs[y]:
            ambiguous.append(dict(pair=pair,reason='Would merge distinct co-occurring lanes from one source log'));continue
        parent[y]=x;members[x]|=members[y];seen_logs[x]|=seen_logs[y];accepted.append(pair)
    components=defaultdict(list)
    for key in lanes:components[root(key)].append(key)
    aliases={};ownership=[]
    for component in components.values():
        if len(component)==1:continue
        def score(key):
            lane=lanes[key];center=np.concatenate([boundary(lane,s)for s in ('left_lane_boundary','right_lane_boundary')])[:,:2].mean(0)
            distances = [float(np.linalg.norm(center-c)) for log in logs[key] for c in anchors.get(log, [])]
            return min(distances) if distances else None
        owner=min(component,key=lambda key:(score(key) is None, score(key) or 0., key))
        aliases.update({str(k):owner for k in component if k!=owner})
        ownership.append(dict(owner=owner,members=component,owner_logs=sorted(logs[owner]),distance_to_selected_area_m=score(owner),
                              requires_context_owner_review=score(owner) is None,
                              reason=('Within a reviewed whole-lane equivalence component, keep the source nearest its own selected Area; retain exact chosen geometry and all alternate source observations.'
                                      if score(owner) is not None else
                                      'No member has a selected Area anchor. Stable source ID proposes an owner only; Agent context-source review is required. All observed alternatives remain available.')))
    assembly=combine_exact_observations(registered,geometry_aliases=aliases)
    report=dict(proposed_registrations=regs,proposed_geometry_aliases=aliases,ownership_components=ownership,
                pair_measurements=measures,accepted_equivalence_edges=accepted,ambiguous_equivalence_edges=ambiguous,
                assembly_preview={k:v for k,v in assembly.items()if k!='lanes'},preview_lane_count=len(assembly['lanes']),
                conditions=str(a.conditions),registration_source=str(a.registration),
                applied=False,scope='Whole-lane ownership proposal only. Partial overlaps and noncongruent or epoch-dependent roads remain unresolved; no source map, topology or final SceneIR is written.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(pair_count=len(measures),eligible=sum(r['eligible']for r in measures),aliases=len(aliases),
                         ambiguous=len(ambiguous),preview_lanes=len(assembly['lanes']),
                         same_id_conflicts=assembly['source_id_geometry_conflicts'],
                         neighbor_conflicts=assembly['neighbor_conflicts'],
                         owners=dict(Counter(log for r in ownership for log in r['owner_logs']))),indent=2))


if __name__=='__main__':
    main()
