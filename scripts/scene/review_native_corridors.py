"""Identify continuous source lane chains needed by the city multi-lane compiler."""
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import networkx as nx
import numpy as np


def compilation_neighbors(scene):
    """Collect all source-observed lateral neighbors, including compiler subdivisions."""
    from scripts.scene.review_native_lateral import source_candidates
    candidates={}
    for fraction in [.25,.5,.75]:
        for row in source_candidates(scene,minimum_length=.01,neighbor_end_margin=1e-5,
                                     dashed_only=False,query_fraction=fraction):
            candidates[row['source_lane'],row['source_neighbor'],row['side']]=row
    return dict(candidates=list(candidates.values()))


def review(scene,mapping,lateral):
    lanes={l['id']:l for l in scene['fusion_map']['lanes']};predecessors=defaultdict(list)
    junction_ends={};bodies=set()
    for group in mapping['physical_junctions']:
        bodies.update(group['source_lane_ids'])
        for key in group['incoming_lane_ids']:junction_ends[key,'end']=group['id']
        for key in group['outgoing_lane_ids']:junction_ends[key,'start']=group['id']
    for key,lane in lanes.items():
        for target in lane['successors']:
            if target in lanes:predecessors[target].append(key)
    fine={k:l for k,l in lanes.items() if l['source']=='av2' and not l.get('is_intersection') and l.get('lane_type')=='VEHICLE' and k not in bodies}
    graph=nx.DiGraph();graph.add_nodes_from(fine)
    for key,lane in fine.items():
        if len(lane['successors'])!=1:continue
        target=lane['successors'][0]
        if target not in fine or len(predecessors[target])!=1:continue
        if (key,'end') in junction_ends or (target,'start') in junction_ends:continue
        other=fine[target]
        if set(lane['allowed_classes'])!=set(other['allowed_classes']):continue
        if np.linalg.norm(np.asarray(lane['centerline'][-1])-other['centerline'][0])>.02:continue
        graph.add_edge(key,target)
    chains=[];membership={}
    for component in sorted([sorted(c) for c in nx.weakly_connected_components(graph)]):
        sub=graph.subgraph(component)
        if not nx.is_directed_acyclic_graph(sub):raise ValueError('Closed source chain needs an explicit cyclic reference: '+str(component))
        keys=list(nx.topological_sort(sub));index=len(chains)
        chains.append(dict(id=index,source_lane_ids=keys,length_m=sum(float(np.linalg.norm(np.diff(np.asarray(fine[k]['centerline'])[:,:2],axis=0),axis=1).sum()) for k in keys)))
        for key in keys:membership[key]=index
    for chain in chains:
        ids=chain['source_lane_ids'];chain['start_junction']=junction_ends.get((ids[0],'start'));chain['end_junction']=junction_ends.get((ids[-1],'end'))
        chain['predecessor_source_ids']=predecessors[ids[0]]
        chain['successor_source_ids']=lanes[ids[-1]]['successors']
    neighbors=nx.DiGraph()
    for candidate in lateral['candidates']:
        a,b=candidate['source_lane'],candidate['source_neighbor']
        if a not in membership or b not in membership:continue
        left,right=(a,b) if candidate['side']=='right' else (b,a)
        if membership[left]==membership[right]:raise ValueError('A source chain also appears as its lateral neighbor')
        neighbors.add_edge(membership[left],membership[right])
    groups=[]
    for component in sorted([sorted(c) for c in nx.weakly_connected_components(neighbors)]):
        sub=neighbors.subgraph(component);acyclic=nx.is_directed_acyclic_graph(sub)
        ordered=list(nx.topological_sort(sub)) if acyclic else component
        straight_order=acyclic and all(sub.in_degree(n)<=1 and sub.out_degree(n)<=1 for n in ordered)
        group=[chains[n] for n in ordered];starts={g['start_junction'] for g in group};ends={g['end_junction'] for g in group}
        groups.append(dict(id=len(groups),chain_ids=ordered,source_lane_ids=[g['source_lane_ids'] for g in group],
            single_lateral_order=straight_order,common_start_junction=len(starts)==1 and None not in starts,
            common_end_junction=len(ends)==1 and None not in ends,
            start_junctions=[g['start_junction'] for g in group],end_junctions=[g['end_junction'] for g in group],
            lengths_m=[g['length_m'] for g in group]))
    covered={key for g in groups for chain in g['source_lane_ids'] for key in chain}
    return dict(source_external_vehicle_lanes=len(fine),chain_count=len(chains),group_count=len(groups),
        single_lateral_order_groups=sum(g['single_lateral_order'] for g in groups),
        aligned_junction_groups=sum(g['single_lateral_order'] and g['common_start_junction'] and g['common_end_junction'] for g in groups),
        candidate_checks_covered=sum(r['source_lane'] in covered and r['source_neighbor'] in covered for r in lateral['candidates']),
        grouped_chain_counts=dict(Counter(len(g['chain_ids']) for g in groups)),chains=chains,groups=groups,
        neighbor_chain_edges=[list(edge) for edge in neighbors.edges],
        scope='Source-only corridor organization, not native compilation. Side accesses and differing endpoints remain explicit for common cross-section cutting.')


def resolve_longitudinal_branches(report, scene, mapping, group_ids=None, require_common_endpoints=True):
    """Select compatible cross sections across lateral source branches.

    Rank lateral paths by common source length and lane count. Set
    require_common_endpoints=False for packing with endpoint-container
    normalization; the packer checks complete topology and retained terminals.
    """
    from map_generation.lane_chains import concatenate_lanes
    from map_generation.lane_sections import common_source_interval
    from map_generation.road_builder import fit_lane, arc_station_to_chord
    lanes={lane['id']:lane for lane in scene['fusion_map']['lanes']}
    chains={chain['id']:chain for chain in report['chains']}
    graph=nx.DiGraph(report['neighbor_chain_edges']);results=[];selected=[]
    for group in report['groups']:
        if group['single_lateral_order'] or (group_ids is not None and group['id'] not in group_ids):continue
        sub=graph.subgraph(group['chain_ids']);row=dict(group_id=group['id'],candidates=[])
        if not nx.is_directed_acyclic_graph(sub):
            row.update(reason='Cyclic source side relations require evidence review',selected=[])
            results.append(row);continue
        for start in (n for n in sub if sub.in_degree(n)==0):
            for end in (n for n in sub if sub.out_degree(n)==0):
                for path in nx.all_simple_paths(sub,start,end):
                    records=[chains[i] for i in path]
                    candidate=dict(chain_ids=path,source_lane_ids=[r['source_lane_ids'] for r in records])
                    try:
                        if require_common_endpoints and (len({r['start_junction'] for r in records})!=1 or len({r['end_junction'] for r in records})!=1):
                            raise ValueError('Source endpoints require further longitudinal subdivision')
                        sources=[];limits=[]
                        for keys in candidate['source_lane_ids']:
                            source=concatenate_lanes(keys,lanes);sources.append(source)
                            low=next(s for s in mapping['lane_to_road_intervals'][keys[0]] if s['junction_id'] is None)['source_s_start_m']
                            high=next(s for s in mapping['lane_to_road_intervals'][keys[-1]] if s['junction_id'] is None)['source_s_end_m']
                            first=fit_lane(lanes[keys[0]],1.)[0];last=fit_lane(lanes[keys[-1]],1.)[0]
                            limits.append([arc_station_to_chord(lanes[keys[0]],first,low),source['source_chain'][-1]['chord_start_m']+arc_station_to_chord(lanes[keys[-1]],last,high)])
                        interval=common_source_interval(sources,limits)
                        candidate.update(common_interval_m=list(interval),lane_length_m=(interval[1]-interval[0])*len(path))
                    except ValueError as error:candidate['reason']=str(error)
                    row['candidates'].append(candidate)
        claimed=set();chosen=[]
        for candidate in sorted((c for c in row['candidates'] if 'lane_length_m' in c),key=lambda c:(-c['lane_length_m'],c['chain_ids'])):
            if claimed.intersection(candidate['chain_ids']):continue
            claimed.update(candidate['chain_ids']);chosen.append(candidate['chain_ids'])
            selected.append(candidate['source_lane_ids'])
        row.update(selected=chosen,unpacked_chain_ids=sorted(set(group['chain_ids'])-claimed))
        results.append(row)
    return dict(groups=selected,decisions=results,scope='Source-supported common cross sections; omitted chains retain original roads. Native compilation and whole-network review remain required.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','roads','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--lateral',type=Path,help='Optional supplied candidate report; the default compiles short and solid-marking observed neighbors too')
    a=p.parse_args();scene=json.loads(a.scene.read_text())
    result=review(scene,json.loads((a.roads/'road_lane_ids.json').read_text()),
                  json.loads(a.lateral.read_text())if a.lateral else compilation_neighbors(scene))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ['chains','groups']},flush=True)


if __name__=='__main__':main()
