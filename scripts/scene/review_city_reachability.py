"""Measure passenger road coverage and Area reachability on the constructed graph."""
import argparse
import json
from pathlib import Path
import networkx as nx
import numpy as np
from shapely.geometry import LineString,box
from shapely.strtree import STRtree


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();scene=json.loads(a.scene.read_text())
    lanes={l['id']:l for l in scene['fusion_map']['lanes'] if 'passenger' in l['allowed_classes']}
    shapes={key:LineString(np.asarray(l['centerline'])[:,:2]) for key,l in lanes.items()}
    graph=nx.DiGraph();graph.add_nodes_from(lanes)
    graph.add_edges_from((k,t) for k,l in lanes.items() for t in l['successors'] if t in lanes)
    components=list(nx.strongly_connected_components(graph));component_of={k:i for i,c in enumerate(components) for k in c}
    ranked=sorted(range(len(components)),key=lambda i:sum(shapes[k].length for k in components[i]),reverse=True)
    lengths=[sum(shapes[k].length for k in components[i]) for i in ranked]
    keys=list(lanes);tree=STRtree([shapes[k] for k in keys]);area_components={};areas=[]
    for area in scene['areas']:
        region=box(*area['bbox_local_xy'])
        candidates=[keys[i] for i in tree.query(region,predicate='intersects')]
        observed=[k for k in candidates if lanes[k]['source']=='av2']
        selected=observed or candidates;area_components[area['id']]={component_of[k] for k in selected}
        areas.append(dict(area_id=area['id'],observed_passenger_lanes=len(observed),
            passenger_lanes=len(candidates),in_largest_strong_component=sum(component_of[k]==ranked[0] for k in selected)))
    condensed=nx.condensation(graph,components)
    descendants={i:nx.descendants(condensed,i)|{i} for group in area_components.values() for i in group}
    pairs=[]
    for source,start in area_components.items():
        reach=set().union(*(descendants[i] for i in start))
        for target,end in area_components.items():
            if source!=target:pairs.append(dict(source=source,target=target,reachable=bool(reach&end)))
    result=dict(passenger_lane_count=len(lanes),passenger_directed_length_m=sum(lengths),
        strong_components=len(components),largest_components=[dict(lanes=len(components[i]),length_m=lengths[n]) for n,i in enumerate(ranked[:10])],
        areas=areas,area_pairs=pairs,unreachable_area_pairs=[r for r in pairs if not r['reachable']],
        scope='Directed source-successor reachability for at least one passenger lane per Area. Lateral changes, native navigation, controls and driving require separate checks.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(dict(passenger_lanes=len(lanes),strong_components=len(components),
        largest_component_length_fraction=lengths[0]/sum(lengths),unreachable_area_pairs=len(result['unreachable_area_pairs'])),flush=True)


if __name__=='__main__':main()
