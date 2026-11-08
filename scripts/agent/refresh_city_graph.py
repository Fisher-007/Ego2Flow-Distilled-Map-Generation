"""Refresh current road facts while preserving unchanged reviewed city evidence."""
import argparse,json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from shapely.geometry import LineString,box
from shapely.strtree import STRtree
from map_generation.information_graph import enrich,compile_conditions


def refresh(previous,scene,conditions):
    graph=deepcopy(previous);lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    prior_lanes={n['id']for n in graph['nodes']if n['kind']=='source_lane'}
    retired=prior_lanes-set(lanes)
    history=dict(conditions=conditions,retired_nodes=[n for n in graph['nodes']if n['id']in retired],
        retired_edges=[e for e in graph['edges']if e['source']in retired or e['target']in retired],
        retired_decisions=[d for d in graph['decisions']if d['source_id']in retired])
    graph['nodes']=[n for n in graph['nodes']if n['id']not in retired]
    graph['edges']=[e for e in graph['edges']if e['source']not in retired and e['target']not in retired and e['relation']!='successor']
    graph['decisions']=[d for d in graph['decisions']if d['source_id']not in retired]
    graph=enrich(graph,scene,conditions['source_scene'])
    area_ids={'region:'+a['id']for a in scene['areas']}
    graph['edges']=[e for e in graph['edges']if not(e['source']in area_ids and e['relation']=='depends_on'and e['target']in lanes)]
    keys=list(lanes);tree=STRtree([LineString(np.asarray(lanes[k]['centerline'])[:,:2])for k in keys])
    for area in scene['areas']:
        for i in tree.query(box(*area['influence_bbox_local_xy']),predicate='intersects'):
            graph['edges'].append(dict(source='region:'+area['id'],relation='depends_on',target=keys[int(i)],
                scope='Current road inside the declared Area influence region'))
    task='task:city_source_revision'
    if not any(n['id']==task for n in graph['nodes']):
        graph['nodes'].append(dict(id=task,kind='simulation_task'))
    for i,path in enumerate(conditions['source_reviews']):
        key=f'source_revision:{conditions["revision"]}:{i}'
        graph['nodes'].append(dict(id=key,kind='reviewed_source_revision',source=path,
            agent=conditions['agent'],reason=conditions['reason'],
            task_status={task:dict(missing=['native_continuity_and_geometry'])}))
        graph['edges'].append(dict(source=task,relation='depends_on',target=key))
        graph['decisions'].append(dict(source_id=key,task=task,choice='retain',agent=conditions['agent'],
            reason=conditions['reason'],required_properties=['preserved_source_observations','reviewed_source_ownership','native_continuity_and_geometry'],evidence_ids=[key]))
    for observation in scene['fusion_map'].get('observed_source_interval_aliases',[]):
        key='interval_observation:'+observation['alias']
        node=dict(id=key,kind='source_interval_observation',content=observation,
            interpretation='Preserved observation on an existing canonical physical road, not another lane.')
        existing=next((n for n in graph['nodes']if n['id']==key),None)
        if existing is None:graph['nodes'].append(node)
        else:existing.update(node)
        if 'canonical_path'in observation:
            for index,target in enumerate(observation['canonical_path']):
                edge=dict(source=key,relation='observed_on_source_path',target=target,
                    path_index=index,canonical_path_stations_m=observation['canonical_stations_m'],
                    status='reviewed',scope='Stations refer to the archived complete canonical path at ownership review, not a later split lane interval.')
                if edge not in graph['edges']:graph['edges'].append(edge)
        else:
            edge=dict(source=key,relation='observed_on_interval',target=observation['canonical'],
                interval_m=observation['canonical_interval_m'],status='reviewed')
            if edge not in graph['edges']:graph['edges'].append(edge)
    graph.setdefault('source_revision_history',[]).append(history)
    graph.update(revision=conditions['revision'],previous_graph=conditions['previous_graph'],
        refresh_reason=conditions['reason'],generation_regions=conditions['generation_regions'])
    summary=dict(current_source_lanes=len(lanes),retired_physical_source_nodes=len(retired),
        preserved_interval_observations=len(scene['fusion_map'].get('observed_source_interval_aliases',[])),
        node_kinds=dict(Counter(n['kind']for n in graph['nodes'])),nodes=len(graph['nodes']),edges=len(graph['edges']))
    return graph,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','previous','conditions','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();scene=json.loads(a.scene.read_text());conditions=json.loads(a.conditions.read_text())
    graph,summary=refresh(json.loads(a.previous.read_text()),scene,conditions)
    compiled=compile_conditions(graph,scene)
    a.output.mkdir(parents=True,exist_ok=True)
    for filename,value in [('graph.json',graph),('conditions.json',compiled),('summary.json',summary)]:
        (a.output/filename).write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    print(summary,flush=True)

if __name__=='__main__':main()
