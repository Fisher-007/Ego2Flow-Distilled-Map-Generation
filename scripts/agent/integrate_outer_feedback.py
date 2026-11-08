"""Turn reviewed outer-drive failure evidence into executable graph conditions."""
import argparse,json
from pathlib import Path
import numpy as np
from map_generation.bridge_decks import profiles,weights,shared_road_profiles
from map_generation.information_graph import add_node,add_edge,compile_conditions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['graph','scene','config','failure','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);read=lambda p:json.loads(p.read_text())
    graph,scene,config=map(read,[a.graph,a.scene,a.config])
    # Reuse the bridge profile to avoid compounding approach-height corrections.
    previous=next((n['native_construction']['profiles'] for n in graph['nodes']
                   if n.get('native_construction',{}).get('kind')=='bridge_decks'),None)
    plans=previous if previous is not None else profiles(scene)
    task='task:source_faithful_simulation';failure='evidence:outer_native_drive:'+str(a.failure)
    add_node(graph,{'id':failure,'kind':'native_runtime_evidence','report':str(a.failure),
        'finding':'Reviewed outer driving/visibility feedback; construction choices below distinguish bridge, shared surface and visible context.',
        'surface_finding':config.get('road_surface_feedback',{}).get('finding')})
    key='construction:tagged_bridge_decks'
    mutable=[l['id'] for l in scene['fusion_map']['lanes'] if l['source']=='osm_sumo'
             and any(np.any(weights(np.array(l['centerline'])[:,:2],plan)[0]>0) for plan in plans)]
    choices=[(key,'bridge_surface',{'kind':'bridge_decks','profiles':plans,'mutable_height_lane_ids':mutable},
        ['deck_separate_from_bare_earth','road_compilation_continuity','native_drive_continuity','bridge_edge_support'],
        'Preserve bridge/layer meaning, road XY and legal topology; estimate unsupported deck z from the banks and update all native surfaces including junction closures.'),
       ('construction:route_visible_context','visual_context',{'kind':'route_visible_context','visual_context':config['visual_context']},
        ['route_view_ground_coverage','source_building_coverage','style_continuity','access_clearance','road_relative_outer_curb','final_surface_edge_closure'],
        'Extend information selection to the route viewing area. Retain source building footprints and known accesses; extrapolate only unsupported vegetation/detail using AV2-supported families.')]
    if config.get('road_surface_feedback'):
        selection=config['road_surface_feedback'];shared=shared_road_profiles(scene,selection['source_way_ids'])
        mutable=[l['id'] for l in scene['fusion_map']['lanes'] if l['source']=='osm_sumo'
                 and any(np.any(weights(np.asarray(l['centerline'])[:,:2],plan)[0]>0) for plan in shared)]
        choices.append(('construction:shared_road_surface','road_surface',
            {'kind':'shared_road_surface','profiles':shared,'mutable_height_lane_ids':mutable},
            ['same_level_surface_continuity','native_visibility','native_drive_continuity'],selection['finding']))
    if config['visual_context'].get('frontage'):
        frontage=config['visual_context']['frontage']
        choices.append(('construction:outer_frontage_quality','visual_context',
            {'kind':'outer_frontage_quality','settings':frontage},
            ['observed_frontage_retained','continuous_ground_profile','building_ground_support',
             'source_conditioned_architecture','native_visibility','native_drive_continuity'],frontage['finding']))
        evidence='evidence:outer_architecture:'+frontage['source_style_evidence']
        add_node(graph,{'id':evidence,'kind':'source_image_evidence','report':frontage['source_style_evidence'],
                        'scope':frontage['source_use']})
        add_edge(graph,'construction:outer_frontage_quality','supported_by',evidence,status='AV2_style_review')
    for key,kind,construction,properties,reason in choices:
        add_node(graph,{'id':key,'kind':kind,'native_construction':construction,
            'task_status':{task:{'complete':False,'missing':properties,'review':'New native build required'}}})
        add_edge(graph,key,'supported_by',failure,status='native_feedback')
        add_edge(graph,task,'depends_on',key)
        graph['decisions'].append({'source_id':key,'choice':'enhance','task':task,'required_properties':properties,
            'evidence_ids':[failure],'reason':reason,'agent':'construction_agent; reviewed source tags, geometry and previous runtime evidence'})
    for plan in plans:
        for lane in plan['lane_ids']:add_edge(graph,'construction:tagged_bridge_decks','separates_deck_from_ground',lane,status='OSM_bridge_layer_supported')
    for n in graph['nodes']:
        if n.get('kind')=='observed_access':add_edge(graph,'construction:route_visible_context','preserves_access_clearance',n['id'],status='source_supported')
    (a.output/'graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    (a.output/'conditions.json').write_text(json.dumps(compile_conditions(graph,scene,task),indent=2)+'\n')
    print({'bridge_profiles':len(plans),'mutable_coarse_heights':len(mutable),'scope':'Graph decisions and executable conditions for native review.'})


if __name__=='__main__':main()
