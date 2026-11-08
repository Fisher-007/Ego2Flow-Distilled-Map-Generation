"""Extend accepted source crowns into explicitly inferred outer roadside context."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import shapely
from scipy.spatial import cKDTree

from map_generation.coordinates import LocalFrame
from map_generation.information_graph import add_node, add_edge, compile_conditions
from scripts.scene.prepare_city_greenery import final_support, nearest_distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','terrain','evidence','selection','graph','osm','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    scene,terrain,evidence,spec,graph=map(read,(a.scene,a.terrain,a.evidence,a.selection,a.graph))
    accepted=deepcopy(evidence['canopy_candidates']);cfg=spec['settings']
    rng=np.random.default_rng(cfg['seed'])
    def polygon(row):
        if 'left_boundary' in row and 'right_boundary' in row:
            return shapely.Polygon(row['left_boundary']+row['right_boundary'][::-1]).buffer(0)
        return shapely.LineString(row['centerline']).buffer(row['width_m']/2)
    road_rows=scene['fusion_map']['lanes']
    roads=shapely.STRtree([polygon(r) for r in road_rows])
    fine=shapely.STRtree([polygon(r) for r in scene['lanes']])
    buildings=shapely.STRtree([shapely.Polygon(b['exterior'],b.get('holes',[])) for b in scene['buildings']])
    junctions=shapely.STRtree([polygon(r) for r in road_rows if r.get('is_intersection',False)])
    frame=LocalFrame(scene['coordinates']['city'],scene['coordinates']['origin_city_xyz'])
    osm=ET.parse(a.osm).getroot()
    nodes={n.get('id'):[float(n.get('lon')),float(n.get('lat'))] for n in osm.findall('node')}
    parking=[]
    for way in osm.findall('way'):
        tags={t.get('k'):t.get('v') for t in way.findall('tag')}
        refs=[n.get('ref') for n in way.findall('nd')]
        if tags.get('amenity')=='parking' and tags.get('parking') not in ('underground','multi-storey') and len(refs)>3 and refs[0]==refs[-1]:
            parking.append(shapely.Polygon(frame.lonlat_to_local([nodes[n] for n in refs])).buffer(0))
    parking_tree=shapely.STRtree(parking)
    vertices=np.asarray(terrain['vertices']);faces=np.asarray(terrain['faces'])
    lo,hi=vertices[:,:2].min(0),vertices[:,:2].max(0)
    step=cfg['lattice_spacing_m']
    xy=np.array(np.meshgrid(np.arange(lo[0]+step,hi[0],step),np.arange(lo[1]+step,hi[1],step))).reshape(2,-1).T
    xy+=rng.uniform(-cfg['jitter_m'],cfg['jitter_m'],xy.shape)
    pts=shapely.points(xy)
    source_xy=np.array([r['position_enu_m'][:2] for r in accepted])
    near=cKDTree(source_xy);source_distance,parent=near.query(xy)
    rd=nearest_distance(roads,pts);bd=nearest_distance(buildings,pts);fd=nearest_distance(fine,pts)
    jd=nearest_distance(junctions,pts)
    radius=np.array([r['crown_radius_m'] for r in accepted])[parent]
    boundary=np.minimum(xy-lo,hi-xy).min(1)
    # Use a boundary-faded roadside probability for inferred scenery.
    probability=cfg['maximum_occupancy']*np.clip((fd-cfg['source_buffer_m'])/cfg['transition_m'],0,1)*np.clip(boundary/cfg['boundary_fade_m'],0,1)
    keep=(rd>radius+cfg['road_crown_clearance_m'])&(rd<cfg['roadside_depth_m'])&(bd>radius+cfg['building_crown_clearance_m'])&(jd>cfg['junction_clearance_m'])&(source_distance>radius+cfg['source_crown_spacing_m'])&(rng.random(len(xy))<probability)
    keep[parking_tree.query(pts,predicate='intersects')[0]]=False
    xy=xy[keep];parent=parent[keep];rd=rd[keep];bd=bd[keep];fd=fd[keep];jd=jd[keep]
    print(dict(selected_outer_candidates=len(xy),retained_source=len(accepted)),flush=True)
    heights=final_support(vertices,faces,xy,allow_missing=True) if len(xy) else []
    inferred=[];unsupported=[]
    for i,(position,source,z) in enumerate(zip(xy,parent,heights)):
        if not np.isfinite(z):
            unsupported.append(dict(xy_m=position.tolist(),reason='No final terrain support triangle'));continue
        ref=accepted[int(source)]
        row={k:deepcopy(ref[k]) for k in ('asset','native_foliage_mode','appearance_condition','native_approximation') if k in ref}
        row.update(id='inferred_outer_'+str(i),position_enu_m=[*position.tolist(),float(z)],
            target_height_m=ref['target_height_m'],crown_radius_m=ref['crown_radius_m'],
            context_kind='inferred_outer',nearest_source_candidate=int(source),nearest_source_id=ref.get('id'),
            yaw_deg=float(rng.uniform(0,360)),native_support_method='Actual final terrain triangle',
            road_clearance_m=float(rd[i]),building_clearance_m=float(bd[i]),
            observed_lane_distance_m=float(fd[i]),junction_clearance_m=float(jd[i]),
            provenance='Inferred roadside placement and nearest reviewed source appearance family; no individual tree observation.')
        inferred.append(row)
    key='appearance:outer_source_canopy';task='task:outer_source_canopy'
    add_node(graph,dict(id=task,kind='simulation_task'))
    add_node(graph,dict(id=key,kind='native_appearance_condition',content=spec,
        native_construction=dict(kind='outer_canopy',settings=cfg),
        task_status={task:dict(complete=False,missing=['native_visibility_support_and_collision_review'])}))
    add_edge(graph,task,'depends_on',key)
    source_key='source:accepted_canopy_for_outer'
    add_node(graph,dict(id=source_key,kind='source_lidar_evidence',content=dict(path=str(a.evidence),accepted_count=len(accepted))))
    add_edge(graph,key,'supported_by',source_key)
    graph['decisions'].append(dict(source_id=key,task=task,choice='enhance',agent=spec['agent'],
        reason=spec['reason'],evidence_ids=[source_key],required_properties=spec['required_properties']))
    conditions=compile_conditions(graph,scene,task)
    output=deepcopy(evidence);output['canopy_candidates']=accepted+inferred
    output['outer_completion']=dict(inferred_count=len(inferred),unsupported=unsupported,selection=str(a.selection),
        source_evidence=str(a.evidence),scope=spec['scope'],native_review_complete=False)
    review=dict(source_count=len(accepted),source_records_unchanged=output['canopy_candidates'][:len(accepted)]==accepted,
        candidate_count=len(xy),inferred_count=len(inferred),unsupported_count=len(unsupported),
        source_parking_polygons=len(parking),native_review_complete=False,
        minimum_road_clearance_m=min((r['road_clearance_m'] for r in inferred),default=None),
        minimum_junction_clearance_m=min((r['junction_clearance_m'] for r in inferred),default=None))
    for name,data in [('evidence.json',output),('review.json',review),('graph.json',graph),('conditions.json',conditions)]:
        (a.output/name).write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(review,flush=True)


if __name__=='__main__':main()
