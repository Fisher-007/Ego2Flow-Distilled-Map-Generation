"""Extend selected tree families over inferred roadside support and planting beds."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import shapely
from scipy.spatial import cKDTree
import cv2
from map_generation.coordinates import LocalFrame
from scripts.scene.prepare_city_greenery import final_support


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['scene','native-snapshot','profiles','osm','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);read=lambda f:json.loads(f.read_text());scene=read(a.scene);native=read(a.native_snapshot);spec=read(a.profiles);profiles=spec['area_style_profiles'];settings=spec['streetscape'];rng=np.random.default_rng(spec['seed']);centers=np.array([np.asarray(x['bbox_local_xy']).reshape(2,2).mean(0)for x in scene['areas']]);area_ids=[x['id']for x in scene['areas']]
 # Use explicit coverage to distinguish observed and inferred context.
 context_policy=settings.get('source_context_policy','observed_coverage')
 if context_policy not in ('observed_coverage','legacy_anchor_radius'):raise ValueError('Unknown source context policy: '+context_policy)
 coverage_rows=settings.get('observed_coverage_polygons',[]);gap_rows=settings.get('source_visible_gaps',[])
 def context_polygon(row):return shapely.Polygon(row['exterior'],row.get('holes',[])).buffer(0)
 coverage=shapely.union_all([context_polygon(x)for x in coverage_rows]);gaps=[context_polygon(x)for x in gap_rows]
 gap_task='task:source_context_coverage';gap_nodes=[dict(id=gap_task,kind='simulation_task')];gap_decisions=[];gap_edges=[]
 for row in gap_rows:
  evidence_ids=[]
  for index,evidence in enumerate(row.get('evidence',[])):
   identity='evidence:street_context_gap:'+row['id']+':'+str(index);evidence_ids.append(identity);gap_nodes.append(dict(id=identity,kind='source_visual_evidence',content=dict(reference=evidence)))
  key='condition:street_context_gap:'+row['id'];gap_nodes.append(dict(id=key,kind='native_appearance_condition',content=row,native_construction=dict(kind='source_neighborhood_gap',settings=row),task_status={gap_task:dict(complete=False,missing=['source_gap_native_realization','independent_visual_review'])}));gap_decisions.append(dict(source_id=key,task=gap_task,choice='enhance',agent=spec.get('agent','city_builder'),reason='Source-visible context gap remains unresolved; source anchor alone is not coverage.',evidence_ids=evidence_ids,required_properties=['native_visibility','source_gap_native_realization','independent_visual_review']));gap_edges.append(dict(source=gap_task,relation='depends_on',target=key))
 (a.output/'source_gap_conditions.json').write_text(json.dumps(dict(nodes=gap_nodes,decisions=gap_decisions,edges=gap_edges,source_context_policy=context_policy,scope='Graph conditions and candidate placements for source-gap enhancement.'),indent=2))
 polygons=[];junctions=[]
 for lane in scene['fusion_map']['lanes']:
  if 'left_boundary'in lane and 'right_boundary'in lane:poly=shapely.Polygon(np.asarray(lane['left_boundary']+lane['right_boundary'][::-1])[:,:2]).buffer(0)
  else:poly=shapely.LineString(np.asarray(lane['centerline'])[:,:2]).buffer(lane['width_m']/2,cap_style='flat')
  if poly.is_empty:continue
  polygons.append(poly)
  if lane.get('is_intersection'):junctions.append(poly)
 road=shapely.union_all(polygons);buildings=[shapely.Polygon(x['exterior'],x.get('holes',[]))for x in scene['buildings']];building_tree=shapely.STRtree(buildings);junction_tree=shapely.STRtree(junctions)
 old_trees=[x for x in native['components']if x['visible']and not x['hidden_in_game']and ('Canopy'in x['label']or '/Vegetation/'in x['mesh'])];known=np.array([[x['actor_center_m'][0],-x['actor_center_m'][1]]for x in old_trees]);known_tree=cKDTree(known)
 frame=LocalFrame(scene['coordinates']['city'],scene['coordinates']['origin_city_xyz']);osm=ET.parse(a.osm).getroot();nodes={x.get('id'):[float(x.get('lon')),float(x.get('lat'))]for x in osm.findall('node')};parking=[];green=[];osm_rows=[]
 for way in osm.findall('way'):
  tags={x.get('k'):x.get('v')for x in way.findall('tag')};refs=[x.get('ref')for x in way.findall('nd')]
  isparking=tags.get('amenity')=='parking'and tags.get('parking')not in ['underground','multi-storey'];isgreen=tags.get('landuse')in ['grass','forest','meadow','recreation_ground']or tags.get('leisure')in ['park','garden']or tags.get('natural')in ['wood','grassland','scrub']
  if not(isparking or isgreen)or len(refs)<4 or refs[0]!=refs[-1]:continue
  poly=shapely.Polygon(frame.lonlat_to_local([nodes[x]for x in refs])).buffer(0)
  (parking if isparking else green).append(poly);osm_rows.append(dict(way=way.get('id'),kind='surface_parking'if isparking else 'green_landcover',tags=tags,area_m2=poly.area))
 parking_union=shapely.union_all(parking);parking_exclusion=parking_union.buffer(.5);boundary=road.buffer(settings['roadside_offset_m'],join_style='round').boundary;rows=[];pockets=[];links=[];accepted_xy=[];selection_counts=dict(boundary_candidates=0,rejected_source_context=0,rejected_building_or_parking=0,rejected_existing_or_spacing=0,rejected_confirmed_coverage=0)
 for line in shapely.get_parts(boundary):
  if line.length<25:continue
  previous=None
  for s in np.arange(8,line.length-8,10.):
   point=shapely.line_interpolate_point(line,float(s));xy=np.array(point.coords[0]);selection_counts['boundary_candidates']+=1;dist=np.linalg.norm(centers-xy,axis=1);near=np.argsort(dist)[:3];weights=np.exp(-(dist[near]-dist[near[0]])/spec['reference_transition_m']);weights/=weights.sum();profile=profiles[area_ids[int(near[0])]];vegetation=profile['vegetation'];spacing=sum(weights[k]*profiles[area_ids[int(j)]]['vegetation']['spacing_m']for k,j in enumerate(near))
   if coverage.covers(point):selection_counts['rejected_confirmed_coverage']+=1;continue
   if (context_policy=='legacy_anchor_radius' and dist.min()<settings.get('source_anchor_exclusion_m',0)) or (len(junctions)and point.distance(junctions[int(junction_tree.nearest(point))])<settings['junction_clearance_m']):selection_counts['rejected_source_context']+=1;continue
   if rng.random()>10./spacing:continue
   radius=vegetation['crown_radius_m']
   if point.distance(buildings[int(building_tree.nearest(point))])<radius+.5 or parking_exclusion.covers(point):selection_counts['rejected_building_or_parking']+=1;continue
   if known_tree.query(xy)[0]<settings['existing_tree_spacing_m'] or (accepted_xy and np.min(np.linalg.norm(np.array(accepted_xy)-xy,axis=1))<spacing*.72):selection_counts['rejected_existing_or_spacing']+=1;continue
   choices=vegetation['assets'];asset_weights=np.array([x['weight']for x in choices],float);asset_weights/=asset_weights.sum();asset=choices[int(rng.choice(len(choices),p=asset_weights))]['asset'];height=float(rng.uniform(*vegetation['tree_height_m']));row=dict(id=settings['id_prefix']+str(len(rows)),position_enu_m=[*xy,0.],target_height_m=height,crown_radius_m=radius,context_kind='inferred_outer',asset=asset,yaw_deg=float(rng.uniform(0,360)),appearance_reference=area_ids[int(near[0])],road_clearance_m=float(point.distance(road)),source_view_evidence=profile['evidence'],native_approximation='Existing source-reviewed native tree family; no measured outer tree location or exact species claim.');matched_gaps=[gap_rows[j]['id']for j,poly in enumerate(gaps)if poly.covers(point)];row['source_gap_ids']=matched_gaps
   if matched_gaps or dist.min()<settings.get('source_review_neighborhood_m',65.):
    row['context_kind']='source_neighborhood_inference';row['requires_source_review']=True;row['native_approximation']='Proposed native tree family near source evidence; unresolved gaps require source-supported geometry and independent image review. Anchor presence is not completed coverage.'
   rows.append(row);accepted_xy.append(xy)
   pit=point.buffer(vegetation['planting_bed_radius_m']);pockets.append(pit)
   if previous is not None and vegetation.get('connect_planting_beds',False) and np.linalg.norm(xy-previous)<spacing*2.5:
    strip=shapely.LineString([previous,xy]).buffer(.65,cap_style='flat')
    if strip.distance(road)>1.4 and not strip.intersects(parking_union)and all(not strip.intersects(buildings[int(j)])for j in building_tree.query(strip)):links.append(strip)
   previous=xy
 vertices=np.asarray(scene['terrain']['vertices']);faces=np.asarray(scene['terrain']['faces']);heights=final_support(vertices,faces,np.array(accepted_xy),allow_missing=True);supported=[];missing=[]
 for row,z in zip(rows,heights):
  if not np.isfinite(z):missing.append(row['id']);continue
  row['position_enu_m'][2]=float(z);row['native_support_method']='Existing final terrain triangle';supported.append(row)
 (a.output/'canopy.json').write_text(json.dumps(dict(canopy_candidates=supported,replacement_label_prefix=settings['label_prefix'],outer_completion=dict(inferred_count=len(supported),source_anchors_are_not_coverage=True,confirmed_coverage_polygons=len(coverage_rows),selection=str(a.profiles),scope='Nearby AV2 style continuation on unchanged outer support; no individual-tree reconstruction.')),indent=2))
 lo,hi=vertices[:,:2].min(0),vertices[:,:2].max(0);size=4096;ground=np.full((size,size),255,np.uint8);park=np.zeros((size,size),np.uint8)
 def pixel(ring):
  uv=(shapely.get_coordinates(ring)-lo)/(hi-lo);uv[:,1]=1-uv[:,1];return np.rint(uv*(size-1)).astype(np.int32)
 def draw(mask,geometry,value,holes):
  for poly in shapely.get_parts(geometry):
   if poly.geom_type!='Polygon':continue
   cv2.fillPoly(mask,[pixel(poly.exterior)],value)
   for ring in poly.interiors:cv2.fillPoly(mask,[pixel(ring)],holes)
 green_union=shapely.union_all(green+pockets+links).difference(road).difference(parking_union).difference(shapely.union_all(buildings));draw(ground,green_union,0,255);draw(park,parking_union,255,0)
 cv2.imwrite(str(a.output/'ground_mask.png'),ground);cv2.imwrite(str(a.output/'parking_mask.png'),park);(a.output/'ground_mask.json').write_text(json.dumps(dict(lower=lo.tolist(),upper=hi.tolist(),size=size,pixel_size_xy_m=((hi-lo)/(size-1)).tolist(),scope='Existing terrain material partition. OSM green/parking polygons plus explicitly inferred narrow planting beds at selected roadside trees; no new road or parking connectivity.',source_scene=str(a.scene),source_profiles=str(a.profiles)),indent=2));(a.output/'parking.json').write_text(json.dumps(dict(source=str(a.osm),polygons=[x for x in osm_rows if x['kind']=='surface_parking'],scope='Existing mapped surface parking appearance only; occupancy, exact epoch and driving permissions remain unresolved.'),indent=2));report=dict(source_context_policy=context_policy,anchor_radius_implies_completion=False,confirmed_coverage_polygons=len(coverage_rows),source_visible_gap_ids=[x['id']for x in gap_rows],source_gap_conditions=str(a.output/'source_gap_conditions.json'),source_neighborhood_candidates=sum(x['context_kind']=='source_neighborhood_inference'for x in supported),inferred_trees=len(supported),unsupported_ids=missing,retained_existing_vegetation=len(old_trees),reference_counts={key:sum(x['appearance_reference']==key for x in supported)for key in area_ids},green_area_m2=green_union.area,inferred_planting_strips=len(links),osm_landcover=osm_rows,selection_counts=selection_counts,road_geometry_changed=False,native_review_pending=True);(a.output/'review.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='osm_landcover'}),flush=True)
if __name__=='__main__':main()
