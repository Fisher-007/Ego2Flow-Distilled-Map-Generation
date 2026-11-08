"""Return accepted group paths to the original source road graph and intervals."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from shapely.geometry import LineString,Polygon
from shapely.ops import unary_union
from map_generation.road_builder import fit_lane,chord_station_to_arc
from map_generation.information_graph import check_road_candidate
from map_generation.building_passages import BuildingVolumes, realize, eligible_vehicle_height
from scripts.scene.apply_access_corridor import slice_geometry


def apply(source,condition,paths,footprint=None,refresh_covered_passages=False):
    if not paths['passed']:raise ValueError('Group projection did not pass')
    before={l['id']:l for l in source['fusion_map']['lanes']}
    inputs={l['id']:l for l in condition['lanes']};outputs={r['id']:r for r in paths['paths']};replacements=defaultdict(list)
    # Clear the whole source width envelope against observed OSM buildings.
    if footprint is None:footprint=BuildingVolumes(source['buildings'])
    building_review=[];generated_segments=[]
    for path in condition['paths']:
        key=path['id'];lane=dict(inputs[key],centerline=outputs[key]['centerline'],boundary_shape=outputs[key]['boundary_shape'])
        g,_=fit_lane(lane,1.);segments=path['segments'];lengths=np.array([r['end_m']-r['start_m'] for r in segments]);cuts=np.r_[0.,lengths.cumsum()]/lengths.sum()*g.length
        for i,row in enumerate(segments):
            coefficients,points=slice_geometry(g,cuts[i],cuts[i+1]);replacements[row['lane_id']].append((row,coefficients,points))
            width=before[row['lane_id']]['width_m'];line=LineString(points[:,:2])
            generated_segments.append((row['lane_id'],points,width))
            vehicle_height,height_condition=eligible_vehicle_height(source,row['lane_id'])
            obstacles=footprint.obstacles(points,width,vehicle_height) if isinstance(footprint,BuildingVolumes) else footprint
            building_review.append(dict(lane_id=row['lane_id'],centerline_inside_building_m=float(line.intersection(obstacles).length),
                width_envelope_inside_building_m2=float(line.buffer(width/2,cap_style='flat').intersection(obstacles).area),
                checked_vehicle_height_m=vehicle_height if isinstance(footprint,BuildingVolumes) else None,
                vehicle_height_condition=height_condition))
    updated=dict(source);updated['fusion_map']=dict(source['fusion_map']);after=dict(before);preservation=[];piece_gaps=[]
    for key,revisions in replacements.items():
        old=before[key];old_geometry,_=fit_lane(old,1.);original=np.asarray(old['centerline']);old_chord=np.linalg.norm(np.diff(original[:,:2],axis=0),axis=1).sum()
        coefficients=[];chunks=[];cursor=0.;modified=[]
        for row,c,points in sorted(revisions,key=lambda v:v[0]['start_m']):
            if row['start_m']<cursor-.001:raise ValueError('Overlapping generated intervals on '+key)
            if row['start_m']>cursor+.001:
                cc,pp=slice_geometry(old_geometry,chord_station_to_arc(old,old_geometry,cursor),chord_station_to_arc(old,old_geometry,row['start_m']))
                coefficients.extend(cc);chunks.append(pp)
            coefficients.extend(c);chunks.append(points);cursor=row['end_m'];modified.append([row['start_m'],row['end_m']])
        if cursor<old_chord-.001:
            cc,pp=slice_geometry(old_geometry,chord_station_to_arc(old,old_geometry,cursor),old_geometry.length);coefficients.extend(cc);chunks.append(pp)
        for a,b in zip(chunks,chunks[1:]):
            gap=float(np.linalg.norm(a[-1]-b[0]))
            if gap>1e-5:piece_gaps.append(dict(lane_id=key,gap_m=gap))
        points=np.vstack([p if i==0 else p[1:] for i,p in enumerate(chunks)])
        lane=deepcopy(old);lane['centerline']=points.tolist();lane.pop('boundary_shape',None);lane['parametric_xy_segments']=coefficients
        lane['group_geometry_revision']=dict(region_id=condition['region']['id'],source_intervals_m=modified,
            model_paths=[p['id'] for p in condition['paths'] if any(r['lane_id']==key for r in p['segments'])],
            source_topology_and_permissions_retained=True)
        after[key]=lane;preservation.append(dict(lane_id=key,source_intervals_m=modified))
    updated['fusion_map']['lanes']=[after[l['id']] for l in source['fusion_map']['lanes']]
    if refresh_covered_passages:
        # Reconstruct building openings for the accepted road before checking vehicle clearance.
        updated,passage_review=realize(updated)
        volumes=BuildingVolumes(updated['buildings']);building_review=[]
        for key,points,width in generated_segments:
            vehicle_height,height_condition=eligible_vehicle_height(updated,key)
            line=LineString(points[:,:2]);obstacles=volumes.obstacles(points,width,vehicle_height)
            building_review.append(dict(lane_id=key,centerline_inside_building_m=float(line.intersection(obstacles).length),
                width_envelope_inside_building_m2=float(line.buffer(width/2,cap_style='flat').intersection(obstacles).area),
                checked_vehicle_height_m=vehicle_height,vehicle_height_condition=height_condition))
    requests=[]
    for key in replacements:
        requests.append(dict(source_id=key,incoming=[l for l in before.values() if key in l['successors']],
            outgoing=[before[t] for t in before[key]['successors'] if t in before]))
    conditions=dict(road=dict(generate=requests,retain_lane_ids=[k for k in before if k not in replacements],preserve_all_topology=True))
    if condition.get('graph_conditions'):
        conditions=json.loads(Path(condition['graph_conditions']).read_text())
        if {r['source_id'] for r in conditions['road']['generate']}!=set(replacements):
            raise ValueError('Applied source intervals differ from the graph-selected generation set')
    review=check_road_candidate(source,updated,conditions)
    review.update(region=condition['region']['id'],source_intervals=preservation,building_review=building_review,
        changed_av2=[k for k,l in before.items() if l['source']=='av2' and after[k]!=l],internal_piece_gaps=piece_gaps)
    if refresh_covered_passages:review['covered_passage_feedback']=passage_review
    review['passed'] &= not review['changed_av2'] and not piece_gaps and all(r['centerline_inside_building_m']<.01 and r['width_envelope_inside_building_m2']<.01 for r in building_review)
    updated['fusion_map']['group_revisions']=[*source['fusion_map'].get('group_revisions',[]),review]
    return updated,review,conditions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('condition','paths','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scene',type=Path,help='Source roads with reviewed building realization updates')
    p.add_argument('--passage-source-scene',type=Path,help='Original building evidence when road feedback requires rebuilding source-tagged covered space')
    a=p.parse_args();condition=json.loads(a.condition.read_text());source=json.loads((a.scene or Path(condition['source_scene'])).read_text())
    if a.passage_source_scene:
        parents={b['passage_parent_id'] for b in source['buildings'] if 'passage_parent_id' in b}
        originals=json.loads(a.passage_source_scene.read_text())['buildings']
        source['building_passage_source_buildings']=[b for b in originals if b['id'] in parents]
    paths=json.loads(a.paths.read_text());result,review,conditions=apply(source,condition,paths,refresh_covered_passages=bool(a.passage_source_scene));a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    (a.output/'conditions.json').write_text(json.dumps(conditions,separators=(',',':'))+'\n')
    print(review,flush=True)
    if not review['passed']:raise SystemExit(1)
    (a.output/'scene.json').write_text(json.dumps(result,separators=(',',':'))+'\n')


if __name__=='__main__':main()
