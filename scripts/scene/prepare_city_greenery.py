"""Assemble source-supported canopy and continuous, explicitly inferred city context."""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
import shapely

from map_generation.coordinates import LocalFrame
from map_generation.information_graph import add_node, add_edge, compile_conditions
from map_generation.mesh_geometry import boundary_segments
from scripts.scene.build_assets import surface_heights


def nearest_distance(tree, points):
    return shapely.distance(points, tree.geometries[tree.nearest(points)])


def final_support(vertices, faces, xy, allow_missing=False):
    # Select actual terrain triangles near queries before constructing GEOS
    # objects; a city ground mesh has millions of faces outside tree locations.
    queries = cKDTree(xy); selected = []
    for start in range(0, len(faces), 200000):
        tri = vertices[faces[start:start+200000]]
        projected = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])[:, 2]
        tri = tri[np.abs(projected) > 1e-10]
        center = tri[:, :, :2].mean(1)
        radius = np.linalg.norm(tri[:, :, :2]-center[:, None], axis=2).max(1)
        selected.append(tri[queries.query(center)[0] <= radius+1e-6])
    triangles = np.concatenate(selected)
    tree = shapely.STRtree(shapely.polygons(triangles[:, :, :2]))
    if allow_missing:
        covered = np.unique(tree.query(shapely.points(xy), predicate='dwithin', distance=2e-5)[0])
        result = np.full(len(xy), np.nan)
        result[covered] = surface_heights(triangles, tree, xy[covered], 'final native canopy support')
        return result
    return surface_heights(triangles, tree, xy, 'final native canopy support')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene','graph','selection','source-canopy','visual-review','road-footprint','osm','output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(path.read_text())
    scene, graph, spec = map(read, (a.scene, a.graph, a.selection))
    views = read(a.visual_review)['areas']; source = read(a.source_canopy/'review.json')
    key, task = 'appearance:city_canopy', 'task:city_greenery'
    add_node(graph, dict(id=task, kind='simulation_task'))
    add_node(graph, dict(id=key, kind='native_appearance_condition', content=spec,
        native_construction=dict(kind='city_canopy', settings=spec),
        task_status={task: dict(complete=False, missing=['native_occlusion_and_driving_review'])}))
    add_edge(graph, task, 'depends_on', key)
    add_edge(graph, key, 'depends_on', 'ground_support:city_road_foundations')
    evidence = []
    for item in source['logs']:
        identity = 'canopy_evidence:'+item['log_id']; evidence.append(identity)
        add_node(graph, dict(id=identity, kind='source_lidar_evidence', content=item))
        add_edge(graph, key, 'supported_by', identity)
    graph['decisions'].append(dict(source_id=key, task=task, choice='enhance', agent=spec['agent'],
        reason=spec['reason'], evidence_ids=evidence, required_properties=spec['required_properties']))
    conditions = compile_conditions(graph, scene, task)
    selected = next(n for n in conditions['native_objects'] if n['source_id']==key)
    settings = selected['native_construction']['settings']
    rng = np.random.default_rng(settings['seed'])
    footprint = shapely.from_wkt(a.road_footprint.read_text())
    road_tree = shapely.STRtree(boundary_segments(footprint.boundary))
    shapely.prepare(footprint)
    def road_distance(points):
        result = nearest_distance(road_tree, points)
        result[shapely.covers(footprint, points)] = 0.
        return result
    buildings = [shapely.Polygon(b['exterior'], b.get('holes', [])) for b in scene['buildings']]
    building_tree = shapely.STRtree(buildings)
    fine = shapely.STRtree([shapely.Polygon(l['left_boundary']+l['right_boundary'][::-1]).buffer(0)
                           for l in scene['lanes']])
    frame = LocalFrame(scene['coordinates']['city'], scene['coordinates']['origin_city_xyz'])
    root = ET.parse(a.osm).getroot()
    nodes = {n.get('id'): [float(n.get('lon')),float(n.get('lat'))] for n in root.findall('node')}
    parking = []
    for way in root.findall('way'):
        tags = {t.get('k'):t.get('v') for t in way.findall('tag')}
        refs = [n.get('ref') for n in way.findall('nd')]
        if tags.get('amenity')=='parking' and tags.get('parking') not in ('underground','multi-storey') and refs[0]==refs[-1]:
            parking.append(shapely.Polygon(frame.lonlat_to_local([nodes[n] for n in refs])).buffer(0))
    parking_tree = shapely.STRtree(parking)
    candidates = []
    for log in source['logs']:
        data = read(Path(log['evidence']))
        candidates.extend(dict(row, source_log=log['log_id'], source_evidence=log['evidence'],
            context_kind='lidar_supported') for row in data['canopy_candidates'])
    # Repeated views/logs can describe the same crown; keep its stronger
    # LiDAR support, retaining duplicate identities in the result.
    observed, rejected, merged = [], [], []
    for item in sorted(candidates, key=lambda x:-x['support_points']):
        center = np.array(item['position_enu_m'][:2]); point = shapely.Point(center)
        d = float(road_distance([point])[0]); b = float(nearest_distance(building_tree, [point])[0])
        if d < settings['observed_trunk_clearance_m'] or b < settings['observed_building_clearance_m']:
            rejected.append(dict(item, reason='Canopy cluster is not a clear trunk placement', final_road_clearance_m=d, building_clearance_m=b)); continue
        near = next((i for i,r in enumerate(observed) if np.linalg.norm(center-np.array(r['position_enu_m'][:2]))<settings['duplicate_spacing_m']),None)
        if near is not None:
            merged.append(dict(source_log=item['source_log'], cluster=item['cluster'], retained_candidate=near)); continue
        observed.append(dict(item, final_road_clearance_m=d, building_clearance_m=b))
    areas = {area['id']:area for area in scene['areas']}
    centers = np.array([np.array(areas[v['area_id']]['bbox_local_xy']).reshape(2,2).mean(0) for v in views])
    known_xy = np.array([r['position_enu_m'][:2] for r in observed]); known_tree = cKDTree(known_xy)
    heights = np.array([r['target_height_m'] for r in observed])
    radii = np.array([r['crown_radius_m'] for r in observed])
    # A jittered lattice extends the reviewed roadside family. It is a
    # construction choice; positions outside AV2 have no observation identity.
    ground = np.asarray(scene['terrain']['vertices']); faces = np.asarray(scene['terrain']['faces'])
    lo,hi=ground[:,:2].min(0)+5.,ground[:,:2].max(0)-5.
    spacing=settings['outer_spacing_m']
    xy=np.array(np.meshgrid(np.arange(lo[0],hi[0],spacing),np.arange(lo[1],hi[1],spacing))).reshape(2,-1).T
    xy+=rng.uniform(-settings['position_jitter_m'],settings['position_jitter_m'],xy.shape)
    points=shapely.points(xy);rd=road_distance(points);bd=nearest_distance(building_tree,points)
    fd=nearest_distance(fine,points);nearest,indices=known_tree.query(xy)
    eligible=(rd>settings['inferred_road_clearance_m'])&(rd<settings['context_depth_m'])&(bd>settings['inferred_building_clearance_m'])&(fd>settings['observed_extension_buffer_m'])&(nearest>settings['duplicate_spacing_m'])
    eligible[parking_tree.query(points,predicate='intersects')[0]]=False
    xy=xy[eligible];ids=indices[eligible];inferred=[]
    for center, parent, d,b in zip(xy,ids,rd[eligible],bd[eligible]):
        inferred.append(dict(position_enu_m=[*center.tolist(),0.], target_height_m=float(heights[parent]),
            crown_radius_m=float(radii[parent]), context_kind='inferred_outer',
            nearest_supported_candidate=int(parent), road_clearance_m=float(d), building_clearance_m=float(b)))
    trees=observed+inferred;xy=np.array([r['position_enu_m'][:2] for r in trees])
    print(dict(source_candidates=len(candidates),retained_source=len(observed),inferred=len(inferred),stage='final support'),flush=True)
    z=final_support(ground,faces,xy)
    distances=np.linalg.norm(xy[:,None,:]-centers[None,:,:],axis=2)
    for i,(tree,height) in enumerate(zip(trees,z)):
        nearest=np.argsort(distances[i])[:3];w=np.exp(-(distances[i,nearest]-distances[i,nearest[0]])/settings['reference_transition_m']);w/=w.sum()
        view=views[int(rng.choice(nearest,p=w))];area=view['area_id']
        family=settings['area_assets'].get(area,settings['default_assets'])
        tree.update(source_ground_z_m=tree['position_enu_m'][2] if tree['context_kind']=='lidar_supported' else None,
            appearance_reference=area, asset=str(rng.choice(family)), yaw_deg=float(rng.uniform(0,360)), ground_source='Final constructed terrain triangle')
        tree['position_enu_m'][2]=float(height)
    result=dict(canopy_candidates=trees,rejected_candidates=rejected,merged_observations=merged,
        graph_condition=selected,source_scene=str(a.scene),scope=settings['scope'])
    review=dict(source_logs=len(source['logs']),source_candidates=len(candidates),native_candidates=len(trees),
        retained_source_candidates=len(observed),inferred_outer_candidates=len(inferred),rejected_source_candidates=len(rejected),
        duplicate_observations=len(merged),assets=dict(Counter(t['asset'] for t in trees)),native_review_complete=False)
    for name,data in [('canopy.json',result),('review.json',review),('graph.json',graph),('conditions.json',conditions)]:
        (a.output/name).write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(review,flush=True)


if __name__=='__main__':main()
