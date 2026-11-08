"""Rasterize graph-selected ground appearance at whole-map metric resolution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import shapely

from map_generation.information_graph import add_node, add_edge, compile_conditions


def source_pavement(scene, settings):
    """Separate observed pavement extents and inferred tagged parking widths."""
    from collections import defaultdict
    from map_generation.roadside_parking import parallel_sides
    observed, tagged, records = [], [], []
    if settings.get('av2_drivable_pavement'):
        for row in scene['assets']:
            if row['kind'] == 'drivable_area' and row['source'] == 'av2':
                polygon = shapely.make_valid(shapely.Polygon(np.asarray(row['boundary'], dtype=float)[:, :2]))
                observed.append(polygon)
                records.append(dict(source_id=row['id'], kind='av2_drivable_pavement', area_m2=polygon.area))
    width = settings.get('source_tagged_roadside_parking_width_m', 0.)
    if width:
        by_way = defaultdict(list)
        for lane in scene['fusion_map']['lanes']:
            if lane['source'] != 'osm_sumo' or lane.get('internal'):
                continue
            by_way[str(lane['edge_id']).lstrip('-').split('#')[0]].append(lane)
        for road in scene['roads']:
            tags = road['tags']; sides = parallel_sides(tags)
            if not sides or any(tags.get(k) not in (None, 'no', 'false', '0') for k in ('bridge', 'tunnel')) or str(tags.get('layer', '0')) != '0':
                continue
            members = by_way[road['id'].split(':')[-1]]
            if not members:
                continue
            line = shapely.LineString(np.asarray(road['reference_line'], dtype=float)[:, :2])
            road_area = shapely.union_all([shapely.LineString(np.asarray(lane['centerline'])[:, :2]).buffer(
                lane['width_m'] / 2, cap_style='flat', join_style='mitre') for lane in members])
            side = shapely.union_all([line.buffer(30 if name == 'left' else -30, single_sided=True) for name in sides])
            region = road_area.buffer(width, join_style='mitre').intersection(side)
            tagged.append(region)
            records.append(dict(source_id=road['id'], kind='source_tagged_roadside_parking',
                source_tags=tags, sides=sides, inferred_width_m=width, area_m2=region.area))
    buildings = shapely.union_all([shapely.Polygon(row['exterior'], row.get('holes', [])) for row in scene['buildings']])
    # AV2 owns the measured surface. Building disagreement is reported; the
    # inferred roadside extension alone is clipped at existing foundations.
    source = shapely.union_all(observed)
    inferred = shapely.union_all(tagged).difference(buildings).difference(source)
    return source.union(inferred), dict(regions=records, observed_area_m2=source.area,
        inferred_tagged_area_m2=inferred.area, observed_building_overlap_m2=source.intersection(buildings).area,
        scope='AV2 pavement footprint and OSM-tagged parking appearance. Width outside AV2 is inferred. Terrain support, curbs and observed-body contact require native review. No navigation lane or parking regulation added.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','graph','selection','road-footprint','osm','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    read=lambda path:json.loads(path.read_text())
    scene,graph,spec=map(read,(a.scene,a.graph,a.selection))
    key,task='appearance:city_ground_cover','task:city_landcover'
    add_node(graph,dict(id=task,kind='simulation_task'))
    add_node(graph,dict(id=key,kind='native_appearance_condition',content=spec,
        native_construction=dict(kind='city_landcover',settings=spec),
        task_status={task:dict(complete=False,missing=['native_ground_appearance_review'])}))
    add_edge(graph,task,'depends_on',key)
    add_edge(graph,key,'depends_on',spec.get('ground_support_condition','ground_support:city_road_foundations'))
    graph['decisions'].append(dict(source_id=key,task=task,choice='enhance',agent=spec['agent'],
        reason=spec['reason'],evidence_ids=[str(a.road_footprint),str(a.osm)],required_properties=spec['required_properties']))
    conditions=compile_conditions(graph,scene,task)
    settings=next(row for row in conditions['native_objects'] if row['source_id']==key)['native_construction']['settings']
    ground=np.asarray(scene['terrain']['vertices'])
    lo,hi=ground[:,:2].min(0),ground[:,:2].max(0)
    size=settings['mask_size'];mask=np.zeros((size,size),np.uint8)
    physical_road=shapely.from_wkt(a.road_footprint.read_text())
    extension,extension_review=source_pavement(scene,settings)
    pavement=physical_road.union(extension).buffer(settings['inferred_sidewalk_width_m'])
    def pixels(ring):
        xy=shapely.get_coordinates(ring);uv=(xy-lo)/(hi-lo);uv[:,1]=1-uv[:,1]
        return np.rint(uv*(size-1)).astype(np.int32)
    def draw_region(target, region):
        for part in shapely.get_parts(region):
            if part.geom_type!='Polygon':continue
            cv2.fillPoly(target,[pixels(part.exterior)],255)
            for ring in part.interiors:cv2.fillPoly(target,[pixels(ring)],0)
    draw_region(mask,pavement)
    bare=a.output/'base';bare.mkdir(exist_ok=True)
    cv2.imwrite(str(bare/'ground_mask.png'),mask)
    report=dict(lower=lo.tolist(),upper=hi.tolist(),size=size,
        pixel_size_xy_m=((hi-lo)/(size-1)).tolist(),sidewalk_width_m=settings['inferred_sidewalk_width_m'],
        scope=settings['scope'],source_road_footprint=str(a.road_footprint),graph_condition=key)
    (bare/'ground_mask.json').write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run([sys.executable,'-m','scripts.assets.prepare_roadside_landcover',
        '--scene',str(a.scene),'--osm',str(a.osm),'--ground-root',str(bare),'--output',str(a.output)],check=True)
    extra=np.zeros_like(mask);draw_region(extra,extension)
    parking=cv2.imread(str(a.output/'parking_mask.png'),cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(a.output/'parking_mask.png'),np.maximum(parking,extra))
    report['pavement_extension']=extension_review
    for row in extension_review['regions']:
        evidence='evidence:pavement_extent:'+row['source_id']
        add_node(graph,dict(id=evidence,kind='source_pavement_evidence',content=row))
        add_edge(graph,key,'supported_by',evidence)
    if spec.get('review_evidence'):
        evidence='feedback:city_parked_pavement'
        add_node(graph,dict(id=evidence,kind='static_review_failure',source=spec['review_evidence'],
            resolution='Revised source pavement conditions; native appearance and contact review pending'))
        add_edge(graph,key,'responds_to',evidence)
    conditions=compile_conditions(graph,scene,task)
    (a.output/'pavement_extension.json').write_text(json.dumps(extension_review,indent=2)+'\n')
    (a.output/'pavement_extension.wkt').write_text(shapely.to_wkt(extension,rounding_precision=8))
    for name,data in [('graph.json',graph),('conditions.json',conditions),('review.json',report)]:
        (a.output/name).write_text(json.dumps(data,separators=(',',':'))+'\n')
    print({k:v for k,v in report.items() if k!='pavement_extension'},flush=True)
    print({k:v for k,v in extension_review.items() if k!='regions'},flush=True)


if __name__=='__main__':main()
