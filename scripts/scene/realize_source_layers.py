"""Realize source building layers and repair measured coarse surface defects."""
import argparse
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
from shapely.geometry import Polygon
from shapely import make_valid
from map_generation.building_layers import realize, measure_solids
from map_generation.buildings import build_buildings
from map_generation.coarse import route_check
from map_generation.terrain import Terrain


def repair_junctions(coarse):
    records=[]
    for junction in coarse['junctions']:
        original=Polygon(junction['boundary'])
        if original.is_valid and original.area>0:
            continue
        valid=make_valid(original)
        # Use convex inferred envelopes containing the supplied source boundaries and endpoints.
        hull=original.convex_hull
        if hull.geom_type!='Polygon' or hull.area<=0:
            raise ValueError(f"Degenerate junction needs source evidence: {junction['id']}")
        source=np.asarray(junction['boundary'])
        points=[]
        for xy in np.asarray(hull.exterior.coords)[:,:2]:
            i=np.linalg.norm(source[:,:2]-xy,axis=1).argmin()
            if np.linalg.norm(source[i,:2]-xy)>1e-7:
                raise ValueError('Hull introduced a non-source vertex')
            points.append(source[i].tolist())
        record=dict(id=junction['id'],original_boundary=junction['boundary'],
                    source_valid_area_m2=valid.area,constructed_area_m2=hull.area,
                    inferred_added_surface_m2=hull.difference(valid).area,
                    removed_valid_surface_m2=valid.difference(hull).area,
                    method='Minimum convex envelope of the original inferred SUMO boundary',
                    scope='Coarse surface inference; lane geometry, permissions and graph unchanged. Fine source surfaces and native support must supersede or verify this envelope.')
        junction['boundary']=points
        junction['surface_repair']=record
        records.append(record)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','config','data-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--rebuild-buildings',action='store_true')
    p.add_argument('--repair-junctions',action='store_true')
    p.add_argument('--default-clearance',type=float,default=4.5)
    a=p.parse_args()
    scene=json.loads(a.scene.read_text());config=json.loads(a.config.read_text())
    a.output.mkdir(parents=True,exist_ok=True)
    original_roads=deepcopy(scene['coarse_map']['lanes'])
    if a.rebuild_buildings:
        terrain=Terrain(scene,config,a.data_root,bounds=config['coarse']['terrain_bounds_local_xy'])
        scene['buildings']=build_buildings(scene,config,a.data_root,terrain)
        scene.pop('building_layer_source_buildings',None)
    route=route_check(scene,config)
    before=measure_solids(scene['buildings'],route['positions'])
    repairs=repair_junctions(scene['coarse_map']) if a.repair_junctions else []
    result,layers=realize(scene,default_clearance_m=a.default_clearance)
    after=measure_solids(result['buildings'],route['positions'])
    report=dict(source_scene=str(a.scene),before=before,after=after,
                junction_surface_repairs=repairs,
                all_junction_surfaces_valid=all(Polygon(j['boundary']).is_valid and Polygon(j['boundary']).area>0 for j in result['coarse_map']['junctions']),
                source_lanes_unchanged=result['coarse_map']['lanes']==original_roads,
                configured_route_unchanged=route==route_check(result,config),
                layer_summary={k:v for k,v in layers.items()if k!='passages'},
                passage_summary={k:v for k,v in layers['passages'].items()if k!='records'},
                legacy_check_scope='Physical solids and vehicle-space measurements complement the 2D footprint checks in check_coarse.py.')
    report['passed']=all([after['passed'],report['all_junction_surfaces_valid'],
                          report['source_lanes_unchanged'],report['configured_route_unchanged'],
                          abs(layers['solid_union']['union_volume_error_m3'])<1e-4,
                          all(r['status']=='constructed_pending_native_review' for r in layers['passages']['records'])])
    for name,value in [('scene.json',result),('layers.json',layers),('physical_review.json',report)]:
        (a.output/name).write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],building_count=len(result['buildings']),
                         original_collision_count=before['route_body_collision_count'],
                         final_collision_count=after['route_body_collision_count'],
                         original_overlap_volume=before['overlap_volume_m3'],
                         final_overlap_volume=after['overlap_volume_m3'],
                         junctions_repaired=len(repairs),passages=report['passage_summary'],
                         union=layers['solid_union']),indent=2))
    if not report['passed']:raise SystemExit('Physical source-layer construction requires further review')


if __name__=='__main__':
    main()
