"""Prepare explicit physical building-clearance constraints for access projection."""
import argparse,json
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import LineString,Polygon,mapping,shape
from shapely.ops import unary_union
from map_generation.building_passages import BuildingVolumes


def prepare(scene,condition,scene_path,clearance_m=.02,movement_limit_m=3.):
    volumes=BuildingVolumes(scene['buildings']);lanes={x['id']:x for x in condition['lanes']};paths={}
    for key in condition['generated_path_ids']:
        lane=lanes[key];p=np.asarray(lane['centerline']);width=max(lane['end_widths_m'])
        query=LineString(p[:,:2]).buffer(width/2+movement_limit_m+clearance_m)
        indices=[int(i)for i in volumes.tree.query(query,predicate='intersects')
                 if volumes.buildings[int(i)]['base_z_m']<p[:,2].max()+2.2 and
                    volumes.buildings[int(i)]['roof_z_m']>p[:,2].min()-.01]
        obstacles=unary_union([volumes.shapes[i]for i in indices])
        paths[key]=dict(building_ids=[volumes.buildings[i]['id']for i in indices],
            obstacle_geometry=mapping(obstacles),source_road_z_range_m=[float(p[:,2].min()),float(p[:,2].max())])
    return dict(source_scene=scene_path,clearance_m=clearance_m,vehicle_height_m=2.2,
        paths=paths,method='Height-qualified actual realized building solids. Constrain signed centerline distance by half the unchanged source width plus explicit geometric clearance; exact road-envelope overlap remains an independent final check.')


def obstacle_shapes(condition):
    spec=condition.get('physical_clearance',{})
    return {key:shape(row['obstacle_geometry'])for key,row in spec.get('paths',{}).items()}


def margins(xy,widths,obstacle,clearance_m):
    if obstacle.is_empty:return np.full(len(xy),1000.)
    points=shapely.points(xy);distance=shapely.distance(points,obstacle.boundary)
    signed=np.where(shapely.covers(obstacle,points),-distance,distance)
    return signed-np.asarray(widths)/2-clearance_m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','batch-root','selection','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();scene=json.loads(a.scene.read_text());selection=json.loads(a.selection.read_text());a.output.mkdir(parents=True,exist_ok=True);records=[]
    for item in selection['regions']:
        key=item['region'];c=json.loads((a.batch_root/key/'condition.json').read_text())
        c['physical_clearance']=prepare(scene,c,str(a.scene.resolve()),float(item.get('clearance_m',.02)))
        c['physical_clearance']['decision']=item
        out=a.output/key;out.mkdir(exist_ok=True);path=out/'condition.json';path.write_text(json.dumps(c,separators=(',',':'))+'\n')
        records.append(dict(region=key,condition=str(path.resolve()),constrained_paths=len(c['physical_clearance']['paths']),building_relations=sum(len(q['building_ids'])for q in c['physical_clearance']['paths'].values())))
    (a.output/'manifest.json').write_text(json.dumps(dict(records=records,source_scene=str(a.scene),selection=str(a.selection)),indent=2)+'\n');print(records,flush=True)


if __name__=='__main__':main()
