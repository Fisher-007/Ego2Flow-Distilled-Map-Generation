"""Measure source-way corridor space independently of inferred SUMO lane offsets."""
import argparse,json,re,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point,LineString,Polygon,box
from shapely import get_coordinates
from map_generation.building_passages import BuildingVolumes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def review(scene,osm,way_ids,output,step_m=2.,reach_m=15.):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root=ET.parse(osm).getroot();nodes={x.get('id'):x for x in root.findall('node')}
    ways={x.get('id'):x for x in root.findall('way')}
    transform=Transformer.from_crs('EPSG:4326',scene['coordinates']['horizontal_crs'],always_xy=True)
    origin=np.asarray(scene['coordinates']['origin_utm_xy'])
    lanes=scene['fusion_map']['lanes'];volumes=BuildingVolumes(scene['buildings']);records=[];plots=[]
    for way_id in way_ids:
        way=ways[str(way_id)];refs=[n.get('ref')for n in way.findall('nd')]
        coords=np.asarray([transform.transform(float(nodes[k].get('lon')),float(nodes[k].get('lat')))for k in refs])-origin
        line=LineString(coords)
        selected=[l for l in lanes if re.sub(r'^[+-]','',l.get('edge_id','')).split('#')[0]==str(way_id)]
        if not selected:raise ValueError('No retained inferred lanes for source way '+str(way_id))
        lane_lines=[LineString(np.asarray(l['centerline'])[:,:2])for l in selected]
        stations=np.linspace(0,line.length,max(3,int(np.ceil(line.length/step_m))+1));xyz=[]
        for station in stations:
            q=line.interpolate(station);index=min(range(len(selected)),key=lambda i:lane_lines[i].distance(q))
            p=np.asarray(selected[index]['centerline']);arc=np.r_[0.,np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).cumsum()]
            z=np.interp(lane_lines[index].project(q),arc,p[:,2]);xyz.append([q.x,q.y,z])
        xyz=np.asarray(xyz);obstacles=volumes.obstacles(xyz,2*reach_m);measurements=[]
        for station,p in zip(stations,xyz):
            before=np.asarray(line.interpolate(max(0.,station-.05)).coords[0]);after=np.asarray(line.interpolate(min(line.length,station+.05)).coords[0])
            direction=after-before;direction/=np.linalg.norm(direction);normal=np.array([-direction[1],direction[0]])
            inside=obstacles.contains(Point(p[:2]));distances=[]
            for sign in [1,-1]:
                ray=LineString([p[:2],p[:2]+sign*reach_m*normal]);hit=ray.intersection(obstacles)
                distances.append(0. if inside else None if hit.is_empty else float(Point(p[:2]).distance(hit)))
            measurements.append(dict(station_m=float(station),xyz=p.tolist(),left_m=distances[0],right_m=distances[1],source_center_inside_building=inside,
                bounded_free_width_m=None if None in distances else sum(distances)))
        finite=[x['bounded_free_width_m']for x in measurements if x['bounded_free_width_m']is not None]
        row=dict(osm_way_id=str(way_id),source_tags={t.get('k'):t.get('v')for t in way.findall('tag')},
            original_way_xy=coords.tolist(),lane_ids=[l['id']for l in selected],lane_widths_m=[l['width_m']for l in selected],
            permissions=[l['allowed_classes']for l in selected],stations=measurements,
            source_center_inside_samples=sum(x['source_center_inside_building']for x in measurements),
            bounded_width_min_m=min(finite)if finite else None,bounded_width_median_m=float(np.median(finite))if finite else None)
        records.append(row);plots.append((row,selected,obstacles));print({k:v for k,v in row.items()if k not in ['stations','original_way_xy','permissions','lane_widths_m','lane_ids']},flush=True)
    for start in range(0,len(plots),4):
        fig,axes=plt.subplots(2,2,figsize=(16,14))
        for ax,(row,selected,obstacles)in zip(axes.flat,plots[start:start+4]):
            q=np.asarray(row['original_way_xy']);lo=q.min(axis=0)-8;hi=q.max(axis=0)+8
            for i in volumes.tree.query(box(*lo,*hi),predicate='intersects'):
                shape=volumes.shapes[int(i)]
                ax.fill(*shape.exterior.xy,color='#d59c95',alpha=.4)
            ax.plot(q[:,0],q[:,1],color='black',lw=1.8,label='OSM source way')
            for lane in selected:
                p=np.asarray(lane['centerline']);ax.plot(p[:,0],p[:,1],color='#1769aa',lw=1)
                poly=LineString(p[:,:2]).buffer(lane['width_m']/2,cap_style='flat')
                if poly.geom_type=='Polygon':ax.fill(*poly.exterior.xy,color='#1769aa',alpha=.12)
            for x in row['stations'][::max(1,len(row['stations'])//12)]:
                ax.text(*x['xyz'][:2],'/'.join('open'if v is None else f'{v:.1f}'for v in [x['left_m'],x['right_m']]),fontsize=6)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_title('OSM '+row['osm_way_id']+' | source centre vs default lane footprints')
        for ax in list(axes.flat)[len(plots[start:start+4]):]:ax.axis('off')
        fig.tight_layout();fig.savefig(output/f'corridors_{start//4:02d}.png',dpi=130);plt.close(fig)
    report=dict(records=records,station_step_m=step_m,ray_reach_m=reach_m,scope='Source-way normal-ray free space in the recorded UTM frame and current height-qualified building volumes. Does not decide permissions, modify widths, move source geometry or accept generated roads. Open rays are not measured corridor edges.')
    (output/'review.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','osm','conditions','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();conditions=json.loads(a.conditions.read_text());review(json.loads(a.scene.read_text()),a.osm,conditions['osm_way_ids'],a.output)
if __name__=='__main__':main()
