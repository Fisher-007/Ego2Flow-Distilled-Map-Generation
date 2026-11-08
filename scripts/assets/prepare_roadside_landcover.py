"""Create a terrain-material mask from OSM surface-parking polygons."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def parking_coverage(root, xy):
    """Query the same source parking mask used by the native ground material."""
    import cv2
    import numpy as np
    root=Path(root);spec=json.loads((root/'ground_mask.json').read_text())
    mask=cv2.imread(str(root/'parking_mask.png'),cv2.IMREAD_GRAYSCALE)
    lo,hi=np.asarray(spec['lower']),np.asarray(spec['upper']);xy=np.asarray(xy)
    uv=(xy-lo)/(hi-lo);uv[:,1]=1-uv[:,1]
    pixels=np.rint(np.clip(uv,0,1)*[mask.shape[1]-1,mask.shape[0]-1]).astype(int)
    return mask[pixels[:,1],pixels[:,0]]>127


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--osm',type=Path,required=True)
    p.add_argument('--ground-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import cv2
    import numpy as np
    import shapely
    from pyproj import Transformer
    scene=json.loads(a.scene.read_text());spec=json.loads((a.ground_root/'ground_mask.json').read_text())
    root=ET.parse(a.osm).getroot()
    nodes={n.get('id'):(float(n.get('lon')),float(n.get('lat'))) for n in root.findall('node')}
    transform=Transformer.from_crs('EPSG:4326',scene['coordinates']['horizontal_crs'],always_xy=True)
    origin=np.asarray(scene['coordinates']['origin_utm_xy'])
    rows=[];polygons=[]
    for way in root.findall('way'):
        tags={t.get('k'):t.get('v') for t in way.findall('tag')}
        if tags.get('amenity')!='parking' or tags.get('parking') in ['underground','multi-storey']:continue
        refs=[n.get('ref') for n in way.findall('nd')]
        if refs[0]!=refs[-1]:continue
        xy=np.asarray([transform.transform(*nodes[n]) for n in refs])-origin
        poly=shapely.Polygon(xy).buffer(0);polygons.append(poly)
        rows.append({'osm_way':way.get('id'),'tags':tags,'area_m2':float(poly.area),'local_bounds':list(poly.bounds)})
    parking=shapely.union_all(polygons);shapely.prepare(parking)
    size=spec['size'];lo,hi=spec['lower'],spec['upper']
    xs=np.linspace(lo[0],hi[0],size);ys=np.linspace(hi[1],lo[1],size)
    mask=np.empty((size,size),np.uint8)
    for i,y in enumerate(ys):mask[i]=shapely.contains(parking,shapely.points(xs,np.full(size,y))).astype(np.uint8)*255
    cv2.imwrite(str(a.output/'parking_mask.png'),np.repeat(mask[:,:,None],3,axis=2))
    for name in ['ground_mask.json','ground_mask.png']:shutil.copy2(a.ground_root/name,a.output/name)
    (a.output/'parking.json').write_text(json.dumps({'source':str(a.osm.resolve()),'polygons':rows,
        'projection':'OSM lon/lat -> source horizontal CRS, subtract origin_utm_xy; same local east/north as the map.',
        'scope':'Source land-cover annotation used as terrain appearance. No new drivable topology, traffic rules or parking task.',
        'limits':'Current OSM dates differ from AV2; visible parking is checked in source views. Small tree islands and curb geometry are not recovered.'},indent=2)+'\n')
    print({'parking_polygons':len(rows),'output':str(a.output)})


if __name__=='__main__':main()
