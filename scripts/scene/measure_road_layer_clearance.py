"""Measure vertical clearance between source-owned road layers.

Evaluate triangle intersections at their vertices and account for the
declared upper-surface thickness.
"""
import argparse,json
from pathlib import Path
import numpy as np
import shapely
from map_generation.av2_surfaces import mesh_triangles
from map_generation.junction_surfaces import plane_z
from scripts.scene.rebase_physical_road_heights import source_ids

def measure(roads,selection):
    upper=set(selection['clearance_source_lane_ids'])
    lower=set(selection['lower_source_lane_ids'])
    meshes=roads['meshes']
    top=[m for m in meshes if source_ids(m)&upper and m.get('kind')!='inferred_shared_road_ground' and m['faces']]
    bottom=[m for m in meshes if source_ids(m)&lower and m.get('kind')!='inferred_shared_road_ground' and m['faces']]
    a=np.concatenate([mesh_triangles(m)for m in top]);b=np.concatenate([mesh_triangles(m)for m in bottom])
    pa=shapely.polygons(a[:,:,:2]);pb=shapely.polygons(b[:,:,:2])
    pairs=shapely.STRtree(pb).query(pa,predicate='intersects')
    intersections=shapely.intersection(pa[pairs[0]],pb[pairs[1]])
    area=shapely.area(intersections);valid=area>1e-10
    pairs=pairs[:,valid];intersections=intersections[valid];area=area[valid]
    xy,owners=shapely.get_coordinates(intersections,return_index=True)
    top_z=plane_z(a[pairs[0,owners]],xy);bottom_z=plane_z(b[pairs[1,owners]],xy)
    headroom=top_z-bottom_z-selection['surface_thickness_allowance_m']
    minima=np.full(len(area),np.inf);np.minimum.at(minima,owners,headroom)
    worst=int(np.argmin(headroom));index=int(owners[worst])
    report=dict(method=__doc__,upper_mesh_ids=[m['id']for m in top],lower_mesh_ids=[m['id']for m in bottom],
        source_selection=selection,upper_triangles=len(a),lower_triangles=len(b),
        triangle_intersections=len(area),intersection_vertex_samples=len(xy),
        minimum_clear_headroom_m=float(headroom.min()),minimum_top_separation_m=float((top_z-bottom_z).min()),
        maximum_clear_headroom_m=float(headroom.max()),
        declared_headroom_m=selection['clear_headroom_m'],
        below_declared_envelope_intersection_area_m2=float(area[minima<selection['clear_headroom_m']-1e-4].sum()),
        worst_contact=dict(xy_m=xy[worst].tolist(),upper_z_m=float(top_z[worst]),lower_z_m=float(bottom_z[worst]),
                           upper_triangle=a[pairs[0,index]].tolist(),lower_triangle=b[pairs[1,index]].tolist()),
        scope='Exact current physical triangle relation only. Surveyed structure dimensions, actual native collision and driving require their own evidence.')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('roads','selection','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(path.read_text());report=measure(read(a.roads),read(a.selection))
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ['method','upper_mesh_ids','lower_mesh_ids','source_selection']},indent=2),flush=True)

if __name__=='__main__':main()
