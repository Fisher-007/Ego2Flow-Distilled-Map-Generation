"""Remove selected superseded paint intervals from installed meshes."""
import argparse,json
from pathlib import Path
import numpy as np,orjson,shapely
from map_generation.source_paint import PATTERNS
from map_generation.shared_road_ground import clip_outside
from scripts.scene.replace_observed_marking_patterns import source_strokes


def resolve(scene,original,current,conditions):
    by={lane['id']:lane for lane in scene['fusion_map']['lanes']};domains={};decisions=[]
    width=original['report']['parameters']['width_m']
    for selection in conditions['reviewed_paint_ownership']:
        if not selection.get('evidence')or not selection.get('reason'):
            raise ValueError('Display ownership needs an explicit evidence-based decision')
        region=shapely.Polygon(selection['domain']['exterior'],selection['domain'].get('holes',[]));retired={};retained={};rows=[]
        for row in original['report']['boundary_records']:
            if 'id'not in row or row['mark_type']not in selection['mark_types']:continue
            line=shapely.LineString(row['centerline_xy_m'])
            if not line.intersects(region.buffer(width*2)):continue
            logs=set()
            for ref in row['sources']:
                if ref.get('observation_source'):logs.add(ref['observation_source'])
                else:logs.update(x['source']for x in by[ref['lane_id']].get('source_observations',[]))
            remove=any(k in source for k in selection['retire_source_logs']for source in logs)and not any(k in source for k in selection['keep_source_logs']for source in logs)
            color=PATTERNS[row['mark_type']][0];footprint=source_strokes(row,width).intersection(region)
            (retired if remove else retained).setdefault(color,[]).append(footprint)
            rows.append(dict(boundary_id=row['id'],source_sides=row['sources'],source_maps=sorted(logs),retire_display=remove,selected_paint_area_m2=footprint.area))
        for color,parts in retired.items():
            # Retain source-owner paint in overlapping footprints.
            target=shapely.union_all(parts);keep=shapely.union_all(retained.get(color,[]))
            domains.setdefault(color,[]).append(target.difference(keep))
        decisions.append(dict(condition=selection,boundaries=rows))
    output=[];checks=[]
    selected_mesh_ids=set(conditions['current_mesh_ids'])
    for mesh in current['meshes']:
        if mesh['id']not in selected_mesh_ids:continue
        domain=shapely.union_all(domains.get(mesh['color'],[]));tri=np.asarray(mesh['vertices'])[np.asarray(mesh['faces'],int)]
        touched=shapely.intersects(shapely.polygons(tri[:,:,:2]),domain)
        removed=0.;retained_fragments=np.empty((0,3,3))
        if touched.any():
            part=tri[touched];before=shapely.union_all(shapely.polygons(part[:,:,:2]));clipped=clip_outside(dict(vertices=part.reshape(-1,3).tolist(),faces=np.arange(part.size//3).reshape(-1,3).tolist()),domain);retained_fragments=np.asarray(clipped['vertices']).reshape(-1,3,3);removed=before.intersection(domain).area
        combined=np.concatenate([tri[~touched],retained_fragments]);assert np.array_equal(combined[:int((~touched).sum())],tri[~touched])
        output.append(dict(mesh,id=mesh['id']+conditions.get('mesh_suffix','_owner'),vertices=combined.reshape(-1,3).tolist(),faces=np.arange(combined.size//3).reshape(-1,3).tolist()))
        checks.append(dict(source_mesh=mesh['id'],original_triangles=len(tri),untouched_triangles=int((~touched).sum()),retired_touched_triangles=int(touched.sum()),retained_boundary_fragments=len(retained_fragments),removed_paint_area_m2=float(removed),unselected_triangles_exact=True,retirement_domain=shapely.to_geojson(domain)))
    return dict(meshes=output,report=dict(decisions=decisions,checks=checks,source_lane_geometry_modified=False,navigation_modified=False,road_support_modified=False,phase='Original installed retained-owner phases unchanged; no full repaint or automatic geometric merge.',scope='Only explicitly selected current mesh and source/epoch paint footprints retired; raw original observations and all prior failures remain.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','original-paint','current-paint','conditions','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:orjson.loads(p.read_bytes());result=resolve(read(a.scene),read(a.original_paint),read(a.current_paint),read(a.conditions));a.output.mkdir(parents=True,exist_ok=True)
    result['report'].update(scene=str(a.scene),original_paint=str(a.original_paint),current_paint=str(a.current_paint),conditions=str(a.conditions))
    (a.output/'markings.json').write_bytes(orjson.dumps(result));(a.output/'report.json').write_text(json.dumps(result['report'],indent=2));print(json.dumps(dict(meshes=len(result['meshes']),checks=[{k:v for k,v in x.items()if k!='retirement_domain'}for x in result['report']['checks']])))

if __name__=='__main__':main()
