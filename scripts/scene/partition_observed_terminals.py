"""Partition source terminal curves into external roads and junction segments."""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from map_generation.access_fusion import section
from map_generation.road_builder import fit_lane,arc_station_to_chord
from scripts.scene.apply_access_corridor import slice_geometry


def partition(scene,conditions):
    originals={l['id']:l for l in scene['fusion_map']['lanes']}
    lanes=deepcopy(originals);records=[]
    for row in conditions['terminals']:
        key=row['lane_id'];lane=originals[key]
        if lane['source']!='av2' or not lane.get('is_intersection') or lane['successors']:
            raise ValueError('Reviewed observed terminal must be an AV2 junction lane with no represented successor: '+key)
        if not row.get('evidence') or row.get('continuation_status')!='unresolved':
            raise ValueError('Native source terminal requires explicit unresolved source evidence')
        geometry=fit_lane(lane,1.)[0];tail_length=float(row['terminal_subcurve_m'])
        if not 0<tail_length<geometry.length:
            raise ValueError('Terminal partition must lie strictly inside the existing observed curve')
        split=geometry.length-tail_length;tail=key+':observed_terminal_tail'
        if tail in lanes:raise ValueError('Terminal compiler part already exists')
        pieces=[];maximum_error=0.;maximum_width_error=0.
        for name,start,end,successors in [(key,0.,split,[tail]),(tail,split,geometry.length,[])]:
            coefficients,points=slice_geometry(geometry,start,end);part=deepcopy(lane)
            widths=[float(geometry.width(start)),float(geometry.width(end))]
            part.update(id=name,centerline=points.tolist(),parametric_xy_segments=coefficients,
                parametric_elevation_rows=[list(v)for v in geometry.elevation_rows(start,end,0.)],
                end_widths_m=widths,successors=successors,
                native_source_interval=dict(source_lane_id=key,start_m=start,end_m=end),
                native_terminal_partition=dict(condition=row,original_is_intersection=lane['is_intersection'],
                    role='unresolved_terminal_subcurve'if name==tail else 'observed_junction_body',
                    source_geometry_scope='Exact existing fitted source subcurve; no added spatial extent.'))
            part.pop('boundary_shape',None)
            if name==tail:part['internal']=False;part['is_intersection']=False
            chord=[arc_station_to_chord(lane,geometry,v)for v in [start,end]]
            length=LineString(np.asarray(lane['centerline'])[:,:2]).length
            for field in ['left_boundary','right_boundary']:
                if field in lane:
                    boundary_length=LineString(np.asarray(lane[field])[:,:2]).length
                    part[field]=section(lane[field],chord[0]/length*boundary_length,chord[1]/length*boundary_length).tolist()
            compiled=fit_lane(part,1.)[0]
            for value in np.linspace(0.,end-start,max(21,int((end-start)/.1)+1)):
                maximum_error=max(maximum_error,float(np.linalg.norm(compiled.xyz(value)-geometry.xyz(start+value))))
                maximum_width_error=max(maximum_width_error,abs(compiled.width(value)-geometry.width(start+value)))
            if abs(compiled.length-(end-start))>1e-6:
                raise ValueError('Native subdivision changed source arc length')
            lanes[name]=part;pieces.append(dict(id=name,start_m=start,end_m=end))
        if maximum_error>1e-6 or maximum_width_error>1e-6:
            raise ValueError('Native terminal partition did not preserve source geometry')
        records.append(dict(condition=row,original_observation=deepcopy(lane),parts=pieces,
            maximum_xyz_error_m=maximum_error,maximum_width_error_m=maximum_width_error,
            source_end_error_m=float(np.linalg.norm(np.asarray(lanes[tail]['centerline'][-1])-lane['centerline'][-1])),
            original_successors=lane['successors'],native_terminal_lane_id=tail))
    affected={r['condition']['lane_id']for r in records}
    changed_unselected=[k for k,l in originals.items()if k not in affected and lanes[k]!=l]
    if changed_unselected:raise ValueError('Native partition changed unselected source lanes')
    out=deepcopy(scene);out['fusion_map']['lanes']=list(lanes.values())
    out.setdefault('native_compilation',{}).setdefault('observed_terminal_partitions',[]).extend(records)
    report=dict(source_lane_count=len(originals),compiler_lane_count=len(lanes),partitions=records,
        changed_unselected=changed_unselected,source_observations_unchanged=out['lanes']==scene['lanes'],
        added_spatial_extent_m=0.,added_automobile_continuations=0,
        status='Native representation candidate. Unresolved continuation remains explicit; independent native geometry/routing and reviewer acceptance required.')
    return out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','conditions','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();scene,report=partition(json.loads(a.scene.read_text()),json.loads(a.conditions.read_text()))
    a.output.mkdir(parents=True,exist_ok=True)
    for name,value in [('scene',scene),('review',report)]:
        (a.output/(name+'.json')).write_text(json.dumps(value,separators=(',',':')if name=='scene'else None,indent=None if name=='scene'else 2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='partitions'},indent=2))
if __name__=='__main__':main()
