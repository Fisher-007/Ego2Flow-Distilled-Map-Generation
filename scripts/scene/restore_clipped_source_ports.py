"""Restore selected clipped source ports and prepare inferred connector candidates."""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
from shapely.geometry import LineString,Point
from map_generation.fusion import seam_geometry,tangent
from scripts.scene.review_clipped_junction_ports import at

def repair(scene,conditions):
    out=deepcopy(scene);lanes={l['id']:l for l in out['fusion_map']['lanes']}
    before=deepcopy(lanes);retired={};alias_rows=[];edges=[];new=[];replaced=[]
    for row in conditions.get('aliases',[]):
        key=row['alias'];lane=lanes[key];path=row['canonical_path']
        incoming=[l['id']for l in lanes.values()if key in l['successors']]
        if incoming and not row.get('redirect_incoming_at_same_mouth',False):
            raise ValueError('Incoming source branches need an explicit same-mouth ownership decision: '+key)
        if any(k not in path for k in lane['successors']):
            raise ValueError('Alias has an independent outgoing observation')
        for a,b in zip(path[:-1],path[1:]):
            if b not in lanes[a]['successors']:raise ValueError('Canonical path is not already connected')
        if any(set(lane['allowed_classes'])!=set(lanes[k]['allowed_classes'])for k in path):
            raise ValueError('An observed alias cannot change source modes')
        p=np.asarray(lane['centerline']);q=np.vstack([lanes[k]['centerline']for k in path])
        line=LineString(q[:,:2]);stations=np.array([line.project(Point(v[:2]))for v in p])
        hits=np.array([at(q,s)for s in stations]);xy=float(np.linalg.norm(p[:,:2]-hits[:,:2],axis=1).max());z=float(abs(p[:,2]-hits[:,2]).max())
        if xy>row['maximum_xy_difference_m']or z>row['maximum_z_difference_m']:
            raise ValueError('Reviewed source owner exceeds actual geometric bounds: '+key)
        if row['kind']=='complete_observation_alias':
            if stations[-1]<=stations[0]or np.min(np.diff(stations))<-.001:
                raise ValueError('Complete source alias reverses its canonical interval')
        elif row['kind']=='unattached_duplicate_prefix':
            if len(lane['successors'])!=1 or stations[-1]<stations[0]-.001:
                raise ValueError('Unattached prefix is not a reviewed terminal duplicate')
        else:raise ValueError('Unknown alias decision')
        if incoming:
            start_xy=float(np.linalg.norm(p[0,:2]-q[0,:2]));start_z=float(abs(p[0,2]-q[0,2]))
            if stations[0]>row['maximum_start_station_m'] or start_xy>row['maximum_xy_difference_m'] or start_z>row['maximum_z_difference_m']:
                raise ValueError('Alias incoming mouth is not the reviewed canonical start')
            for source in incoming:
                lanes[source]['successors']=[path[0]if target==key else target for target in lanes[source]['successors']]
                lanes[source]['successors']=list(dict.fromkeys(lanes[source]['successors']))
                edges.append([source,path[0]])
        record=dict(row,redirected_incoming=incoming,maximum_xy_difference_m=xy,maximum_z_difference_m=z,
            canonical_stations_m=[float(stations[0]),float(stations[-1])],
            original_observation=deepcopy(lane))
        alias_rows.append(record);retired[key]=path[0];del lanes[key]
    for lane in lanes.values():
        for field in ['left_neighbor','right_neighbor']:
            if lane.get(field)in retired:lane[field]=retired[lane[field]]
    for row in conditions.get('relations',[]):
        a,b=row['from'],row['to'];pa=np.asarray(lanes[a]['centerline'][-1]);pb=np.asarray(lanes[b]['centerline'][0])
        gap=float(np.linalg.norm(pa[:2]-pb[:2]));dz=float(abs(pa[2]-pb[2]))
        if gap>row['maximum_xy_gap_m']or dz>row['maximum_z_gap_m']:
            raise ValueError('Observed relation exceeds reviewed port bounds')
        if b not in lanes[a]['successors']:lanes[a]['successors'].append(b);edges.append([a,b])
    for row in conditions.get('connectors',[]):
        sources=[lanes[k]for k in row['from']];targets=[lanes[k]for k in row['to']]
        starts=np.asarray([l['centerline'][-1]for l in sources]);ends=np.asarray([l['centerline'][0]for l in targets])
        if np.max(np.linalg.norm(starts-starts[0],axis=1))>1e-6 or np.max(np.linalg.norm(ends-ends[0],axis=1))>1e-6:
            raise ValueError('A shared connector must have actual common source mouths')
        allowed=set.intersection(*(set(l['allowed_classes'])for l in sources+targets))
        if not allowed:raise ValueError('No common source permission for connector')
        if 'allowed_classes'in row and not set(row['allowed_classes'])<=allowed:raise ValueError('Connector adds unsupported permission')
        allowed=sorted(row.get('allowed_classes',allowed))
        da=np.mean([tangent(l['centerline'],True)for l in sources],axis=0);db=np.mean([tangent(l['centerline'],False)for l in targets],axis=0)
        da/=np.linalg.norm(da);db/=np.linalg.norm(db)
        ident=row['id']
        if ident in lanes:raise ValueError('Connector ID already exists')
        widths=[float(np.median([l['width_m']for l in sources])),float(np.median([l['width_m']for l in targets]))]
        lane=dict(id=ident,source='source_clipped_access_transition',internal=False,
            centerline=seam_geometry(starts[0],ends[0],da,db),width_m=max(widths),end_widths_m=widths,
            speed_mps=min(l['speed_mps']for l in sources+targets),allowed_classes=allowed,
            successors=list(row['to']),generation_eligible=True,source_port_condition=deepcopy(row),
            source_geometry_status='Agent-reviewed source cut; actual constrained model generation and physical acceptance pending.')
        lanes[ident]=lane;new.append(ident)
        if row.get('replace_existing_direct_relation',False):
            if len(sources)!=1 or len(targets)!=1 or targets[0]['id']not in sources[0]['successors']:
                raise ValueError('Replacing a direct relation requires the exact reviewed edge')
            sources[0]['successors'].remove(targets[0]['id'])
            replaced.append([sources[0]['id'],targets[0]['id'],ident])
        for l in sources:
            l['successors'].append(ident);edges.append([l['id'],ident])
        edges.extend([ident,k]for k in row['to'])
    geometry_changes=[]
    for key,lane in before.items():
        if key in retired:continue
        a=deepcopy(lane);b=deepcopy(lanes[key]);a.pop('successors');b.pop('successors')
        for field in ['left_neighbor','right_neighbor']:
            if a.get(field)in retired:a[field]=retired[a[field]]
        if a!=b:geometry_changes.append(key)
        expected={retired.get(target,target)for target in lane['successors']}
        for source,target,connector in replaced:
            if source==key and target in expected:
                expected.remove(target);expected.add(connector)
                if target not in lanes[connector]['successors']:raise ValueError('Replacement did not retain the full branch')
        if not expected<=set(lanes[key]['successors']):raise ValueError('A retained branch was lost')
    if geometry_changes:raise ValueError('Retained source properties changed')
    report=dict(conditions=conditions,original_lane_count=len(before),resulting_lane_count=len(lanes),
        original_area_count=len(scene['areas']),resulting_area_count=len(out['areas']),
        aliases=alias_rows,replaced_direct_relations=replaced,retired_duplicate_ids=list(retired),new_connector_ids=new,added_edges=edges,
        retained_geometry_changes=0,retained_permission_changes=0,
        geometry_status='Source topology and ownership only. New connectors require actual model generation; existing conflicting geometry and building/native checks remain open.')
    out['fusion_map']['lanes']=list(lanes.values())
    out['fusion_map'].setdefault('clipped_source_port_reviews',[]).append(report)
    out['fusion_map'].setdefault('observed_source_interval_aliases',[]).extend(alias_rows)
    return out,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','conditions','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();scene,report=repair(json.loads(a.scene.read_text()),json.loads(a.conditions.read_text()))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(scene,separators=(',',':'),allow_nan=False)+'\n')
    (a.output/'source_port_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ['conditions','aliases']},indent=2),flush=True)
if __name__=='__main__':main()
