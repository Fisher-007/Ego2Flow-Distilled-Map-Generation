"""Transfer native elevation changes to source-owned physical surfaces.

Retained XY and triangulation receive corresponding native height deltas.
Closure heights interpolate the deltas of their two parent surfaces.
"""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
import shapely
from map_generation.av2_surfaces import mesh_triangles
from map_generation.junction_surfaces import query_surface,plane_z
from scripts.scene.deform_physical_profiles import sample_changes
from scripts.scene.revise_physical_profiles import native_height_delta

def source_ids(mesh):
    keys=set(mesh.get('source_lane_ids',[]))|{mesh.get('source_lane_id')}
    keys.update(mesh.get(side+'_source_lane_id') for side in ('from','to'))
    for key in list(keys):
        if isinstance(key,str) and key.startswith('bridge:'):
            keys.update(key[7:].split('->'))
    return keys-{None}

def sample_native_interval(native, mesh, road_length, step=.5, knots=()):
    """Sample an actual mapped native lane interval, including packed lane IDs."""
    from map_generation.packed_surfaces import strip_mesh
    low, high = mesh['native_interval_m']
    grid = np.unique(np.r_[np.linspace(low, high, max(3, int(np.ceil((high-low)/step))+1)),
                           [s for s in knots if low < s < high]])
    left, right = [], []
    margin = min(1e-4, road_length*1e-3)
    for station in grid:
        s = float(np.float32(np.clip(station, margin, road_length-margin)))
        waypoint = native.get_waypoint_xodr(int(mesh['xodr_road_id']), int(mesh['xodr_lane_id']), s)
        if waypoint is None:
            raise ValueError('Missing mapped native support: '+str((mesh['id'],s)))
        p, forward = waypoint.transform.location, waypoint.transform.get_forward_vector()
        center = np.array([p.x,-p.y,p.z])
        normal = np.array([forward.y,forward.x,0.]);normal /= np.linalg.norm(normal)
        left.append(center+normal*waypoint.lane_width/2)
        right.append(center-normal*waypoint.lane_width/2)
    metadata = {k:mesh[k] for k in ('id','source_lane_id','source_interval_m',
                                   'xodr_road_id','xodr_lane_id','native_interval_m')}
    return strip_mesh(left,right,**metadata)


def sample_grouped_interval(native, mesh, section, road_length):
    """Carry measured native Z on the retained grouped cross-section stations.

    Packed physical boundaries retain source transverse heights and are not the
    same XY polygon as the native reference lane. For an elevation-only revision,
    sample the same native lane/station before and after on the original physical
    support. Subtraction preserves its existing transverse shape and contact cuts.
    """
    from map_generation.packed_surfaces import strip_mesh
    parts = section.get('native_sections', [dict(
        road_id=section['source_lanes'][0]['road_id'], start_m=0., end_m=section['length_m'])])
    part = next(p for p in parts if int(p['road_id']) == int(mesh['xodr_road_id']))
    offset = part['start_m']; low, high = np.asarray(mesh['native_interval_m']) + offset
    grid = np.asarray([s['s_m'] for s in section['samples']])
    stations = np.unique(np.r_[low, grid[(grid > low) & (grid < high)], high]) - offset
    boundaries = [np.asarray(mesh[side],float).copy() for side in ('left_boundary','right_boundary')]
    if any(len(b) != len(stations) for b in boundaries):
        raise ValueError('Grouped physical station correspondence changed: '+mesh['id'])
    lane_index = next(i for i,e in enumerate(section['source_lanes'])
                      if int(e['lane_id']) == int(mesh['xodr_lane_id']))
    origins = np.asarray([s['reference_xyz'][:2] for s in section['samples']])
    normals = np.asarray([s['normal_xy'] for s in section['samples']])
    offsets = np.asarray([s['boundary_offsets_m'] for s in section['samples']])
    maximum_xy_error = 0.
    for boundary, index in zip(boundaries, (lane_index,lane_index+1)):
        original_xy = origins + normals*offsets[:,index,None]
        expected = np.column_stack([np.interp(stations+offset,grid,original_xy[:,d]) for d in range(2)])
        maximum_xy_error = max(maximum_xy_error,float(abs(boundary[:,:2]-expected).max(initial=0)))
    if maximum_xy_error > 1e-6:
        raise ValueError('Grouped physical station XY no longer matches source mapping: '+mesh['id'])
    z = []; maximum_native_center_offset = 0.; margin = min(1e-4,road_length*1e-3)
    for i,station in enumerate(stations):
        s = float(np.float32(np.clip(station,margin,road_length-margin)))
        waypoint = native.get_waypoint_xodr(int(mesh['xodr_road_id']),int(mesh['xodr_lane_id']),s)
        if waypoint is None:
            raise ValueError('Missing native grouped station: '+str((mesh['id'],s)))
        p = waypoint.transform.location; z.append(p.z)
        center = (boundaries[0][i,:2]+boundaries[1][i,:2])/2
        maximum_native_center_offset = max(maximum_native_center_offset,
                                           float(np.linalg.norm(center-np.array([p.x,-p.y]))))
    for boundary in boundaries: boundary[:,2] = z
    metadata = {k:mesh[k] for k in ('id','source_lane_id','source_interval_m',
                                   'xodr_road_id','xodr_lane_id','native_interval_m')}
    result = strip_mesh(*boundaries,**metadata)
    result['native_station_mapping'] = dict(
        method='Actual native lane Z at each retained grouped cross-section station; original physical XY retained for delta interpolation',
        stations=len(stations),maximum_source_mapping_xy_error_m=maximum_xy_error,
        maximum_native_center_offset_m=maximum_native_center_offset,
        native_sample_stations_m=stations.tolist())
    return result


def sample_owned(mesh,xy,tolerance):
    triangles=mesh_triangles(mesh)
    tree=shapely.STRtree(shapely.polygons(triangles[:,:,:2]))
    z=query_surface(triangles,xy,tree,allow_missing=True)
    missing=~np.isfinite(z);maximum=0.
    if missing.any():
        points=shapely.points(xy[missing]);idx=tree.nearest(points)
        distance=shapely.distance(points,tree.geometries[idx]);maximum=float(distance.max())
        if maximum>tolerance:
            raise ValueError('Retained physical cut leaves its native source support: '+mesh['id']+' distance='+str(maximum))
        z[missing]=plane_z(triangles[idx],xy[missing])
    return z,int(missing.sum()),maximum

def sample_owned_change(old, new, points, tolerance):
    """Follow the original 3D support through a native elevation-only change.

    A source strip can overlap itself in XY. Independently taking its old and
    new highest planes can select different cross sections and corrupt the
    transferred height. Match the retained point to its original native plane,
    then interpolate the corresponding native vertex-height difference there.
    """
    delta,xy_error=native_height_delta(old,new)
    values,diagnostic=sample_changes(dict(vertices=points),mesh_triangles(old),delta)
    maximum=diagnostic['maximum_native_support_distance_m']
    if maximum>tolerance:
        raise ValueError('Retained physical cut leaves its native source support: '+old['id']+' distance='+str(maximum))
    return values,diagnostic['extrapolated_vertices'],maximum,xy_error


def closure_parent_delta(mesh, original, updated):
    """Interpolate parent deltas across the retained connecting footprint."""
    xy=np.asarray(mesh['vertices'],float)[:,:2]
    values=[];distances=[];remeshed=[]
    for side in ('from','to'):
        key=mesh[side+'_mesh_id'];old=original[key];new=updated[key]
        points=np.asarray(old['vertices'],float)
        same_xy_faces=(old['faces']==new['faces'] and
            np.array_equal(points[:,:2],np.asarray(new['vertices'],float)[:,:2]))
        if same_xy_faces:
            delta=points.copy()
            delta[:,2]=np.asarray(new['vertices'],float)[:,2]-points[:,2]
            triangles=delta[np.asarray(old['faces'],int)]
        else:
            triangles=points[np.asarray(old['faces'],int)]
        polygons=shapely.polygons(triangles[:,:,:2])
        tree=shapely.STRtree(polygons)
        query=shapely.points(xy);indices=tree.nearest(query)
        nearest=shapely.get_point(shapely.shortest_line(query,polygons[indices]),-1)
        contact_xy=shapely.get_coordinates(nearest)
        distances.append(shapely.distance(query,nearest))
        if same_xy_faces:
            values.append(plane_z(triangles[indices],contact_xy))
        else:
            current_z,count,maximum=sample_owned(new,contact_xy,1e-4)
            values.append(current_z-plane_z(triangles[indices],contact_xy))
            remeshed.append(dict(mesh_id=key,contact_samples=len(contact_xy),
                precision_boundary_samples=count,maximum_boundary_distance_m=maximum))
    d0,d1=distances;v0,v1=values
    coincident=(d0<=1e-8)&(d1<=1e-8)
    total=d0+d1
    weight=np.divide(d1,total,out=np.full_like(total,.5),where=total>1e-12)
    delta=v0*weight+v1*(1-weight)
    return delta,dict(
        method='Actual retained parent delta, interpolated by distance to the two parent surfaces; exact at each unique contact',
        from_mesh_id=mesh['from_mesh_id'],to_mesh_id=mesh['to_mesh_id'],
        remeshed_parent_samples=remeshed,
        maximum_parent_distance_m=float(max(d0.max(initial=0),d1.max(initial=0))),
        coincident_contact_vertices=int(coincident.sum()),
        maximum_coincident_delta_disagreement_m=float(abs(v0[coincident]-v1[coincident]).max(initial=0)))

def rebase(roads,previous,current,revision,tolerance=1e-4):
    if not revision['native_graph_XY_widths_and_permissions_unchanged']:
        raise ValueError('Physical height transfer requires an elevation-only native revision')
    changed=set(revision['changed_source_lane_ids'])
    before={m['id']:m for m in previous['meshes']}
    after={m['id']:m for m in current['meshes']}
    result=deepcopy(roads);records=[];pending=[];closures=[]
    original={m['id']:m for m in roads['meshes']}
    updated={m['id']:m for m in result['meshes']}
    protected={'av2_boundary_surface','av2_grounded_junction'}
    for mesh in result['meshes']:
        keys=source_ids(mesh)
        if not keys&changed or mesh.get('kind') in protected:continue
        if mesh.get('kind')=='inferred_shared_road_ground':
            pending.append(dict(mesh_id=mesh['id'],changed_source_ids=sorted(keys&changed)))
            continue
        if not mesh['faces']:continue
        if mesh.get('kind')=='connection_closure':
            closures.append(mesh)
            continue
        old,new=before[mesh['id']],after[mesh['id']]
        if source_ids(old)!=source_ids(new):
            raise ValueError('Native source ownership changed: '+mesh['id'])
        p=np.asarray(mesh['vertices'],float)
        delta,n0,d0,xy_error=sample_owned_change(old,new,p,tolerance)
        p[:,2]+=delta;mesh['vertices']=p.tolist()
        for side in ('left_boundary','right_boundary'):
            if side not in mesh:continue
            boundary=np.asarray(mesh[side],float)
            shift,_,_,_=sample_owned_change(old,new,boundary,tolerance)
            boundary[:,2]+=shift;mesh[side]=boundary.tolist()
        record=dict(mesh_id=mesh['id'],source_ids=sorted(keys),maximum_height_change_m=float(abs(delta).max()),
                    precision_boundary_samples=n0,maximum_boundary_distance_m=d0,
                    native_vertex_xy_error_m=xy_error,
                    height_transfer='Original native 3D support owns the interpolated native vertex-height change')
        if 'native_station_mapping' in new:
            record['native_station_mapping']={k:v for k,v in new['native_station_mapping'].items() if k!='native_sample_stations_m'}
        mesh['native_height_rebase']=record;records.append(record)
    pending_ids={r['mesh_id'] for r in pending}
    closure_pending=[]
    for mesh in closures:
        parents={mesh['from_mesh_id'],mesh['to_mesh_id']}
        # Resolve consumed parents through their recorded shared-floor lineage.
        replacements={}
        for key in parents-original.keys():
            owners={k for k in pending_ids
                    if key in original[k].get('selection',{}).get('mesh_ids',[])}
            if not owners:
                raise ValueError('Closure parent has no current shared-floor lineage: '+key)
            replacements[key]=sorted(owners)
        dependencies=(parents&pending_ids)|set().union(*(set(v)for v in replacements.values()))
        if dependencies:
            closure_pending.append(dict(mesh_id=mesh['id'],parent_dependencies=sorted(dependencies),
                                        replaced_parent_meshes=replacements))
            continue
        delta,contact=closure_parent_delta(mesh,original,updated)
        p=np.asarray(mesh['vertices'],float);p[:,2]+=delta;mesh['vertices']=p.tolist()
        record=dict(mesh_id=mesh['id'],source_ids=sorted(source_ids(mesh)),
                    maximum_height_change_m=float(abs(delta).max(initial=0)),
                    parent_contact_delta=contact)
        mesh['native_height_rebase']=record;records.append(record)
    fixed={m['id']:m for m in roads['meshes'] if m.get('kind') in protected}
    current_fixed={m['id']:m for m in result['meshes'] if m['id'] in fixed}
    changed_ids={r['mesh_id'] for r in records}
    exact_outside=all(m==original[m['id']]for m in result['meshes']if m['id']not in changed_ids)
    xy_faces=all(m['faces']==original[m['id']]['faces'] and
        np.array_equal(np.asarray(m['vertices'])[:,:2],np.asarray(original[m['id']]['vertices'])[:,:2])
        for m in result['meshes'] if m['id'] in changed_ids)
    report=dict(method=__doc__,changed_source_lane_ids=sorted(changed),updated_meshes=records,
        common_floor_dependencies=pending,closure_dependencies=closure_pending,observed_AV2_exact=fixed==current_fixed,
        unmodified_meshes_exact=exact_outside,all_physical_xy_and_faces_exact=xy_faces,
        source_boundary_tolerance_m=tolerance,
        maximum_closure_contact_delta_disagreement_m=max((r.get('parent_contact_delta',{}).get('maximum_coincident_delta_disagreement_m',0)for r in records),default=0),
        scope='Only ordinary retained source meshes updated. Reported common floors, bridge clearance, terrain, native support and driving remain dependent validation.')
    report['transfer_passed']=report['observed_AV2_exact'] and exact_outside and xy_faces and report['maximum_closure_contact_delta_disagreement_m']<=tolerance
    result['native_height_rebase']=report
    return result,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('roads','previous-native','current-native','height-revision','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-boundary-tolerance',type=float,default=1e-4,
        help='Metres of same-source triangle-edge extrapolation; actual maximum distances are reported. Does not alter physical geometry or collision checks.')
    a=p.parse_args();read=lambda path:json.loads(path.read_text())
    result,report=rebase(read(a.roads),read(a.previous_native),read(a.current_native),read(a.height_revision),a.source_boundary_tolerance)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'road_meshes.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(dict(transfer_passed=report['transfer_passed'],updated_meshes=len(report['updated_meshes']),
               common_floor_dependencies=report['common_floor_dependencies'],closure_dependencies=report['closure_dependencies'],maximum_closure_contact_delta_disagreement_m=report['maximum_closure_contact_delta_disagreement_m']),flush=True)
    if not report['transfer_passed']:raise SystemExit('Source-owned physical height transfer requires feedback')

if __name__=='__main__':main()
