"""Refresh changed source profiles and reopen their dependent shared surfaces.

Retain observed AV2 meshes and unrelated compiled geometry. Shared inferred
patches depending on changed sources are reopened from their recorded parents,
then passed to the existing source-connected physical-ground builder.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree

from map_generation.av2_surfaces import mesh_triangles, plane_z
from map_generation.shared_road_ground import clip_outside
from map_generation.mesh_geometry import spatial_union


def native_height_delta(old,new):
    # At steep short connectors, native resampling can exchange the two
    # cross-section endpoints. Match their positions, retaining the previous
    # physical footprint and triangulation while transferring only height.
    a,b=np.asarray(old['vertices'],float),np.asarray(new['vertices'],float)
    distance,indices=cKDTree(b[:,:2]).query(a[:,:2])
    reverse=cKDTree(a[:,:2]).query(b[:,:2])[0]
    error=float(max(distance.max(),reverse.max()))
    if error>.001:raise ValueError('Height revision changed native vertex positions: '+old['id'])
    field=mesh_triangles(old).copy()
    field[:,:,2]=(b[indices,2]-a[:,2])[np.asarray(old['faces'],int).reshape(-1,3)]
    return field,error


def closure_height_delta(mesh,previous,revised):
    # Closure numbering depends on the set of tiny uncovered cap fragments.
    # Its stable identity is the two source parents. Transfer their endpoint
    # height changes over the same cap triangulation used by the compiler.
    caps=[];deltas=[];errors=[]
    for side,index in [('from',-1),('to',0)]:
        old=previous[mesh[side+'_mesh_id']];new=revised[old['id']]
        xy=np.asarray([old['left_boundary'][index],old['right_boundary'][index]])[:,:2]
        field,error=native_height_delta(old,new)
        tree=shapely.STRtree(shapely.polygons(field[:,:,:2]));ids=tree.nearest(shapely.points(xy))
        caps.extend(xy);deltas.extend(plane_z(field[ids],xy));errors.append(error)
    caps=np.asarray(caps);deltas=np.asarray(deltas)
    pieces=shapely.get_parts(shapely.constrained_delaunay_triangles(shapely.convex_hull(shapely.multipoints(caps))))
    xy=shapely.get_coordinates(pieces).reshape(-1,4,2)[:,:3]
    ids=cKDTree(caps).query(xy.reshape(-1,2))[1]
    return np.concatenate([xy,deltas[ids].reshape(-1,3,1)],axis=2),max(errors)


def reopened_selections(previous, reopened, revision):
    """Rebuild active patches from retained footprints and source-parent layer membership."""
    reviews={r['id']:r for r in previous.get('shared_inferred_ground',[])}
    old={m['id']:m for m in previous['meshes']}
    available={m['id']for m in reopened['meshes']}
    fine=[m for m in reopened['meshes']if m.get('kind')in ('av2_boundary_surface','av2_grounded_junction')]
    boxes=[]
    for mesh in fine:
        v=np.asarray(mesh['vertices']);boxes.append(shapely.box(*v[:,:2].min(0),*v[:,:2].max(0)))
    tree=shapely.STRtree(boxes)
    def parents(key):
        if key in reviews:return {p for child in reviews[key]['original_mesh_ids']for p in parents(child)}
        return {key}
    selections=[]
    for item in revision['reopened_shared_surfaces']:
        mesh=old[item['id']]
        # Rebuild active patch coverage from surviving faces after contact revisions.
        patch_tri=mesh_triangles(mesh)
        domain=spatial_union(shapely.polygons(patch_tri[:,:,:2]),grid_size=1e-7)
        ids=parents(mesh['id'])&available
        explicit=set(mesh['selection'].get('mesh_ids',[]))
        for i in tree.query(domain,predicate='dwithin',distance=1e-6):
            observed=fine[i]
            if observed['id']in explicit:
                ids.add(observed['id']);continue
            # Observed roads at the same XY can be an independent bridge.
            # Recover only an existing co-height contact of the active patch.
            tri=mesh_triangles(observed);points=np.asarray(mesh['vertices'])
            observed_tree=shapely.STRtree(shapely.polygons(tri[:,:,:2]))
            hits=observed_tree.query(shapely.points(points[:,:2]),predicate='dwithin',distance=1e-6)
            if hits.size and np.min(abs(plane_z(tri[hits[1]],points[hits[0],:2])-points[hits[0],2]))<.002:
                ids.add(observed['id'])
        for part in shapely.get_parts(domain):
            if part.area==0:continue
            selections.append(dict(mesh['selection'],polygon_xy_m=list(part.exterior.coords),
                holes_xy_m=[list(r.coords)for r in part.interiors],mesh_ids=sorted(ids),
                reopened_patch_id=mesh['id'],boundary_condition='retained_shared_perimeter',
                reason='Rebuild the active shared footprint from its recorded source parents; retain its perimeter, observed contacts and independent crossing levels.'))
    return selections


def revise(roads, before, after, restored, changed):
    previous={m['id']:m for m in before['meshes']}
    revised={m['id']:m for m in after['meshes']}
    source_restored={m['id']:m for m in restored['meshes']}
    reviews={r['id']:r for r in roads.get('shared_inferred_ground',[])}
    previous_closures={(m['from_mesh_id'],m['to_mesh_id']):m for m in before['meshes']
                       if m.get('kind')=='connection_closure'}
    shared=[m for m in roads['meshes']if changed.intersection(m.get('source_lane_ids',[]))
            and m.get('kind')=='inferred_shared_road_ground']
    shared_ids={m['id']for m in shared};by_id={m['id']:m for m in roads['meshes']if m['id']not in shared_ids}
    domains=defaultdict(list)
    def parents(key):
        if key in reviews:
            return {p for child in reviews[key]['original_mesh_ids']for p in parents(child)}
        return {key}
    for mesh in shared:
        domain=spatial_union(shapely.polygons(mesh_triangles(mesh)[:,:,:2]),grid_size=1e-7)
        for key in parents(mesh['id']):domains[key].append(domain)
    print(dict(reopening_shared_surfaces=len(shared),source_parents=len(domains)),flush=True)
    for count,(key,parts) in enumerate(domains.items(),1):
        original=source_restored[key]
        if original.get('kind')in ('av2_boundary_surface','av2_grounded_junction'):
            raise ValueError('Observed geometry unexpectedly owned by an inferred surface')
        domain=shapely.union_all(parts);v=np.asarray(original['vertices'])
        box=shapely.box(*(v[:,:2].min(0)-1.),*(v[:,:2].max(0)+1.))
        fragment=clip_outside(original,box.difference(domain))
        if key in by_id:
            kept=clip_outside(by_id[key],domain);offset=len(kept['vertices'])
            faces=kept['faces']+[[i+offset for i in f]for f in fragment['faces']]
            if faces:
                by_id[key]=dict(kept,vertices=kept['vertices']+fragment['vertices'],faces=faces)
            else:del by_id[key]
        elif fragment['faces']:by_id[key]=fragment
        if count%250==0:print(dict(reopened_source_parents=count,total=len(domains)),flush=True)
    def affected(mesh):
        key=mesh.get('source_lane_id')or ''
        if key in changed:return True
        if key.startswith('bridge:')and changed.intersection(key[7:].split('->')):return True
        if mesh.get('kind')=='connection_closure':
            return any(affected(previous[mesh[s+'_mesh_id']])for s in ('from','to'))
        return False
    updates=[]
    for key,mesh in list(by_id.items()):
        old=(previous_closures.get((mesh['from_mesh_id'],mesh['to_mesh_id']))
             if mesh.get('kind')=='connection_closure' else previous.get(key))
        if old is None or not affected(old):continue
        if not old['faces']:continue
        if old.get('kind')=='connection_closure':
            field,xy_error=closure_height_delta(old,previous,revised)
        else:
            new=revised[key]
            if (old.get('source_lane_id'),old.get('xodr_road_id'))!=(new.get('source_lane_id'),new.get('xodr_road_id')):
                raise ValueError('Resampled native mesh identity changed: '+key)
            field,xy_error=native_height_delta(old,new)
        if np.max(abs(field[:,:,2]))<1e-7:continue
        polygons=shapely.polygons(field[:,:,:2]);tree=shapely.STRtree(polygons)
        xyz=np.asarray(mesh['vertices'],float);points=shapely.points(xyz[:,:2]);ids=tree.nearest(points)
        delta=plane_z(field[ids],xyz[:,:2]);xyz[:,2]+=delta
        replacement=dict(mesh,vertices=xyz.tolist(),height_revision='Source-native profile correction; retained XY clips and source contact offsets preserved')
        for boundary in ('left_boundary','right_boundary'):
            if boundary in mesh:
                values=np.asarray(mesh[boundary],float);owners=tree.nearest(shapely.points(values[:,:2]))
                values[:,2]+=plane_z(field[owners],values[:,:2]);replacement[boundary]=values.tolist()
        by_id[key]=replacement
        updates.append(dict(id=key,source_lane_id=mesh.get('source_lane_id'),maximum_z_change_m=float(abs(delta).max()),native_vertex_xy_error_m=xy_error,
            maximum_distance_from_native_parent_m=float(shapely.distance(points,polygons[ids]).max())))
    result=dict(roads,meshes=list(by_id.values()))
    result['shared_inferred_ground']=[r for r in roads.get('shared_inferred_ground',[])if r['id']not in shared_ids]
    fine={m['id']:m for m in roads['meshes']if m.get('kind')in ('av2_boundary_surface','av2_grounded_junction')}
    if fine!={k:by_id[k]for k in fine}:raise ValueError('Observed physical surface changed')
    review=dict(updated_sources=sorted(changed),updated_meshes=updates,
        reopened_shared_surfaces=[dict(id=m['id'],selection=m['selection'])for m in shared],
        observed_surfaces_exactly_preserved=True,
        scope='Intermediate geometry with dependent shared patches reopened; source-connected shared realization, ground and paint must follow before native installation.')
    return result,review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('roads','previous-native','revised-native','source-restored','height-review','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(path.read_text())
    changed={r['source_lane_id']for r in read(a.height_review)['changed_sources']}
    roads=read(a.roads);print('Loaded current physical surfaces',flush=True)
    before=read(a.previous_native);print('Loaded previous source-native surfaces',flush=True)
    after=read(a.revised_native);print('Loaded revised source-native surfaces',flush=True)
    restored=read(a.source_restored);print('Loaded source restoration',flush=True)
    result,review=revise(roads,before,after,restored,changed)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'road_meshes.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    print(dict(updated_meshes=len(review['updated_meshes']),reopened_shared_surfaces=len(review['reopened_shared_surfaces'])),flush=True)


if __name__=='__main__':main()
