"""Transfer a source height correction through existing physical road surfaces."""
from collections import defaultdict

import numpy as np
import shapely
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg

from map_generation.av2_surfaces import mesh_triangles,plane_z
from scripts.scene.revise_physical_profiles import native_height_delta,closure_height_delta


def sample_changes(mesh,base,delta):
    points=np.asarray(mesh['vertices'],float)
    tree=shapely.STRtree(shapely.polygons(base[:,:,:2]))
    hits=tree.query(shapely.points(points[:,:2]),predicate='dwithin',distance=1e-6)
    owners=np.full(len(points),-1,dtype=int)
    if hits.size:
        residual=abs(plane_z(base[hits[1]],points[hits[0],:2])-points[hits[0],2])
        order=np.lexsort((residual,hits[0]));hits=hits[:,order]
        first=np.r_[True,np.diff(hits[0])!=0];owners[hits[0,first]]=hits[1,first]
    missing=owners<0
    owners[missing]=tree.nearest(shapely.points(points[missing,:2]))
    values=plane_z(delta[owners],points[:,:2])
    distance=shapely.distance(shapely.points(points[:,:2]),tree.geometries[owners])
    return values,dict(maximum_native_support_distance_m=float(distance.max(initial=0.)),
                       extrapolated_vertices=int(missing.sum()))


def smooth_changes(mesh,values,fine,screen=.02):
    vertices,first,inverse=np.unique(np.asarray(mesh['vertices'],float),axis=0,return_index=True,return_inverse=True)
    faces=inverse[np.asarray(mesh['faces'],int)];values=values[first]
    used=np.unique(faces)
    all_edges=np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    edges,counts=np.unique(all_edges,axis=0,return_counts=True)
    fixed=np.ones(len(vertices),bool);fixed[used]=False
    fixed[np.unique(edges[counts==1])]=True
    observed=np.zeros(len(vertices),bool)
    for surface in fine:
        tri=mesh_triangles(surface);tree=shapely.STRtree(shapely.polygons(tri[:,:,:2]))
        hits=tree.query(shapely.points(vertices[:,:2]),predicate='dwithin',distance=1e-6)
        if hits.size:
            close=abs(plane_z(tri[hits[1]],vertices[hits[0],:2])-vertices[hits[0],2])<.002
            observed[hits[0,close]]=True
    fixed[observed]=True;values=values.copy();values[observed]=0.
    lengths=np.linalg.norm(vertices[edges[:,0],:2]-vertices[edges[:,1],:2],axis=1)
    weights=1/np.maximum(lengths,.1)
    matrix=coo_matrix((np.tile(weights,2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    degree=np.asarray(matrix.sum(axis=1)).ravel()
    laplacian=diags(degree*(1+screen))-matrix;free=np.flatnonzero(~fixed)
    system=laplacian[free][:,free]
    rhs=screen*degree[free]*values[free]-laplacian[free][:,fixed]@values[fixed]
    result,status=cg(system,rhs,x0=values[free],rtol=1e-9,atol=1e-10,maxiter=3000,M=diags(1/system.diagonal()))
    if status:raise RuntimeError('Shared height correction did not converge: '+str(status))
    values[free]=result
    return values[inverse],dict(fixed_perimeter_vertices=int(fixed.sum()),observed_contact_vertices=int(observed.sum()),
                       correction_screening_weight=screen)


def deform(roads,before,after,changed,shared_correction_screening_weight=.02):
    old={m['id']:m for m in before['meshes']};new={m['id']:m for m in after['meshes']}
    closures={(m['from_mesh_id'],m['to_mesh_id']):m for m in before['meshes']if m.get('kind')=='connection_closure'}
    by_source=defaultdict(list)
    for mesh in before['meshes']:
        if mesh.get('kind')=='connection_closure' or not mesh['faces']:continue
        by_source[mesh.get('source_lane_id')].append(mesh)
    def affected(mesh):
        key=mesh.get('source_lane_id')or ''
        return key in changed or (key.startswith('bridge:')and bool(changed.intersection(key[7:].split('->'))))
    cache={}
    def field(mesh):
        if mesh['id']not in cache:
            base=mesh_triangles(mesh)
            if mesh.get('kind')=='connection_closure':
                delta,_=closure_height_delta(mesh,old,new)
                tri=base;tree=shapely.STRtree(shapely.polygons(tri[:,:,:2]))
                xy=delta[:,:,:2].reshape(-1,2);owners=tree.nearest(shapely.points(xy))
                base=delta.copy();base[:,:,2]=plane_z(tri[owners],xy).reshape(-1,3)
            else:delta,_=native_height_delta(mesh,new[mesh['id']])
            cache[mesh['id']]=(base,delta)
        return cache[mesh['id']]
    fine=[m for m in roads['meshes']if m.get('kind')in ('av2_boundary_surface','av2_grounded_junction')]
    bounds=[]
    for m in fine:
        v=np.asarray(m['vertices']);bounds.append(shapely.box(*v[:,:2].min(0),*v[:,:2].max(0)))
    fine_tree=shapely.STRtree(bounds)
    result=[];records=[]
    for mesh in roads['meshes']:
        if mesh.get('kind')in ('av2_boundary_surface','av2_grounded_junction'):
            result.append(mesh);continue
        shared=mesh.get('kind')=='inferred_shared_road_ground'
        if shared:
            sources=set(mesh.get('source_lane_ids',[]))
            if not sources&changed:result.append(mesh);continue
            parents={m['id']:m for key in sources for m in by_source[key]}
            fields=[field(m)for m in parents.values()]
            if not fields:raise ValueError('Shared surface lacks source-native height fields: '+mesh['id'])
            base=np.concatenate([f[0]for f in fields]);delta=np.concatenate([f[1]for f in fields])
        else:
            parent=closures.get((mesh.get('from_mesh_id'),mesh.get('to_mesh_id')))if mesh.get('kind')=='connection_closure'else old.get(mesh['id'])
            if parent is None or not parent['faces']:
                result.append(mesh);continue
            selected=any(affected(old[parent[s+'_mesh_id']])for s in ('from','to'))if parent.get('kind')=='connection_closure'else affected(parent)
            if not selected:result.append(mesh);continue
            base,delta=field(parent)
        values,diagnostic=sample_changes(mesh,base,delta)
        if shared:
            v=np.asarray(mesh['vertices']);box=shapely.box(*v[:,:2].min(0),*v[:,:2].max(0))
            values,contacts=smooth_changes(mesh,values,[fine[i]for i in fine_tree.query(box,predicate='intersects')],shared_correction_screening_weight)
            diagnostic.update(contacts)
        vertices=np.asarray(mesh['vertices'],float).copy();vertices[:,2]+=values
        revised=dict(mesh,vertices=vertices.tolist(),height_revision='Source height delta on the retained physical footprint and triangulation')
        for side in ('left_boundary','right_boundary'):
            if side in mesh:
                boundary=np.asarray(mesh[side],float).copy()
                shift,_=sample_changes(dict(vertices=boundary),base,delta);boundary[:,2]+=shift;revised[side]=boundary.tolist()
        result.append(revised)
        records.append(dict(id=mesh['id'],shared=shared,maximum_z_change_m=float(abs(values).max(initial=0.)),**diagnostic))
        if len(records)%250==0:print(dict(updated_physical_meshes=len(records)),flush=True)
    return dict(roads,meshes=result),dict(records=records,observed_AV2_geometry_unchanged=True,
        all_XY_faces_and_physical_coverage_unchanged=True,
        method='Transfer native source height changes; smooth the correction on existing shared triangles with fixed perimeter changes and zero change at observed co-height contacts. Actual clearance and native driving follow.')
