"""Fit a common physical surface across selected same-level road overlaps.

A harmonic height field connects perimeter constraints and fixed AV2 surfaces.
"""
import numpy as np
import shapely
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg, spsolve

from .av2_surfaces import mesh_triangles, remove_zero_area_faces
from scripts.scene.build_assets import surface_heights, plane_heights, BOOLEAN_GRID_M
from .mesh_geometry import spatial_union


def profile_height_conditions(xy, profiles, radius_m):
    """Combine inferred approach profiles on one reviewed physical floor.

    Conflicting coarse heights supply soft evidence for the common surface;
    measured AV2 contacts remain fixed by the shared-ground solver.
    """
    total=np.zeros(len(xy));weighted=np.zeros(len(xy));query=shapely.points(xy)
    for profile in profiles:
        points=np.asarray(profile['points_enu_m'],dtype=float)
        station=np.r_[0.,np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).cumsum()]
        line=shapely.LineString(points[:,:2]);distance=shapely.distance(query,line)
        weight=np.clip(1.-distance/radius_m,0.,1.)**2
        value=np.interp(shapely.line_locate_point(line,query),station,points[:,2])
        total+=weight;weighted+=weight*value
    return total,np.divide(weighted,total,out=np.zeros_like(total),where=total>0)


def surface_laplacian(xy, faces):
    """Assemble triangle FEM stiffness and lumped area using each face's area."""
    tri=xy[faces];x,y=tri[:,:,0],tri[:,:,1]
    twice=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0])
    area=abs(twice)/2
    b=np.stack([y[:,1]-y[:,2],y[:,2]-y[:,0],y[:,0]-y[:,1]],axis=1)
    c=np.stack([x[:,2]-x[:,1],x[:,0]-x[:,2],x[:,1]-x[:,0]],axis=1)
    local=(b[:,:,None]*b[:,None,:]+c[:,:,None]*c[:,None,:])/(4*area[:,None,None])
    rows=np.repeat(faces,3,axis=1).ravel();cols=np.tile(faces,(1,3)).ravel()
    matrix=coo_matrix((local.ravel(),(rows,cols)),shape=(len(xy),len(xy))).tocsr()
    mass=np.zeros(len(xy));np.add.at(mass,faces.ravel(),np.repeat(area/3,3))
    return matrix,mass


def reference_gradient_rhs(xy, faces, meshes):
    """Fit the actual parent tangent field, retaining nonzero road grades."""
    triangles=np.concatenate([mesh_triangles(m)for m in meshes])
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    valid=abs(normal[:,2])>1e-10
    triangles,normal=triangles[valid],normal[valid]
    gradient=-normal[:,:2]/normal[:,2,None]
    centers=xy[faces].mean(1);points=shapely.points(centers)
    polygons=shapely.polygons(triangles[:,:,:2]);tree=shapely.STRtree(polygons)
    pairs=tree.query(points,predicate='dwithin',distance=2*BOOLEAN_GRID_M)
    target=np.zeros((len(faces),2));counts=np.zeros(len(faces))
    np.add.at(target,pairs[0],gradient[pairs[1]]);np.add.at(counts,pairs[0],1)
    covered=counts>0;target[covered]/=counts[covered,None]
    missing=~covered;distance=0.
    if missing.any():
        nearest=tree.nearest(points[missing]);target[missing]=gradient[nearest]
        distance=float(shapely.distance(points[missing],polygons[nearest]).max())
    tri=xy[faces];x,y=tri[:,:,0],tri[:,:,1]
    twice=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0])
    b=np.stack([y[:,1]-y[:,2],y[:,2]-y[:,0],y[:,0]-y[:,1]],axis=1)
    c=np.stack([x[:,2]-x[:,1],x[:,0]-x[:,2],x[:,1]-x[:,0]],axis=1)
    basis=np.stack([b,c],axis=2)/twice[:,None,None]
    local=abs(twice)[:,None]/2*np.sum(basis*target[:,None,:],axis=2)
    rhs=np.zeros(len(xy));np.add.at(rhs,faces.ravel(),local.ravel())
    return rhs,dict(reference_mesh_ids=[m['id']for m in meshes],
        nearest_reference_distance_m=distance,target_grade_max=float(np.linalg.norm(target,axis=1).max()))


def triangulate_domain(domain, spacing_m):
    x0,y0,x1,y1=domain.bounds
    xx,yy=np.meshgrid(np.arange(np.floor(x0/spacing_m)*spacing_m,x1,spacing_m),
                      np.arange(np.floor(y0/spacing_m)*spacing_m,y1,spacing_m))
    cells=shapely.box(xx.ravel(),yy.ravel(),xx.ravel()+spacing_m,yy.ravel()+spacing_m)
    cells=cells[shapely.intersects(cells,domain)]
    pieces=shapely.get_parts(shapely.constrained_delaunay_triangles(shapely.intersection(cells,domain)))
    pieces=pieces[shapely.area(pieces)>1e-12]
    xy=shapely.get_coordinates(pieces).reshape(-1,4,2)[:,:3]
    unique,inverse=np.unique(np.round(xy.reshape(-1,2),8),axis=0,return_inverse=True)
    faces=inverse.reshape(-1,3);tri=unique[faces]
    area2=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    # Rounding can collapse an already tiny polygon/grid intersection. Drop
    # those faces before constructing the height system, then compact nodes.
    faces=faces[abs(area2)>2e-12]
    used,indices=np.unique(faces,return_inverse=True)
    return unique[used],indices.reshape(-1,3)


def clip_outside(mesh,domain):
    if domain.geom_type=='GeometryCollection':
        # Keep polygonal components for surface overlays.
        parts=shapely.get_parts(domain)
        domain=shapely.union_all(parts[shapely.area(parts)>0])
    tri=mesh_triangles(mesh);polygons=shapely.polygons(tri[:,:,:2])
    shapely.prepare(domain)
    hit=shapely.intersects(domain,polygons)
    if not hit.any():return mesh
    # Interior faces vanish exactly. Only boundary-crossing faces need a
    # polygon difference against a potentially large city support domain.
    boundary=hit & ~shapely.covers(domain,polygons)
    cutting=polygons[boundary]
    if len(cutting) and shapely.get_num_coordinates(domain)>2048:
        # Clip batches within their complete triangle bounds.
        bounds=shapely.bounds(cutting)
        cells=np.floor((bounds[:,:2]+bounds[:,2:])/40.).astype(np.int64)
        _,owners=np.unique(cells,axis=0,return_inverse=True)
        remnants=np.empty(len(cutting),dtype=object)
        for group in range(int(owners.max())+1):
            selected=np.flatnonzero(owners==group)
            local=domain.intersection(shapely.box(*bounds[selected,:2].min(0),*bounds[selected,2:].max(0)))
            remnants[selected]=shapely.difference(cutting[selected],local)
    else:
        remnants=shapely.difference(cutting,domain)
    precision=None
    try:
        triangulated=shapely.constrained_delaunay_triangles(remnants)
    except shapely.errors.GEOSException:
        # Normalize collapsed clipping remnants at submicron precision before triangulation.
        repaired=shapely.set_precision(shapely.make_valid(remnants,method='structure',keep_collapsed=False),1e-7)
        precision=dict(grid_m=1e-7,area_change_m2=float(shapely.area(repaired).sum()-shapely.area(remnants).sum()))
        triangulated=shapely.constrained_delaunay_triangles(repaired)
    pieces,owners=shapely.get_parts(triangulated,return_index=True)
    positive=shapely.area(pieces)>1e-12;pieces,owners=pieces[positive],owners[positive]
    xy=shapely.get_coordinates(pieces).reshape(-1,4,2)[:,:3]
    z=plane_heights(np.repeat(tri[boundary][owners],3,axis=0),xy.reshape(-1,2)).reshape(-1,3)
    kept=np.concatenate([tri[~hit],np.concatenate([xy,z[:,:,None]],axis=2)])
    winding=np.cross(kept[:,1]-kept[:,0],kept[:,2]-kept[:,0])[:,2]
    kept[winding<0]=kept[winding<0,::-1]
    result=dict(mesh,vertices=kept.reshape(-1,3).tolist(),faces=np.arange(kept.size//3).reshape(-1,3).tolist())
    if precision:result['clip_precision_repair']=precision
    return result


def remove_cut_boundary_residue(mesh, domain, tolerance_m=2*BOOLEAN_GRID_M):
    """Remove inferred clipping fragments contained in the Boolean precision band.

    Repeated snapped overlays can leave micrometre-wide strips of a previous
    elevated road beside its replacement. Native collision cooking can retain
    those strips. Both their width and every vertex must lie within the known
    cut precision; observed AV2 meshes and other thin geometry stay intact.
    """
    if mesh.get('kind') in ('av2_boundary_surface','av2_grounded_junction') or not mesh['faces']:
        return mesh
    # Observed contacts can leave isolated lines beside the polygonal cut.
    # GeometryCollection.boundary is undefined; use the actual surface cut.
    parts=shapely.get_parts(domain)
    domain=shapely.union_all(parts[shapely.area(parts)>0])
    tri=mesh_triangles(mesh);xy=tri[:,:,:2]
    a,b=xy[:,1]-xy[:,0],xy[:,2]-xy[:,0]
    area2=abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    longest=np.linalg.norm(xy-np.roll(xy,1,axis=1),axis=2).max(1)
    candidates=np.flatnonzero(area2<tolerance_m*longest)
    if not len(candidates):return mesh
    distance=shapely.distance(shapely.points(xy[candidates].reshape(-1,2)),domain.boundary).reshape(-1,3)
    removed=candidates[(distance<=tolerance_m).all(1)]
    if not len(removed):return mesh
    keep=np.ones(len(tri),bool);keep[removed]=False
    result=dict(mesh,faces=np.asarray(mesh['faces'])[keep].tolist())
    result['cut_boundary_residue_cleanup']=dict(tolerance_m=tolerance_m,
        removed_faces=len(removed),removed_xy_area_m2=float(area2[removed].sum()/2),
        removed_3d_area_m2=float(np.linalg.norm(np.cross(tri[removed,1]-tri[removed,0],tri[removed,2]-tri[removed,0]),axis=1).sum()/2))
    return result


def preserve_support_instances(roads, selections):
    """Assign distinct identities to support pieces sharing a patch name.

    Retain source membership and the shared selection alias for each piece.
    """
    reserved={m['id'] for m in roads['meshes']};seen=set();aliases={};meshes=[];repairs=[]
    for index,mesh in enumerate(roads['meshes']):
        key=mesh['id']
        if key in seen:
            unique=key+'_instance_2'
            while unique in reserved:unique+='_'
            reserved.add(unique)
            mesh=dict(mesh,id=unique,legacy_mesh_id=key)
            if mesh.get('source_lane_id')==key:mesh['source_lane_id']=unique
            aliases.setdefault(key,[]).append(unique)
            repairs.append(dict(original_id=key,retained_id=unique,source_mesh_index=index))
        seen.add(key);meshes.append(mesh)
    if not repairs:return roads,selections
    updated=[]
    for selection in selections:
        selection=dict(selection)
        for field in ('mesh_ids','fixed_support_mesh_ids','reconstructed_support_mesh_ids'):
            if field in selection:
                selection[field]=sorted(set(selection[field])|{name for key in selection[field] for name in aliases.get(key,[])})
        updated.append(selection)
    return dict(roads,meshes=meshes,mesh_identity_repairs=roads.get('mesh_identity_repairs',[])+repairs),updated


def reconstructed_support_selected(mesh, selection):
    """Select recorded physical reconstruction without reclassifying raw AV2."""
    if mesh['id'] not in selection.get('reconstructed_support_mesh_ids', []):
        return False
    if mesh.get('kind') != 'av2_grounded_junction' or not mesh.get('road_height_reconstruction'):
        raise ValueError('Selected support lacks recorded physical reconstruction: '+mesh['id'])
    return True


def realize_shared_ground(roads, selections, spacing_m=.5, workers=1):
    """Consume explicit same-level patch selections; preserve the other meshes."""
    roads,selections=preserve_support_instances(roads,selections)
    if workers>1:return realize_disjoint_patches(roads,selections,spacing_m,workers)
    result=dict(roads, meshes=[dict(m) for m in roads['meshes']]);reviews=[]
    # Whole-city feedback can contain hundreds of disjoint junctions. Index
    # original mesh bounds once; each patch replaces only its local entries.
    by_id={m['id']:m for m in result['meshes']}
    originals=list(by_id)
    boxes=[]
    for mesh in result['meshes']:
        v=np.asarray(mesh['vertices'])
        boxes.append(shapely.box(*v[:,:2].min(0),*v[:,:2].max(0)))
    bounds_tree=shapely.STRtree(boxes)
    created={}
    for index,selection in enumerate(selections):
        region=shapely.Polygon(selection['polygon_xy_m'],selection.get('holes_xy_m',[]))
        candidates=[originals[i] for i in bounds_tree.query(region,predicate='intersects')]
        candidates.extend(key for key,box in created.items() if key in by_id and region.intersects(box))
        # Select shared support by source connectivity and layer.
        selected_ids=selection.get('mesh_ids')
        if selected_ids is not None:
            candidates=[key for key in candidates if key in selected_ids]
        original=[by_id[key] for key in candidates if key in by_id];fine=[];mutable=[];retained=[]
        fixed_support_ids=set(selection.get('fixed_support_mesh_ids',[]))
        for m in original:
            if reconstructed_support_selected(m,selection):mutable.append(m)
            elif m.get('kind') in ('av2_boundary_surface','av2_grounded_junction'):fine.append(m)
            elif m['id'] in fixed_support_ids:retained.append(m)
            else:mutable.append(m)
        tri=np.concatenate([mesh_triangles(m) for m in mutable]) if mutable else np.empty((0,3,3))
        polys=shapely.polygons(tri[:,:,:2])
        domain=spatial_union(polys,grid_size=BOOLEAN_GRID_M).intersection(region)
        observed_tri=np.concatenate([mesh_triangles(m) for m in fine]) if fine else np.empty((0,3,3))
        protected=fine+retained
        fine_tri=np.concatenate([mesh_triangles(m) for m in protected]) if protected else np.empty((0,3,3))
        fine_polys=shapely.polygons(fine_tri[:,:,:2]);fine_domain=shapely.union_all(fine_polys)
        # Keep positive-area polygon components from apron cuts.
        parts=shapely.get_parts(domain);positive=shapely.area(parts)>0
        discarded_zero_area_parts=int((~positive).sum())
        domain=shapely.union_all(parts[positive])
        retained_observed_overlap=domain.intersection(fine_domain)
        actual_observed_domain=(shapely.union_all(shapely.polygons(observed_tri[:,:,:2])) if retained else fine_domain)
        actual_observed_overlap=domain.intersection(actual_observed_domain)
        domain=domain.difference(fine_domain)
        source_domain=domain
        gap=float(selection.get('contact_gap_m',0.))
        if gap:
            # Close same-layer gaps against retained AV2/floor contacts to attach height constraints.
            coverage=domain.union(fine_domain.intersection(region))
            closed=coverage.buffer(gap/2,join_style='mitre').buffer(-gap/2,join_style='mitre')
            closed=closed.intersection(domain.buffer(gap))
            scopes=selection.get('contact_gap_regions',[])
            if scopes:
                # Apply the selected contact to its carriageway.
                scope=shapely.union_all([shapely.Polygon(
                    item['polygon_xy_m'],item.get('holes_xy_m',[])) for item in scopes])
                # Clip the added candidate at full precision.
                closed=shapely.set_precision(closed,0).intersection(scope)
                closed=closed.intersection(region).difference(fine_domain)
                domain=shapely.set_precision(domain,0).union(closed)
            else:
                domain=domain.union(closed).intersection(region).difference(fine_domain)
        # Extract polygonal road area before establishing height anchors.
        parts=shapely.get_parts(domain)
        domain=shapely.union_all(parts[shapely.area(parts)>0])
        if domain.is_empty:
            # Remove inferred overlap where canonical source surfaces cover the entire patch.
            replacement_domain=retained_observed_overlap
            ids={m['id'] for m in mutable}
            for m in mutable:
                clipped=clip_outside(m,replacement_domain)
                clipped=remove_cut_boundary_residue(clipped,replacement_domain)
                if clipped['faces']:by_id[m['id']]=clipped
                else:del by_id[m['id']]
            reviews.append(dict(id='retained_source_ground_'+str(index),selection=selection,
                status='retained_existing_source_support',area_m2=0.,vertices=0,triangles=0,
                original_mesh_ids=sorted(ids),replacement_domain_wkt=replacement_domain.wkt,
                retained_source_mesh_ids=[m['id']for m in protected],
                redundant_observed_overlap_removed_m2=actual_observed_overlap.area,
                redundant_retained_support_overlap_removed_m2=retained_observed_overlap.area-actual_observed_overlap.area,
                max_height_change_m=0.,source_xy_coverage_area_error_m2=0.,
                method='Current source floor covers the complete selected inferred area; retain it and clip redundant inferred overlap.'))
            print(dict(completed_patches=index+1,total_patches=len(selections),
                status='retained_existing_source_support',source_junctions=selection.get('source_junctions'),
                redundant_overlap_area_m2=replacement_domain.area),flush=True)
            continue
        xy,faces=triangulate_domain(domain,spacing_m)
        support=np.concatenate([tri,fine_tri])
        support_tree=shapely.STRtree(shapely.polygons(support[:,:,:2]))
        if gap:
            covered=np.zeros(len(xy),dtype=bool)
            covered[support_tree.query(shapely.points(xy),predicate='dwithin',distance=2*BOOLEAN_GRID_M)[0]]=True
            z=np.empty(len(xy))
            z[covered]=surface_heights(support,support_tree,xy[covered],'reviewed road patch support')
            nearest=support_tree.nearest(shapely.points(xy[~covered]))
            z[~covered]=plane_heights(support[nearest],xy[~covered])
        else:
            z=surface_heights(support,support_tree,xy,'reviewed road patch support')
        fixed_outline=(region.boundary if selection.get('boundary_condition')=='source_approach_cuts'
                       else domain.boundary)
        # Use the Boolean grid precision for contact tests.
        cut_tolerance_m=2*BOOLEAN_GRID_M
        boundary=shapely.distance(shapely.points(xy),fixed_outline)<=cut_tolerance_m
        exterior_cut_vertices=0
        exterior_cut_height_change_m=0.
        if selection.get('boundary_condition')=='source_approach_cuts':
            # Anchor the local solve to triangles extending outside the replacement domain.
            exterior=~shapely.covers(region.buffer(cut_tolerance_m),
                shapely.polygons(support[:,:,:2]))
            cut_ids=np.flatnonzero(boundary)
            pairs=shapely.STRtree(shapely.polygons(support[exterior,:,:2])).query(
                shapely.points(xy[cut_ids]),predicate='dwithin',distance=cut_tolerance_m)
            retained_cut_z=np.full(len(cut_ids),-np.inf)
            if pairs.shape[1]:
                heights=plane_heights(support[exterior][pairs[1]],xy[cut_ids[pairs[0]]])
                np.maximum.at(retained_cut_z,pairs[0],heights)
            supported=np.isfinite(retained_cut_z)
            boundary[cut_ids[~supported]]=False
            if supported.any():
                ids=cut_ids[supported]
                exterior_cut_height_change_m=float(np.max(abs(z[ids]-retained_cut_z[supported])))
                z[ids]=retained_cut_z[supported]
                exterior_cut_vertices=len(ids)
        # Anchor observed contacts to AV2 and other contacts to the existing upper surface.
        # Use the shared precision bound for source-edge queries.
        fine_pairs=shapely.STRtree(fine_polys).query(shapely.points(xy),predicate='dwithin',distance=cut_tolerance_m)
        observed=np.unique(fine_pairs[0])
        if len(observed):z[observed]=surface_heights(fine_tri,shapely.STRtree(fine_polys),xy[observed],'observed patch perimeter')
        boundary[observed]=True
        actual_observed_vertices=(np.unique(shapely.STRtree(shapely.polygons(observed_tri[:,:,:2])).query(
            shapely.points(xy),predicate='dwithin',distance=cut_tolerance_m)[0]) if retained else observed)
        if selection.get('boundary_condition')!='source_approach_cuts' and not boundary.any():
            raise ValueError('Shared road support has no retained perimeter height anchors')
        edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        edges=edges[edges[:,0]!=edges[:,1]]
        lengths=np.linalg.norm(xy[edges[:,0]]-xy[edges[:,1]],axis=1)
        weights=1/np.maximum(lengths,.1)
        graph=coo_matrix((np.tile(weights,2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(xy),len(xy))).tocsr()
        degree=np.asarray(graph.sum(axis=1)).ravel()
        # Preserve fitted bridge and underpass profiles in the shared-surface solve.
        screen=float(selection.get('profile_screening_weight',0.))
        laplacian=diags(degree*(1+screen))-graph
        if selection.get('height_energy')=='triangle_fem':
            stiffness,_=surface_laplacian(xy,faces)
            laplacian=stiffness+diags(screen*degree)
        profiles=selection.get('source_height_profiles',[])
        profile_weight=np.zeros(len(xy));profile_target=np.zeros(len(xy))
        if profiles:
            support_weight,profile_target=profile_height_conditions(xy,profiles,
                selection['profile_support_radius_m'])
            stiffness,mass=surface_laplacian(xy,faces)
            profile_weight=mass*selection['profile_anchor_weight']/spacing_m**2*support_weight
            laplacian=stiffness+diags(profile_weight+screen*degree)
        profile_anchored_components=[]
        if selection.get('boundary_condition')=='source_approach_cuts':
            from scipy.sparse.csgraph import connected_components
            count,labels=connected_components(graph,directed=False)
            isolated_support=[]
            for component in set(range(count))-set(labels[boundary]):
                ids=np.flatnonzero(labels==component)
                anchored=ids[profile_weight[ids]>0]
                if len(anchored):
                    # Use source approaches as height anchors for floors enclosed by the solve region.
                    profile_anchored_components.append(dict(
                        vertices=len(ids),source_anchored_vertices=len(anchored),
                        total_profile_weight=float(profile_weight[anchored].sum())))
                    continue
                # Retain the heights of isolated end caps with one supporting surface.
                pairs=support_tree.query(shapely.points(xy[ids]),predicate='dwithin',distance=2*BOOLEAN_GRID_M)
                values=plane_heights(support[pairs[1]],xy[ids[pairs[0]]])
                if len(np.unique(pairs[0]))!=len(ids) or np.any(abs(values-z[ids[pairs[0]]])>1e-6):
                    raise ValueError('Selected road support contains conflicting unanchored surfaces: '+str(dict(
                        component=component,vertices=len(ids),covered_vertices=len(np.unique(pairs[0])),
                        maximum_height_difference_m=float(abs(values-z[ids[pairs[0]]]).max()) if len(values) else None)))
                boundary[ids]=True
                isolated_support.append(dict(vertices=len(ids),bounds_xy_m=[xy[ids].min(0).tolist(),xy[ids].max(0).tolist()],
                    maximum_candidate_height_difference_m=float(abs(values-z[ids[pairs[0]]]).max()),action='Retain the existing isolated single-surface heights'))
        else:isolated_support=[]
        gradient_rhs=np.zeros(len(xy));gradient_reference=None
        gradient_ids=selection.get('height_gradient_reference_mesh_ids',[])
        if gradient_ids:
            if selection.get('height_energy')!='triangle_fem':raise ValueError('Parent gradient reference requires triangle FEM')
            gradient_rhs,gradient_reference=reference_gradient_rhs(xy,faces,[m for m in original if m['id']in gradient_ids])
        free=np.flatnonzero(~boundary);system=laplacian[free][:,free]
        rhs=(gradient_rhs[free]+screen*degree[free]*z[free]+profile_weight[free]*profile_target[free]
             -laplacian[free][:,boundary]@z[boundary])
        solver=selection.get('height_solver','cg')
        if solver=='direct':
            # Use sparse factorization to retain fixed contacts on thin triangles.
            heights=spsolve(system.tocsc(),rhs)
            if not np.isfinite(heights).all():raise RuntimeError('Shared road ground has no finite direct solution')
        elif solver=='cg':
            heights,status=cg(system,rhs,x0=z[free],rtol=1e-9,atol=1e-9,maxiter=3000,M=diags(1/system.diagonal()))
            if status:raise RuntimeError('Shared road ground did not converge: '+str(status))
        else:raise ValueError('Unknown shared road height solver: '+solver)
        before=z.copy();z[free]=heights
        vertices=np.c_[xy,z];normal=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])[:,2]
        faces[normal<0]=faces[normal<0,::-1]
        ids={m['id'] for m in mutable}
        replacement_domain=domain.union(retained_observed_overlap)
        for m in mutable:
            clipped=clip_outside(m,replacement_domain)
            clipped=remove_cut_boundary_residue(clipped,replacement_domain)
            if clipped['faces']:by_id[m['id']]=clipped
            else:del by_id[m['id']]
        key='inferred_shared_ground_'+str(index)
        while key in by_id or key in originals:key+='_' # Preserve existing support identities.
        by_id[key]=dict(id=key,kind='inferred_shared_road_ground',source_lane_id=key,
            source_lane_ids=sorted({source for m in mutable
                for source in (m.get('source_lane_ids')or [m.get('source_lane_id')])if source}),
            vertices=vertices.tolist(),faces=faces.tolist(),selection=selection,
            support_domain_wkt=domain.wkt,
            replacement_domain_wkt=replacement_domain.wkt,
            physical_role='Single inferred same-level support; original lane graph and AV2 observations retained')
        created[key]=shapely.box(*domain.bounds)
        report=dict(id=key,selection=selection,discarded_zero_area_contact_parts=discarded_zero_area_parts,area_m2=domain.area,vertices=len(xy),triangles=len(faces),
            retained_isolated_support=isolated_support,
            profile_anchored_components=profile_anchored_components,
            original_mesh_ids=sorted(ids),observed_contact_vertices=len(actual_observed_vertices),fixed_perimeter_vertices=int(boundary.sum()),retained_exterior_cut_vertices=exterior_cut_vertices,retained_exterior_cut_height_change_m=exterior_cut_height_change_m,cut_anchor_tolerance_m=cut_tolerance_m,
            fixed_support_mesh_ids=[m['id']for m in retained],fixed_support_contact_vertices=len(observed),observed_contact_tolerance_m=cut_tolerance_m,
            reconstructed_support_mesh_ids=[m['id'] for m in mutable if reconstructed_support_selected(m,selection)],
            redundant_observed_overlap_removed_m2=actual_observed_overlap.area,
            redundant_retained_support_overlap_removed_m2=retained_observed_overlap.area-actual_observed_overlap.area,
            max_height_change_m=float(abs(z-before).max()),source_xy_coverage_area_error_m2=abs(shapely.area(shapely.polygons(vertices[faces][:,:,:2])).sum()-domain.area),
            source_contact_gap_fill_area_m2=domain.difference(source_domain).area,
            boundary_condition=selection.get('boundary_condition','entire_perimeter'),
            source_gradient_reference=gradient_reference,source_height_profile_count=len(profiles),height_solver=solver,height_energy=selection.get('height_energy','edge_graph'),
            source_height_profile_scope='Joint soft inferred heights on the reviewed floor; observed AV2, explicitly retained support contacts and selected perimeter remain fixed.',
            method=('Linear triangle finite-element height field with area-weighted inferred profiles, fixed source approach contacts and measured AV2 support.' if profiles or selection.get('height_energy')=='triangle_fem' else
                    'Harmonic height field on a clipped regular grid; fixed retained approach contacts and measured AV2 contact, with source-selected physical-layer membership.'))
        reviews.append(report)
        print(dict(completed_patches=index+1,total_patches=len(selections),area_m2=domain.area,
                   max_height_change_m=report['max_height_change_m']),flush=True)
    result['meshes']=list(by_id.values())
    retained={m['id'] for m in result['meshes']}
    if 'connection_closure' in result:
        closure=dict(result['connection_closure']);closure['patches']=[p for p in closure['patches'] if p['id'] in retained];closure['patch_count']=len(closure['patches'])
        result['connection_closure']=closure
    result['shared_inferred_ground']=roads.get('shared_inferred_ground',[])+reviews
    remove_zero_area_faces(result)
    return result,reviews


def realize_disjoint_patches(roads,selections,spacing_m,workers):
    """Solve independent selected junctions concurrently; clip shared meshes once per result."""
    from concurrent.futures import ThreadPoolExecutor
    regions=[shapely.Polygon(s['polygon_xy_m'],s.get('holes_xy_m',[])) for s in selections]
    pairs=shapely.STRtree(regions).query(regions,predicate='intersects')
    if any(i<j and regions[i].intersection(regions[j]).area>1e-8
           and (selections[i].get('mesh_ids')is None or selections[j].get('mesh_ids')is None
                or set(selections[i]['mesh_ids'])&set(selections[j]['mesh_ids']))for i,j in pairs.T):
        raise ValueError('Merge overlapping selected support regions before concurrent realization')
    originals=roads['meshes'];boxes=[]
    for mesh in originals:
        v=np.asarray(mesh['vertices']);boxes.append(shapely.box(*v[:,:2].min(0),*v[:,:2].max(0)))
    tree=shapely.STRtree(boxes)
    def solve(item):
        index,selection=item
        local=[originals[i] for i in tree.query(regions[index],predicate='intersects')]
        result,reviews=realize_shared_ground(dict(meshes=local),[selection],spacing_m)
        review=reviews[0];surface=next((m for m in result['meshes'] if m['id']==review['id']),None)
        return index,surface,review
    by_id={m['id']:dict(m) for m in originals};reserved=set(by_id);reviews=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,surface,review in pool.map(solve,enumerate(selections)):
            domain=shapely.from_wkt(surface['replacement_domain_wkt'] if surface is not None else review['replacement_domain_wkt'])
            for key in review['original_mesh_ids']:
                if key not in by_id:continue
                clipped=clip_outside(by_id[key],domain)
                clipped=remove_cut_boundary_residue(clipped,domain)
                if clipped['faces']:by_id[key]=clipped
                else:del by_id[key]
            if surface is not None:
                key='inferred_shared_ground_'+str(index)
                while key in by_id or key in reserved:key+='_'
                surface.update(id=key,source_lane_id=key);review['id']=key
                by_id[key]=surface
            else:review['id']='retained_source_ground_'+str(index)
            reviews.append(review)
            print(dict(completed_city_patches=len(reviews),total_patches=len(selections),workers=workers),flush=True)
    result=dict(roads,meshes=list(by_id.values()),shared_inferred_ground=roads.get('shared_inferred_ground',[])+reviews)
    if 'connection_closure' in roads:
        closure=dict(roads['connection_closure']);closure['patches']=[p for p in closure['patches'] if p['id'] in by_id]
        closure['patch_count']=len(closure['patches']);result['connection_closure']=closure
    remove_zero_area_faces(result)
    return result,reviews
