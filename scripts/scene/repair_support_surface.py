"""Fit occluded ground within a selected single-level region using local LiDAR."""
import xml.etree.ElementTree as ET
import numpy as np
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree


def close_surface_edges(terrain, roads, road_footprint=None, nearest_terrain_layer=False,
                        road_boundary_offsets_m=None):
    """Build skirts from the carved terrain boundary."""
    from scripts.scene.build_assets import surface_heights, BOOLEAN_GRID_M
    from map_generation.mesh_geometry import spatial_union, boundary_segments
    triangles=np.concatenate([np.array(m['vertices'])[m['faces']] for m in roads['meshes']])
    road_polys=shapely.polygons(triangles[:,:,:2]);road_tree=shapely.STRtree(road_polys)
    offsets = road_boundary_offsets_m or {}
    # Reconstruct the recorded carve footprint at the common mesh precision.
    if offsets:
        cut_shapes = []
        for mesh in roads['meshes']:
            support = np.asarray(mesh['vertices'])[mesh['faces']]
            shapes = shapely.polygons(support[:, :, :2])
            offset = offsets.get(mesh['id'], 0.)
            cut_shapes.extend(shapely.buffer(shapes[shapely.area(shapes) > 1e-12], offset,
                                            join_style='mitre', mitre_limit=5.)
                              if offset else shapes)
        match_polys = np.asarray(cut_shapes, dtype=object)
    else:
        match_polys = road_polys
    footprint=(road_footprint if road_footprint is not None else shapely.set_precision(
        shapely.set_precision(spatial_union(match_polys, grid_size=BOOLEAN_GRID_M), BOOLEAN_GRID_M), 0))
    boundary=footprint.boundary
    boundary_tree=shapely.STRtree(boundary_segments(boundary))
    vertices=np.array(terrain['vertices']);faces=np.array(terrain['faces'])
    top=vertices[faces,:2]
    ab,ac=top[:,1]-top[:,0],top[:,2]-top[:,0]
    signed_area2=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
    longest=np.maximum.reduce([np.linalg.norm(ab,axis=1),np.linalg.norm(ac,axis=1),
                               np.linalg.norm(top[:,2]-top[:,1],axis=1)])
    # Exclude clipping slivers below the ground-interior resolution.
    resolved=(abs(signed_area2)>BOOLEAN_GRID_M*longest)
    excluded=int((~resolved).sum())
    # Retain boundary triangles adjoining a resolved ground interior.
    ground=shapely.union_all(shapely.polygons(top[resolved]))
    shapely.prepare(ground)
    faces=faces.copy()
    reversed_top=signed_area2<0
    faces[reversed_top]=faces[reversed_top][:,[0,2,1]]
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    parent_resolved=np.tile(resolved,3)
    # The carved terrain and the reproduced road union each underwent the
    # common Boolean grid operation; their combined matching bound is two grids.
    match_tolerance=2*BOOLEAN_GRID_M
    near=np.zeros(len(vertices),dtype=bool)
    near[boundary_tree.query(shapely.points(vertices[:,:2]),predicate='dwithin',distance=match_tolerance)[0]]=True
    selected=near[edges].all(axis=1)
    edges=edges[selected];parent_resolved=parent_resolved[selected]
    middle=vertices[edges,:2].mean(axis=1)
    on_boundary=np.zeros(len(edges),dtype=bool)
    on_boundary[boundary_tree.query(shapely.points(middle),predicate='dwithin',distance=match_tolerance)[0]]=True
    edges=edges[on_boundary];parent_resolved=parent_resolved[on_boundary]
    a,b=vertices[edges[:,0],:2],vertices[edges[:,1],:2]
    direction=b-a;length=np.linalg.norm(direction,axis=1)
    nonzero=length>1e-10
    edges=edges[nonzero];parent_resolved=parent_resolved[nonzero]
    a,b,direction,length=a[nonzero],b[nonzero],direction[nonzero],length[nonzero]
    inward=np.c_[-direction[:,1],direction[:,0]]/length[:,None]
    probes=a[:,None,:]+direction[:,None,:]*np.array([.25,.5,.75])[None,:,None]
    probes+=inward[:,None,:]*(2*BOOLEAN_GRID_M)
    supported=shapely.covers(ground,shapely.points(probes.reshape(-1,2))).reshape(-1,3).all(axis=1)
    attached_thin_edges=int((supported & ~parent_resolved).sum())
    unsupported_edges=int((~supported).sum())
    edges=edges[supported]
    # Clipped triangles can carry duplicate indices at an identical endpoint.
    unique={}
    for a,b in edges:
        key=tuple(sorted((tuple(np.round(vertices[a],8)),tuple(np.round(vertices[b],8)))))
        unique[key]=(vertices[a],vertices[b])
    starts=[];ends=[]
    for a,b in unique.values():
        count=max(1,int(np.ceil(np.linalg.norm(b[:2]-a[:2]))))
        samples=a+(b-a)*np.linspace(0,1,count+1)[:,None]
        starts.extend(samples[:-1]);ends.extend(samples[1:])
    starts=np.array(starts);ends=np.array(ends)
    # Combine outline and face precision bounds for height lookup.
    support_tolerance=match_tolerance+2*BOOLEAN_GRID_M
    def support(points):
        if not nearest_terrain_layer:
            return surface_heights(triangles,road_tree,points[:,:2],'curb closure',coverage_tolerance_m=support_tolerance)
        from map_generation.av2_surfaces import plane_z
        # Mitre corners in the recorded carve extend up to five offsets.
        pairs=road_tree.query(shapely.points(points[:,:2]),predicate='dwithin',distance=support_tolerance + 5. * max(offsets.values(), default=0.))
        values=plane_z(triangles[pairs[1]],points[pairs[0],:2])
        order=np.lexsort((abs(values-points[pairs[0],2]),pairs[0]))
        ids,first=np.unique(pairs[0][order],return_index=True)
        result=np.full(len(points),np.nan);result[ids]=values[order[first]]
        if not np.isfinite(result).all():raise ValueError('Terrain boundary has no source road layer support')
        return result
    za=support(starts);zb=support(ends)
    quads=np.stack([np.c_[starts[:,:2],za],np.c_[ends[:,:2],zb],ends,starts],axis=1)
    flat=quads.reshape(-1,3);index=np.arange(len(flat)).reshape(-1,4)
    faces=np.concatenate([index[:,[0,1,2]],index[:,[0,2,3]]])
    normals=np.cross(flat[faces[:,1]]-flat[faces[:,0]],flat[faces[:,2]]-flat[faces[:,0]])
    # The directed top edge has ground on its left. Face the actual road
    # free space on its right, even when the ground lies below the road.
    outward=np.c_[ends[:,1]-starts[:,1],starts[:,0]-ends[:,0]]
    flip=(normals[:,:2]*np.concatenate([outward,outward])).sum(axis=1)<0
    faces[flip]=faces[flip][:,[0,2,1]]
    valid=np.linalg.norm(normals,axis=1)>1e-10
    terrain['seam_vertices']=flat.tolist();terrain['seam_faces']=faces[valid].tolist()
    terrain['report']['final_surface_edge_closure']={
        'terrain_boundary_segments':len(unique),'subdivided_segments':len(quads),
        'method':'Resolved final terrain boundary endpoints and linear edge subdivision; native road height at identical XY; winding faces out of the terrain into road free space.',
        'unresolved_planar_parent_faces_excluded':excluded,
        'minimum_parent_altitude_m':BOOLEAN_GRID_M,
        'curb_faces_reoriented':int(flip.sum()),
        'boundary_match_precision_m':match_tolerance,
        'inward_resolved_ground_probe_distance_m':2*BOOLEAN_GRID_M,
        'thin_boundary_edges_with_resolved_interior':attached_thin_edges,
        'unsupported_boundary_edges_rejected':unsupported_edges,
        'source_clearance_boundary_offsets_m':offsets,
        'clearance_buffer_mitre_limit':5.,
        'support_search_radius_m':support_tolerance + 5. * max(offsets.values(), default=0.),
        'support_precision_basis':'terrain-to-outline matching bound plus outline-to-original-face rounding bound; source XY and triangle planes retained',
        'terrain_endpoint_gap_m':0.,'road_layer_selection':'nearest terrain edge height' if nearest_terrain_layer else 'upper road surface'}


def repair(scene, roads, points, selection):
    root=ET.parse(selection['osm']).getroot()
    way=next(w for w in root.findall('way') if w.get('id')==selection['osm_way'])
    refs=[n.get('ref') for n in way.findall('nd')]
    nodes={n.get('id'):(float(n.get('lon')),float(n.get('lat'))) for n in root.findall('node')}
    transform=Transformer.from_crs('EPSG:4326',scene['coordinates']['horizontal_crs'],always_xy=True)
    polygon=shapely.Polygon(np.array([transform.transform(*nodes[n]) for n in refs])-scene['coordinates']['origin_utm_xy'])
    roadverts=np.concatenate([np.array(m['vertices']) for m in roads['meshes']])
    roadtree=cKDTree(roadverts[:,:2]);area=shapely.contains(polygon,shapely.points(points[:,:2]))
    pts=points[area];nominal=roadverts[roadtree.query(pts[:,:2])[1],2]
    pts=pts[abs(pts[:,2]-nominal)<1.]
    cells=np.floor(pts[:,:2]).astype(int);keys,ids=np.unique(cells,axis=0,return_inverse=True)
    anchors=[]
    for index,key in enumerate(keys):
        values=pts[ids==index]
        if len(values)>=15:
            anchors.append([*np.median(values[:,:2],axis=0),np.quantile(values[:,2],.15)])
    anchors=np.array(anchors);center=anchors[:,:2].mean(0)
    design=np.c_[anchors[:,:2]-center,np.ones(len(anchors))]
    coef=np.linalg.lstsq(design,anchors[:,2],rcond=None)[0]
    for _ in range(10):
        residual=anchors[:,2]-design@coef
        weight=np.minimum(1.,.08/np.maximum(abs(residual),1e-6))
        coef=np.linalg.lstsq(design*weight[:,None],anchors[:,2]*weight,rcond=None)[0]
    residual=abs(anchors[:,2]-design@coef)
    report={'source':selection,'support_cells':len(anchors),'plane_center_xy_m':center.tolist(),
            'plane_coefficients':coef.tolist(),'anchor_abs_residual_m_quantiles':np.quantile(residual,[.5,.9,1]).tolist(),
            'method':'One-meter cells, lower static LiDAR height population, robust local plane; occluded ground interpolated within reviewed single-level source polygon.'}
    if len(anchors)<30 or np.quantile(residual,.9)>.3:
        raise ValueError('Ground does not support the reviewed single-level approximation: '+str(report))
    verts=np.array(scene['terrain']['vertices']);xy=shapely.points(verts[:,:2])
    active=shapely.contains(polygon,xy);ids=np.flatnonzero(active)
    triangles=np.concatenate([np.array(m['vertices'])[m['faces']] for m in roads['meshes']])
    roadunion=shapely.union_all(shapely.polygons(triangles[:,:,:2]))
    weight=np.clip(shapely.distance(xy[ids],polygon.boundary)/1.,0,1)*np.clip(shapely.distance(xy[ids],roadunion)/.75,0,1)
    target=np.c_[verts[ids,:2]-center,np.ones(len(ids))]@coef
    delta=(target-verts[ids,2])*weight;verts[ids,2]+=delta
    scene['terrain']['vertices']=verts.tolist()
    report.update(vertices_changed=int((weight>0).sum()),max_abs_height_change_m=float(abs(delta).max()),
                  scope='Repaired terrain support only; existing roads unchanged. Plane is inferred beneath occluding vehicles, not directly observed there.')
    return report
