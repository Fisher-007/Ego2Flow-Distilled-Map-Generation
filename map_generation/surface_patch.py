"""Select complete existing road tiles touched by a reviewed geometry revision."""
from collections import Counter
import numpy as np


def refresh_edge_support(terrain, previous_roads, revised_roads, tolerance_m=.002,
                         source_height_matched_vertices=False):
    """Move road ends of existing closure quads with their identified floor.

    Closure quads store two road endpoints followed by two terrain endpoints.
    Road XY and the terrain endpoints stay fixed. Previous height identifies
    the affected floor where bridges share the same plan-view position.
    """
    import shapely
    from .av2_surfaces import mesh_triangles, plane_z
    points=np.asarray(terrain['seam_vertices'],dtype=float)
    if not source_height_matched_vertices and len(points)%4:
        raise ValueError('Expected the existing four-vertex closure layout')
    old=np.concatenate([mesh_triangles(m)for m in previous_roads['meshes']if m['faces']])
    new=np.concatenate([mesh_triangles(m)for m in revised_roads['meshes']if m['faces']])
    old_tree=shapely.STRtree(shapely.polygons(old[:,:,:2]))
    new_tree=shapely.STRtree(shapely.polygons(new[:,:,:2]))
    candidates=(np.arange(len(points)) if source_height_matched_vertices
                else np.flatnonzero(np.arange(len(points))%4<2))
    pairs=old_tree.query(shapely.points(points[candidates,:2]),predicate='dwithin',distance=1e-7)
    heights=plane_z(old[pairs[1]],points[candidates[pairs[0]],:2])
    matched=np.unique(candidates[pairs[0][abs(heights-points[candidates[pairs[0]],2])<=tolerance_m]])
    pairs=new_tree.query(shapely.points(points[matched,:2]),predicate='dwithin',distance=1e-7)
    heights=plane_z(new[pairs[1]],points[matched[pairs[0]],:2])
    order=np.lexsort((abs(heights-points[matched[pairs[0]],2]),pairs[0]))
    ids,first=np.unique(pairs[0][order],return_index=True)
    indices=matched[ids];target=heights[order[first]];delta=target-points[indices,2]
    changed=abs(delta)>1e-7;indices=indices[changed];delta=delta[changed]
    points[indices,2]+=delta
    terrain['seam_vertices']=points.tolist()
    return dict(changed_vertex_indices=indices.tolist(),changed_endpoints=len(indices),
        maximum_height_change_m=float(abs(delta).max())if len(delta)else 0.,
        terrain_endpoints_and_all_XY_preserved=not source_height_matched_vertices,
        endpoint_selection='source floor height after triangulated clearance' if source_height_matched_vertices else 'original closure quad road endpoints',
        method='Match previous road-floor height at each existing closure endpoint, then use revised support at the same XY.')


def triangle_cells(mesh, size):
    return np.floor(np.asarray(mesh['vertices'])[mesh['faces'],:2].mean(1)/size).astype(np.int64)


def prepare_road_patch(before, after, assets, touched_ids):
    size=assets['surface_tile_size_m']
    if not size:raise ValueError('Partial replacement needs spatially tiled source assets')
    old={m['id']:m for m in before['meshes']};new={m['id']:m for m in after['meshes']}
    changed={key for key in touched_ids if key not in old or key not in new
             or old[key]['vertices']!=new[key]['vertices'] or old[key]['faces']!=new[key]['faces']}
    selected=set()
    for meshes in (old,new):
        for key in changed&meshes.keys():
            if meshes[key]['faces']:selected.update(map(tuple,np.unique(triangle_cells(meshes[key],size),axis=0)))
    counts=Counter()
    for mesh in before['meshes']:
        if not mesh['faces']:continue
        cells,n=np.unique(triangle_cells(mesh,size),axis=0,return_counts=True)
        counts.update({tuple(c):int(k)for c,k in zip(cells,n)})
    records=[dict(r)for r in assets['objects']if r['kind']=='road']
    if not all('surface_tile_cell_xy'in r for r in records):
        layout=[(cell,min(250000,count-start))for cell,count in sorted(counts.items())for start in range(0,count,250000)]
        indexed={r['name']:r for r in records}
        if len(layout)!=len(indexed):raise ValueError('Source roads do not match the installed tile partition')
        for index,(cell,count) in enumerate(layout):
            row=indexed['road_'+str(index)]
            if row['faces']!=count:raise ValueError('Source triangle count differs from installed tile: '+row['name'])
            row['surface_tile_cell_xy']=list(map(int,cell))
    installed=Counter()
    for row in records:installed[tuple(row['surface_tile_cell_xy'])]+=row['faces']
    if installed!=counts:raise ValueError('Installed tile cells differ from the supplied source roads')
    retired=['map_assets_'+r['name']for r in records if tuple(r['surface_tile_cell_xy'])in selected]
    result=[]
    for mesh in after['meshes']:
        if not mesh['faces']:continue
        cells=triangle_cells(mesh,size);keep=np.zeros(len(cells),bool)
        for cell in selected:keep|=(cells==cell).all(1)
        if not keep.any():continue
        faces=np.asarray(mesh['faces'])[keep];used,indices=np.unique(faces,return_inverse=True)
        result.append(dict(mesh,vertices=np.asarray(mesh['vertices'])[used].tolist(),faces=indices.reshape(-1,3).tolist()))
    report=dict(changed_mesh_ids=sorted(changed),selected_cells=[list(map(int,c))for c in sorted(selected)],
        replace_surface_names=sorted(retired),source_road_assets=records,
        surface_name_offset=max(int(r['name'].removeprefix('road_'))for r in records)+1,
        triangles=sum(len(m['faces'])for m in result),
        method='Whole source triangles assigned by centroid to complete affected cells; retained cells keep their native assets.')
    return dict(meshes=result),report

def prepare_context_patch(before, after, assets, bounds_xy_m):
    """Select complete terrain/closure cells around a reviewed local change."""
    size = assets['surface_tile_size_m']
    if not size:
        raise ValueError('Partial replacement needs spatially tiled source assets')
    bounds = np.asarray(bounds_xy_m, dtype=float)
    regions = bounds[None] if bounds.ndim == 2 else bounds
    result = dict(report=dict(scope='Complete terrain and closure cells for a reviewed local revision.'))
    retired = []
    offsets = {}
    rows = []
    for kind, vk, fk in [('terrain', 'vertices', 'faces'), ('road_edge', 'seam_vertices', 'seam_faces')]:
        old_vertices, new_vertices = np.asarray(before[vk]), np.asarray(after[vk])
        old_faces, new_faces = np.asarray(before[fk], int), np.asarray(after[fk], int)
        old_triangles, new_triangles = old_vertices[old_faces], new_vertices[new_faces]
        old_cells, new_cells = (np.floor(t[:, :, :2].mean(1)/size).astype(np.int64)
                                for t in (old_triangles, new_triangles))
        selected = set()
        for triangles, cells in [(old_triangles, old_cells), (new_triangles, new_cells)]:
            minimum, maximum = triangles[:, :, :2].min(1), triangles[:, :, :2].max(1)
            near = np.zeros(len(triangles), bool)
            for region in regions:
                near |= ((minimum <= region[1]).all(1)
                         & (maximum >= region[0]).all(1))
            selected.update(map(tuple, np.unique(cells[near], axis=0)))
        def in_cells(cells):
            keep = np.zeros(len(cells), bool)
            for cell in selected:
                keep |= (cells == cell).all(1)
            return keep
        old_keep, new_keep = in_cells(old_cells), in_cells(new_cells)
        unchanged = np.array_equal(old_triangles[~old_keep], new_triangles[~new_keep])
        if not unchanged:
            raise ValueError('Context geometry changed outside selected complete cells: '+kind)
        records = [r for r in assets['objects'] if r['kind'] == kind]
        if not all('surface_tile_cell_xy' in r for r in records):
            raise ValueError('Context patch requires recorded native cell identities')
        retired.extend('map_assets_'+r['name'] for r in records
                       if tuple(r['surface_tile_cell_xy']) in selected)
        offsets[kind] = max(int(r['name'].removeprefix(kind+'_')) for r in records)+1
        faces = new_faces[new_keep]
        used, inverse = np.unique(faces, return_inverse=True)
        result[vk] = new_vertices[used].tolist()
        result[fk] = inverse.reshape(-1, 3).tolist()
        rows.append(dict(kind=kind,selected_cells=[list(map(int,c)) for c in sorted(selected)],
                         previous_source_triangles=int(old_keep.sum()),
                         current_source_triangles=int(new_keep.sum()),
                         retained_triangles=int((~old_keep).sum()),
                         unselected_geometry_exact=unchanged))
    return result, dict(surface_tile_size_m=size,selection_bounds_xy_m=bounds.tolist(),
                        replace_surface_names=sorted(retired),surface_name_offsets=offsets,
                        records=rows,all_unselected_geometry_exact=True,
                        method='Retain complete source triangles in affected native cells. New unique asset names replace only recorded old cell identities; roads and other cell assets remain unchanged.')
