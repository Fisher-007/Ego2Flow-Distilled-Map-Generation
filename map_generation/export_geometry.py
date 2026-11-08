"""Numerical geometry checks at the native-asset export boundary."""


def missing_surface_omission(entry):
    """Explain an observed FBX omission only when the source remnant is negligible."""
    p=entry.get('native_precision')
    if not p or p['bounds_cm'] is not None or p['resolved_source_triangles']!=0 or p['unresolved_source_triangles']!=entry['faces']:
        return None
    if p['unresolved_source_area_m2']==0.:
        return 'All source faces have zero area at native coordinate precision; FBX created no asset.'
    low,high=entry['bounds_m']['min'],entry['bounds_m']['max']
    diagonal=sum((b-a)**2 for a,b in zip(low,high))**.5
    if entry['kind']=='road_edge' and diagonal<.001:
        return 'All closure faces are unresolved and the complete remnant is smaller than the existing 1 mm asset-bounds tolerance; FBX rejected its faces as degenerate.'
    return None


def footprint_area(building):
    def area(ring):
        if len(ring)<3:return 0.
        x0,y0=ring[0][:2]
        xy=[(p[0]-x0,p[1]-y0) for p in ring]
        return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(xy,xy[1:]+xy[:1])))/2
    return area(building['exterior'])-sum(area(ring) for ring in building.get('holes',[]))


def resolved_skirt_faces(vertices,faces,minimum_altitude_m=1e-6):
    """Keep closure faces with nonzero thickness after Blender float export."""
    import numpy as np
    triangles=np.asarray(vertices,dtype=np.float32).astype(np.float64)[faces]
    edges=np.stack([triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,1],
                    triangles[:,0]-triangles[:,2]],axis=1)
    double_area=np.linalg.norm(np.cross(edges[:,0],-edges[:,2]),axis=1)
    longest=np.linalg.norm(edges,axis=2).max(1)
    altitude=double_area/np.maximum(longest,1e-30)
    keep=altitude>minimum_altitude_m
    return keep,dict(removed_triangles=int((~keep).sum()),
        removed_area_m2=float(double_area[~keep].sum()/2),
        maximum_removed_altitude_m=float(altitude[~keep].max()) if (~keep).any() else 0.,
        threshold_m=minimum_altitude_m)


def resolved_source_faces(vertices, faces, minimum_altitude_m=1e-6):
    """Filter terrain and closure faces below the double-precision resolution threshold."""
    import numpy as np
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    edges = np.stack([triangles[:, 1] - triangles[:, 0],
                      triangles[:, 2] - triangles[:, 1],
                      triangles[:, 0] - triangles[:, 2]], axis=1)
    double_area = np.linalg.norm(np.cross(edges[:, 0], -edges[:, 2]), axis=1)
    longest = np.linalg.norm(edges, axis=2).max(1)
    altitude = double_area / np.maximum(longest, 1e-30)
    keep = altitude > minimum_altitude_m
    return keep, dict(
        removed_triangles=int((~keep).sum()),
        removed_source_area_m2=float(double_area[~keep].sum() / 2),
        maximum_removed_source_altitude_m=(
            float(altitude[~keep].max()) if (~keep).any() else 0.),
        threshold_m=minimum_altitude_m,
        source_face_indices=np.flatnonzero(~keep).tolist(),
        reason='Source-degenerate context triangles excluded before float rounding; '
               'retained source geometry is unchanged.')


def native_surface_bounds(vertices, faces):
    """Compute inner and outer source envelopes in native centimeters.

    The inner envelope uses faces exceeding the native squared-normal threshold,
    0.1 mm thickness, and local float spacing. The outer envelope contains the
    complete source mesh. These bounds account for native welding and rounding.
    """
    import numpy as np
    xyz = (np.asarray(vertices, dtype=np.float32).astype(np.float64) * [100., -100., 100.]).astype(np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = xyz[faces]
    edges = [triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0],
             triangles[:, 2] - triangles[:, 1]]
    cross = np.cross(edges[0], edges[1])
    square = np.sum(cross * cross, axis=1)
    keep = square >= 2e-5
    for edge in edges:
        keep &= np.max(np.abs(edge), axis=1) > 2e-5
    longest = np.maximum.reduce([np.linalg.norm(edge.astype(np.float64),axis=1) for edge in edges])
    altitude_cm = np.sqrt(square.astype(np.float64))/np.maximum(longest,1e-30)
    # Each rounded position has at most half a spacing of error per axis.
    # Two opposing positions can therefore close one spacing diagonal.
    spacing = np.spacing(np.abs(triangles)).astype(np.float64)
    rounding_span_cm = np.linalg.norm(spacing, axis=2).max(1)
    minimum_altitude_cm = np.maximum(.01, rounding_span_cm)
    keep &= altitude_cm >= minimum_altitude_cm
    used = xyz[np.unique(faces[keep])]
    return dict(bounds_cm=dict(min=used.min(0).tolist(), max=used.max(0).tolist()) if len(used) else None,
        resolved_source_triangles=int(keep.sum()), unresolved_source_triangles=int((~keep).sum()),
        unresolved_source_area_m2=float(np.sqrt(square[~keep].astype(np.float64)).sum() / 20000),
        normal_squared_threshold_cm4=2e-5, equal_corner_threshold_cm=2e-5,
        minimum_resolved_triangle_altitude_m=.0001,
        maximum_native_position_rounding_span_m=float(rounding_span_cm.max()/100),
        scope='Bounds of source faces resolved above 0.1 mm and the native-position float spacing. Native bounds must cover these and stay within complete source bounds; thinner slivers can survive or disappear in native import. Source vertices and the 1 mm bounds tolerance remain unchanged.')
