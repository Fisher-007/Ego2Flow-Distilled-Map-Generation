"""Source-supported physical ground for AV2 junctions and reviewed shared spaces."""

from pathlib import Path
import json

import numpy as np
import shapely
from shapely.geometry import Polygon
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from av2.map.map_api import ArgoverseStaticMap

from .coordinates import LocalFrame, source_log_registration
from .av2_surfaces import mesh_triangles, plane_z, center_residuals, overlap_report


def query_surface(triangles, xy, tree=None, allow_missing=False):
    if tree is None:
        tree = shapely.STRtree(shapely.polygons(triangles[:, :, :2]))
    pairs = tree.query(shapely.points(xy), predicate="dwithin", distance=1e-7)
    z = np.full(len(xy), -np.inf)
    np.maximum.at(z, pairs[0], plane_z(triangles[pairs[1]], xy[pairs[0]]))
    if not np.isfinite(z).all() and not allow_missing:
        raise ValueError(f"Surface misses {np.count_nonzero(~np.isfinite(z))} requested boundary points")
    return z


def contact_segments(domain, kept_native):
    """Use actual shared edges, with heights from the unchanged lane triangles."""
    rows = []
    for mesh in kept_native:
        tri = mesh_triangles(mesh)
        polygon = shapely.union_all(shapely.polygons(tri[:, :, :2]))
        common = domain.boundary.intersection(polygon.boundary)
        for part in shapely.get_parts(common):
            if part.geom_type != "LineString":
                continue
            xy = np.asarray(part.coords)[:, :2]
            z = query_surface(tri, xy)
            for i in range(len(xy) - 1):
                rows.append({"lane_id": mesh["source_lane_id"],
                             "xyz": np.column_stack((xy[i:i+2], z[i:i+2])).tolist()})
    return rows


def component_contacts(domain, contacts):
    """Clip a shared cap to this component, excluding corner-only neighbors."""
    rows=[]
    for row in contacts:
        cap=np.asarray(row['xyz']);direction=cap[1,:2]-cap[0,:2]
        common=domain.boundary.intersection(shapely.linestrings(cap[:,:2]))
        for part in shapely.get_parts(common):
            if part.geom_type!='LineString' or part.length==0:continue
            xy=shapely.get_coordinates(part)
            t=((xy-cap[0,:2])@direction)/(direction@direction)
            z=cap[0,2]+t*(cap[1,2]-cap[0,2])
            for i in range(len(xy)-1):
                rows.append(dict(row,xyz=np.c_[xy[i:i+2],z[i:i+2]].tolist()))
    return rows


def raster_grid(domain, ground, frame, registration=None, reference_z=0., boundary_support=None):
    """Sample official 0.3 m cell centers; interpolate only finite grid support."""
    sim = ground.array_Sim2_city
    if not np.allclose(sim.R, np.eye(2)):
        raise ValueError("The AV2 raster grid must be axis aligned")
    bounds = np.asarray(domain.bounds).reshape(2, 2)
    if registration is not None:
        basis=np.asarray(registration['rotation_matrix']);translation=np.asarray(registration['translation_local_m'])
        corners=np.array([[x,y,reference_z]for x in bounds[:,0]for y in bounds[:,1]])
        source=(corners-translation)@basis
        px=sim.transform_point_cloud(frame.local_to_city(source[:,:2]))
        lo,hi=np.floor(px.min(0)).astype(int)-3,np.ceil(px.max(0)).astype(int)+3
    else:
        px = sim.transform_point_cloud(frame.local_to_city(bounds))
        lo, hi = np.floor(px[0]).astype(int) - 1, np.ceil(px[1]).astype(int) + 1
    cols, rows = np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1)
    cc, rr = np.meshgrid(cols + .5, rows + .5)
    city = sim.inverse().transform_point_cloud(np.column_stack((cc.ravel(), rr.ravel())))
    xy = frame.city_to_local(city).reshape(*cc.shape, 2)
    z = ground.get_ground_height_at_xy(city).reshape(cc.shape) - frame.origin[2]
    if registration is None:
        interp = RegularGridInterpolator((xy[:, 0, 1], xy[0, :, 0]), z,
                                         bounds_error=False, fill_value=np.nan)

    def sample_height(points):
        return interp(np.asarray(points)[:, ::-1])

    indices = np.arange(len(city)).reshape(cc.shape)
    a, b = indices[:-1, :-1].ravel(), indices[:-1, 1:].ravel()
    c, d = indices[1:, 1:].ravel(), indices[1:, :-1].ravel()
    faces = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
    if registration is not None:
        xyz=np.c_[xy.reshape(-1,2),z.ravel()]@basis.T+translation
        triangles3=xyz[faces]
        triangles3=triangles3[np.isfinite(triangles3).all(axis=(1,2))]
        support_tree=shapely.STRtree(shapely.polygons(triangles3[:,:,:2]))
        def sample_height(points):
            return query_surface(triangles3,np.asarray(points),support_tree,allow_missing=True)
        triangles=triangles3[:,:,:2]
        if boundary_support is not None:
            # Complete the footprint grid using measured raster and boundary heights.
            xx,yy=np.meshgrid(np.arange(bounds[0,0]-.3,bounds[1,0]+.6,.3),
                              np.arange(bounds[0,1]-.3,bounds[1,1]+.6,.3))
            ix=np.arange(xx.size).reshape(xx.shape)
            aa,bb,cc_,dd=ix[:-1,:-1].ravel(),ix[:-1,1:].ravel(),ix[1:,1:].ravel(),ix[1:,:-1].ravel()
            triangles=np.c_[xx.ravel(),yy.ravel()][np.r_[np.c_[aa,bb,cc_],np.c_[aa,cc_,dd]]]
    else:
        triangles = xy.reshape(-1, 2)[faces]
    polys = shapely.polygons(triangles)
    clipped = shapely.intersection(polys[shapely.intersects(polys, domain)], domain)
    pieces = shapely.get_parts(shapely.constrained_delaunay_triangles(clipped))
    positive = pieces[shapely.area(pieces) > 1e-12]
    tri_xy = shapely.get_coordinates(positive).reshape(-1, 4, 2)[:, :3]
    points, indices = np.unique(tri_xy.reshape(-1, 2), axis=0, return_inverse=True)
    completion={"enabled":boundary_support is not None,"source_boundary_points":0,
                "missing_grid_vertices_completed":0}
    if boundary_support is not None:
        observed_xyz=(np.c_[xy.reshape(-1,2),z.ravel()] if registration is None else xyz)
        observed_xyz=observed_xyz[np.isfinite(observed_xyz).all(axis=1)]
        support=np.r_[observed_xyz,np.asarray(boundary_support)]
        unique,inverse=np.unique(support[:,:2],axis=0,return_inverse=True)
        sums=np.bincount(inverse,weights=support[:,2]);counts=np.bincount(inverse)
        support_interp=LinearNDInterpolator(unique,sums/counts,fill_value=np.nan)
        completion.update(source_boundary_points=len(boundary_support),
            missing_grid_vertices_completed=int((~np.isfinite(sample_height(points))).sum()),
            method='Interpolate missing raster support from finite official grid points and measured AV2 boundary XYZ in the same reviewed frame. Preserve finite raster heights; keep this completion separate from direct raster observations.')
    def height(requested):
        requested=np.asarray(requested);values=np.asarray(sample_height(requested))
        missing=~np.isfinite(values)
        if missing.any() and boundary_support is not None:
            values[missing] = support_interp(requested[missing])
            # Allow sub-micrometer hull roundoff when querying the measured triangulation.
            retry = np.flatnonzero(~np.isfinite(values))
            if len(retry):
                hull = shapely.MultiPoint(unique).convex_hull
                near = shapely.distance(shapely.points(requested[retry]), hull) <= 1e-7
                rows = retry[near]
                simplex = support_interp.tri.find_simplex(requested[rows], tol=1e-9)
                rows, simplex = rows[simplex >= 0], simplex[simplex >= 0]
                if len(rows):
                    transform = support_interp.tri.transform[simplex]
                    bary = np.einsum('nij,nj->ni', transform[:,:2], requested[rows]-transform[:,2])
                    weights = np.c_[bary, 1-bary.sum(axis=1)]
                    values[rows] = (weights * (sums/counts)[support_interp.tri.simplices[simplex]]).sum(axis=1)
                    completion['boundary_simplex_roundoff_queries'] = completion.get('boundary_simplex_roundoff_queries', 0) + len(rows)
        if not np.isfinite(values).all():
            missing=requested[~np.isfinite(values)]
            raise ValueError(f"Official raster misses interpolation support for {len(missing)} points; first local XY: {missing[:5].tolist()}")
        return values
    height.completion=completion
    return points, indices.reshape(-1, 3), height


def join_contacts(xy, base_z, contacts, height, distance, absolute_target=False, contact_height=None, coincident_contact_limit_m=1e-6):
    """Match shared-edge heights, then fade their residual inside the junction."""
    targets, gaps, deltas = [], [], []
    for row in contacts:
        cap = np.asarray(row["xyz"])
        direction = cap[1, :2] - cap[0, :2]
        t = np.clip(((xy - cap[0, :2]) @ direction) / (direction @ direction), 0., 1.)
        closest = cap[0, :2] + t[:, None] * direction
        gap = np.linalg.norm(xy - closest, axis=1)
        target = cap[0, 2] + t * (cap[1, 2] - cap[0, 2])
        near = gap < distance
        delta = np.zeros(len(xy))
        if absolute_target:
            delta[near] = target[near] - base_z[near]
        elif "residual_z" in row:
            residual = row["residual_z"]
            delta[near] = residual[0] + t[near] * (residual[1] - residual[0])
        else:
            delta[near] = target[near] - height(closest[near])
        gaps.append(gap); targets.append(target); deltas.append(delta)
    if not contacts:
        return base_z, {"contact_segments": 0, "modified_vertices": 0, "max_correction_m": 0.}
    gaps, targets, deltas = map(np.asarray, (gaps, targets, deltas))
    nearest = gaps.min(axis=0)
    active = nearest < distance
    weights = np.maximum(1 - gaps / distance, 0.)**2 / np.maximum(gaps, 1e-8)**2
    correction = np.zeros(len(xy))
    fade = np.clip(nearest[active] / distance, 0., 1.)
    correction[active] = ((weights[:, active] * deltas[:, active]).sum(axis=0) /
                          weights[:, active].sum(axis=0)) * (1 - 3*fade**2 + 2*fade**3)
    result = base_z + correction
    # Require matching height values at shared lane-cap endpoints.
    max_contact_conflict = 0.
    resolved_contacts = 0
    max_resolved_gap = 0.
    on_contact = np.flatnonzero(nearest < 1e-7)
    actual_contact_z = contact_height(xy[on_contact]) if contact_height is not None else None
    for index, j in enumerate(on_contact):
        values = targets[gaps[:, j] < 1e-7, j]
        max_contact_conflict = max(max_contact_conflict, float(np.ptp(values)))
        disagreement = float(np.ptp(values))
        if disagreement > coincident_contact_limit_m + 1e-9 and contact_height is None:
            raise ValueError(f"Non-intersection contact heights disagree at {xy[j].tolist()}: {values.tolist()}")
        if actual_contact_z is not None:
            result[j] = actual_contact_z[index]
        elif disagreement > 1e-6:
            # Use the midpoint to minimize the maximum residual to the two retained edges.
            result[j] = (values.min() + values.max()) / 2
            resolved_contacts += 1
            max_resolved_gap = max(max_resolved_gap, disagreement / 2)
        else:
            result[j] = values[0]
    return result, {
        "method": ("Blend to the measured absolute contact height using a common distance-weighted field and cubic fade; each surface uses its own original vertex height as the outside target." if absolute_target else "Inverse-distance weighted contact-height residual, compact support inside the junction; cubic smoothstep fades to zero at the stated distance. Shared-edge vertices take the unchanged lane's exact linear edge height."),
        "transition_distance_m": distance, "contact_segments": len(contacts),
        "modified_vertices": int(active.sum()), "max_correction_m": float(abs(result - base_z).max(initial=0.)),
        "max_coincident_contact_z_disagreement_m": max_contact_conflict,
        "bounded_coincident_contact_resolution": {"limit_m": coincident_contact_limit_m, "vertices": resolved_contacts, "max_residual_to_retained_edge_m": max_resolved_gap, "method": "Midrange at inconsistent coincident edge points only; unique source contacts remain exact."},
        "exact_contact_source": "Actual retained fine-surface query at the constructed vertex" if contact_height is not None else ("Source contact-line interpolation with explicitly bounded midrange resolution" if resolved_contacts else "Consistent contact-line interpolation"),
    }


def ground_junctions(scene, roads, report, original_meshes, map_dir=None, transition_distance=2., data_root=None, shared_surface_review=None,
                     geometry_output=None, resume_geometry=None):
    """Realize reviewed same-level source footprints as one physical ground layer."""
    native = [m for m in roads["meshes"] if m.get("kind") == "av2_boundary_surface"]
    shared_ids=set((shared_surface_review or {}).get('source_lane_ids',[]))
    keep_native = [m for m in native if not m["is_intersection"] and m['source_lane_id']not in shared_ids]
    replace = [m for m in native if m["is_intersection"] or m['source_lane_id']in shared_ids]
    replace_ids = {m["id"] for m in replace}
    source_meshes = {m['id']: m for m in roads['meshes']}
    affected_closures = [m for m in roads['meshes'] if m.get('kind') == 'connection_closure'
                         and ({m['from_mesh_id'], m['to_mesh_id']} & replace_ids)]
    internal_closures = [m for m in affected_closures if all(
        source_meshes[m[name]].get('kind') == 'av2_boundary_surface' for name in ('from_mesh_id', 'to_mesh_id'))]
    absorbed_ids = {m['id'] for m in internal_closures}
    replace_polygons = [shapely.union_all(shapely.polygons(mesh_triangles(m)[:, :, :2])) for m in replace]
    domain = shapely.union_all(replace_polygons)
    closure_footprints = [(m, shapely.union_all(shapely.polygons(mesh_triangles(m)[:, :, :2])))
                          for m in internal_closures]
    closure_polygons = [shapely.polygons(mesh_triangles(m)[:, :, :2]) for m in internal_closures]
    # Use the shared layer for successor-closure heights and record subthreshold triangle area.
    closure_area, numerical_area = 0., 0.
    for polygons in closure_polygons:
        areas = shapely.area(polygons)
        closure_area += float(areas.sum())
        numerical_area += float(areas[areas <= 1e-12].sum())
        domain = shapely.union_all([domain, *polygons[areas > 1e-12]])
    kept_union = shapely.union_all([shapely.union_all(shapely.polygons(mesh_triangles(m)[:, :, :2])) for m in keep_native])
    domain = shapely.difference(domain, kept_union)
    contacts = contact_segments(domain, keep_native)
    coords = scene["coordinates"]
    frame = LocalFrame(coords["city"], coords["origin_city_xyz"])
    source_lanes={l['id']:l for l in scene['lanes']+scene.get('av2_context',[])}
    ground_cache={}
    if map_dir is not None:
        ground_cache[str(map_dir)]=ArgoverseStaticMap.from_map_dir(Path(map_dir), build_raster=True).raster_ground_height_layer
    retained = [m for m in roads["meshes"] if m["id"] not in replace_ids | absorbed_ids]
    prior_geometry = json.loads(Path(resume_geometry).read_text()) if resume_geometry else None
    grounded, records = (prior_geometry['meshes'], prior_geometry['components']) if prior_geometry else ([], [])
    numerical_components = prior_geometry.get('numerical_components', []) if prior_geometry else []
    if prior_geometry:
        cached_domain=shapely.union_all([shapely.union_all(shapely.polygons(mesh_triangles(m)[:,:,:2]))for m in grounded])
        if cached_domain.symmetric_difference(domain).area>1e-6:
            raise ValueError('Current fine ground footprint differs from the selected geometry checkpoint')
    for index, part in enumerate([] if prior_geometry else shapely.get_parts(domain)):
        if part.geom_type != "Polygon":
            continue
        # Use the same area cutoff as raster_grid's retained triangles.
        # A whole sub-cutoff component cannot contain a retained face.
        if part.area <= 1e-12:
            numerical_components.append(dict(component=index, area_m2=part.area,
                bounds_xy_m=list(part.bounds), footprint_wkt=part.wkt))
            continue
        local_contacts = component_contacts(part,contacts)
        source_ids = [m["source_lane_id"] for m, polygon in zip(replace, replace_polygons)
                      if part.intersects(polygon)]
        closure_source_ids = []
        if not source_ids:
            # Assign detached closure caps to their recorded source endpoints.
            for closure, polygon in closure_footprints:
                if part.distance(polygon) <= 1e-7:
                    closure_source_ids.extend(source_meshes[closure[key]]["source_lane_id"]
                                              for key in ("from_mesh_id", "to_mesh_id"))
            source_ids = sorted(set(closure_source_ids))
        if not source_ids:
            raise ValueError(f"Ground component lacks source lane or connection provenance: area={part.area}, bounds={part.bounds}")
        candidates={}
        if map_dir is not None:
            candidates[str(map_dir)]=part.area
        else:
            for key in source_ids:
                lane=source_lanes[key]
                overlap=part.intersection(Polygon(lane['left_boundary']+lane['right_boundary'][::-1])).area
                for observation in lane['source_observations']:
                    path=str((Path(data_root)/observation['source']).parent)
                    candidates[path]=candidates.get(path,0.)+overlap
            # Query all selected source rasters before assigning inferred heights.
            for source in scene.get('source_info',{}).get('assembly_review',{}).get('source_maps',[]):
                candidates.setdefault(str((Path(data_root)/source).parent),0.)
        attempted=[]
        reference_z=float(np.median([p[2]for key in source_ids for p in source_lanes[key]['centerline']]))
        for selected_map,area in sorted(candidates.items(),key=lambda r:-r[1]):
            if selected_map not in ground_cache:
                ground_cache[selected_map]=ArgoverseStaticMap.from_map_dir(Path(selected_map),build_raster=True).raster_ground_height_layer
            ground=ground_cache[selected_map]
            registration=source_log_registration(scene,Path(selected_map).parent.name)if data_root else None
            try:
                xy,faces,height=raster_grid(part,ground,frame,registration,reference_z)
                height(xy)
                coverage=shapely.union_all(shapely.polygons(xy[faces]))
                if coverage.symmetric_difference(part).area>1e-6:
                    raise ValueError('Selected official ground lacks complete junction coverage')
                break
            except ValueError as error:
                attempted.append(dict(map_dir=selected_map,reason=str(error)))
        else:
            # The shared-space polygons may extend beyond every ground raster.
            # AV2 boundary XYZ is separate measured support, recorded as such.
            selected_map,area=max(candidates.items(),key=lambda r:r[1])
            ground=ground_cache[selected_map]
            registration=source_log_registration(scene,Path(selected_map).parent.name)if data_root else None
            support=np.asarray([p for key in source_ids for side in ['left_boundary','right_boundary']
                                for p in source_lanes[key][side]])
            xy,faces,height=raster_grid(part,ground,frame,registration,reference_z,support)
            height(xy)
        base = height(xy)
        z, transition = join_contacts(xy, base, local_contacts, height, transition_distance,
            coincident_contact_limit_m=(shared_surface_review or {}).get("coincident_contact_resolution", {}).get("max_source_disagreement_m", 1e-6))
        vertices = np.column_stack((xy, z))
        tri = vertices[faces]
        down = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2] < 0
        faces[down] = faces[down, ::-1]
        mesh = {"id": f"av2_grounded_junction_{index}", "kind": "av2_grounded_junction",
                "source_lane_id": f"av2:grounded_junction:{index}", "source_lane_ids": source_ids,
                "is_intersection": any(source_lanes[k]['is_intersection']for k in source_ids),
                "physical_role":"Single observed ground layer across source junction and shared-space footprints",
                "source_interval_m": None, "xodr_road_id": None,
                "xodr_lane_id": -1, "junction_id": None,
                "vertices": vertices.tolist(), "faces": faces.tolist()}
        grounded.append(mesh)
        coverage = shapely.union_all(shapely.polygons(mesh_triangles(mesh)[:, :, :2]))
        interfaces = []
        for row in local_contacts:
            cap = np.asarray(row["xyz"])
            t = np.linspace(0, 1, int(np.ceil(np.linalg.norm(cap[1, :2] - cap[0, :2]) / .1)) + 1)
            samples = cap[0] + t[:, None] * (cap[1] - cap[0])
            actual = query_surface(mesh_triangles(mesh), samples[:, :2], allow_missing=True)
            roundoff_rows = np.flatnonzero(~np.isfinite(actual))
            roundoff_distances = []
            if len(roundoff_rows):
                distances = shapely.distance(shapely.points(samples[roundoff_rows, :2]), coverage)
                near = roundoff_rows[distances <= 1e-6]
                if len(near):
                    # Query adjacent edges within one micrometer for subthreshold triangles.
                    links = shapely.shortest_line(shapely.points(samples[near, :2]), coverage)
                    edge_xy = shapely.get_coordinates(links).reshape(-1, 2, 2)[:, -1]
                    actual[near] = query_surface(mesh_triangles(mesh), edge_xy)
                    roundoff_distances = distances[distances <= 1e-6].tolist()
            if not np.isfinite(actual).all():
                missing = samples[~np.isfinite(actual)]
                diagnostic = dict(component=index, mesh=mesh, footprint_wkt=part.wkt,
                    contact=row, missing_samples_xyz_m=missing.tolist(),
                    distance_to_mesh_m=shapely.distance(shapely.points(missing[:, :2]), coverage).tolist(),
                    distance_to_footprint_m=shapely.distance(shapely.points(missing[:, :2]), part).tolist(),
                    union_symmetric_difference_m2=coverage.symmetric_difference(part).area)
                if geometry_output:
                    diagnostic_path = Path(geometry_output).with_name('boundary_query_failure.json')
                    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                    diagnostic_path.write_text(json.dumps(diagnostic, separators=(',', ':')) + '\n')
                raise ValueError(f"Constructed component {index} misses {len(missing)} contact points; distances to mesh {diagnostic['distance_to_mesh_m']}")
            interfaces.append({"source_lane_id": row["lane_id"], "cap_xyz_m": row["xyz"],
                               "samples": len(t), "max_height_gap_m": float(abs(actual - samples[:, 2]).max()),
                               "submicrometre_boundary_queries": len(roundoff_distances),
                               "max_boundary_query_shift_m": max(roundoff_distances, default=0.)})
        if registration is None:
            observed = ground.get_ground_height_at_xy(frame.local_to_city(xy)) - frame.origin[2]
        else:
            basis=np.asarray(registration['rotation_matrix']);translation=np.asarray(registration['translation_local_m'])
            raw=(np.c_[xy,base]-translation)@basis
            raw_z=ground.get_ground_height_at_xy(frame.local_to_city(raw[:,:2]))-frame.origin[2]
            observed=(np.c_[raw[:,:2],raw_z]@basis.T+translation)[:,2]
        valid_observed=np.isfinite(observed)
        if not valid_observed.all() and not height.completion['enabled']:
            raise ValueError("A constructed junction vertex lacks direct official ground-raster coverage")
        record = {"id": mesh["id"], "source_lane_ids": source_ids, "area_m2": part.area,
                  "triangles": len(faces), "vertices": len(vertices), "bounds_xy_m": list(part.bounds),
                  "union_symmetric_difference_m2": coverage.symmetric_difference(part).area,
                  "triangle_overlap_area_m2": float(shapely.area(shapely.polygons(mesh_triangles(mesh)[:, :, :2])).sum() - coverage.area),
                  "ground_covered_vertices":int(valid_observed.sum()),
                  "ground_missing_vertices":int((~valid_observed).sum()),
                  "source_boundary_height_completion":height.completion,
                  "source_map_dir":selected_map,"source_log_registration":registration,
                  "source_identity_from_connection_closures": sorted(set(closure_source_ids)),
                  "source_selection":"Contributing lane area first, then other selected city logs; require complete interpolation support in the reviewed source frame",
                  "selected_raster_lane_support_area_m2":area,
                  "unsupported_source_candidates":attempted,
                  "abs_surface_minus_official_cell_quantiles_50_95_max_m": np.quantile(abs(z[valid_observed] - observed[valid_observed]), [.5, .95, 1]).tolist() if valid_observed.any()else [],
                  "contact_transition": transition, "interfaces": interfaces}
        records.append(record)
        print(f"Grounded {mesh['id']}: {part.area:.2f} m2, {len(faces)} faces, {len(interfaces)} contacts", flush=True)

    if geometry_output:
        geometry_output = Path(geometry_output)
        geometry_output.parent.mkdir(parents=True, exist_ok=True)
        geometry_output.write_text(json.dumps(dict(meshes=grounded, components=records,
            numerical_components=numerical_components,
            scope='Completed physical ground geometry; full source-center, overlap and connector checks follow.'),
            separators=(',', ':')) + '\n')
        print(f"Ground geometry saved: {geometry_output}", flush=True)
    # Keep source endpoints on closures whose roads use shared-ground components.
    replacement_by_source = {}
    for mesh in grounded:
        polygon = shapely.union_all(shapely.polygons(mesh_triangles(mesh)[:, :, :2]))
        for key in mesh['source_lane_ids']:
            replacement_by_source.setdefault(key, []).append((mesh['id'], polygon))
    retargeted = []
    for i, old in enumerate(retained):
        if old.get('kind') != 'connection_closure':
            continue
        changed = dict(old)
        for side in ('from', 'to'):
            key = side + '_mesh_id'
            original = source_meshes[old[key]]
            changed[side + '_source_lane_id'] = original.get('source_lane_id')
            if old[key] in replace_ids:
                polygon = shapely.union_all(shapely.polygons(mesh_triangles(old)[:, :, :2]))
                candidates = replacement_by_source[original['source_lane_id']]
                target = min(candidates, key=lambda r: r[1].distance(polygon))[0]
                changed[side + '_source_mesh_id'] = old[key]
                changed[key] = target
                retargeted.append(dict(closure_id=old['id'], side=side, source_mesh_id=old[key], physical_mesh_id=target))
        retained[i] = changed

    physical = keep_native + grounded
    meshes = retained + grounded
    source_ids = {m["source_lane_id"] for m in native}
    source_lanes = {l["id"]: l for l in scene["lanes"] + scene.get("av2_context", []) if l["id"] in source_ids}
    center_surfaces = {m["source_lane_id"]: m for m in keep_native}
    for mesh in grounded:
        center_surfaces.update({key: mesh for key in mesh["source_lane_ids"]})
    source_polys = {key: Polygon(np.concatenate((lane["left_boundary"], lane["right_boundary"][::-1])))
                    for key, lane in source_lanes.items()}
    aliases = report.get("compiler_children_to_source", {})
    center_original = [dict(m, source_lane_id=aliases[m["source_lane_id"]])
                       if m.get("source_lane_id") in aliases else m for m in original_meshes]
    center_sources = {key: source_lanes[key] for key in center_surfaces}
    centers, samples = center_residuals(center_sources, source_polys, center_original, physical)
    print(f"Measured complete physical layer at {len(samples)} source centers", flush=True)
    overlaps = overlap_report({m["id"]: m for m in physical})
    print(f"Measured fine-surface overlaps: {overlaps['pairs']} pairs", flush=True)
    other = [m for m in retained if m.get("kind") != "av2_boundary_surface"]
    overlaps_retained = overlap_report({m["id"]: m for m in physical}, other)
    print(f"Measured fine/coarse overlaps: {overlaps_retained['pairs']} pairs", flush=True)
    details = {
        "method": "Union AV2 intersection and Agent-reviewed shared-space polygons into one physical surface per connected component. Prefer complete official raster support from selected city logs in their reviewed frames; record any missing-raster completion from measured boundary XYZ separately. Match retained source-lane contacts with a bounded interior height transition.",
        "map_dir": str(map_dir)if map_dir else None,
        "raster_files": [str(p)for folder in sorted({r['source_map_dir'] for r in records})
                         for p in sorted(Path(folder).glob("*ground_height_surface*"))],
        "native_intersection_lanes_replaced": sum(m['is_intersection']for m in replace),
        "nonintersection_shared_space_lanes_replaced":sum(not m['is_intersection']for m in replace),
        "shared_space_agent_review":shared_surface_review,
        "source_connection_closures": dict(absorbed_ids=sorted(absorbed_ids),
            absorbed_area_m2=closure_area, numerical_triangle_area_m2=numerical_area,
            retained_endpoint_mapping=retargeted,
            coarse_contact_status='Source identities retained; outside patch heights handled by connector joining'),
        "single_layer_components": len(grounded),
        "sub_triangle_area_components": dict(area_cutoff_m2=1e-12,
            components=numerical_components,
            total_area_m2=sum(row['area_m2'] for row in numerical_components)),
        "grounded_area_m2": domain.area, "retained_nonintersection_lanes": len(keep_native),
        "retained_nonintersection_meshes_and_boundary_xyz_unchanged": all(m in meshes for m in keep_native),
        "missing_ground_fill": "Where all selected logs lack raster interpolation support, use explicitly recorded interpolation from finite official ground and measured source boundary XYZ; no zero fill or extrapolation beyond measured support.",
        "max_contact_height_gap_m": max(r["max_height_gap_m"] for component in records for r in component["interfaces"]),
        "components": records,
        "limitations": [
            "Intersection source lane XYZ and native navigation remain unchanged; sparse virtual boundaries do not constrain this physical surface.",
            "The official raster is 0.3 m and float16; interpolation and the recorded contact correction do not add survey accuracy.",
            "Fine/coarse and generated connector surfaces are retained in this step. Their remaining overlap and edge-height differences require a separate local transition.",
        ],
    }
    report.update(method=details["method"], junction_ground_restoration=details,
                  native_centerline_to_physical_surface=centers,
                  native_surface_overlaps=overlaps, native_vs_retained_surface_overlaps=overlaps_retained,
                  output_mesh_count=len(meshes), native_triangles=sum(len(m["faces"]) for m in physical))
    report["limitations"] = details["limitations"] + [
        "Physical road surface heights can depart from the old native center elevation; CARLA 0.9.16 native roll remains zero."]
    report["shared_boundaries"]["scope"] = "Original AV2 source-boundary check; physical junction contacts are measured separately in junction_ground_restoration."
    closure = dict(roads.get('connection_closure', {}))
    if closure:
        closure['patches'] = [r for r in closure['patches'] if r['id'] not in absorbed_ids]
        closure['patch_count'] = len(closure['patches'])
        closure['absorbed_into_shared_ground'] = sorted(absorbed_ids)
    output = {**roads, "meshes": meshes, "connection_closure": closure, "av2_surface_restoration": report,
              "surface_scope": "Retained AV2 boundary meshes unchanged; reviewed junction and shared-space unions use one source-supported physical ground layer with continuous lane contacts. OpenDRIVE unchanged."}
    return output, report, samples
