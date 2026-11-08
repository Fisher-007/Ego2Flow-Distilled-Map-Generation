"""Drape AV2 paint observations over exported road triangles.

Dash patterns follow Blender Driving Scenario Creator road_properties.py and
road.get_strips_s_boundaries. GEOS handles buffering and triangulation.
"""

from collections import Counter
import json

import numpy as np
import shapely
from shapely.ops import substring


DSC_URL = "https://github.com/johschmitz/blender-driving-scenario-creator"
SUPPORTED = {
    "SOLID_WHITE": ("white", "solid"), "DASHED_WHITE": ("white", "broken"),
    "SOLID_YELLOW": ("yellow", "solid"), "DASHED_YELLOW": ("yellow", "broken"),
    "DOUBLE_SOLID_YELLOW": ("yellow", "double_solid"),
}


def plane_heights(triangles, xy):
    """Same triangle-plane interpolation used by build_assets.surface_heights."""
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return triangles[:, 0, 2] - ((xy - triangles[:, 0, :2]) * normal[:, :2]).sum(axis=1) / normal[:, 2]


def source_surface_layers(scene, roads):
    """Separate paint support using source road levels before XY projection.

    Connected internal movements inherit an unambiguous adjacent level.
    At-grade building passages remain at grade. Layer tags are relational
    identities; they are never converted into measured vertical coordinates.
    """
    lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    levels={};internal=set();neighbors={k:set() for k in lanes}
    for k,lane in lanes.items():
        tags=lane.get('source_tags',{})
        if 'layer' in tags:level=float(tags['layer'])
        elif tags.get('bridge') not in (None,'no','false','0'):level=1.
        elif tags.get('tunnel') not in (None,'no','false','0','building_passage'):level=-1.
        else:level=0.
        levels[k]=level
        if lane.get('internal') and not any(k in tags for k in ('layer','bridge','tunnel')):internal.add(k)
        for target in lane.get('successors',[]):
            if target in neighbors:neighbors[k].add(target);neighbors[target].add(k)
    pending=set(internal)
    while pending:
        seed=pending.pop();component={seed};queue=[seed];boundary=set()
        while queue:
            for target in neighbors[queue.pop()]:
                if target in pending:pending.remove(target);component.add(target);queue.append(target)
                elif target not in internal:boundary.add(levels[target])
        if len(boundary)==1:
            level=boundary.pop()
            for k in component:levels[k]=level
    by_id={m['id']:m for m in roads['meshes'] if 'id' in m}
    def source_levels(key):
        if key and key.startswith('bridge:'):
            return set().union(*(source_levels(k) for k in key[7:].split('->')))
        # Keep the source lane's layer identity on native longitudinal partitions.
        if key not in levels and key and ':native_part:' in key:
            parent, part = key.rsplit(':native_part:', 1)
            if part.isdigit() and parent in levels:
                key = parent
        return {levels.get(key, 0.)}
    grouped={}
    for mesh in roads['meshes']:
        keys={mesh.get('source_lane_id')}|set(mesh.get('source_lane_ids',[]));keys.discard(None)
        if mesh.get('kind')=='connection_closure':
            for side in ('from','to'):
                key=mesh.get(side+'_source_lane_id')
                if key is None:key=by_id.get(mesh.get(side+'_mesh_id'),{}).get('source_lane_id')
                if key is not None:keys.add(key)
        memberships=set().union(*(source_levels(k) for k in keys)) if keys else {0.}
        for level in memberships:grouped.setdefault(level,[]).append(mesh)
    return levels,grouped


class RoadSurface:
    def __init__(self, roads, region=None, diagnostics=None, resume_tiles=False):
        self.diagnostics = diagnostics
        self.resume_tiles = resume_tiles
        self.triangles = np.concatenate([
            np.asarray(mesh["vertices"], dtype=float)[np.asarray(mesh["faces"], dtype=int)]
            for mesh in roads["meshes"] if mesh["faces"]
        ])
        normals = np.cross(self.triangles[:, 1] - self.triangles[:, 0],
                           self.triangles[:, 2] - self.triangles[:, 0])
        zero = (normals == 0).all(axis=1)
        self.zero_area_triangle_count = int(zero.sum())
        self.triangles, normals = self.triangles[~zero], normals[~zero]
        if not np.isfinite(self.triangles).all() or np.any(normals[:, 2] <= 0):
            raise ValueError("Expected finite road triangles with upward ENU winding")
        self.polygons = shapely.polygons(self.triangles[:, :, :2])
        if region is not None:
            # Index individual paint islands before intersecting road triangles.
            pairs = shapely.STRtree(self.polygons).query(shapely.get_parts(region), predicate='intersects')
            ids = np.unique(pairs[1])
            self.triangles, normals, self.polygons = self.triangles[ids], normals[ids], self.polygons[ids]
        self.normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        self.tree = shapely.STRtree(self.polygons)
        from .mesh_geometry import spatial_union
        # A common 0.1 micrometre footprint grid resolves coincident slivers
        # from shared-road remeshing. Height projection uses original triangles.
        self.footprint = spatial_union(self.polygons, grid_size=1e-7)

    def project(self, xy):
        points = shapely.points(xy)
        pairs = self.tree.query(points, predicate="covered_by")
        missing = np.ones(len(xy), dtype=bool)
        missing[pairs[0]] = False
        if missing.any():
            # Apply the proximity tolerance only to samples outside all road triangles.
            indices = np.flatnonzero(missing)
            # Use a two-micrometer bound for exterior noding roundoff.
            nearby = self.tree.query(points[indices], predicate="dwithin", distance=2e-6)
            pairs = np.concatenate([pairs, np.vstack([indices[nearby[0]], nearby[1]])], axis=1)
        z = np.full(len(xy), -np.inf)
        values = plane_heights(self.triangles[pairs[1]], xy[pairs[0]])
        np.maximum.at(z, pairs[0], values)
        if not np.isfinite(z).all():
            failed=xy[~np.isfinite(z)]
            distances=shapely.distance(shapely.points(failed),self.footprint)
            order=np.argsort(distances)[-8:]
            raise ValueError("Clipped marking vertex has no road triangle support: " + str({
                'count':len(failed),'xy_distance_m':np.c_[failed[order],distances[order]].tolist()}))
        # Choose a stable visible parent plane at shared triangle edges.
        chosen = np.full(len(xy), len(self.triangles), dtype=int)
        top = np.abs(values - z[pairs[0]]) <= 1e-10
        np.minimum.at(chosen, pairs[0, top], pairs[1, top])
        return np.column_stack((xy, z)), self.normals[chosen], chosen

    def visible_parent_patch(self, polygon):
        """Resolve a failed paint cell by clipping each physical parent plane.

        Use half-plane clipping to remove higher neighbors explicitly. This
        avoids depending on a tiny equal-height cut joining a large noded ring.
        """
        ids=self.tree.query(polygon,predicate='intersects')
        tri=self.triangles[ids]
        normals=self.normals[ids]
        slopes=-normals[:,:2]/normals[:,2,None]
        constants=tri[:,0,2]-(slopes*tri[:,0,:2]).sum(1)
        result=[];parents=[]
        for i,parent in enumerate(ids):
            visible=self.polygons[parent].intersection(polygon)
            for j,other in enumerate(ids):
                if i==j or visible.is_empty:continue
                gradient=slopes[j]-slopes[i];constant=constants[j]-constants[i]
                values=tri[j,:,:2]@gradient+constant
                if np.max(np.abs(values))<1e-10:
                    if other<parent:visible=visible.difference(self.polygons[other])
                    continue
                if values.max()<=0:continue
                vertices=[]
                for k in range(3):
                    p0,p1=tri[j,k,:2],tri[j,(k+1)%3,:2]
                    d0,d1=values[k],values[(k+1)%3]
                    if d0>=0:vertices.append(p0)
                    if (d0>0 and d1<0) or (d0<0 and d1>0):
                        vertices.append(p0+(p1-p0)*d0/(d0-d1))
                if len(vertices)>=3:
                    visible=visible.difference(shapely.Polygon(vertices))
            parts=shapely.get_parts(visible)
            parts=parts[(shapely.get_type_id(parts)==3)&(shapely.area(parts)>1e-12)]
            cells=shapely.get_parts(shapely.constrained_delaunay_triangles(parts))
            cells=cells[shapely.area(cells)>1e-12]
            if not len(cells):continue
            xy=shapely.get_coordinates(cells).reshape(-1,4,2)[:,:3]
            flat=xy.reshape(-1,2)
            xyz=np.column_stack((flat,plane_heights(np.repeat(self.triangles[parent][None],len(flat),axis=0),flat)))
            result.extend(xyz.reshape(-1,3,3));parents.extend([parent]*len(xy))
        triangles=np.asarray(result).reshape(-1,3,3)
        area=shapely.area(shapely.polygons(triangles[:,:,:2])).sum()
        if abs(area-polygon.area)>max(1e-9,polygon.area*1e-6):
            raise ValueError('Visible-parent paint repair did not preserve requested area')
        return triangles,np.asarray(parents,dtype=int)

    def cut_and_drape(self, paint, offset, _local=False):
        """Node paint against every intersecting road edge before triangulating."""
        if paint.is_empty:
            return np.empty((0, 3, 3)), {"cells": 0, "max_interpolation_gap_m": 0.0}
        islands = shapely.get_parts(paint)
        if not _local and np.max(np.ptp(shapely.get_coordinates(paint), axis=0)) > 100.:
            # A connected boundary stripe can span the whole city. Centroid
            # grouping leaves that stripe unpartitioned; cut actual 100 m cells.
            bounds = np.asarray(paint.bounds).reshape(2, 2)
            low, high = np.floor(bounds / 100.).astype(int)
            tile_tree = shapely.STRtree(islands)
            if np.any(high > low):
                batches, reviews = [], []
                for x in range(low[0], high[0]+1):
                    for y in range(low[1], high[1]+1):
                        tile = shapely.box(x*100., y*100., (x+1)*100., (y+1)*100.)
                        ids = tile_tree.query(tile, predicate='intersects')
                        if not len(ids):
                            continue
                        local = shapely.union_all(shapely.intersection(islands[ids], tile))
                        if local.area < 1e-12:
                            continue
                        index=len(batches)
                        cache=self.diagnostics / ('paint_tile_%04d.npz'%index) if self.diagnostics else None
                        report_file=cache.with_suffix('.json') if cache else None
                        if self.resume_tiles and cache.is_file() and report_file.is_file():
                            review=json.loads(report_file.read_text())
                            with np.load(cache) as saved:
                                if saved['paint_wkb'].item()==shapely.to_wkb(local) and review['max_interpolation_gap_m']<=.0001:
                                    batches.append(saved['triangles']);reviews.append(review)
                                    continue
                        result, review = self.cut_and_drape(local, offset, _local=True)
                        batches.append(result); reviews.append(review)
                        if self.diagnostics is not None:
                            index = len(batches)-1
                            np.savez_compressed(self.diagnostics / ('paint_tile_%04d.npz' % index),
                                                triangles=result, paint_wkb=shapely.to_wkb(local))
                            (self.diagnostics / ('paint_tile_%04d.json' % index)).write_text(json.dumps(review))
                            print(json.dumps({'paint_tile': [x, y], 'index': index, **review}), flush=True)
                checks = {}
                for key in set().union(*(r.keys() for r in reviews)):
                    values = [r[key] for r in reviews if key in r]
                    checks[key] = max(values) if key.startswith('max_') else min(values) if key.startswith('min_') else sum(values)
                checks['spatial_batches'] = len(batches)
                return np.concatenate(batches), checks
        ids = np.unique(self.tree.query(islands, predicate="intersects")[1])
        parts = shapely.intersection(self.polygons[ids], paint)
        # Extract polygon components and road-parent indices before noding paint boundaries.
        parts, owners = shapely.get_parts(parts, return_index=True)
        usable = (shapely.get_type_id(parts) == 3) & (shapely.area(parts) > 1e-12)
        parts, ids = parts[usable], ids[owners[usable]]
        if not len(parts):
            return np.empty((0, 3, 3)), {"cells": 0, "max_interpolation_gap_m": 0.0}
        # Partition paint along road edges and equal-height lines between overlapping planes.
        pairs = shapely.STRtree(parts).query(parts, predicate="intersects")
        pairs = pairs[:, pairs[0] < pairs[1]]
        overlap = shapely.intersection(parts[pairs[0]], parts[pairs[1]])
        usable = shapely.area(overlap) > 1e-12
        pairs, overlap = pairs[:, usable], overlap[usable]
        tri = self.triangles[ids]
        normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        slopes = -normals[:, :2]/normals[:, 2, None]
        constants = tri[:, 0, 2] - (slopes*tri[:, 0, :2]).sum(axis=1)
        cuts = []
        for (a, b), region in zip(pairs.T, overlap):
            gradient, constant = slopes[a]-slopes[b], constants[a]-constants[b]
            coords = shapely.get_coordinates(region)
            dz = coords @ gradient + constant
            if dz.min() >= -1e-9 or dz.max() <= 1e-9:
                continue
            center = coords.mean(axis=0)
            center -= (center @ gradient + constant) * gradient / (gradient @ gradient)
            tangent = np.array([-gradient[1], gradient[0]])/np.linalg.norm(gradient)
            radius = np.linalg.norm(np.ptp(coords, axis=0))+1.
            cut = shapely.LineString([center-radius*tangent, center+radius*tangent])
            cuts.append(shapely.intersection(cut, region))
        # Submicron snapping collapses GEOS roundoff slivers at newly noded
        # plane intersections. AV2 is cm-quantized; road XYZ is not modified.
        lines = shapely.union_all(shapely.set_precision([*shapely.boundary(parts), *cuts], 1e-8))
        cells = shapely.get_parts(shapely.polygonize(shapely.get_parts(lines)))
        inside = shapely.covers(paint, shapely.point_on_surface(cells))
        cells = cells[inside & (shapely.area(cells) > 1e-12)]
        # Apply the declared precision to invalid output rings.
        invalid = ~shapely.is_valid(cells)
        area_before = shapely.area(cells[invalid]).sum()
        cells[invalid] = shapely.set_precision(cells[invalid], 1e-8)
        precision_area_delta = float(shapely.area(cells[invalid]).sum() - area_before)
        # Clip complete cells to retain narrow paint/road holes.
        paint_tree = shapely.STRtree(shapely.get_parts(paint))
        contained = np.zeros(len(cells), dtype=bool)
        contained[paint_tree.query(cells, predicate='covered_by')[0]] = True
        clip_area_before = shapely.area(cells).sum()
        retried_intersections = 0
        for i in np.flatnonzero(~contained):
            local = shapely.union_all(paint_tree.geometries[paint_tree.query(cells[i], predicate='intersects')])
            try:
                cells[i] = cells[i].intersection(local)
            except shapely.errors.GEOSException:
                # Use the common partition grid when re-noding rebuilt contacts.
                if self.diagnostics is not None:
                    np.save(self.diagnostics / 'failed_cell_overlay.npy',
                            shapely.to_wkb([cells[i], local]))
                cells[i] = shapely.intersection(cells[i], local, grid_size=1e-8)
                retried_intersections += 1
        cells = shapely.get_parts(cells)
        cells = cells[(shapely.get_type_id(cells) == 3) & (shapely.area(cells) > 1e-12)]
        clipped_cell_area = float(clip_area_before - shapely.area(cells).sum())
        try:
            triangulated = shapely.constrained_delaunay_triangles(cells)
        except shapely.errors.GEOSException:
            if self.diagnostics is not None:
                np.save(self.diagnostics / 'failed_paint_cells.npy', shapely.to_wkb(cells))
            raise
        triangles, cell_indices = shapely.get_parts(triangulated, return_index=True)
        usable = shapely.area(triangles) > 1e-12
        triangles, cell_indices = triangles[usable], cell_indices[usable]
        xy = shapely.get_coordinates(triangles).reshape(-1, 4, 2)[:, :3]
        areas = shapely.area(triangles)
        longest = np.max(np.linalg.norm(xy-np.roll(xy, 1, axis=1), axis=2), axis=1)
        # Remove subprecision triangles at height discontinuities and record their area.
        usable = 2*areas/longest > 1e-7
        sliver_count, sliver_area = int((~usable).sum()), float(areas[~usable].sum())
        xy, cell_indices = xy[usable], cell_indices[usable]
        _, _, triangle_parents = self.project(xy.mean(axis=1))
        parents = np.repeat(triangle_parents, 3)
        # Use each visible parent plane for both interior and boundary paint vertices.
        flat_xy = xy.reshape(-1, 2)
        xyz = np.column_stack((flat_xy, plane_heights(self.triangles[parents], flat_xy)))
        normals = self.normals[parents]
        original = xyz.reshape(-1, 3, 3)
        # Check interior residuals where overlapping road planes cross within a cell.
        weights = np.asarray([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])
        samples = np.einsum("ki,nij->nkj", weights, original).reshape(-1, 3)
        substrate, _, _ = self.project(samples[:, :2])
        gaps = samples[:, 2] - substrate[:, 2]
        raw_gap=float(np.max(np.abs(gaps)))
        repair=np.max(np.abs(gaps.reshape(-1,len(weights))),axis=1)>.0001
        if repair.any():
            repaired=[];repaired_parents=[]
            for index in np.flatnonzero(repair):
                patch,patch_parents=self.visible_parent_patch(shapely.Polygon(original[index,:,:2]))
                repaired.append(patch);repaired_parents.append(patch_parents)
            original=np.concatenate([original[~repair],*repaired])
            triangle_parents=np.concatenate([triangle_parents[~repair],*repaired_parents])
            xyz=original.reshape(-1,3);normals=np.repeat(self.normals[triangle_parents],3,axis=0)
            samples=np.einsum('ki,nij->nkj',weights,original).reshape(-1,3)
            substrate,_,_=self.project(samples[:,:2]);gaps=samples[:,2]-substrate[:,2]
        if self.diagnostics is not None and np.max(np.abs(gaps)) > .001:
            worst = np.argsort(np.abs(gaps))[-16:]
            path = self.diagnostics / ('drape_failure_%05d.npz' % len(list(self.diagnostics.glob('drape_failure_*.npz'))))
            np.savez_compressed(path, samples=samples[worst], substrate=substrate[worst],
                                paint_triangles=original[worst//len(weights)],
                                parent_triangles=self.triangles[triangle_parents[worst//len(weights)]],
                                road_triangles=self.triangles[ids], paint_wkb=shapely.to_wkb(paint))
        lifted = (xyz + offset * normals).reshape(-1, 3, 3)
        winding = np.cross(lifted[:, 1] - lifted[:, 0], lifted[:, 2] - lifted[:, 0])[:, 2]
        lifted[winding < 0] = lifted[winding < 0, ::-1]
        keep = np.abs(winding) > 1e-12
        return lifted[keep], {"cells": len(cells), "max_interpolation_gap_m": float(np.max(np.abs(gaps))),
                              'max_before_local_repair_gap_m':raw_gap,
                              'locally_repaired_paint_triangles':int(repair.sum()),
                              "invalid_noded_cells_repaired": int(invalid.sum()),
                              "partition_precision_area_change_m2": precision_area_delta,
                              "partition_overlay_grid_retries": retried_intersections,
                              "outside_paint_cell_area_removed_m2": clipped_cell_area,
                              "min_interpolation_gap_m": float(gaps.min()),
                              "crossing_plane_partitions": len(cuts),
                              "numerical_sliver_triangles_removed": sliver_count,
                              "numerical_sliver_area_removed_m2": sliver_area,
                              "degenerate_after_offset": int((~keep).sum())}



def connect_reviewed_paint_lines(dissolved, groups, selections, snap_tolerance):
    """Join only Agent-selected display boundaries across a source interface.

    Source lanes and their boundaries are untouched. The finite paint-only
    collar shares one dash station sequence; outside it original line points
    are retained exactly.
    """
    result=dict(dissolved); reviews=[]
    for selection in selections:
        kind=selection['mark_type']
        lines=[line for line in shapely.get_parts(result[kind])
               if line.geom_type=='LineString']
        selected=[]
        for part in selection['parts']:
            source=[line for line,ref in groups[kind]
                    if ref['lane_id']==part['lane_id'] and ref['side']==part['side']]
            if not source:
                raise ValueError('Reviewed paint source is absent: '+str(part))
            anchor=np.asarray(part['endpoint_xy_m'],float)
            candidates=[]
            for index,line in enumerate(lines):
                ends=np.asarray(line.coords)[[0,-1]]
                distance=float(np.linalg.norm(ends-anchor,axis=1).min())
                if distance<=float(selection['endpoint_tolerance_m']):
                    near=shapely.buffer(line,max(snap_tolerance,1e-6))
                    if any(shapely.intersection(item,near).length>1e-6 for item in source):
                        candidates.append((index,line,int(np.argmin(np.linalg.norm(ends-anchor,axis=1)))))
            if len(candidates)!=1:
                raise ValueError('Reviewed paint endpoint is not unique: '+str(part))
            selected.append(candidates[0])
        if len(selected)!=2 or selected[0][0]==selected[1][0]:
            raise ValueError('Paint continuation requires two distinct source parts')
        ia,a,enda=selected[0]; ib,b,endb=selected[1]
        if enda==0:a=shapely.reverse(a)
        if endb==1:b=shapely.reverse(b)
        gap=float(np.linalg.norm(np.asarray(a.coords[-1])-b.coords[0]))
        if gap>float(selection['max_endpoint_gap_m']):
            raise ValueError('Reviewed paint continuation exceeds its finite gap')
        collar=float(selection['blend_length_m'])
        if min(a.length,b.length)<=collar:
            raise ValueError('Reviewed paint collar exceeds a retained source part')
        left=substring(a,0,a.length-collar); right=substring(b,collar,b.length)
        pa=np.asarray(left.coords[-1]); pb=np.asarray(right.coords[0])
        da=pa-np.asarray(a.interpolate(a.length-collar-min(.1,collar/2)).coords[0])
        db=np.asarray(b.interpolate(collar+min(.1,collar/2)).coords[0])-pb
        da/=np.linalg.norm(da); db/=np.linalg.norm(db)
        scale=float(np.linalg.norm(pb-pa))
        t=np.linspace(0.,1.,max(8,int(np.ceil(scale/.1))))[:,None]
        curve=(2*t**3-3*t**2+1)*pa+(t**3-2*t**2+t)*scale*da+(-2*t**3+3*t**2)*pb+(t**3-t**2)*scale*db
        joined=shapely.LineString([*list(left.coords)[:-1],*curve.tolist(),*list(right.coords)[1:]])
        if not joined.is_simple:raise ValueError('Reviewed paint continuation folded')
        result[kind]=shapely.MultiLineString([line for i,line in enumerate(lines) if i not in (ia,ib)]+[joined])
        reviews.append(dict(id=selection['id'],mark_type=kind,source_parts=selection['parts'],
            original_endpoint_gap_m=gap,blend_length_each_side_m=collar,
            joined_length_m=float(joined.length),connector_xy_m=curve.tolist(),
            unchanged_outside_connector=True,scope='Display paint only; one explicit default pattern phase. Original source boundaries, road support and navigation unchanged.'))
    return result,reviews


def build_markings(scene, roads, width=0.12, dash=3.0, gap=6.0, offset=0.002, diagnostics=None, conditions=None):
    if min(width, dash, gap, offset) <= 0:
        raise ValueError("Mark width, dash, gap and normal offset must be positive")
    source_lanes = [lane for lane in scene["fusion_map"]["lanes"] if lane["source"] == "av2"]
    from .source_paint import collect_source_lines, dissolve_and_conflicts, PATTERNS, reviewed_paint_ownership
    groups, skipped, counts = collect_source_lines(
        scene, (conditions or {}).get("reviewed_retained_source_sides", ()),
        (conditions or {}).get("reviewed_observation_sides", ()))
    selected = (conditions or {}).get('source_side_selection')
    if selected is not None:
        keys = {(row['lane_id'], row['side']) for row in selected}
        groups = {kind: [(line, ref) for line, ref in rows
                         if (ref['lane_id'], ref['side']) in keys]
                  for kind, rows in groups.items()}
        groups = {kind: rows for kind, rows in groups.items() if rows}
    groups, ownership_review = reviewed_paint_ownership(
        groups, (conditions or {}).get('reviewed_paint_ownership', []))
    snap_tolerance = float((conditions or {}).get('same_type_snap_tolerance_m', 0.))
    dissolved, conflicts, conflict_regions = dissolve_and_conflicts(groups, snap_tolerance)
    held_length = 0.
    if conflicts:
        if not conditions or conditions.get('conflict_policy') != 'hold_exact_shared_conflicts':
            raise ValueError("Conflicting AV2 paint types on one boundary: " + str(conflicts))
        # Remove unresolved shared display intervals while retaining source lane records.
        ambiguity = shapely.union_all(conflict_regions)
        held_length = float(ambiguity.length)
        dissolved = {kind: shapely.line_merge(shapely.difference(line, ambiguity))
                     for kind, line in dissolved.items()}
    dissolved, continuation_review = connect_reviewed_paint_lines(
        dissolved, groups, (conditions or {}).get('reviewed_continuations', []), snap_tolerance)
    kinds = sorted(dissolved)

    # Select complete road triangles touching the paint corridor.
    region = shapely.buffer(shapely.union_all(list(dissolved.values())), 2 * width,
                            join_style='mitre')
    # Project source-ground paint onto its road layer, excluding overpasses.
    _, layers = source_surface_layers(scene, roads)
    surface = RoadSurface(dict(meshes=layers[0.]), region, diagnostics)
    records, meshes, paint_by_color = [], [], {}
    for kind in kinds:
        color, stripe_patterns = PATTERNS[kind]
        pieces = []
        lines = [line for line in shapely.get_parts(dissolved[kind]) if line.geom_type == "LineString"]
        lines.sort(key=lambda line: (line.bounds, line.length))
        for line in lines:
            # Dash phase is a rendering parameter.
            coords = list(line.coords)
            if coords[-1] < coords[0]:
                line = shapely.reverse(line)
            sources = [ref for candidate, ref in groups[kind]
                       if shapely.intersection(candidate, shapely.buffer(line, snap_tolerance)
                                               if snap_tolerance else line).length > 1e-6]
            intervals = []
            offsets = [0.] if len(stripe_patterns) == 1 else [width, -width]
            for stripe_pattern, lateral_offset in zip(stripe_patterns, offsets):
                stripe_intervals = [(0., line.length)]
                if stripe_pattern == 'broken':
                    stripe_intervals = [(float(s), min(float(s) + dash, line.length))
                                        for s in np.arange(0, line.length, dash + gap)]
                intervals.append(dict(pattern=stripe_pattern, lateral_offset_m=lateral_offset,
                                      intervals_m=stripe_intervals))
                for start, end in stripe_intervals:
                    piece = substring(line, start, end)
                    if piece.length <= 1e-8:
                        continue
                    stroke = (shapely.offset_curve(piece, lateral_offset, join_style='mitre')
                              if lateral_offset else piece)
                    pieces.append(shapely.buffer(stroke, width / 2, cap_style='flat', join_style='mitre'))
            records.append({"id": "boundary_%03d" % len(records), "mark_type": kind,
                            "sources": sources, "centerline_xy_m": list(line.coords),
                            "length_m": line.length, "paint_intervals_m": intervals})
        paint = shapely.union_all(pieces)
        clipped = shapely.intersection(paint, surface.footprint)
        paint_by_color.setdefault(color, []).append(clipped)
        records.append({"mark_type": kind, "summary": True,
                        "source_side_count": len(groups[kind]), "dissolved_line_count": len(lines),
                        "raw_boundary_length_m": sum(line.length for line, _ in groups[kind]),
                        "unique_boundary_length_m": float(dissolved[kind].length),
                        "requested_paint_area_m2": paint.area, "clipped_paint_area_m2": clipped.area,
                        "outside_road_area_removed_m2": max(0.0, paint.area - clipped.area)})

    for color, parts in sorted(paint_by_color.items()):
        paint = shapely.union_all(parts)
        lifted, checks = surface.cut_and_drape(paint, offset)
        meshes.append({"id": "road_markings_" + color, "color": color,
                       "vertices": lifted.reshape(-1, 3).tolist(),
                       "faces": np.arange(lifted.size // 3).reshape(-1, 3).tolist(),
                       "semantic_class": "RoadLines", "semantic_id": 24,
                       "ue_asset_category": "RoadLine", "collision_required": "NoCollision",
                       "cast_shadow_required": False, "paint_area_m2": paint.area, "projection": checks})
    report = {"source_lane_count": len(source_lanes), "source_mark_type_counts": dict(counts),
              "zero_area_source_triangles_excluded": surface.zero_area_triangle_count,
              "paint_support_triangles": len(surface.triangles),
              "painted_source_sides": sum(len(items) for items in groups.values()),
              "boundary_records": records, "skipped": skipped, "conflicts": conflicts,
              "held_conflicting_boundary_length_m": held_length, "agent_conditions": conditions,
              "reviewed_continuations": continuation_review,
              "reviewed_paint_ownership": ownership_review,
              "mesh_count": len(meshes), "triangles": sum(len(mesh["faces"]) for mesh in meshes),
              "parameters": {"width_m": width, "dash_m": dash, "gap_m": gap,
                             "double_solid_inner_gap_m": width,
                             "normal_offset_m": offset, "line_dissolve_precision_m": 1e-7,
                             "same_type_snap_tolerance_m": snap_tolerance,
                             "surface_partition_precision_m": 1e-8,
                             "minimum_paint_triangle_altitude_m": 1e-7},
              "provenance": {"boundary_xy_and_paint_type": "retained AV2 source lane boundaries",
                             "surface_z_and_normal": "supplied source-ground road triangles; highest surface at same-level overlaps; separately tagged decks excluded",
                             "display_defaults": DSC_URL + "/blob/main/addon/road_properties.py",
                             "pattern_method": DSC_URL + "/blob/main/addon/road.py:get_strips_s_boundaries",
                             "geometry_operations": "GEOS line union/merge, substring buffer, road-edge polygonize, constrained Delaunay"},
              "assumptions": ["AV2 does not provide paint width, dash length, gap or phase; all are explicit display defaults.",
                              "NONE is no paint, UNKNOWN stays unknown; intersection status never substitutes for mark_type.",
                              "Unsupported observed types are reported, not converted to solid paint.",
                              "Compound stripe order follows official AV2 side semantics; reciprocal observations are normalized before comparison.",
                              "Any exact conflicting source line is held only under an explicit recorded condition; original source facts and navigation remain unchanged.",
                              "Independent visual meshes only; importer must set NoCollision and CastShadow=false.",
                              "Source lane, road mesh, OpenDRIVE and source elevation are unchanged."],
              "max_surface_interpolation_gap_m": max((m["projection"]["max_interpolation_gap_m"] for m in meshes), default=0.0)}
    report["passed"] = bool(meshes) and report["max_surface_interpolation_gap_m"] <= 0.001
    return {"frame": "SceneIR/Blender ENU metres", "meshes": meshes, "report": report}
