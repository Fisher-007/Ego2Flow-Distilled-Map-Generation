"""Normalize directional AV2 paint descriptions without changing observations."""
from collections import Counter
import numpy as np
import shapely
from map_generation.lane_sections import marking_pattern

PATTERNS = {
    "SOLID_WHITE": ("white", ("solid",)),
    "DASHED_WHITE": ("white", ("broken",)),
    "SOLID_YELLOW": ("yellow", ("solid",)),
    "DASHED_YELLOW": ("yellow", ("broken",)),
    "DOUBLE_SOLID_WHITE": ("white", ("solid", "solid")),
    "DOUBLE_SOLID_YELLOW": ("yellow", ("solid", "solid")),
    "DOUBLE_DASH_YELLOW": ("yellow", ("broken", "broken")),
    "COMPOUND_SOLID_BROKEN_YELLOW": ("yellow", ("solid", "broken")),
    "COMPOUND_BROKEN_SOLID_YELLOW": ("yellow", ("broken", "solid")),
}


def collect_source_lines(scene, reviewed_retained_source_sides=(), reviewed_observation_sides=()):
    """Use canonical line direction so reciprocal compound markings agree.

    Two-stripe tuples are positive-normal then negative-normal. The original
    AV2 side/type stays in references; reversing a line reverses stripe order.
    """
    reviewed = {(row["lane_id"], row["side"]): row
                for row in reviewed_retained_source_sides}
    observation_keys = {(row["lane_id"], row["side"]) for row in reviewed_observation_sides}
    found = set()
    groups, skipped, counts = {}, [], Counter()
    for lane in scene["fusion_map"]["lanes"]:
        for side in ("left", "right"):
            key = (lane["id"], side)
            retained = reviewed.get(key)
            if key in observation_keys:
                continue
            if lane["source"] != "av2":
                if retained is None:
                    continue
                # Reconstruction may change the lane category while preserving
                # its observed boundary. Only reviewed exact sides are restored.
                if lane.get("original_source") != "av2" or not lane.get("source_observations"):
                    raise ValueError("Reviewed paint side lacks AV2 provenance: " + str(key))
                observed_xy = np.asarray(lane[side + "_boundary"])[:, :2]
                if (lane.get(side + "_mark_type") != retained["mark_type"]
                        or not np.array_equal(observed_xy, retained["boundary_xy_m"])):
                    raise ValueError("Reviewed retained boundary changed: " + str(key))
            if retained is not None:
                found.add(key)
            kind = lane.get(side + "_mark_type", "UNKNOWN")
            counts[kind] += 1
            ref = dict(lane_id=lane["id"], side=side, mark_type=kind,
                       is_intersection=lane["is_intersection"],
                       source_observations=lane.get("source_observations", []))
            if retained is not None:
                ref.update(retained_source_category=lane["source"],
                           original_source=lane.get("original_source", lane["source"]),
                           source_observations=lane.get("source_observations", []))
            if kind not in PATTERNS and kind not in ("SOLID_DASH_YELLOW", "DASH_SOLID_YELLOW"):
                reason = {"NONE": "source explicitly has no painted marking",
                          "UNKNOWN": "source marking remains unknown"}.get(kind, "known type not implemented")
                skipped.append(dict(ref, reason=reason))
                continue
            points = np.asarray(lane[side + "_boundary"], dtype=float)
            reverse = tuple(points[-1, :2]) < tuple(points[0, :2])
            normalized = kind
            if kind in ("SOLID_DASH_YELLOW", "DASH_SOLID_YELLOW"):
                positive, negative, color = marking_pattern(kind, side)
                stripes = tuple("BROKEN" if s == "DASH" else s for s in (positive, negative))
                if reverse:
                    stripes = stripes[::-1]
                normalized = "COMPOUND_" + "_".join(s.upper() for s in stripes) + "_" + color
            if reverse:
                points = points[::-1]
            line = shapely.set_precision(shapely.LineString(points[:, :2]), 1e-7)
            groups.setdefault(normalized, []).append((line, ref))
    if found != set(reviewed):
        raise ValueError("Reviewed paint sides missing from scene: " + str(set(reviewed) - found))
    if reviewed_observation_sides:
        lanes = {lane["id"]: lane for lane in scene["fusion_map"]["lanes"]}
        origin = np.asarray(scene["coordinates"]["origin_city_xyz"], dtype=float)
        for selected in reviewed_observation_sides:
            lane = lanes[selected["lane_id"]]
            side = selected["side"]
            matches = [row for row in lane["source_observations"]
                       if row["source"] == selected["observation_source"]
                       and row["source_lane_id"] == selected["observation_lane_id"]]
            if len(matches) != 1:
                raise ValueError("Expected one reviewed source observation: " + str(selected))
            observation = matches[0]
            difference = observation["definition_differences"]
            boundary = difference[side + "_lane_boundary"]
            points = np.asarray([[point[axis] for axis in ("x", "y", "z")]
                                 for point in boundary], dtype=float) - origin
            kind = difference.get(side + "_lane_mark_type", lane[side + "_mark_type"])
            if kind != selected["mark_type"]:
                raise ValueError("Reviewed observation marking type changed")
            # Normalize paint observations in the registered city frame.
            observed = dict(lane, source="av2",
                            is_intersection=difference.get("is_intersection", lane["is_intersection"]),
                            left_mark_type="NONE", right_mark_type="NONE")
            observed[side + "_boundary"] = points.tolist()
            observed[side + "_mark_type"] = kind
            extra, _, extra_counts = collect_source_lines({"fusion_map": {"lanes": [observed]}})
            for normalized, rows in extra.items():
                for line, ref in rows:
                    ref.update(observation_source=observation["source"],
                               observation_lane_id=observation["source_lane_id"],
                               geometry_source="registered AV2 source observation",
                               reconstructed_lane_source=lane["source"])
                    groups.setdefault(normalized, []).append((line, ref))
            counts.update(extra_counts)
    return groups, skipped, counts


def coalesce_source_lines(rows, tolerance):
    """Unify reviewed nearly coincident paint observations, retaining tails.

    Longest original observations anchor the result. Other source vertices and
    segments snap only within the explicit paint tolerance. Source lane data,
    access rules and physical road meshes are not edited.
    """
    from shapely.ops import nearest_points
    accepted = shapely.GeometryCollection()
    for line, _ in sorted(rows, key=lambda row: (-row[0].length, row[0].bounds)):
        if not accepted.is_empty:
            points = shapely.get_coordinates(accepted).tolist()
            for xy in line.coords:
                point = shapely.Point(xy)
                nearest = nearest_points(point, accepted)[1]
                if point.distance(nearest) <= tolerance:
                    points.append(list(nearest.coords[0]))
            line = shapely.snap(line, shapely.MultiPoint(points), tolerance).simplify(1e-7)
        accepted = shapely.union_all([accepted, line], grid_size=1e-6)
    return shapely.line_merge(accepted)


def dissolve_and_conflicts(groups, snap_tolerance_m=0.):
    dissolved = {
        kind: (coalesce_source_lines(rows, snap_tolerance_m) if snap_tolerance_m > 0
               else shapely.line_merge(shapely.union_all([line for line, _ in rows])))
        for kind, rows in groups.items()}
    conflicts, regions = [], []
    kinds = sorted(dissolved)
    for i, first in enumerate(kinds):
        for second in kinds[i + 1:]:
            overlap = shapely.intersection(dissolved[first], dissolved[second])
            if overlap.length <= 1e-6:
                continue
            refs = {kind: [ref for line, ref in groups[kind]
                          if line.intersection(overlap).length > 1e-6]
                    for kind in (first, second)}
            conflicts.append(dict(types=[first, second], shared_length_m=overlap.length,
                                  source_sides=refs, geometry=shapely.to_geojson(overlap)))
            regions.append(overlap)
    return dissolved, conflicts, regions


def reviewed_paint_ownership(groups, selections):
    """Apply source-log and epoch ownership to selected paint intervals."""
    report=[]
    for selection in selections:
        domain=shapely.Polygon(selection['domain']['exterior'],selection['domain'].get('holes',[]))
        keep=set(selection['keep_source_logs']);retire=set(selection['retire_source_logs'])
        if not selection.get('reason') or not selection.get('evidence'):
            raise ValueError('Paint ownership requires a recorded source decision')
        changed=[]
        for kind,rows in groups.items():
            if kind not in selection['mark_types']:continue
            result=[]
            for line,ref in rows:
                sources=[ref['observation_source']]if ref.get('observation_source')else[x['source']for x in ref.get('source_observations',[])]
                selected=any(log in source for log in retire for source in sources)and not any(log in source for log in keep for source in sources)
                if selected:
                    after=shapely.difference(line,domain)
                    removed=line.length-after.length
                    if removed>1e-7:changed.append(dict(reference=ref,retired_display_length_m=float(removed)))
                    for part in shapely.get_parts(after):
                        if part.geom_type=='LineString':result.append((part,ref))
                else:result.append((line,ref))
            groups[kind]=result
        report.append(dict(condition=selection,changed_source_sides=changed,raw_lane_geometry_modified=False,navigation_permissions_modified=False))
    return groups,report
