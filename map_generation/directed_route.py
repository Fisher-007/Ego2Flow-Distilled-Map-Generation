"""Coalesce selected duplicate controller targets along a native source route."""
import numpy as np


def coalesce_directed_indices(points_enu, road_ids, conditions):
    points = np.asarray(points_enu, dtype=float)
    roads = np.asarray(road_ids)
    removed = set()
    reviews = []
    for condition in conditions:
        axis = np.asarray(condition["direction_xy"], dtype=float)
        axis /= np.linalg.norm(axis)
        relative = points[:, :2] - condition["origin_xy_m"]
        station = relative @ axis
        lateral = relative @ np.array([-axis[1], axis[0]])
        low, high = condition["station_interval_m"]
        eligible = (np.isin(roads, condition["native_road_ids"])
                    & (station >= low) & (station <= high)
                    & (abs(lateral) <= condition["corridor_half_width_m"]))
        selected = np.flatnonzero(eligible)
        blocks = np.split(selected, np.flatnonzero(np.diff(selected) != 1) + 1)
        skipped = []
        maximum_retrace = 0.
        anchor_side = condition.get("anchor_side", "upstream")
        # For overlapping route tails, preserve the selected upstream or downstream handoff.
        direction = -1 if anchor_side == "downstream" else 1
        for block in blocks:
            if len(block) < 3:
                continue
            block = block[::direction]
            anchor = int(block[0])
            for index in block[1:]:
                index = int(index)
                retrace = direction * float(station[anchor] - station[index])
                if retrace < -condition.get("station_tolerance_m", .05):
                    anchor = index
                    continue
                if index in (0, len(points) - 1):
                    raise ValueError("Reviewed overlap would remove a route endpoint")
                if retrace > condition["maximum_retrace_m"]:
                    raise ValueError("Route retraces beyond its reviewed source overlap")
                if abs(points[index, 2] - points[anchor, 2]) > condition["maximum_height_difference_m"]:
                    raise ValueError("Reviewed duplicate targets are on different height profiles")
                skipped.append(index)
                removed.add(index)
                maximum_retrace = max(maximum_retrace, retrace)
        skipped.sort()
        reviews.append(dict(condition_id=condition["id"], anchor_side=anchor_side,
                            skipped_original_indices=skipped,
                            skipped_native_road_ids=[int(roads[i]) for i in skipped],
                            maximum_retrace_m=maximum_retrace,
                            source_evidence=condition["evidence"]))
    indices = [i for i in range(len(points)) if i not in removed]
    return indices, dict(original_targets=len(points), controller_targets=len(indices),
                         conditions=reviews, original_route_and_control_references_retained=True,
                         scope="Explicit source-qualified controller targets only; native geometry and graph remain unchanged.")
