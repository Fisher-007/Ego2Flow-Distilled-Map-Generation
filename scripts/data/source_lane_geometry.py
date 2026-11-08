"""Read measured source lane centerlines while retaining unknown height."""
import numpy as np
from av2.geometry.interpolate import compute_midpoint_line


def source_centerline(static_map, lane):
    left, right = lane.left_lane_boundary.xyz, lane.right_lane_boundary.xyz
    if np.isfinite(left).all() and np.isfinite(right).all():
        return static_map.get_lane_segment_centerline(lane.id)
    # Use the 2D midpoint operation for lane lookup when source Z is unavailable.
    if not np.isfinite(left[:, :2]).all() or not np.isfinite(right[:, :2]).all():
        raise ValueError(f"Nonfinite source XY in lane {lane.id}")
    xy, _ = compute_midpoint_line(left[:, :2], right[:, :2])
    return np.column_stack([xy, np.full(len(xy), np.nan)])

