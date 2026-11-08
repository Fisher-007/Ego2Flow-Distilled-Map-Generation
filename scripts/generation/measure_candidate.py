"""Measure a decoded local road against its source boundary conditions."""
import numpy as np
from scripts.generation.generate_coarse_conditioned import curvature


def measure(sample, prepared):
    mask = np.asarray(sample["lane_mask"], dtype=bool)
    ids = np.asarray(sample["lane_ids"], dtype=int)
    index = int(np.flatnonzero(~mask)[0])
    points = sample["road_points"][index] @ prepared["rotation"] + prepared["center"]
    links = np.asarray(sample["road_connection_types"]).argmax(-1).reshape(len(mask), -1).T
    positions = {prepared["context"][int(ids[i])]["id"]: i for i in np.flatnonzero(mask)}
    return {
        "centerline_xy_enu_m": points.tolist(),
        "endpoint_gaps_m": np.linalg.norm(points[[0, -1]] - prepared["ends"][:, :2], axis=1).tolist(),
        "boundary_relation_types": [int(links[positions[prepared["incoming"]["id"]], index]),
                                    int(links[index, positions[prepared["outgoing"]["id"]]])],
        "required_relation_type": 2,
        "max_curvature_per_m": curvature(points),
    }

