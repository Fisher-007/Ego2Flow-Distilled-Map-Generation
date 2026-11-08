"""Prepare source geometry and latent inputs for conditional road generation."""

import argparse


from collections import Counter


import json
import os


from pathlib import Path


import sys


import numpy as np


from shapely.geometry import LineString, Polygon, box


from shapely.ops import unary_union


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def configuration(source):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(source / "cfgs"), version_base=None):
        cfg = compose(config_name="config", overrides=["dataset_name=waymo", "model_name=ldm",
            "ldm.eval.run_name=scenario_dreamer_ldm_large_waymo", "ldm.model.num_l2l_blocks=3",
            "ldm.model.autoencoder_run_name=scenario_dreamer_autoencoder_waymo"])
    for child in (cfg.ldm, cfg.ae):
        OmegaConf.set_struct(child, False)
        child.dataset_name = "waymo"
    if os.environ.get("MAP_SD_CHECKPOINT"):
        checkpoint = Path(os.environ["MAP_SD_CHECKPOINT"])
        cfg.ldm.eval.save_dir = str(checkpoint.parent.parent)
        cfg.ldm.eval.run_name = checkpoint.parent.name
    if os.environ.get("MAP_SD_AUTOENCODER"):
        cfg.ldm.model.autoencoder_path = os.environ["MAP_SD_AUTOENCODER"]
    return cfg.ldm, cfg.ae


def prepare(scene, seam_id, cfg):
    from utils.lane_graph_helpers import resample_polyline
    lanes = {row["id"]: row for row in scene["fusion_map"]["lanes"]}
    seam = lanes[seam_id]
    incoming = next(lane for lane in lanes.values() if seam_id in lane["successors"])
    outgoing = lanes[seam["successors"][0]]
    ends = np.asarray([incoming["centerline"][-1], outgoing["centerline"][0]])
    center = ends[:, :2].mean(axis=0)
    delta = ends[1, :2] - ends[0, :2]
    yaw = float(np.arctan2(delta[1], delta[0]))
    angle = np.pi / 2 - yaw
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    field = box(-cfg.fov / 2 + .01, -cfg.fov / 2 + .01, cfg.fov / 2 - .01, cfg.fov / 2 - .01)
    context, points = [], []
    for lane in sorted(lanes.values(), key=lambda row: row["id"]):
        if lane["source"] not in {"av2", "osm_sumo"}:
            continue
        normalized = (np.asarray(lane["centerline"])[:, :2] - center) @ rotation.T
        clipped = LineString(normalized).intersection(field)
        pieces = [p for p in getattr(clipped, "geoms", [clipped]) if p.geom_type == "LineString" and p.length > .1]
        if not pieces:
            continue
        # The adapter retains the longest connected in-FOV portion per source lane.
        clipped = max(pieces, key=lambda p: p.length)
        points.append(resample_polyline(np.asarray(clipped.coords), num_points=20))
        context.append({"id": lane["id"], "source": lane["source"], "source_lane_id": lane.get("source_lane_id"),
                        "width_m": lane["width_m"], "clipped_component_count": len(pieces)})
    ids = {row["id"]: i for i, row in enumerate(context)}
    if incoming["id"] not in ids or outgoing["id"] not in ids:
        raise ValueError("Both source boundaries must be present in the context")
    relations = np.zeros((len(context), len(context)), dtype=np.int64)
    for key, i in ids.items():
        for target in lanes[key]["successors"]:
            if target in ids:
                relations[i, ids[target]], relations[ids[target], i] = 2, 1
    for key, i in ids.items():
        for neighbor, relation in (("left_neighbor", 3), ("right_neighbor", 4)):
            if lanes[key].get(neighbor) in ids:
                relations[i, ids[lanes[key][neighbor]]] = relation
    np.fill_diagonal(relations, 5)
    return {"seam": seam, "incoming": incoming, "outgoing": outgoing, "ends": ends,
            "center": center, "yaw": yaw, "rotation": rotation, "context": context,
            "points": np.asarray(points), "relations": relations}


def encoder_data(prepared, cfg):
    import torch
    from cfgs.config import PARTITIONED
    from utils.data_container import ScenarioDreamerData
    from utils.data_helpers import normalize_scene
    from utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite
    from utils.torch_helpers import from_numpy
    fields = ("fov", "min_speed", "max_speed", "min_length", "max_length", "min_width", "max_width",
              "min_lane_x", "max_lane_x", "min_lane_y", "max_lane_y")
    agents, lanes = normalize_scene(np.array([[0., 0., 15 / 3.6, 0., 1., 4.7, 1.85]]), prepared["points"].copy(),
                                   **{key: cfg[key] for key in fields})
    count = len(lanes)
    d = ScenarioDreamerData()
    d["map_id"], d["lg_type"], d["num_lanes"], d["num_agents"] = 0, PARTITIONED, count, 1
    d["agent"].x, d["agent"].type = from_numpy(agents), from_numpy(np.array([[1., 0., 0.]]))
    d["lane"].x = from_numpy(lanes)
    d["lane"].partition_mask = torch.ones(count).bool()
    d["lane"].ids = torch.arange(count)
    d["lane", "to", "lane"].edge_index = get_edge_index_complete_graph(count)
    # Official training edge(src,dst) uses adjacency[dst,src], not adjacency[src,dst].
    d["lane", "to", "lane"].type = from_numpy(np.eye(6)[prepared["relations"].T.reshape(-1)])
    d["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(1)
    d["lane", "to", "agent"].edge_index = get_edge_index_bipartite(count, 1)
    for key in (("lane", "to", "lane"), ("lane", "to", "agent"), ("agent", "to", "agent")):
        d[key].encoder_mask = torch.ones(d[key].edge_index.shape[1]).bool()
    return d




def curvature(points):
    d = np.diff(points, axis=0)
    length = np.linalg.norm(d, axis=1)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return float(np.max(np.abs(np.diff(heading)) / np.maximum((length[1:] + length[:-1]) / 2, 1e-6)))
