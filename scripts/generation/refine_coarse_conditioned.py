"""Prepare, condition, and decode local Scenario Dreamer road refinements."""

import argparse


from copy import deepcopy


import json


from pathlib import Path


import sys


import numpy as np


from shapely.geometry import LineString, Point


from scripts.generation.generate_coarse_conditioned import (
    configuration, curvature, encoder_data, prepare, save,
)


def add_initial_seam(prepared):
    from utils.lane_graph_helpers import resample_polyline
    seeded = deepcopy(prepared)
    index = len(prepared["context"])
    seam = prepared["seam"]
    xy = (np.asarray(seam["centerline"])[:, :2] - prepared["center"]) @ prepared["rotation"].T
    seed_points = resample_polyline(xy, num_points=20)
    seeded["points"] = np.concatenate([prepared["points"], seed_points[None]])
    seeded["context"].append({"id": seam["id"], "source": seam["source"], "width_m": seam["width_m"],
                              "role": "rule-based initialization; this latent will be diffused"})
    relations = np.pad(prepared["relations"], ((0, 1), (0, 1)))
    ids = {row["id"]: i for i, row in enumerate(seeded["context"])}
    incoming, outgoing = ids[prepared["incoming"]["id"]], ids[prepared["outgoing"]["id"]]
    relations[incoming, index], relations[index, incoming] = 2, 1
    relations[index, outgoing], relations[outgoing, index] = 2, 1
    relations[index, index] = 5
    seeded["relations"] = relations
    return seeded, index


def refinement_data(model, seeded, seed_index):
    import torch
    from torch_geometric.data import Batch
    from utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite
    count = len(seeded["context"])
    d = model._build_ldm_dset_from_ae_dset_for_inpainting([encoder_data(seeded, model.cfg_dataset)], 1, 1)[0]
    d["num_lanes"], d["num_agents"] = count, 1
    for kind, n in (("lane", count), ("agent", 1)):
        d[kind].latents = d[kind].latents[:n]
        d[kind].x = torch.zeros_like(d[kind].latents)
    d["lane"].ids = d["lane"].ids[:count]
    generated_ids = torch.as_tensor(np.atleast_1d(seed_index), dtype=d['lane'].ids.dtype)
    d["lane"].mask = ~torch.isin(d["lane"].ids, generated_ids)
    d["agent"].mask = torch.ones(1).bool()
    d["lane", "to", "lane"].edge_index = get_edge_index_complete_graph(count)
    d["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(1)
    d["lane", "to", "agent"].edge_index = get_edge_index_bipartite(count, 1)
    return Batch.from_data_list([d]).to(model.device)


def decode(model, data, agents, lanes, output, name):
    import torch
    from utils.data_helpers import unnormalize_latents, unnormalize_scene, convert_batch_to_scenarios
    cfg = model.cfg_dataset
    agents, lanes = unnormalize_latents(agents, lanes, cfg.agent_latents_mean, cfg.agent_latents_std,
                                        cfg.lane_latents_mean, cfg.lane_latents_std)
    a, l, a_types, _, links = model.autoencoder.model.forward_decoder(agents, lanes, data)
    fields = ("fov", "min_speed", "max_speed", "min_length", "max_length", "min_width", "max_width",
              "min_lane_x", "max_lane_x", "min_lane_y", "max_lane_y")
    a, l = unnormalize_scene(a, l, **{key: cfg[key] for key in fields})
    decoded = data.clone()
    decoded["agent"].x, decoded["lane"].x = a, l
    decoded["agent"].type = torch.nn.functional.one_hot(a_types, num_classes=cfg.num_agent_types)
    decoded["lane", "to", "lane"].type = links
    return next(iter(convert_batch_to_scenarios(decoded, 1, 0, str(output), conditioning_filenames=[name],
                                              cache_samples=True, cache_lane_types=False, mode="inpainting").values()))

