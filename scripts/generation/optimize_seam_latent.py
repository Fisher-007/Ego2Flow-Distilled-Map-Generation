"""Optimize road latents under endpoint, tangent, and shape constraints."""

import argparse


import json


from pathlib import Path


import sys


import numpy as np


from shapely.geometry import LineString, Polygon


from shapely.ops import unary_union


from scripts.generation.measure_candidate import measure


from scripts.generation.refine_coarse_conditioned import (
    add_initial_seam, configuration, decode, prepare, refinement_data, save,
)


PARAMETERS = {"steps": 100, "learning_rate": .005, "endpoint_weight": 1.,
              "tangent_weight": .25, "smoothness_weight": .1, "latent_weight": .01}


def inspect_unprojected(sample, prepared, buildings, tangents):
    """Use the existing relation convention and eligibility, without projection."""
    result = measure(sample, prepared)
    points = np.asarray(result["centerline_xy_enu_m"])
    directions = np.stack([points[1] - points[0], points[-1] - points[-2]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.sum(directions * tangents, axis=1), -1, 1)))
    inside = float(LineString(points).intersection(buildings).length)
    result.update(heading_errors_deg=angles.tolist(), inside_buildings_m=inside)
    ids = np.asarray(sample["lane_ids"], dtype=int)
    mask = np.asarray(sample["lane_mask"], dtype=bool)
    matrix = np.asarray(sample["road_connection_types"]).argmax(-1).reshape(len(mask), -1).T
    expected = prepared["relations"][np.ix_(ids[mask], ids[mask])]
    predicted = matrix[np.ix_(mask, mask)]
    known_ids = [prepared["context"][i]["id"] for i in ids[mask]]
    result["known_relation_changes"] = [
        {"from": known_ids[i], "to": known_ids[j], "expected_type": int(expected[i, j]),
         "decoded_type": int(predicted[i, j])} for i, j in zip(*np.nonzero(expected != predicted))]
    result["existing_eligibility_passed"] = bool(
        max(result["endpoint_gaps_m"]) <= .5 and max(angles) <= 20 and inside < .01
        and result["boundary_relation_types"] == [2, 2])
    result["builder_lane_proposal_written"] = False
    result["endpoint_projection"] = None
    result["builder_status"] = "Unprojected candidate only; exact endpoint connection is a separate builder step"
    return result


def continuous_geometry(model, data, agents, lanes):
    from utils.data_container import get_edge_indices
    from utils.data_helpers import unnormalize_latents
    import torch
    cfg = model.cfg_dataset
    a, l = unnormalize_latents(agents, lanes, cfg.agent_latents_mean, cfg.agent_latents_std,
                               cfg.lane_latents_mean, cfg.lane_latents_std)
    decoded = model.autoencoder.model.decoder(a, l, *get_edge_indices(data))
    points = decoded[3]
    # Use functional affine transforms to retain gradients through lane unnormalization.
    return torch.stack([
        ((points[..., 0].clamp(-1000, 1000) + 1) / 2) * (cfg.max_lane_x - cfg.min_lane_x) + cfg.min_lane_x,
        ((points[..., 1].clamp(-1000, 1000) + 1) / 2) * (cfg.max_lane_y - cfg.min_lane_y) + cfg.min_lane_y,
    ], dim=-1)


def optimize(model, data, agents, lanes, seam_index, endpoints, tangents, output):
    import torch
    output.mkdir(parents=True, exist_ok=True)
    initial = lanes[seam_index].detach().clone()
    parameter = initial.clone().requires_grad_(True)
    indices = torch.tensor([seam_index], device=model.device)
    optimizer = torch.optim.Adam([parameter], lr=PARAMETERS["learning_rate"])
    history = []
    with torch.no_grad():
        raw = decode(model, data, agents, lanes, output, "raw")
    for step in range(PARAMETERS["steps"]):
        optimizer.zero_grad(set_to_none=True)
        current = lanes.index_copy(0, indices, parameter[None])
        points = continuous_geometry(model, data, agents, current)[seam_index]
        endpoint_loss = (points[[0, -1]] - endpoints).square().sum(-1).mean()
        directions = torch.stack([points[1] - points[0], points[-1] - points[-2]])
        directions = torch.nn.functional.normalize(directions, dim=-1)
        tangent_loss = (directions - tangents).square().sum(-1).mean()
        smoothness_loss = (points[2:] - 2 * points[1:-1] + points[:-2]).square().sum(-1).mean()
        latent_loss = (parameter - initial).square().sum()
        terms = {"endpoint": endpoint_loss, "tangent": tangent_loss,
                 "smoothness": smoothness_loss, "latent": latent_loss}
        loss = sum(PARAMETERS[key + "_weight"] * value for key, value in terms.items())
        loss.backward()
        history.append({"step_before_update": step, "loss": float(loss.detach().cpu()),
                        **{key: float(value.detach().cpu()) for key, value in terms.items()}})
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0:
            print(output.name, "completed_updates", step + 1, "loss_before_update", history[-1]["loss"], flush=True)
    final = lanes.index_copy(0, indices, parameter.detach()[None])
    with torch.no_grad():
        optimized = decode(model, data, agents, final, output, "optimized")
        functional_points = continuous_geometry(model, data, agents, final)[seam_index].cpu().numpy()
    torch.save({"raw_lanes": lanes.cpu(), "optimized_lanes": final.cpu(), "agents": agents.cpu(),
                "lane_mask": data["lane"].mask.cpu()}, output / "latents.pt")
    save(output / "optimization_history.json", history)
    return raw, optimized, {
        "completed_updates": len(history), "seam_latent_l2_change": float(torch.linalg.norm(parameter.detach() - initial).cpu()),
        "fixed_context_latent_max_change": float((final[data["lane"].mask] - lanes[data["lane"].mask]).abs().max().cpu()),
        "agent_latent_max_change": 0.,
        "functional_vs_official_decode_max_coordinate_difference_m": float(np.abs(functional_points - optimized["road_points"][seam_index]).max()),
    }

