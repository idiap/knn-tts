# SPDX-FileCopyrightText: 2023 MediaLab, Department of Electrical & Electronic Engineering, Stellenbosch University
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import os

import torch


def load_target_style_feats(feats_base_path, max_num_files=1000):
    feats = []
    for filepath in os.listdir(feats_base_path)[:max_num_files]:
        if ".pt" in filepath:
            filepath = os.path.join(feats_base_path, filepath)
            feats.append(torch.load(filepath, weights_only=False))
    feats = torch.concat(feats, dim=0).cpu()
    return feats


def fast_cosine_dist(source_feats, matching_pool, device):
    """Like torch.cdist, but fixed dim=-1 and for cosine distance."""
    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -(torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0] ** 2) + source_norms[:, None] ** 2 + matching_norms[None] ** 2
    dotprod /= 2

    dists = 1 - (dotprod / (source_norms[:, None] * matching_norms[None]))
    return dists


@torch.inference_mode()
def knn_vc(source_frames, target_style_set, topk=4, weighted_average=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    target_style_set = target_style_set.to(device)
    source_frames = source_frames.to(device)

    dists = fast_cosine_dist(source_frames, target_style_set, device=device)
    best = dists.topk(k=topk, largest=False, dim=-1)

    if weighted_average:
        weights = 1 / (best.values + 1e-8)  # Adding a small value to avoid division by zero
        weights /= weights.sum(dim=-1, keepdim=True)  # Normalize weights
        selected_frames = (target_style_set[best.indices] * weights[..., None]).sum(dim=1)
    else:
        selected_frames = target_style_set[best.indices].mean(dim=1)

    return selected_frames

