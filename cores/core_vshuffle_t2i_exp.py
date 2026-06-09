"""T2I V-shuffle baseline wrapper for paper experiments.

Keeps production core untouched. The wrapper replaces Sinkhorn with standard
attention over a deterministic shuffle of style V tokens.
"""

import torch
import torch.nn.functional as F

import cores.core_ot_attn as core_ot_attn
import utils


def _vshuffle_attn(q, k, v, n_iters=None, tau=None):
    generator = torch.Generator(device=v.device).manual_seed(42)
    perm = torch.randperm(v.shape[2], device=v.device, generator=generator)
    return F.scaled_dot_product_attention(q, k, v[:, :, perm, :])


def step(**kwargs):
    original = utils.sinkhorn_attn
    utils.sinkhorn_attn = _vshuffle_attn
    try:
        return core_ot_attn.step(**kwargs)
    finally:
        utils.sinkhorn_attn = original
