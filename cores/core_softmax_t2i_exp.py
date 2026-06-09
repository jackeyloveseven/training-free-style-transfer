"""T2I softmax ablation wrapper for paper experiments.

This module keeps the production core untouched and only replaces Sinkhorn
attention with standard scaled dot-product attention inside this process.
"""

import torch.nn.functional as F

import cores.core_ot_attn as core_ot_attn
import utils


def _softmax_attn(q, k, v, n_iters=None, tau=None):
    return F.scaled_dot_product_attention(q, k, v)


def step(**kwargs):
    original = utils.sinkhorn_attn
    utils.sinkhorn_attn = _softmax_attn
    try:
        return core_ot_attn.step(**kwargs)
    finally:
        utils.sinkhorn_attn = original
