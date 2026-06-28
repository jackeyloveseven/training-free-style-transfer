"""
Klein 9B (FLUX.2) OT-Attention Style Transfer — T2I version

Pure text-to-image generation with style optimization.
No content image — relies on text prompt for semantics,
style optimization for visual appearance.
"""
import glob
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

DIFFSYNTH_ROOT = os.environ.get("DIFFSYNTH_ROOT", "/mnt/local/yzh/prog/DiffSynth-Studio")
KLEIN9B_ROOT = os.environ.get("KLEIN9B_ROOT", "/mnt/local/klein9b")

# --- Parameters ---
seed = int(os.environ.get("SEED", "43"))
width = int(os.environ.get("WIDTH", "512"))
height = int(os.environ.get("HEIGHT", "512"))
num_inference_steps = int(os.environ.get("NUM_STEPS", "25"))
start_time = int(os.environ.get("OVERRIDE_START_TIME",
                                os.environ.get("START_TIME", "850")))
os.environ["OVERRIDE_START_TIME"] = str(start_time)
embedded_guidance = float(os.environ.get("EMBEDDED_GUIDANCE", "0"))
lr = float(os.environ.get("LR", "0.01"))
iters = int(os.environ.get("ITERS", "5"))
start_layer, end_layer = 12, 28
style_beta = float(os.environ.get("STYLE_BETA", "0.7"))
prompt = os.environ.get("PROMPT", "A panda eat bamboo")

style_image_path = os.environ.get("STY_IMAGE",
    "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty/27.png")


# --- DiffSynth setup ---
def _install_modelscope_stub():
    if "modelscope" in sys.modules:
        return
    module = types.ModuleType("modelscope")
    module.snapshot_download = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("modelscope not installed; use local weights"))
    sys.modules["modelscope"] = module

def _sorted_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files

if DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)
_install_modelscope_stub()

from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.diffusion.flow_match import FlowMatchScheduler


# --- Attention hooking (same as i2i version) ---
class Klein9BCache:
    def __init__(self): self.q, self.k, self.v, self.out = [], [], [], []
    def clear(self): self.q.clear(); self.k.clear(); self.v.clear(); self.out.clear()
    def add(self, q, k, v, out): self.q.append(q); self.k.append(k); self.v.append(v); self.out.append(out)
    def get(self): return self.q.copy(), self.k.copy(), self.v.copy(), self.out.copy()


class Klein9BController:
    def __init__(self, self_layers=(0, 16)):
        self.cur_self_layer = 0
        self.self_layers = list(range(*self_layers))
        self.txt_len = 0
        # Injection mode: "cache" (default) or "inject"
        self.mode = "cache"
        self.inject_ref_k = []
        self.inject_ref_v = []
        self.inject_beta = 0.0
        self.inject_idx = 0

    def step(self):
        self.cur_self_layer = 0
        self.inject_idx = 0

    def set_inject(self, ref_k_list, ref_v_list, beta):
        self.mode = "inject"
        self.inject_ref_k = ref_k_list
        self.inject_ref_v = ref_v_list
        self.inject_beta = beta
        self.inject_idx = 0

    def set_sinkhorn(self, ref_k_list, ref_v_list, tau=0.1, n_iters=6, beta=1.0, only_last_n=0, use_pattern=False):
        self.mode = "sinkhorn"
        self.inject_ref_k = ref_k_list
        self.inject_ref_v = ref_v_list
        self.sinkhorn_tau = tau
        self.sinkhorn_iters = n_iters
        self.inject_beta = beta
        self.inject_idx = 0
        self.sinkhorn_only_last_n = only_last_n
        # use_pattern: apply style's K-K self-correlation to content's V (no style content leaks)
        self.sinkhorn_use_pattern = use_pattern

    def reset_mode(self):
        self.mode = "cache"
        self.inject_ref_k = []
        self.inject_ref_v = []
        self.inject_beta = 0.0
        self.inject_idx = 0


def register_klein9b_attn_hooks(dit, controller, cache):
    from diffsynth.core.attention.attention import torch_sdpa

    def make_double_hook(original_block_attn):
        def hooked_processor(attn, hidden_states, encoder_hidden_states=None,
                             attention_mask=None, image_rotary_emb=None):
            from diffsynth.models.flux2_dit import _get_qkv_projections, apply_rotary_emb

            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states
            )
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            query = attn.norm_q(query)
            key = attn.norm_k(key)

            txt_len = 0
            if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
                encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
                encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
                encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
                encoder_query = attn.norm_added_q(encoder_query)
                encoder_key = attn.norm_added_k(encoder_key)
                txt_len = encoder_hidden_states.shape[1]
                controller.txt_len = txt_len
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            query = query.to(hidden_states.dtype)
            key = key.to(hidden_states.dtype)
            value = value.to(hidden_states.dtype)

            q_t = query.transpose(1, 2)
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)

            if controller.cur_self_layer in controller.self_layers:
                if controller.mode == "cache":
                    cache.add(q_t, k_t, v_t, attn_out_t)
                elif controller.mode == "inject" and controller.inject_idx < len(controller.inject_ref_k):
                    ref_k = controller.inject_ref_k[controller.inject_idx]
                    ref_v = controller.inject_ref_v[controller.inject_idx]
                    beta = controller.inject_beta
                    _tl = txt_len if txt_len > 0 else controller.txt_len
                    q_img = q_t[:, :, _tl:, :]
                    ref_out = F.scaled_dot_product_attention(q_img, ref_k, ref_v)
                    blended = (1.0 - beta) * attn_out_t[:, :, _tl:, :] + beta * ref_out
                    attn_out_t = attn_out_t.clone()
                    attn_out_t[:, :, _tl:, :] = blended
                    controller.inject_idx += 1
                elif controller.mode == "sinkhorn" and controller.inject_idx < len(controller.inject_ref_k):
                    ref_k = controller.inject_ref_k[controller.inject_idx]
                    ref_v = controller.inject_ref_v[controller.inject_idx]
                    _n_total = len(controller.inject_ref_k)
                    _oln = controller.sinkhorn_only_last_n
                    _apply_ot = (_oln <= 0) or (controller.inject_idx >= _n_total - _oln)
                    if _apply_ot:
                        _tl = txt_len if txt_len > 0 else controller.txt_len
                        q_img = q_t[:, :, _tl:, :]
                        v_img = v_t[:, :, _tl:, :]
                        if getattr(controller, 'sinkhorn_use_pattern', False):
                            ref_out = sinkhorn_style_pattern(ref_k, v_img,
                                                            n_iters=controller.sinkhorn_iters,
                                                            tau=controller.sinkhorn_tau,
                                                            txt_len=_tl)
                        else:
                            ref_out = sinkhorn_attn(q_img, ref_k, ref_v,
                                                   n_iters=controller.sinkhorn_iters,
                                                   tau=controller.sinkhorn_tau)
                        beta = controller.inject_beta
                        blended = (1.0 - beta) * attn_out_t[:, :, _tl:, :] + beta * ref_out
                        attn_out_t = attn_out_t.clone()
                        attn_out_t[:, :, _tl:, :] = blended
                    controller.inject_idx += 1

            controller.cur_self_layer += 1

            attn_output = attn_out_t.transpose(1, 2)
            attn_output = attn_output.flatten(2, 3).to(query.dtype)

            if encoder_hidden_states is not None:
                encoder_out, img_out = attn_output.split(
                    [txt_len, attn_output.shape[1] - txt_len], dim=1)
                img_out = attn.to_out[0](img_out)
                img_out = attn.to_out[1](img_out)
                encoder_out = attn.to_add_out(encoder_out)
                return img_out, encoder_out
            else:
                attn_output = attn.to_out[0](attn_output)
                attn_output = attn.to_out[1](attn_output)
                return attn_output

        return hooked_processor

    def make_single_hook(original_attn):
        def hooked_processor(attn, hidden_states, attention_mask=None,
                             image_rotary_emb=None):
            from diffsynth.models.flux2_dit import apply_rotary_emb

            proj_out = attn.to_qkv_mlp_proj(hidden_states)
            qkv, mlp_hidden_states = torch.split(
                proj_out, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
            )
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            query = attn.norm_q(query)
            key = attn.norm_k(key)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            query = query.to(hidden_states.dtype)
            key = key.to(hidden_states.dtype)
            value = value.to(hidden_states.dtype)

            q_t = query.transpose(1, 2)
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)

            if controller.cur_self_layer in controller.self_layers:
                if controller.mode == "cache":
                    cache.add(q_t, k_t, v_t, attn_out_t)
                elif controller.mode == "inject" and controller.inject_idx < len(controller.inject_ref_k):
                    ref_k = controller.inject_ref_k[controller.inject_idx]
                    ref_v = controller.inject_ref_v[controller.inject_idx]
                    beta = controller.inject_beta
                    _tl = getattr(controller, 'txt_len', 0)
                    q_img = q_t[:, :, _tl:, :]
                    ref_out = F.scaled_dot_product_attention(q_img, ref_k, ref_v)
                    blended = (1.0 - beta) * attn_out_t[:, :, _tl:, :] + beta * ref_out
                    attn_out_t = attn_out_t.clone()
                    attn_out_t[:, :, _tl:, :] = blended
                    controller.inject_idx += 1
                elif controller.mode == "sinkhorn" and controller.inject_idx < len(controller.inject_ref_k):
                    ref_k = controller.inject_ref_k[controller.inject_idx]
                    ref_v = controller.inject_ref_v[controller.inject_idx]
                    _n_total = len(controller.inject_ref_k)
                    _oln = controller.sinkhorn_only_last_n
                    _apply_ot = (_oln <= 0) or (controller.inject_idx >= _n_total - _oln)
                    if _apply_ot:
                        _tl = getattr(controller, 'txt_len', 0)
                        q_img = q_t[:, :, _tl:, :]
                        v_img = v_t[:, :, _tl:, :]
                        if getattr(controller, 'sinkhorn_use_pattern', False):
                            ref_out = sinkhorn_style_pattern(ref_k, v_img,
                                                            n_iters=controller.sinkhorn_iters,
                                                            tau=controller.sinkhorn_tau,
                                                            txt_len=_tl)
                        else:
                            ref_out = sinkhorn_attn(q_img, ref_k, ref_v,
                                                   n_iters=controller.sinkhorn_iters,
                                                   tau=controller.sinkhorn_tau)
                        beta = controller.inject_beta
                        blended = (1.0 - beta) * attn_out_t[:, :, _tl:, :] + beta * ref_out
                        attn_out_t = attn_out_t.clone()
                        attn_out_t[:, :, _tl:, :] = blended
                    controller.inject_idx += 1

            controller.cur_self_layer += 1

            attn_output = attn_out_t.transpose(1, 2).flatten(2, 3).to(query.dtype)

            mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
            hidden_states = attn.to_out(hidden_states)
            return hidden_states

        return hooked_processor

    for block in dit.transformer_blocks:
        block.attn.processor = make_double_hook(block.attn)
    for block in dit.single_transformer_blocks:
        block.attn.processor = make_single_hook(block.attn)


# --- Utility functions ---
def sinkhorn_attn(Q, K_style, V_style, n_iters=6, tau=0.1):
    B, H, Lc, D = Q.shape
    logits = torch.matmul(Q, K_style.transpose(-1, -2)) / (D ** 0.5 * tau)
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    P = torch.softmax(log_P, dim=-1)
    return torch.matmul(P, V_style)


def sinkhorn_style_pattern(K_style, V_content, n_iters=6, tau=0.1, txt_len=0):
    """Transfer style's internal attention pattern to content's values.
    Computes style's self-attention (K_img @ K_img^T with OT normalization),
    then applies that pattern to V_content.
    This transfers HOW style tokens relate (brushstroke structure) without any style content."""
    # Use only image portion of K_style
    K_img = K_style[:, :, txt_len:, :]
    B, H, Ls, D = K_img.shape
    Lc = V_content.shape[2]
    # If sizes differ, truncate to minimum
    L = min(Ls, Lc)
    K_img = K_img[:, :, :L, :]
    V_c = V_content[:, :, :L, :]
    # Style's self-correlation pattern
    logits = torch.matmul(K_img, K_img.transpose(-1, -2)) / (D ** 0.5 * tau)
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    P_style = torch.softmax(log_P, dim=-1)
    # Apply style's relationship pattern to content values
    out = torch.matmul(P_style, V_c)
    # Pad back if we truncated
    if L < Lc:
        out = F.pad(out, (0, 0, 0, Lc - L))
    return out


def sinkhorn_attn_freq(Q, K_style, V_style, n_iters=6, tau=0.1, hi_boost=2.0):
    """
    Frequency-aware Sinkhorn: boost high-freq components in cost matrix.

    Subtracts spatial mean from Q/K before computing cost, so the OT plan
    is driven by texture variation (high-freq) rather than absolute position (low-freq).
    The plan is then applied to full V_style (preserving all info in transport).

    hi_boost: weight for the high-freq cost relative to low-freq (>1 = prefer texture alignment)
    """
    B, H, Lc, D = Q.shape
    Ls = K_style.shape[2]

    # Decompose Q and K into low-freq (spatial mean) and high-freq (residual)
    Q_lo = Q.mean(dim=2, keepdim=True).expand_as(Q)  # [B, H, Lc, D]
    K_lo = K_style.mean(dim=2, keepdim=True).expand_as(K_style)
    Q_hi = Q - Q_lo
    K_hi = K_style - K_lo

    # Cost = blend of low-freq and high-freq similarity
    logits_hi = torch.matmul(Q_hi, K_hi.transpose(-1, -2)) / (D ** 0.5 * tau)
    logits_lo = torch.matmul(Q_lo, K_lo.transpose(-1, -2)) / (D ** 0.5 * tau)
    logits = logits_lo + hi_boost * logits_hi

    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    P = torch.softmax(log_P, dim=-1)
    return torch.matmul(P, V_style)


def feature_adain(x, y, alpha=1.0, eps=1e-6):
    m_x, s_x = x.mean(dim=2, keepdim=True), x.std(dim=2, keepdim=True) + eps
    m_y, s_y = y.mean(dim=2, keepdim=True), y.std(dim=2, keepdim=True) + eps
    res = (x - m_x) / s_x * s_y + m_y
    return alpha * res + (1 - alpha) * x


def adain_1d(source, target, eps=1e-6):
    s_mean = source.mean(dim=1, keepdim=True)
    s_std = source.std(dim=1, keepdim=True)
    t_mean = target.mean(dim=(0, 1), keepdim=True)
    t_std = target.std(dim=(0, 1), keepdim=True)
    return (source - s_mean) / (s_std + eps) * t_std + t_mean


def sinkhorn_latent_transport(content_lat, style_lat_noisy, style_lat_clean,
                              alpha=0.5, n_iters=10, tau=0.1, grag_delta=1.0,
                              mode="ot_adain"):
    """
    OT-based latent-space style transport (replaces Adam in-loop).

    Modes:
      - "blend": linear interpolation toward transported style (fast but kills texture)
      - "ot_adain": OT-matched per-token AdaIN (preserves content structure, transfers style statistics)

    Cost: ~microseconds (two bmm + 10 iter row/col normalize on 1024x1024).
    """
    B, N, D = content_lat.shape

    # GRAG analog: compress style tokens toward mean to reduce layout leakage
    style_noisy_g = style_lat_noisy
    style_clean_g = style_lat_clean
    if grag_delta < 1.0:
        s_mean = style_lat_noisy.mean(dim=1, keepdim=True)
        style_noisy_g = s_mean + grag_delta * (style_lat_noisy - s_mean)
        sc_mean = style_lat_clean.mean(dim=1, keepdim=True)
        style_clean_g = sc_mean + grag_delta * (style_lat_clean - sc_mean)

    # Cost matrix: frequency-aware cosine similarity
    # Use high-freq residuals (token - spatial_mean) for cost → OT matches by texture, not position
    c_f = content_lat.float()
    s_f = style_noisy_g.float()
    c_hi = c_f - c_f.mean(dim=1, keepdim=True)
    s_hi = s_f - s_f.mean(dim=1, keepdim=True)
    c_norm = F.normalize(c_hi, dim=-1)
    s_norm = F.normalize(s_hi, dim=-1)
    logits = torch.bmm(c_norm, s_norm.transpose(1, 2)) / tau  # [B, N, M]

    # Sinkhorn normalization (doubly-stochastic)
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    T = torch.softmax(log_P, dim=-1)  # [B, N, M]

    if mode == "blend":
        # Direct transport: apply plan to clean style latent
        transported = torch.bmm(T, style_clean_g.float()).to(content_lat.dtype)
        return (1.0 - alpha) * content_lat + alpha * transported

    else:
        # OT-AdaIN: use transport plan to find matched style statistics per token,
        # then transfer mean/std while preserving content's high-freq detail.
        # This is like AdaIN but spatially-aware (each token gets its OT-matched partner's stats).
        style_f = style_clean_g.float()  # [B, M, D]

        # Weighted style mean/std per content token (soft OT matching)
        # T: [B, N, M], style_f: [B, M, D]
        matched_mean = torch.bmm(T, style_f)  # [B, N, D] — OT-weighted mean of style tokens
        matched_var = torch.bmm(T, style_f ** 2) - matched_mean ** 2  # [B, N, D]
        matched_std = (matched_var.clamp(min=1e-6)).sqrt()

        # Content statistics (per-token, over channel groups for stability)
        content_f = content_lat.float()
        # Global channel-wise stats (more stable than per-token)
        c_mean = content_f.mean(dim=1, keepdim=True)  # [B, 1, D]
        c_std = content_f.std(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1, D]

        # Normalize content, apply matched style stats
        normed = (content_f - c_mean) / c_std
        styled = normed * matched_std + matched_mean

        result = ((1.0 - alpha) * content_f + alpha * styled).to(content_lat.dtype)
        return result


def dit_forward_truncated(dit_model, hidden_states, timestep, guidance,
                          encoder_hidden_states, txt_ids, img_ids,
                          grad_single_layers=8):
    """
    Run DiT forward with truncated gradient:
    - All double blocks + first (24 - grad_single_layers) single blocks: no_grad
    - Last grad_single_layers single blocks + output: with_grad

    This makes backward ~3-4x faster since only deep layers build a compute graph.
    """
    from diffsynth.models.flux2_dit import apply_rotary_emb

    num_txt_tokens = encoder_hidden_states.shape[1]

    ts = timestep.to(hidden_states.dtype) * 1000
    g = guidance.to(hidden_states.dtype) * 1000 if guidance is not None else None

    temb = dit_model.time_guidance_embed(ts, g)
    double_stream_mod_img = dit_model.double_stream_modulation_img(temb)
    double_stream_mod_txt = dit_model.double_stream_modulation_txt(temb)
    single_stream_mod = dit_model.single_stream_modulation(temb)[0]

    hs = dit_model.x_embedder(hidden_states)
    ehs = dit_model.context_embedder(encoder_hidden_states)

    _img_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
    _txt_ids = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
    image_rotary_emb = dit_model.pos_embed(_img_ids)
    text_rotary_emb = dit_model.pos_embed(_txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    # Phase A: no_grad for double blocks + early single blocks
    n_single = len(dit_model.single_transformer_blocks)
    n_nograd_single = max(0, n_single - grad_single_layers)

    with torch.no_grad():
        for block in dit_model.transformer_blocks:
            ehs, hs = block(
                hidden_states=hs,
                encoder_hidden_states=ehs,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
            )
        hs = torch.cat([ehs, hs], dim=1)
        for block in dit_model.single_transformer_blocks[:n_nograd_single]:
            hs = block(
                hidden_states=hs,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
            )

    # Detach to break the graph, then re-enable grad for deep layers
    hs = hs.detach().requires_grad_(True)

    # Phase B: with_grad for last N single blocks + output
    for block in dit_model.single_transformer_blocks[n_nograd_single:]:
        hs = block(
            hidden_states=hs,
            encoder_hidden_states=None,
            temb_mod_params=single_stream_mod,
            image_rotary_emb=concat_rotary_emb,
        )

    hs = hs[:, num_txt_tokens:, ...]
    hs = dit_model.norm_out(hs, temb)
    output = dit_model.proj_out(hs)
    return output


def load_image_tensor(path, size=None):
    from torchvision.transforms import ToTensor
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return ToTensor()(img).unsqueeze(0)


# --- Load model ---
print("Loading Klein 9B model...")
model_root = Path(KLEIN9B_ROOT)
pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=_sorted_files(str(model_root / "text_encoder" / "*.safetensors"))),
        ModelConfig(path=_sorted_files(str(model_root / "transformer" / "*.safetensors"))),
        ModelConfig(path=str(model_root / "vae" / "diffusion_pytorch_model.safetensors")),
    ],
    tokenizer_config=ModelConfig(path=str(model_root / "tokenizer")),
)
dit = pipe.dit
vae = pipe.vae
scheduler = FlowMatchScheduler("FLUX.2")
pipe.vram_management_enabled = False

# Setup hooks
cache = Klein9BCache()
controller = Klein9BController(self_layers=(start_layer, end_layer))
register_klein9b_attn_hooks(dit, controller, cache)

# --- Encode style image ---
print(f"Prompt: {prompt}")
print(f"Style: {style_image_path}")

style_img = load_image_tensor(style_image_path, size=(512, 512))

pipe.load_models_to_device(["vae"])
with torch.no_grad():
    sty_pix = pipe.preprocess_image(Image.open(style_image_path).convert("RGB").resize((512, 512), Image.BICUBIC))
    style_latent_4d = vae.encode(sty_pix)
    style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

# Image IDs for the generated image
lat_h, lat_w = height // 16, width // 16
image_ids = torch.cartesian_prod(
    torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)
).unsqueeze(0).to("cuda")

# Style image IDs
sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
style_image_ids = torch.cartesian_prod(
    torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)
).unsqueeze(0).to("cuda")

# --- Encode prompt ---
from diffsynth.pipelines.flux2_image import Flux2Unit_Qwen3PromptEmbedder, Flux2Unit_PromptEmbedder

if pipe.text_encoder_qwen3 is not None:
    pipe.load_models_to_device(["text_encoder_qwen3"])
    embedder = Flux2Unit_Qwen3PromptEmbedder()
    prompt_embeds, text_ids = embedder.encode_prompt(
        pipe.text_encoder_qwen3, pipe.tokenizer, prompt,
        dtype=pipe.torch_dtype, device="cuda"
    )
    null_embeds, null_text_ids = embedder.encode_prompt(
        pipe.text_encoder_qwen3, pipe.tokenizer, "",
        dtype=pipe.torch_dtype, device="cuda"
    )
else:
    pipe.load_models_to_device(["text_encoder"])
    embedder = Flux2Unit_PromptEmbedder()
    prompt_embeds, text_ids = embedder.encode_prompt(
        pipe.text_encoder, pipe.tokenizer, prompt,
        dtype=pipe.torch_dtype, device="cuda"
    )
    null_embeds, null_text_ids = embedder.encode_prompt(
        pipe.text_encoder, pipe.tokenizer, "",
        dtype=pipe.torch_dtype, device="cuda"
    )

# --- Load dit onto GPU ---
pipe.load_models_to_device(["dit"])
dit.to("cuda")
dit.requires_grad_(False)

# --- Generate noise ---
torch.manual_seed(seed)
noise = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
noise = rearrange(noise, "B C H W -> B (H W) C")
latents = noise.clone()

# --- Setup scheduler ---
scheduler.set_timesteps(num_inference_steps, dynamic_shift_len=lat_h * lat_w)
timesteps = scheduler.timesteps
print(f"Timesteps: {timesteps}")

# --- Denoising loop with OT attention (T2I: no content image) ---
loss_fn = torch.nn.L1Loss()
guidance_tensor = None
if embedded_guidance > 0:
    guidance_tensor = torch.tensor([embedded_guidance], device="cuda", dtype=torch.bfloat16)

_DIST_W = float(os.environ.get("DIST_W", "0.15"))
_CONTENT_W = float(os.environ.get("CONTENT_W", "0.2"))
_ADAIN_ALPHA = float(os.environ.get("ADAIN_ALPHA", "0.5"))
_GRAG_DEEP = float(os.environ.get("GRAG_DEEP", "1.0"))
_GRAG_SHALLOW = float(os.environ.get("GRAG_SHALLOW", "1.0"))
_OT_TAU_MAX = float(os.environ.get("OT_TAU_MAX", "0.15"))
_OT_TAU_MIN = float(os.environ.get("OT_TAU_MIN", "0.07"))
_OT_ITERS = int(os.environ.get("OT_ITERS", "6"))
_OT_LAYER_FRAC = float(os.environ.get("OT_LAYER_FRAC", "0.5"))
_STYLE_W = float(os.environ.get("STYLE_W", "2.0"))
# Sinkhorn mode: "standard" | "freq" | "dual"
# freq: frequency-aware (boost texture in cost matrix)
# dual: two-branch loss (full + compressed), GRAG becomes layout dial
_SINKHORN_MODE = os.environ.get("SINKHORN_MODE", "freq")
_FREQ_HI_BOOST = float(os.environ.get("FREQ_HI_BOOST", "2.0"))
# dual mode: weight for the compressed branch (layout-free style matching)
_DUAL_COMPRESS_W = float(os.environ.get("DUAL_COMPRESS_W", "0.3"))
_DUAL_COMPRESS_DELTA = float(os.environ.get("DUAL_COMPRESS_DELTA", "0.3"))
# GRAG repositioned: apply to loss targets instead of ref K/V
# "loss_grag": compress both self_out and ref_out before computing L1
#   → matches global style statistics without requiring positional alignment
# "grad_grag": compress Adam gradient spatially (uniform update direction)
# "adain_grag": compress style latent used for AdaIN warm-start only
_GRAG_MODE = os.environ.get("GRAG_MODE", "off")  # off | loss | grad | adain (loss is useful standalone; redundant with freq sinkhorn)
_GRAG_DELTA = float(os.environ.get("GRAG_DELTA", "0.5"))
# OPTIM_MODE:
#   "slt"    — [Flash/最优传输] Sinkhorn Latent Transport, 0 forward/backward (~2s)
#   "adam"   — [Pro/反向传播] N forward+backward per step (strongest, slowest)
#   "inject" — StyleID-style attn injection, 1 extra forward (weakest, fastest)
#   "scfg"  — Style Classifier-Free Guidance: amplified inject delta (2 extra forward, no backward)
#   "sds"   — Score Distillation Style: 1 forward+backward, gradient as direct update (middle ground)
_OPTIM_MODE = os.environ.get("OPTIM_MODE", "adam")
_INJECT_BETA = float(os.environ.get("INJECT_BETA", "1.0"))
_SCFG_SCALE = float(os.environ.get("SCFG_SCALE", "3.0"))  # style guidance scale (like text CFG scale)
_SDS_LR = float(os.environ.get("SDS_LR", "0.05"))  # one-shot gradient step size
# SLT (Sinkhorn Latent Transport) params
_SLT_ALPHA = float(os.environ.get("SLT_ALPHA", "0.2"))  # transport blending strength
_SLT_TAU = float(os.environ.get("SLT_TAU", "0.02"))  # temperature for cost matrix (lower=sharper)
_SLT_ITERS = int(os.environ.get("SLT_ITERS", "10"))  # Sinkhorn iterations
_SLT_GRAG = float(os.environ.get("SLT_GRAG", "0.7"))  # GRAG delta for style tokens (1.0=no compression)
_SLT_MODE = os.environ.get("SLT_MODE", "ot_adain")  # "ot_adain" (texture-preserving) or "blend" (direct interpolation)
_SLT_GUIDE_SCALE = float(os.environ.get("SLT_GUIDE_SCALE", "3.0"))  # noise steering strength

latent_seq_len = latents.shape[1]

pbar = tqdm(enumerate(timesteps), total=len(timesteps), desc="Klein9B T2I OT-Style")
for i, t_val in pbar:
    timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
    cache.clear(); controller.step()

    # Real attention K/V injection (eq:injection): style K/V at the CURRENT
    # timestep, blended during this step's single denoise forward.
    # Fixes the old double-step bug (Step 1 advanced to sigma_{i+1}, then the
    # inject branch forwarded that latent at t_i and stepped a second time)
    # and skips the AdaIN warm-start for a faithful injection baseline.
    if _OPTIM_MODE == "inject" and t_val < start_time:
        if i < num_inference_steps - 1:
            sty_noise_inj = torch.randn_like(style_latent)
            style_latent_inj = scheduler.add_noise(
                style_latent, sty_noise_inj, timesteps[i:i + 1])
        else:
            style_latent_inj = style_latent
        with torch.no_grad():
            _ = dit(
                hidden_states=style_latent_inj.to(device="cuda", dtype=torch.bfloat16),
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=null_embeds,
                txt_ids=null_text_ids,
                img_ids=style_image_ids,
            )
        _, inj_k_list, inj_v_list, _ = cache.get()
        controller.set_inject(inj_k_list, inj_v_list, _INJECT_BETA)
        cache.clear(); controller.step()
        with torch.no_grad():
            noise_pred_styled = dit(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance_tensor,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
            )
        latents = scheduler.step(noise_pred_styled, t_val, latents)
        controller.reset_mode()
        pbar.set_postfix(mode="inject", beta=f"{_INJECT_BETA:.2f}")
        continue

    # Step 1: Denoising (pure T2I, no content image concat)
    with torch.no_grad():
        noise_pred = dit(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance_tensor,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
        )
    latents = scheduler.step(noise_pred, t_val, latents)

    if t_val >= start_time:
        continue

    # Step 2: Noisy style latent
    if i < num_inference_steps - 1:
        timestep_style = timesteps[i + 1:i + 2]
        sty_noise = torch.randn_like(style_latent)
        style_latent_ = scheduler.add_noise(style_latent, sty_noise, timestep_style)
    else:
        timestep_style = torch.tensor([0.0])
        style_latent_ = style_latent

    # Step 3: Style forward -> ref K/V
    cache.clear(); controller.step()
    with torch.no_grad():
        _ = dit(
            hidden_states=style_latent_.to(device="cuda", dtype=torch.bfloat16),
            timestep=timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000,
            guidance=None,
            encoder_hidden_states=null_embeds,
            txt_ids=null_text_ids,
            img_ids=style_image_ids,
        )
    _, ref_k_list, ref_v_list, _ = cache.get()

    # Step 3.5: GRAG
    _txt_len = controller.txt_len
    n_cached = len(ref_k_list)
    for idx in range(n_cached):
        depth_frac = idx / max(n_cached - 1, 1)
        delta_i = _GRAG_SHALLOW + (_GRAG_DEEP - _GRAG_SHALLOW) * depth_frac
        k = ref_k_list[idx]
        v = ref_v_list[idx]
        k_txt, k_img = k[:, :, :_txt_len, :], k[:, :, _txt_len:, :]
        v_txt, v_img = v[:, :, :_txt_len, :], v[:, :, _txt_len:, :]
        km = k_img.mean(dim=2, keepdim=True)
        vm = v_img.mean(dim=2, keepdim=True)
        k_img = km + delta_i * (k_img - km)
        v_img = vm + delta_i * (v_img - vm)
        ref_k_list[idx] = torch.cat([k_txt, k_img], dim=2)
        ref_v_list[idx] = torch.cat([v_txt, v_img], dim=2)

    # Step 4: AdaIN warm-start
    content_anchor = latents.detach().clone()
    with torch.no_grad():
        if _GRAG_MODE == "adain":
            sl_mean = style_latent_.mean(dim=1, keepdim=True)
            sl_compressed = sl_mean + _GRAG_DELTA * (style_latent_ - sl_mean)
            latents_warmed = feature_adain(
                latents.unsqueeze(1), sl_compressed.unsqueeze(1), alpha=_ADAIN_ALPHA
            ).squeeze(1)
        else:
            latents_warmed = feature_adain(
                latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=_ADAIN_ALPHA
            ).squeeze(1)
    latents = latents_warmed.detach()

    if _OPTIM_MODE == "inject":
        # Injection is handled before Step 1 (single-forward, single-step);
        # this branch is unreachable and kept only to preserve the elif chain.
        pass

    elif _OPTIM_MODE == "scfg":
        # === Style Classifier-Free Guidance ===
        # Analogous to text CFG: pred = pred_normal + scale * (pred_styled - pred_normal)
        # pred_normal: already computed in Step 1 (noise_pred from the denoising step)
        # pred_styled: one forward with inject hooks (attention blended toward style)
        # The delta (pred_styled - pred_normal) IS the style direction.
        # Amplifying it with scale > 1 pushes harder toward style without any backward.
        # Re-run normal forward to get consistent noise_pred (Step 1 already stepped latents)
        cache.clear(); controller.step()
        with torch.no_grad():
            noise_pred_normal = dit(
                hidden_states=content_anchor,  # latent BEFORE adain warm-start
                timestep=timestep / 1000,
                guidance=guidance_tensor,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
            )
        controller.set_inject(ref_k_list, ref_v_list, _INJECT_BETA)
        cache.clear(); controller.step()
        with torch.no_grad():
            noise_pred_styled = dit(
                hidden_states=content_anchor,
                timestep=timestep / 1000,
                guidance=guidance_tensor,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
            )
        controller.reset_mode()
        # CFG-style amplification
        noise_pred_guided = noise_pred_normal + _SCFG_SCALE * (noise_pred_styled - noise_pred_normal)
        latents = scheduler.step(noise_pred_guided, t_val, content_anchor)
        pbar.set_postfix(mode="scfg", scale=f"{_SCFG_SCALE:.1f}")

    elif _OPTIM_MODE == "sds":
        # === Score Distillation Style ===
        # Single forward+backward: compute style loss, use gradient as direct latent update.
        # Like SDS but for style: grad points from current latent toward "styled" latent.
        latents.requires_grad_(True)
        cache.clear(); controller.step()
        _ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000
        with torch.enable_grad():
            _dit_out = dit(
                hidden_states=latents,
                timestep=_ts_cuda.detach(),
                guidance=None,
                encoder_hidden_states=null_embeds.detach(),
                txt_ids=null_text_ids.detach(),
                img_ids=image_ids.detach(),
            )
        del _dit_out
        q_list, _, _, self_out_list = cache.get()

        _t_float = float(t_val.item()) if hasattr(t_val, 'item') else float(t_val)
        _progress = 1.0 - (_t_float / max(start_time, 1.0))
        _tau = _OT_TAU_MAX - (_OT_TAU_MAX - _OT_TAU_MIN) * _progress
        n_layers = len(q_list)
        n_ot_layers = max(1, int(n_layers * _OT_LAYER_FRAC))

        ref_self_out_list = []
        for idx, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
            q_det = q.detach()
            if idx >= n_layers - n_ot_layers:
                ref_self_out_list.append(
                    sinkhorn_attn_freq(q_det, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau, hi_boost=_FREQ_HI_BOOST)
                )
            else:
                ref_self_out_list.append(F.scaled_dot_product_attention(q_det, ref_k, ref_v))
        style_loss = sum(loss_fn(so, rso.detach()) for so, rso in zip(self_out_list, ref_self_out_list))
        style_loss.backward()

        with torch.no_grad():
            latents = latents - _SDS_LR * latents.grad
        latents = latents.detach()
        pbar.set_postfix(mode="sds", sl=f"{style_loss.item():.2f}")

    elif _OPTIM_MODE == "slt":
        # === OT-Attention CFG (Flash mode) ===
        with torch.no_grad():
            cache.clear(); controller.step()
            pred_normal = dit(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance_tensor,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
            )
            controller.set_sinkhorn(ref_k_list, ref_v_list,
                                    tau=_SLT_TAU, n_iters=_SLT_ITERS,
                                    beta=_SLT_ALPHA)
            cache.clear(); controller.step()
            pred_styled = dit(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance_tensor,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
            )
            controller.reset_mode()
            style_direction = pred_styled - pred_normal
            pred_guided = pred_normal + _SLT_GUIDE_SCALE * style_direction
            latents = scheduler.step(pred_guided, t_val, content_anchor)
        pbar.set_postfix(mode="ot-cfg", s=f"{_SLT_GUIDE_SCALE:.1f}")

    elif _OPTIM_MODE == "hybrid":
        # === Hybrid: SLT (color) + Adam (brushstroke) ===
        # Phase 1: Per-token OT in latent space — aligns color/statistics (microseconds, no forward)
        # Phase 2: Adam iters — backward through attention adds brushstroke texture
        # Faster than pure Adam because color pre-aligned → Adam converges in fewer iters.
        _hybrid_adam_iters = int(os.environ.get("HYBRID_ITERS", "2"))

        # Phase 1: SLT color pre-alignment (latent space OT, near-zero cost)
        with torch.no_grad():
            latents = sinkhorn_latent_transport(
                latents, style_latent_, style_latent,
                alpha=_SLT_ALPHA, n_iters=_SLT_ITERS, tau=_SLT_TAU,
                grag_delta=_GRAG_DEEP, mode="ot_adain"
            )

        # Phase 2: Adam iteration(s) for brushstroke texture
        latents.requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=lr)
        _ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000

        for it in range(_hybrid_adam_iters):
            cache.clear(); controller.step()
            optimizer.zero_grad()

            with torch.enable_grad():
                _dit_out = dit(
                    hidden_states=latents,
                    timestep=_ts_cuda.detach(),
                    guidance=None,
                    encoder_hidden_states=null_embeds.detach(),
                    txt_ids=null_text_ids.detach(),
                    img_ids=image_ids.detach(),
                )
            del _dit_out
            q_list, _, _, self_out_list = cache.get()

            _t_float = float(t_val.item()) if hasattr(t_val, 'item') else float(t_val)
            _progress = 1.0 - (_t_float / max(start_time, 1.0))
            _tau = _OT_TAU_MAX - (_OT_TAU_MAX - _OT_TAU_MIN) * _progress
            n_layers = len(q_list)
            n_ot_layers = max(1, int(n_layers * _OT_LAYER_FRAC))

            ref_self_out_list = []
            for idx, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                q_det = q.detach()
                if idx >= n_layers - n_ot_layers:
                    ref_self_out_list.append(
                        sinkhorn_attn_freq(q_det, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau, hi_boost=_FREQ_HI_BOOST)
                    )
                else:
                    ref_self_out_list.append(F.scaled_dot_product_attention(q_det, ref_k, ref_v))

            style_loss = sum(loss_fn(so, rso.detach()) for so, rso in zip(self_out_list, ref_self_out_list))
            content_loss = loss_fn(latents, content_anchor)
            loss = _STYLE_W * style_loss + _CONTENT_W * content_loss
            loss.backward()
            optimizer.step()

        latents = latents.detach()
        pbar.set_postfix(mode="hybrid", sl=f"{style_loss.item():.2f}")

    elif _OPTIM_MODE == "hybrid_trunc":
        # === Hybrid + Truncated Backward ===
        # Same as hybrid but backward only flows through last N single layers.
        # Saves ~60% backward time per iter.
        _hybrid_adam_iters = int(os.environ.get("HYBRID_ITERS", "3"))
        _grad_layers = int(os.environ.get("GRAD_LAYERS", "8"))

        # Phase 1: SLT color pre-alignment
        with torch.no_grad():
            latents = sinkhorn_latent_transport(
                latents, style_latent_, style_latent,
                alpha=_SLT_ALPHA, n_iters=_SLT_ITERS, tau=_SLT_TAU,
                grag_delta=_GRAG_DEEP, mode="ot_adain"
            )

        # Phase 2: Truncated Adam
        latents.requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=lr)
        _ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000

        for it in range(_hybrid_adam_iters):
            cache.clear(); controller.step()
            optimizer.zero_grad()

            with torch.enable_grad():
                _dit_out = dit_forward_truncated(
                    dit, latents,
                    timestep=_ts_cuda.detach(),
                    guidance=None,
                    encoder_hidden_states=null_embeds.detach(),
                    txt_ids=null_text_ids.detach(),
                    img_ids=image_ids.detach(),
                    grad_single_layers=_grad_layers,
                )
            del _dit_out
            q_list, _, _, self_out_list = cache.get()

            _t_float = float(t_val.item()) if hasattr(t_val, 'item') else float(t_val)
            _progress = 1.0 - (_t_float / max(start_time, 1.0))
            _tau = _OT_TAU_MAX - (_OT_TAU_MAX - _OT_TAU_MIN) * _progress
            n_layers = len(q_list)
            n_ot_layers = max(1, int(n_layers * _OT_LAYER_FRAC))

            ref_self_out_list = []
            for idx, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                q_det = q.detach()
                if idx >= n_layers - n_ot_layers:
                    ref_self_out_list.append(
                        sinkhorn_attn_freq(q_det, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau, hi_boost=_FREQ_HI_BOOST)
                    )
                else:
                    ref_self_out_list.append(F.scaled_dot_product_attention(q_det, ref_k, ref_v))

            style_loss = sum(loss_fn(so, rso.detach()) for so, rso in zip(self_out_list, ref_self_out_list))
            content_loss = loss_fn(latents, content_anchor)
            loss = _STYLE_W * style_loss + _CONTENT_W * content_loss
            loss.backward()
            optimizer.step()

        latents = latents.detach()
        pbar.set_postfix(mode="hybrid_trunc", sl=f"{style_loss.item():.2f}")

    else:
        # === Adam-in-loop Mode (default) ===
        # Dynamic iters: early steps (high t) get more iters, late steps (low t) fewer
        _t_float_dyn = float(t_val.item()) if hasattr(t_val, 'item') else float(t_val)
        _progress_dyn = 1.0 - (_t_float_dyn / max(start_time, 1.0))  # 0=early, 1=late
        _dynamic_iters = max(1, int(iters * (1.0 - 0.6 * _progress_dyn)))  # iters..40% of iters

        latents.requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=lr)
        _ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000

        for it in range(_dynamic_iters):
            cache.clear(); controller.step()
            optimizer.zero_grad()

            with torch.enable_grad():
                _dit_out = dit(
                    hidden_states=latents,
                    timestep=_ts_cuda.detach(),
                    guidance=None,
                    encoder_hidden_states=null_embeds.detach(),
                    txt_ids=null_text_ids.detach(),
                    img_ids=image_ids.detach(),
                )
            del _dit_out
            q_list, _, _, self_out_list = cache.get()

            _t_float = float(t_val.item()) if hasattr(t_val, 'item') else float(t_val)
            _progress = 1.0 - (_t_float / max(start_time, 1.0))
            _tau = _OT_TAU_MAX - (_OT_TAU_MAX - _OT_TAU_MIN) * _progress

            n_layers = len(q_list)
            n_ot_layers = max(1, int(n_layers * _OT_LAYER_FRAC))

            def _compute_ref_out(q_det, ref_k, ref_v, use_ot):
                if use_ot:
                    if _SINKHORN_MODE == "freq":
                        return sinkhorn_attn_freq(q_det, ref_k, ref_v,
                                                 n_iters=_OT_ITERS, tau=_tau,
                                                 hi_boost=_FREQ_HI_BOOST)
                    else:
                        return sinkhorn_attn(q_det, ref_k, ref_v,
                                            n_iters=_OT_ITERS, tau=_tau)
                else:
                    return F.scaled_dot_product_attention(q_det, ref_k, ref_v)

            ref_self_out_list = []
            for idx, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                q_det = q.detach()
                use_ot = (idx >= n_layers - n_ot_layers)
                ref_self_out_list.append(_compute_ref_out(q_det, ref_k, ref_v, use_ot))

            if _GRAG_MODE == "loss":
                def _compress_out(out, delta):
                    m = out.mean(dim=2, keepdim=True)
                    return m + delta * (out - m)
                style_loss = sum(
                    loss_fn(_compress_out(so, _GRAG_DELTA), _compress_out(rso, _GRAG_DELTA).detach())
                    for so, rso in zip(self_out_list, ref_self_out_list)
                )
            else:
                style_loss = sum(
                    loss_fn(so, rso.detach())
                    for so, rso in zip(self_out_list, ref_self_out_list)
                )

            if _SINKHORN_MODE == "dual":
                ref_self_out_compressed = []
                for idx, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                    q_det = q.detach()
                    k_m = ref_k[:, :, _txt_len:, :].mean(dim=2, keepdim=True)
                    v_m = ref_v[:, :, _txt_len:, :].mean(dim=2, keepdim=True)
                    k_c = ref_k.clone()
                    v_c = ref_v.clone()
                    k_c[:, :, _txt_len:, :] = k_m + _DUAL_COMPRESS_DELTA * (ref_k[:, :, _txt_len:, :] - k_m)
                    v_c[:, :, _txt_len:, :] = v_m + _DUAL_COMPRESS_DELTA * (ref_v[:, :, _txt_len:, :] - v_m)
                    use_ot = (idx >= n_layers - n_ot_layers)
                    ref_self_out_compressed.append(_compute_ref_out(q_det, k_c, v_c, use_ot))
                style_loss_compressed = sum(
                    loss_fn(so, rso.detach())
                    for so, rso in zip(self_out_list, ref_self_out_compressed)
                )
                style_loss = (1.0 - _DUAL_COMPRESS_W) * style_loss + _DUAL_COMPRESS_W * style_loss_compressed

            lat4d = latents.unsqueeze(1)
            sty4d = style_latent_.unsqueeze(1).detach()
            dist_loss = (
                loss_fn(lat4d.mean(dim=2, keepdim=True), sty4d.mean(dim=2, keepdim=True)) +
                loss_fn(lat4d.std(dim=2, keepdim=True), sty4d.std(dim=2, keepdim=True))
            )

            decay = 1.0 - (it / max(iters, 1)) * 0.5
            content_loss = loss_fn(latents, content_anchor) * decay

            loss = _STYLE_W * style_loss + _DIST_W * dist_loss + _CONTENT_W * content_loss
            loss.backward()

            if _GRAG_MODE == "grad" and latents.grad is not None:
                g = latents.grad
                gm = g.mean(dim=1, keepdim=True)
                latents.grad = gm + _GRAG_DELTA * (g - gm)

            optimizer.step()

            pbar.set_postfix(sl=f"{style_loss.item():.2f}", dl=f"{dist_loss.item():.2f}")

        latents = latents.detach()

# --- Global color transfer ---
latents = adain_1d(latents, style_latent)

# --- Decode ---
pipe.load_models_to_device(["vae"])
latents_4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
image = vae.decode(latents_4d)
image = pipe.vae_output_to_image(image)

# --- Save ---
out_dir = Path(".")
image.save(out_dir / "output_klein9b_t2i.png")
Image.open(style_image_path).convert("RGB").resize((512, 512)).save(out_dir / "style_klein9b_t2i.png")
print(f"Saved: output_klein9b_t2i.png, style_klein9b_t2i.png")
print(f"Prompt: '{prompt}' | Style: {style_image_path}")
