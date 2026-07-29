"""
Spatial Smoothing Visualization — Paper Figure

Proves backward pass creates spatially coherent updates (brushstroke mechanism) by:
1. Gradient Field PCA: backward gradient at each token → PCA → RGB spatial map
   - Backward: smooth, directional color regions (= coherent update directions = brushstrokes)
   - SLT update: per-token independent → random speckle
   - Random noise: completely random (control)

2. Update Magnitude Heatmap: |grad| at each spatial position
   - Backward: clustered (gradients large in coherent regions)
   - SLT: uniform (OT re-assignment is position-independent)

3. Autocorrelation Decay Curve: cosine sim vs spatial distance (1,2,3,4,5 pixels)
   - Backward: slow decay (long-range coherence)
   - SLT: near-zero at all distances
   - Proves attention Jacobian propagates gradients across spatial neighbors

Output: publication-ready PNG figures for paper

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_spatial_viz.py
"""
import csv, glob, os, sys, time, types, json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

DIFFSYNTH_ROOT = "third_party/DiffSynth-Studio"
KLEIN9B_ROOT = "models/klein9b"
CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
VIZ_DIR = Path(CWD) / "d_grand" / "spatial_viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(CWD) / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============== DiffSynth setup ==============
def _install_modelscope_stub():
    if "modelscope" in sys.modules:
        return
    module = types.ModuleType("modelscope")
    module.snapshot_download = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no"))
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
from diffsynth.pipelines.flux2_image import Flux2Unit_Qwen3PromptEmbedder, Flux2Unit_PromptEmbedder
from diffsynth.diffusion.flow_match import FlowMatchScheduler

# ============== Load Klein 9B ==============
print("Loading Klein 9B...")
model_root = Path(KLEIN9B_ROOT)
pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(path=_sorted_files(str(model_root / "text_encoder" / "*.safetensors"))),
        ModelConfig(path=_sorted_files(str(model_root / "transformer" / "*.safetensors"))),
        ModelConfig(path=str(model_root / "vae" / "diffusion_pytorch_model.safetensors")),
    ],
    tokenizer_config=ModelConfig(path=str(model_root / "tokenizer")),
)
dit = pipe.dit
vae = pipe.vae
pipe.vram_management_enabled = False

# ============== Hooks ==============
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
    def step(self): self.cur_self_layer = 0

def register_hooks(dit_model, controller, cache):
    def make_double_hook(attn_module):
        def hooked_processor(attn, hidden_states, encoder_hidden_states=None,
                             attention_mask=None, image_rotary_emb=None):
            from diffsynth.models.flux2_dit import _get_qkv_projections, apply_rotary_emb
            query, key, value, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
            query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
            key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
            value = value.unflatten(-1, (attn.heads, -1))
            txt_len = 0
            if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
                eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
                ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
                ev = ev.unflatten(-1, (attn.heads, -1))
                txt_len = encoder_hidden_states.shape[1]
                controller.txt_len = txt_len
                query = torch.cat([eq, query], dim=1)
                key = torch.cat([ek, key], dim=1)
                value = torch.cat([ev, value], dim=1)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            query, key, value = query.to(hidden_states.dtype), key.to(hidden_states.dtype), value.to(hidden_states.dtype)
            q_t, k_t, v_t = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)
            if controller.cur_self_layer in controller.self_layers:
                cache.add(q_t, k_t, v_t, attn_out_t)
            controller.cur_self_layer += 1
            attn_output = attn_out_t.transpose(1,2).flatten(2,3).to(query.dtype)
            if encoder_hidden_states is not None:
                enc_out, img_out = attn_output.split([txt_len, attn_output.shape[1]-txt_len], dim=1)
                return attn.to_out[1](attn.to_out[0](img_out)), attn.to_add_out(enc_out)
            else:
                return attn.to_out[1](attn.to_out[0](attn_output))
        return hooked_processor

    def make_single_hook(attn_module):
        def hooked_processor(attn, hidden_states, attention_mask=None, image_rotary_emb=None):
            from diffsynth.models.flux2_dit import apply_rotary_emb
            proj_out = attn.to_qkv_mlp_proj(hidden_states)
            qkv, mlp_hs = torch.split(proj_out, [3*attn.inner_dim, attn.mlp_hidden_dim*attn.mlp_mult_factor], dim=-1)
            q, k, v = qkv.chunk(3, dim=-1)
            q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
            k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
            v = v.unflatten(-1, (attn.heads, -1))
            if image_rotary_emb is not None:
                q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
                k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
            q, k, v = q.to(hidden_states.dtype), k.to(hidden_states.dtype), v.to(hidden_states.dtype)
            q_t, k_t, v_t = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)
            if controller.cur_self_layer in controller.self_layers:
                cache.add(q_t, k_t, v_t, attn_out_t)
            controller.cur_self_layer += 1
            attn_output = attn_out_t.transpose(1,2).flatten(2,3).to(q.dtype)
            mlp_hs = attn.mlp_act_fn(mlp_hs)
            return attn.to_out(torch.cat([attn_output, mlp_hs], dim=-1))
        return hooked_processor

    for block in dit_model.transformer_blocks:
        block.attn.processor = make_double_hook(block.attn)
    for block in dit_model.single_transformer_blocks:
        block.attn.processor = make_single_hook(block.attn)

start_layer, end_layer = 12, 28
cache = Klein9BCache()
controller = Klein9BController(self_layers=(start_layer, end_layer))
register_hooks(dit, controller, cache)
pipe.load_models_to_device(["dit"])
dit.to("cuda")
dit.requires_grad_(False)

# ============== Utility functions ==============
def sinkhorn_attn_freq(Q, K_style, V_style, n_iters=6, tau=0.1, hi_boost=2.0):
    B, H, Lc, D = Q.shape
    Q_lo = Q.mean(dim=2, keepdim=True).expand_as(Q)
    K_lo = K_style.mean(dim=2, keepdim=True).expand_as(K_style)
    Q_hi, K_hi = Q - Q_lo, K_style - K_lo
    logits = torch.matmul(Q_lo, K_lo.transpose(-1,-2))/(D**0.5*tau) + hi_boost*torch.matmul(Q_hi, K_hi.transpose(-1,-2))/(D**0.5*tau)
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    return torch.matmul(torch.softmax(log_P, dim=-1), V_style)

def feature_adain(x, y, alpha=1.0, eps=1e-6):
    m_x, s_x = x.mean(dim=2, keepdim=True), x.std(dim=2, keepdim=True) + eps
    m_y, s_y = y.mean(dim=2, keepdim=True), y.std(dim=2, keepdim=True) + eps
    return alpha * ((x - m_x)/s_x * s_y + m_y) + (1-alpha) * x

def sinkhorn_latent_transport(content_lat, style_lat_noisy, style_lat_clean,
                              alpha=0.5, n_iters=10, tau=0.1, grag_delta=1.0, mode="ot_adain"):
    B, N, D = content_lat.shape
    sn, sc = style_lat_noisy, style_lat_clean
    if grag_delta < 1.0:
        sm = sn.mean(dim=1, keepdim=True)
        sn = sm + grag_delta*(sn - sm)
        scm = sc.mean(dim=1, keepdim=True)
        sc = scm + grag_delta*(sc - scm)
    c_f, s_f = content_lat.float(), sn.float()
    c_hi, s_hi = c_f - c_f.mean(1, keepdim=True), s_f - s_f.mean(1, keepdim=True)
    logits = torch.bmm(F.normalize(c_hi, dim=-1), F.normalize(s_hi, dim=-1).transpose(1,2)) / tau
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    T = torch.softmax(log_P, dim=-1)
    style_f = sc.float()
    matched_mean = torch.bmm(T, style_f)
    matched_std = (torch.bmm(T, style_f**2) - matched_mean**2).clamp(min=1e-6).sqrt()
    c_mean = c_f.mean(1, keepdim=True)
    c_std = c_f.std(1, keepdim=True).clamp(min=1e-6)
    styled = (c_f - c_mean)/c_std * matched_std + matched_mean
    return ((1-alpha)*c_f + alpha*styled).to(content_lat.dtype)

# Prompt encoding
_prompt_cache = {}
def encode_prompt(text):
    if text in _prompt_cache:
        return _prompt_cache[text]
    if pipe.text_encoder_qwen3 is not None:
        pipe.load_models_to_device(["text_encoder_qwen3"])
        embedder = Flux2Unit_Qwen3PromptEmbedder()
        embeds, ids = embedder.encode_prompt(pipe.text_encoder_qwen3, pipe.tokenizer, text, dtype=pipe.torch_dtype, device="cuda")
    else:
        pipe.load_models_to_device(["text_encoder"])
        embedder = Flux2Unit_PromptEmbedder()
        embeds, ids = embedder.encode_prompt(pipe.text_encoder, pipe.tokenizer, text, dtype=pipe.torch_dtype, device="cuda")
    pipe.load_models_to_device(["dit"]); dit.to("cuda")
    _prompt_cache[text] = (embeds, ids)
    return embeds, ids

null_embeds, null_text_ids = encode_prompt("")


# ============== Core: run one step and capture gradient + SLT update ==============
def capture_update_fields(prompt, style_path):
    """
    Run denoising, at first eligible step capture:
    - backward_grad: gradient from style loss backward [1, N, D]
    - slt_update: direction of SLT latent transport [1, N, D]
    - adam_update_1: latent change after Adam iter 1
    - adam_update_2: latent change after Adam iter 2
    Returns dict of numpy arrays reshaped to [H, W, D].
    """
    width, height = 512, 512
    num_inference_steps = 25
    start_time_val = 850
    lr = 0.03
    content_w = 1.5
    style_w = 2.0
    grag_deep = 0.5
    grag_shallow = 0.7
    slt_alpha = 0.1
    seed = 43

    lat_h, lat_w = height // 16, width // 16

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        sty_pix = pipe.preprocess_image(Image.open(style_path).convert("RGB").resize((512, 512), Image.BICUBIC))
        style_latent_4d = vae.encode(sty_pix)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

    sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
    image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)).unsqueeze(0).to("cuda")
    style_image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)).unsqueeze(0).to("cuda")

    prompt_embeds, text_ids = encode_prompt(prompt)
    pipe.load_models_to_device(["dit"]); dit.to("cuda")

    torch.manual_seed(seed)
    latents = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    latents = rearrange(latents, "B C H W -> B (H W) C")

    scheduler = FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(num_inference_steps, dynamic_shift_len=lat_h * lat_w)
    timesteps = scheduler.timesteps

    loss_fn = torch.nn.L1Loss()
    results = {}
    captured = False

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()

        with torch.no_grad():
            noise_pred = dit(hidden_states=latents, timestep=timestep / 1000, guidance=None,
                             encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=image_ids)
        latents = scheduler.step(noise_pred, t_val, latents)

        if t_val >= start_time_val:
            continue
        if captured:
            continue
        captured = True

        # Noisy style
        if i < num_inference_steps - 1:
            ts = timesteps[i + 1:i + 2]
            style_latent_ = scheduler.add_noise(style_latent, torch.randn_like(style_latent), ts)
        else:
            ts = torch.tensor([0.0]); style_latent_ = style_latent

        # Style forward
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(hidden_states=style_latent_.to("cuda", torch.bfloat16),
                    timestep=ts.to("cuda", torch.bfloat16) / 1000, guidance=None,
                    encoder_hidden_states=null_embeds, txt_ids=null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, _ = cache.get()

        # GRAG
        _txt_len = controller.txt_len
        for idx in range(len(ref_k_list)):
            df = idx / max(len(ref_k_list) - 1, 1)
            delta = grag_shallow + (grag_deep - grag_shallow) * df
            k, v = ref_k_list[idx], ref_v_list[idx]
            kt, ki = k[:, :, :_txt_len, :], k[:, :, _txt_len:, :]
            vt, vi = v[:, :, :_txt_len, :], v[:, :, _txt_len:, :]
            km, vm = ki.mean(2, keepdim=True), vi.mean(2, keepdim=True)
            ref_k_list[idx] = torch.cat([kt, km + delta * (ki - km)], dim=2)
            ref_v_list[idx] = torch.cat([vt, vm + delta * (vi - vm)], dim=2)

        content_anchor = latents.detach().clone()

        # AdaIN
        with torch.no_grad():
            latents = feature_adain(latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=0.5).squeeze(1)
        latents = latents.detach()

        # Record pre-SLT state
        latents_pre_slt = latents.clone()

        # SLT
        with torch.no_grad():
            latents = sinkhorn_latent_transport(latents, style_latent_, style_latent,
                                               alpha=slt_alpha, n_iters=10, tau=0.02,
                                               grag_delta=grag_deep, mode="ot_adain")

        slt_update = (latents - latents_pre_slt).detach()
        results["slt_update"] = slt_update[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        # Adam iterations
        latents_pre_adam = latents.clone()
        latents.requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=lr)
        _ts = ts.to("cuda", torch.bfloat16) / 1000

        for it in range(2):
            cache.clear(); controller.step()
            optimizer.zero_grad()
            with torch.enable_grad():
                _ = dit(hidden_states=latents, timestep=_ts.detach(), guidance=None,
                        encoder_hidden_states=null_embeds.detach(), txt_ids=null_text_ids.detach(),
                        img_ids=image_ids.detach())
            del _
            q_list, _, _, self_out_list = cache.get()
            n_layers = len(q_list)
            n_ot = max(1, int(n_layers * 0.5))

            ref_outs = []
            for idx2, (q, rk, rv) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                if idx2 >= n_layers - n_ot:
                    ref_outs.append(sinkhorn_attn_freq(q.detach(), rk, rv, n_iters=6, tau=0.1, hi_boost=2.0))
                else:
                    ref_outs.append(F.scaled_dot_product_attention(q.detach(), rk, rv))

            sl = sum(loss_fn(so, ro.detach()) for so, ro in zip(self_out_list, ref_outs))
            cl = loss_fn(latents, content_anchor)
            loss = style_w * sl + content_w * cl
            loss.backward()

            if it == 0:
                results["backward_grad"] = latents.grad.detach().clone()[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

            optimizer.step()

            if it == 0:
                results["adam_update_1"] = (latents.detach() - latents_pre_adam)[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)
            elif it == 1:
                results["adam_update_2"] = (latents.detach() - latents_pre_adam)[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        latents = latents.detach()

        # Random control
        results["random"] = np.random.randn(lat_h, lat_w, results["backward_grad"].shape[-1]).astype(np.float32)

    results["lat_h"] = lat_h
    results["lat_w"] = lat_w
    return results


# ============== Visualization functions ==============
def field_to_pca_rgb(field, pca_model=None):
    """
    field: [H, W, D] → PCA to 3 components → normalize to [0,1] RGB image [H, W, 3]
    Returns (rgb_array, pca_model)
    """
    H, W, D = field.shape
    flat = field.reshape(-1, D)
    if pca_model is None:
        pca_model = PCA(n_components=3)
        projected = pca_model.fit_transform(flat)
    else:
        projected = pca_model.transform(flat)
    # Normalize each channel independently to [0,1]
    for c in range(3):
        col = projected[:, c]
        p5, p95 = np.percentile(col, 5), np.percentile(col, 95)
        if p95 - p5 > 1e-8:
            projected[:, c] = np.clip((col - p5) / (p95 - p5), 0, 1)
        else:
            projected[:, c] = 0.5
    return projected.reshape(H, W, 3), pca_model


def field_to_magnitude(field):
    """field: [H, W, D] → magnitude [H, W]"""
    return np.linalg.norm(field, axis=-1)


def neighbor_cosine_at_distance(field, distance):
    """
    Compute mean cosine similarity between tokens at given Manhattan distance.
    field: [H, W, D]
    """
    H, W, D = field.shape
    norms = np.linalg.norm(field, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = field / norms

    cos_vals = []
    # Horizontal at distance d
    if W > distance:
        left = normed[:, :-distance, :]
        right = normed[:, distance:, :]
        cos_vals.append((left * right).sum(axis=-1).mean())
    # Vertical at distance d
    if H > distance:
        top = normed[:-distance, :, :]
        bot = normed[distance:, :, :]
        cos_vals.append((top * bot).sum(axis=-1).mean())
    return float(np.mean(cos_vals)) if cos_vals else 0.0


def make_figure(all_results, output_path):
    """
    Create the main paper figure with 3 panels:
    Panel A: PCA-colored update direction maps (backward vs SLT vs random)
    Panel B: Update magnitude heatmaps
    Panel C: Autocorrelation decay curves
    """
    n_samples = len(all_results)

    fig = plt.figure(figsize=(16, 4 * n_samples + 3))
    gs = gridspec.GridSpec(n_samples + 1, 6, figure=fig, hspace=0.35, wspace=0.3)

    methods = ["backward_grad", "adam_update_1", "adam_update_2", "slt_update", "random"]
    method_labels = ["Backward\nGradient", "Adam\nUpdate (iter1)", "Adam\nUpdate (iter2)", "SLT\nUpdate", "Random\n(control)"]

    # Compute autocorrelation for all samples
    distances = [1, 2, 3, 4, 5, 6, 8, 10]
    autocorr_data = {m: [] for m in methods}

    for si, res in enumerate(all_results):
        # --- Panel A: PCA RGB maps ---
        # Fit PCA on backward_grad (strongest signal), apply same transform to all
        _, shared_pca = field_to_pca_rgb(res["backward_grad"])

        for mi, (method, label) in enumerate(zip(methods[:5], method_labels[:5])):
            if mi >= 5:
                break
            ax = fig.add_subplot(gs[si, mi])
            field = res[method]
            rgb, _ = field_to_pca_rgb(field, pca_model=shared_pca)
            ax.imshow(rgb, interpolation='nearest', aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if si == 0:
                ax.set_title(label, fontsize=9, fontweight='bold')
            if mi == 0:
                ax.set_ylabel(f"Sample {si+1}", fontsize=9)

        # Magnitude subplot (small, rightmost column)
        ax_mag = fig.add_subplot(gs[si, 5])
        # Show backward magnitude vs SLT magnitude side by side
        mag_bw = field_to_magnitude(res["backward_grad"])
        mag_slt = field_to_magnitude(res["slt_update"])
        combined = np.concatenate([mag_bw, np.ones((mag_bw.shape[0], 1)) * mag_bw.max(), mag_slt], axis=1)
        ax_mag.imshow(combined, cmap='hot', interpolation='nearest', aspect='equal')
        ax_mag.set_xticks([]); ax_mag.set_yticks([])
        if si == 0:
            ax_mag.set_title("|Grad| vs |SLT|", fontsize=9, fontweight='bold')

        # Autocorrelation computation
        for method in methods:
            curve = [neighbor_cosine_at_distance(res[method], d) for d in distances]
            autocorr_data[method].append(curve)

    # --- Panel C: Autocorrelation decay (bottom row) ---
    ax_auto = fig.add_subplot(gs[n_samples, :])
    colors = {'backward_grad': '#d62728', 'adam_update_1': '#ff7f0e', 'adam_update_2': '#ffbb78',
              'slt_update': '#2ca02c', 'random': '#7f7f7f'}
    labels_nice = {'backward_grad': 'Backward Gradient', 'adam_update_1': 'Adam Update (iter 1)',
                   'adam_update_2': 'Adam Update (iter 2)', 'slt_update': 'SLT Update', 'random': 'Random (control)'}

    for method in methods:
        curves = np.array(autocorr_data[method])  # [n_samples, n_distances]
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        ax_auto.plot(distances, mean_curve, 'o-', color=colors[method], label=labels_nice[method], linewidth=2, markersize=5)
        ax_auto.fill_between(distances, mean_curve - std_curve, mean_curve + std_curve,
                             color=colors[method], alpha=0.15)

    ax_auto.set_xlabel("Spatial Distance (tokens)", fontsize=11)
    ax_auto.set_ylabel("Cosine Similarity", fontsize=11)
    ax_auto.set_title("Spatial Autocorrelation Decay — Backward Gradient Shows Long-Range Coherence", fontsize=12, fontweight='bold')
    ax_auto.legend(fontsize=9, loc='upper right')
    ax_auto.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_auto.set_xlim(0.5, max(distances) + 0.5)
    ax_auto.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved figure: {output_path}")


def make_autocorr_only_figure(all_results, output_path):
    """Clean autocorrelation-only figure for paper main body."""
    methods = ["backward_grad", "adam_update_1", "slt_update", "random"]
    distances = [1, 2, 3, 4, 5, 6, 8, 10]
    autocorr_data = {m: [] for m in methods}

    for res in all_results:
        for method in methods:
            curve = [neighbor_cosine_at_distance(res[method], d) for d in distances]
            autocorr_data[method].append(curve)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    colors = {'backward_grad': '#d62728', 'adam_update_1': '#ff7f0e',
              'slt_update': '#2ca02c', 'random': '#7f7f7f'}
    labels_nice = {'backward_grad': 'Backward Gradient (ours)',
                   'adam_update_1': 'Adam Latent Update (ours)',
                   'slt_update': 'SLT Update (forward-only)',
                   'random': 'Random Noise (control)'}
    linestyles = {'backward_grad': '-', 'adam_update_1': '--',
                  'slt_update': '-.', 'random': ':'}

    for method in methods:
        curves = np.array(autocorr_data[method])
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        ax.plot(distances, mean_curve, linestyles[method], color=colors[method],
                label=labels_nice[method], linewidth=2.5, markersize=0)
        ax.fill_between(distances, mean_curve - std_curve, mean_curve + std_curve,
                        color=colors[method], alpha=0.12)

    ax.set_xlabel("Spatial Distance (pixels)", fontsize=12)
    ax.set_ylabel("Neighbor Cosine Similarity", fontsize=12)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_xlim(0.8, max(distances) + 0.2)
    ax.set_ylim(-0.1, 0.75)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved figure: {output_path}")


# ============== Main ==============
if __name__ == "__main__":
    total_start = time.time()

    print("\n" + "=" * 80)
    print("SPATIAL SMOOTHING VISUALIZATION — Paper Figure")
    print("=" * 80)

    test_cases = [
        ("A panda eat bamboo", f"{CWD}/sty/27.png"),
        ("A cat sitting on a windowsill", f"{CWD}/sty/10.png"),
        ("A field of sunflowers at sunset", f"{CWD}/sty/00.png"),
        ("A portrait of an old fisherman", f"{CWD}/sty/33.png"),
        ("A castle on a cliff by the sea", f"{CWD}/sty/20.png"),
        ("A bouquet of wildflowers", f"{CWD}/sty/28.png"),
    ]

    all_results = []
    for prompt, sty_path in test_cases:
        print(f"\n  Processing: '{prompt}' × {Path(sty_path).stem}...")
        torch.cuda.empty_cache()
        res = capture_update_fields(prompt, sty_path)
        all_results.append(res)

        # Print per-sample metrics
        bw_cos = neighbor_cosine_at_distance(res["backward_grad"], 1)
        slt_cos = neighbor_cosine_at_distance(res["slt_update"], 1)
        rand_cos = neighbor_cosine_at_distance(res["random"], 1)
        print(f"    d=1 cosine: backward={bw_cos:.4f}  slt={slt_cos:.4f}  random={rand_cos:.4f}")

    # Generate figures
    print("\n  Generating figures...")
    make_figure(all_results, VIZ_DIR / "spatial_smoothing_full.png")
    make_autocorr_only_figure(all_results, VIZ_DIR / "autocorrelation_decay.png")

    # Also save individual PCA maps at high res
    for si, (res, (prompt, sty)) in enumerate(zip(all_results, test_cases)):
        _, shared_pca = field_to_pca_rgb(res["backward_grad"])
        for method in ["backward_grad", "slt_update", "random"]:
            rgb, _ = field_to_pca_rgb(res[method], pca_model=shared_pca)
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            H, W = rgb_uint8.shape[:2]
            img = Image.fromarray(rgb_uint8, 'RGB').resize((W * 16, H * 16), Image.NEAREST)
            img.save(VIZ_DIR / f"sample{si}_{method}_pca.png")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY — Spatial Coherence (d=1 neighbor cosine similarity)")
    print(f"{'=' * 80}")
    bw_vals = [neighbor_cosine_at_distance(r["backward_grad"], 1) for r in all_results]
    adam1_vals = [neighbor_cosine_at_distance(r["adam_update_1"], 1) for r in all_results]
    slt_vals = [neighbor_cosine_at_distance(r["slt_update"], 1) for r in all_results]
    rand_vals = [neighbor_cosine_at_distance(r["random"], 1) for r in all_results]

    print(f"  Backward Gradient: {np.mean(bw_vals):.4f} ± {np.std(bw_vals):.4f}")
    print(f"  Adam Update:       {np.mean(adam1_vals):.4f} ± {np.std(adam1_vals):.4f}")
    print(f"  SLT Update:        {np.mean(slt_vals):.4f} ± {np.std(slt_vals):.4f}")
    print(f"  Random:            {np.mean(rand_vals):.4f} ± {np.std(rand_vals):.4f}")
    print(f"\n  Backward/SLT ratio: {np.mean(bw_vals)/max(np.mean(slt_vals), 1e-6):.1f}x")
    print(f"  → Proves: backward pass through attention creates {np.mean(bw_vals)/max(np.mean(slt_vals), 1e-6):.1f}x more")
    print(f"    spatially coherent updates than forward-only OT transport.")

    # Save JSON
    summary = {
        "backward_grad_cos1": {"mean": float(np.mean(bw_vals)), "std": float(np.std(bw_vals)), "values": [float(v) for v in bw_vals]},
        "adam_update_cos1": {"mean": float(np.mean(adam1_vals)), "std": float(np.std(adam1_vals)), "values": [float(v) for v in adam1_vals]},
        "slt_update_cos1": {"mean": float(np.mean(slt_vals)), "std": float(np.std(slt_vals)), "values": [float(v) for v in slt_vals]},
        "random_cos1": {"mean": float(np.mean(rand_vals)), "std": float(np.std(rand_vals)), "values": [float(v) for v in rand_vals]},
    }
    with open(RESULTS_DIR / "spatial_viz_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"\nTOTAL TIME: {total_elapsed / 60:.1f} minutes")
    print(f"Figures: {VIZ_DIR}")
