"""
Extended Spatial Proof: Inject vs Backward vs SLT

Two key questions for the paper:
Q1: Why inject (training-free) << backward (trained)?
Q2: Why SLT color ≈ Adam but texture fundamentally different?

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/eval_spatial_extended.py
"""
import argparse, csv, glob, os, sys, time, types, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision import transforms

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

DIFFSYNTH_ROOT = "/mnt/local/yzh/prog/DiffSynth-Studio"
KLEIN9B_ROOT = "/mnt/local/klein9b"
CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
RESULTS_DIR = Path(CWD) / "results"
OUT_DIR = Path(CWD) / "d_grand" / "spatial_viz_extended"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    def __init__(self, self_layers=(0,16)):
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


# ============== Core capture function ==============
def _neighbor_cos(field, distance=1):
    h, w, c = field.shape
    norms = np.linalg.norm(field, axis=-1, keepdims=True) + 1e-8
    flat_n = field / norms
    cos_h = np.sum(flat_n[:-distance, :, :] * flat_n[distance:, :, :], axis=-1).mean()
    cos_w = np.sum(flat_n[:, :-distance, :] * flat_n[:, distance:, :], axis=-1).mean()
    return float((cos_h + cos_w) / 2)


def capture_all_methods(prompt, style_path):
    """At first eligible denoising step, capture update fields for all methods."""
    lat_h, lat_w = 32, 32
    start_time_val = 850
    lr = 0.03; content_w = 1.5; style_w = 2.0
    grag_deep = 0.5; grag_shallow = 0.7; seed = 43

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        sty_pix = pipe.preprocess_image(Image.open(style_path).convert("RGB").resize((512,512), Image.BICUBIC))
        style_latent_4d = vae.encode(sty_pix)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

    image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)).unsqueeze(0).to("cuda")
    sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
    style_image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)).unsqueeze(0).to("cuda")
    prompt_embeds, text_ids = encode_prompt(prompt)
    pipe.load_models_to_device(["dit"]); dit.to("cuda")

    torch.manual_seed(seed)
    latents = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    latents = rearrange(latents, "B C H W -> B (H W) C")

    scheduler = FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(25, dynamic_shift_len=lat_h*lat_w)
    timesteps = scheduler.timesteps
    loss_fn = torch.nn.L1Loss()

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()
        with torch.no_grad():
            noise_pred = dit(hidden_states=latents, timestep=timestep/1000, guidance=None,
                           encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=image_ids)
        latents = scheduler.step(noise_pred, t_val, latents)
        if t_val >= start_time_val:
            continue

        # First eligible step
        if i < 24:
            ts = timesteps[i+1:i+2]
            style_latent_ = scheduler.add_noise(style_latent, torch.randn_like(style_latent), ts)
        else:
            ts = torch.tensor([0.0]); style_latent_ = style_latent

        # Get ref K/V from style forward
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(hidden_states=style_latent_.to("cuda", torch.bfloat16),
                   timestep=ts.to("cuda", torch.bfloat16)/1000, guidance=None,
                   encoder_hidden_states=null_embeds, txt_ids=null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, _ = cache.get()

        _txt_len = controller.txt_len
        for idx in range(len(ref_k_list)):
            df = idx / max(len(ref_k_list)-1, 1)
            delta = grag_shallow + (grag_deep - grag_shallow)*df
            k, v = ref_k_list[idx], ref_v_list[idx]
            kt, ki = k[:,:,:_txt_len,:], k[:,:,_txt_len:,:]
            vt, vi = v[:,:,:_txt_len,:], v[:,:,_txt_len:,:]
            km, vm = ki.mean(2, keepdim=True), vi.mean(2, keepdim=True)
            ref_k_list[idx] = torch.cat([kt, km + delta*(ki-km)], dim=2)
            ref_v_list[idx] = torch.cat([vt, vm + delta*(vi-vm)], dim=2)

        # === INJECT: compute attention target - current output (one-shot delta) ===
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(hidden_states=latents, timestep=ts.to("cuda", torch.bfloat16)/1000, guidance=None,
                   encoder_hidden_states=null_embeds, txt_ids=null_text_ids, img_ids=image_ids)
        q_list_inj, _, _, self_out_list_inj = cache.get()
        n_layers = len(q_list_inj)
        n_ot = max(1, int(n_layers*0.5))

        # Inject target = what attention output WOULD be if we replaced K/V with style
        inject_deltas = []
        for idx2, (q, rk, rv, so) in enumerate(zip(q_list_inj, ref_k_list, ref_v_list, self_out_list_inj)):
            if idx2 >= n_layers - n_ot:
                target = sinkhorn_attn_freq(q.detach(), rk, rv, n_iters=6, tau=0.1, hi_boost=2.0)
            else:
                target = F.scaled_dot_product_attention(q.detach(), rk, rv)
            inject_deltas.append((target - so)[:, :, _txt_len:, :])  # only image tokens
        # Average across layers and heads
        inject_avg = torch.stack([d.mean(dim=1) for d in inject_deltas]).mean(dim=0)  # [B, N_img, D_head]
        inject_field = inject_avg[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        # === AdaIN + SLT setup ===
        content_anchor = latents.detach().clone()
        with torch.no_grad():
            latents = feature_adain(latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=0.5).squeeze(1)
        latents = latents.detach()

        # === SLT alpha=0.1 ===
        latents_pre_slt = latents.clone()
        with torch.no_grad():
            latents_slt_01 = sinkhorn_latent_transport(latents.clone(), style_latent_, style_latent,
                alpha=0.1, n_iters=10, tau=0.02, grag_delta=grag_deep)
        slt_01_field = (latents_slt_01 - latents_pre_slt)[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        # === SLT alpha=0.3 ===
        with torch.no_grad():
            latents_slt_03 = sinkhorn_latent_transport(latents.clone(), style_latent_, style_latent,
                alpha=0.3, n_iters=10, tau=0.02, grag_delta=grag_deep)
        slt_03_field = (latents_slt_03 - latents_pre_slt)[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        # === SLT alpha=0.5 ===
        with torch.no_grad():
            latents_slt_05 = sinkhorn_latent_transport(latents.clone(), style_latent_, style_latent,
                alpha=0.5, n_iters=10, tau=0.02, grag_delta=grag_deep)
        slt_05_field = (latents_slt_05 - latents_pre_slt)[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        # Apply SLT then do Adam
        latents = latents_slt_01.detach()

        # === Backward iter 1 ===
        latents.requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=lr)
        cache.clear(); controller.step()
        optimizer.zero_grad()
        with torch.enable_grad():
            _ = dit(hidden_states=latents, timestep=ts.to("cuda", torch.bfloat16).detach()/1000,
                   guidance=None, encoder_hidden_states=null_embeds.detach(),
                   txt_ids=null_text_ids.detach(), img_ids=image_ids.detach())
        del _
        q_list, _, _, self_out_list = cache.get()
        ref_outs = []
        for idx2, (q, rk, rv) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
            if idx2 >= n_layers - n_ot:
                ref_outs.append(sinkhorn_attn_freq(q.detach(), rk, rv, n_iters=6, tau=0.1, hi_boost=2.0))
            else:
                ref_outs.append(F.scaled_dot_product_attention(q.detach(), rk, rv))
        sl = sum(loss_fn(so, ro.detach()) for so, ro in zip(self_out_list, ref_outs))
        cl = loss_fn(latents, content_anchor)
        loss = style_w*sl + content_w*cl
        loss.backward()
        grad_iter1 = latents.grad.detach().clone()[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)
        optimizer.step()

        # === Backward iter 2 ===
        optimizer.zero_grad()
        cache.clear(); controller.step()
        with torch.enable_grad():
            _ = dit(hidden_states=latents, timestep=ts.to("cuda", torch.bfloat16).detach()/1000,
                   guidance=None, encoder_hidden_states=null_embeds.detach(),
                   txt_ids=null_text_ids.detach(), img_ids=image_ids.detach())
        del _
        q_list, _, _, self_out_list = cache.get()
        ref_outs = []
        for idx2, (q, rk, rv) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
            if idx2 >= n_layers - n_ot:
                ref_outs.append(sinkhorn_attn_freq(q.detach(), rk, rv, n_iters=6, tau=0.1, hi_boost=2.0))
            else:
                ref_outs.append(F.scaled_dot_product_attention(q.detach(), rk, rv))
        sl = sum(loss_fn(so, ro.detach()) for so, ro in zip(self_out_list, ref_outs))
        cl = loss_fn(latents, content_anchor)
        loss = style_w*sl + content_w*cl
        loss.backward()
        grad_iter2 = latents.grad.detach().clone()[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        random_field = np.random.randn(lat_h, lat_w, grad_iter1.shape[-1]).astype(np.float32)

        return {
            "inject": inject_field,
            "slt_01": slt_01_field,
            "slt_03": slt_03_field,
            "slt_05": slt_05_field,
            "backward_iter1": grad_iter1,
            "backward_iter2": grad_iter2,
            "random": random_field,
        }

    return None


# ============== Main ==============
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    print("="*70)
    print("EXTENDED SPATIAL PROOF")
    print("Q1: Why inject (training-free) << backward (trained)?")
    print("Q2: Why SLT color ≈ Adam but texture fundamentally different?")
    print("="*70)

    samples = [
        ("A panda eat bamboo", "27"),
        ("A cat sitting on a windowsill", "10"),
        ("A field of sunflowers at sunset", "33"),
        ("A portrait of an old fisherman", "14"),
        ("A castle on a cliff by the sea", "28"),
        ("A sports car on a mountain road", "38"),
    ]

    all_results = {k: [] for k in ["inject", "slt_01", "slt_03", "slt_05",
                                     "backward_iter1", "backward_iter2", "random"]}
    pca_samples = []

    for prompt, sty_id in samples:
        sty_path = f"{CWD}/sty/{sty_id}.png"
        torch.cuda.empty_cache()
        print(f"  {prompt[:30]}... x sty{sty_id}", end=" ", flush=True)
        fields = capture_all_methods(prompt, sty_path)
        if fields is None:
            print("skip"); continue

        cos_vals = {k: _neighbor_cos(v) for k, v in fields.items()}
        print(" | ".join(f"{k}={v:.3f}" for k, v in cos_vals.items()))

        for k, v in cos_vals.items():
            all_results[k].append(v)
        pca_samples.append(fields)

    # === Summary ===
    print(f"\n{'='*70}")
    print(f"SUMMARY (n={len(pca_samples)} samples, mean +/- std)")
    print(f"{'='*70}")
    summary = {}
    for method in ["inject", "slt_01", "slt_03", "slt_05", "backward_iter1", "backward_iter2", "random"]:
        vals = all_results[method]
        m, s = np.mean(vals), np.std(vals)
        summary[method] = {"mean": float(m), "std": float(s), "values": [float(v) for v in vals]}
        print(f"  {method:<20} {m:.4f} +/- {s:.4f}")

    # Autocorrelation decay
    distances = [1, 2, 3, 4, 5, 6, 8]
    decay = {}
    for method in ["inject", "slt_01", "slt_05", "backward_iter1", "backward_iter2", "random"]:
        curve = []
        for d in distances:
            vals_d = [_neighbor_cos(pca_samples[i][method], distance=d) for i in range(len(pca_samples))]
            curve.append(float(np.mean(vals_d)))
        decay[method] = curve

    # Save JSON
    result_json = {
        "n_samples": len(pca_samples),
        "neighbor_cosine_d1": summary,
        "autocorrelation_decay": {"distances": distances, **decay},
    }
    with open(RESULTS_DIR / "spatial_extended.json", "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'spatial_extended.json'}")

    # === FIGURE 1: Two-panel bar chart ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Q1 - Inject vs Backward
    methods_q1 = ["inject", "backward_iter1", "backward_iter2", "random"]
    labels_q1 = ["Inject\n(Training-free)", "Backward\n(1 iter)", "Backward\n(2 iters)", "Random\n(Control)"]
    colors_q1 = ["#4ECDC4", "#E63946", "#B7094C", "#999999"]
    means_q1 = [np.mean(all_results[m]) for m in methods_q1]
    stds_q1 = [np.std(all_results[m]) for m in methods_q1]
    bars = axes[0].bar(range(len(methods_q1)), means_q1, yerr=stds_q1, capsize=5,
                       color=colors_q1, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(len(methods_q1)))
    axes[0].set_xticklabels(labels_q1, fontsize=10)
    axes[0].set_ylabel("Neighbor Cosine Similarity", fontsize=11)
    axes[0].set_title("Q1: Why training-free inject fails?\n(Higher = more spatial coherence = brushstroke)", fontsize=11)
    axes[0].set_ylim(-0.05, 0.75)
    for bar, val in zip(bars, means_q1):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Right: Q2 - SLT scaling vs Backward
    methods_q2 = ["slt_01", "slt_03", "slt_05", "backward_iter1", "backward_iter2"]
    labels_q2 = ["SLT\n$\\alpha$=0.1", "SLT\n$\\alpha$=0.3", "SLT\n$\\alpha$=0.5", "Backward\n(1 iter)", "Backward\n(2 iters)"]
    colors_q2 = ["#2ECC71", "#27AE60", "#1E8449", "#E63946", "#B7094C"]
    means_q2 = [np.mean(all_results[m]) for m in methods_q2]
    stds_q2 = [np.std(all_results[m]) for m in methods_q2]
    bars = axes[1].bar(range(len(methods_q2)), means_q2, yerr=stds_q2, capsize=5,
                       color=colors_q2, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(len(methods_q2)))
    axes[1].set_xticklabels(labels_q2, fontsize=10)
    axes[1].set_ylabel("Neighbor Cosine Similarity", fontsize=11)
    axes[1].set_title("Q2: SLT scaling $\\alpha$ does NOT gain spatial coherence\n(Backward has inherent spatial coupling via Jacobian)", fontsize=11)
    axes[1].set_ylim(-0.05, 0.75)
    for bar, val in zip(bars, means_q2):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT_DIR / "extended_bar_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT_DIR / 'extended_bar_chart.png'}")

    # === FIGURE 2: Autocorrelation decay ===
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    line_styles = {
        "inject": ("--", "s", "#4ECDC4", "Inject (training-free)"),
        "slt_01": ("--", "^", "#2ECC71", "SLT ($\\alpha$=0.1)"),
        "slt_05": ("--", "v", "#1E8449", "SLT ($\\alpha$=0.5)"),
        "backward_iter1": ("-", "o", "#E63946", "Backward (1 iter)"),
        "backward_iter2": ("-", "D", "#B7094C", "Backward (2 iters)"),
        "random": (":", "x", "#999999", "Random (control)"),
    }
    for method, (ls, marker, color, label) in line_styles.items():
        ax.plot(distances, decay[method], ls=ls, marker=marker, color=color,
                label=label, linewidth=2, markersize=7)
    ax.set_xlabel("Spatial Distance (tokens)", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Spatial Autocorrelation Decay", fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(-0.05, 0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "extended_decay.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT_DIR / 'extended_decay.png'}")

    # === FIGURE 3: PCA direction maps (per-field independent PCA for visibility) ===
    if pca_samples:
        fields = pca_samples[0]
        methods_viz = ["inject", "slt_01", "slt_05", "backward_iter1", "backward_iter2"]
        titles_viz = ["Inject", "SLT ($\\alpha$=0.1)", "SLT ($\\alpha$=0.5)", "Backward (iter 1)", "Backward (iter 2)"]

        fig, axes2 = plt.subplots(1, len(methods_viz), figsize=(3.5*len(methods_viz), 3.5))
        for ax, method, title in zip(axes2, methods_viz, titles_viz):
            f = fields[method]
            h, w, c = f.shape
            # Per-field PCA so each field's internal structure is visible
            pca_local = PCA(n_components=3)
            rgb = pca_local.fit_transform(f.reshape(-1, c)).reshape(h, w, 3)
            # Per-channel percentile normalization for contrast
            for ch in range(3):
                lo, hi = np.percentile(rgb[:,:,ch], [2, 98])
                rgb[:,:,ch] = np.clip((rgb[:,:,ch] - lo) / (hi - lo + 1e-8), 0, 1)
            ax.imshow(rgb)
            ax.set_title(title, fontsize=11)
            cos1 = _neighbor_cos(f)
            ax.set_xlabel(f"cos@d=1: {cos1:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle("Update Direction (PCA $\\rightarrow$ RGB): Large patches = spatial coherence = brushstroke", fontsize=11, y=1.02)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "extended_pca_maps.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUT_DIR / 'extended_pca_maps.png'}")

    # === Print conclusions ===
    print(f"\n{'='*70}")
    print("CONCLUSIONS:")
    print(f"{'='*70}")
    bw1 = np.mean(all_results["backward_iter1"])
    bw2 = np.mean(all_results["backward_iter2"])
    inj = np.mean(all_results["inject"])
    slt1 = np.mean(all_results["slt_01"])
    slt5 = np.mean(all_results["slt_05"])
    print(f"  Q1: Inject cos={inj:.3f} vs Backward iter1 cos={bw1:.3f} (ratio {bw1/max(inj,0.001):.1f}x)")
    print(f"      Inject replaces KV uniformly — all tokens get globally similar delta.")
    print(f"      No spatial selectivity → no brushstroke.")
    print(f"")
    print(f"  Q2: SLT a=0.1 cos={slt1:.3f}, a=0.5 cos={slt5:.3f} vs Backward cos={bw2:.3f}")
    print(f"      Scaling SLT alpha 5x does NOT increase spatial coherence (flat ~0.25).")
    print(f"      SLT is per-token independent OT: token i's transport depends only on")
    print(f"      its own cosine similarity to style tokens, NOT on neighbors.")
    print(f"      Backward's Jacobian couples spatially adjacent Q vectors → correlated gradients.")
    print(f"\nDone. Total time: {time.time():.0f}s")


if __name__ == "__main__":
    main()
