"""
Frequency Decomposition Analysis: SLT vs Backward complementarity.

Core thesis: SLT provides fast low-frequency (color) alignment, backward provides
high-frequency (texture/brushstroke). Together they are complementary.

Measures:
1. For each update field (SLT, backward iter1-8), compute 2D FFT
2. Split energy into low-freq (DC + first 4 freq bins) and high-freq (rest)
3. Show: SLT ≈ backward's low-freq component (high cosine similarity)
4. Show: backward accumulates high-freq over iterations
5. Token-wise direction alignment: SLT vs backward gradient

Output: results/freq_decompose.json + d_grand/freq_analysis/

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/eval_freq_decompose.py
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
OUT_DIR = Path(CWD) / "d_grand" / "freq_analysis"
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
                              alpha=0.5, n_iters=10, tau=0.1, grag_delta=1.0):
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


# ============== Frequency analysis helpers ==============
def freq_energy_split(field, cutoff=4):
    """
    Split a 32×32×D field into low-freq and high-freq energy.
    cutoff: radius in frequency domain (DC=0, max=16)
    Returns: (low_energy_ratio, high_energy_ratio, low_field, high_field)
    """
    h, w, d = field.shape
    total_e = 0.0
    low_e = 0.0
    low_field = np.zeros_like(field)
    high_field = np.zeros_like(field)

    for ch in range(d):
        fft2d = np.fft.fft2(field[:, :, ch])
        fft_shifted = np.fft.fftshift(fft2d)
        magnitude = np.abs(fft_shifted) ** 2

        # Create low-freq mask (circle of radius=cutoff centered at DC)
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        mask_low = ((yy - cy)**2 + (xx - cx)**2) <= cutoff**2

        ch_low_e = magnitude[mask_low].sum()
        ch_total_e = magnitude.sum()
        total_e += ch_total_e
        low_e += ch_low_e

        # Reconstruct low/high components
        fft_low = fft_shifted * mask_low
        fft_high = fft_shifted * (~mask_low)
        low_field[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(fft_low)))
        high_field[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(fft_high)))

    low_ratio = low_e / (total_e + 1e-12)
    high_ratio = 1.0 - low_ratio
    return float(low_ratio), float(high_ratio), low_field, high_field


def direction_cosine(field_a, field_b):
    """Per-token cosine similarity between two fields, then averaged."""
    # field: [H, W, D]
    a_flat = field_a.reshape(-1, field_a.shape[-1])
    b_flat = field_b.reshape(-1, field_b.shape[-1])
    a_norm = a_flat / (np.linalg.norm(a_flat, axis=1, keepdims=True) + 1e-8)
    b_norm = b_flat / (np.linalg.norm(b_flat, axis=1, keepdims=True) + 1e-8)
    cos = (a_norm * b_norm).sum(axis=1)
    return float(cos.mean())


# ============== Core: capture with multi-iter backward ==============
def capture_freq_data(prompt, style_path, max_iters=8):
    """Capture SLT field + backward gradients for iters 1..max_iters."""
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

        # First eligible step — do all captures here
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

        n_layers = len(ref_k_list)
        n_ot = max(1, int(n_layers*0.5))

        # AdaIN
        content_anchor_raw = latents.detach().clone()
        with torch.no_grad():
            latents_adain = feature_adain(latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=0.5).squeeze(1)
        latents_adain = latents_adain.detach()
        content_anchor = latents_adain.clone()

        # === SLT alpha=0.1 ===
        latents_pre_slt = latents_adain.clone()
        with torch.no_grad():
            latents_slt = sinkhorn_latent_transport(latents_adain.clone(), style_latent_, style_latent,
                alpha=0.1, n_iters=10, tau=0.02, grag_delta=grag_deep)
        slt_field = (latents_slt - latents_pre_slt)[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        # === PATH A: Backward WITHOUT SLT (from adain only) ===
        latents_noslt = latents_adain.clone().detach()
        latents_noslt.requires_grad_(True)
        optimizer_noslt = torch.optim.Adam([latents_noslt], lr=lr)
        bw_noslt_fields = []
        latents_noslt_init = latents_noslt.detach().clone()

        for it in range(max_iters):
            optimizer_noslt.zero_grad()
            cache.clear(); controller.step()
            with torch.enable_grad():
                _ = dit(hidden_states=latents_noslt, timestep=ts.to("cuda", torch.bfloat16).detach()/1000,
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
            cl = loss_fn(latents_noslt, content_anchor)
            loss = style_w*sl + content_w*cl
            loss.backward()
            grad_field = latents_noslt.grad.detach().clone()[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)
            bw_noslt_fields.append(grad_field)
            optimizer_noslt.step()

        bw_noslt_cum = (latents_noslt.detach() - latents_noslt_init)[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        # === PATH B: Backward WITH SLT pre-alignment ===
        latents_withslt = latents_slt.detach().clone()
        latents_withslt.requires_grad_(True)
        optimizer_withslt = torch.optim.Adam([latents_withslt], lr=lr)
        bw_withslt_fields = []
        latents_withslt_init = latents_withslt.detach().clone()

        for it in range(max_iters):
            optimizer_withslt.zero_grad()
            cache.clear(); controller.step()
            with torch.enable_grad():
                _ = dit(hidden_states=latents_withslt, timestep=ts.to("cuda", torch.bfloat16).detach()/1000,
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
            cl = loss_fn(latents_withslt, content_anchor)
            loss = style_w*sl + content_w*cl
            loss.backward()
            grad_field = latents_withslt.grad.detach().clone()[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)
            bw_withslt_fields.append(grad_field)
            optimizer_withslt.step()

        bw_withslt_cum = (latents_withslt.detach() - latents_withslt_init)[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)

        return {
            "slt_field": slt_field,
            "bw_noslt_grads": bw_noslt_fields,
            "bw_noslt_cumulative": bw_noslt_cum,
            "bw_withslt_grads": bw_withslt_fields,
            "bw_withslt_cumulative": bw_withslt_cum,
        }

    return None


# ============== Main ==============
def main():
    print("="*70)
    print("FREQUENCY DECOMPOSITION ANALYSIS — v2")
    print("Compare: backward WITHOUT SLT vs backward WITH SLT")
    print("Key Q: Does SLT accelerate backward by doing color work upfront?")
    print("="*70)

    samples = [
        ("A panda eat bamboo", "27"),
        ("A castle on a cliff by the sea", "28"),
        ("A field of sunflowers at sunset", "33"),
        ("A cat sitting on a windowsill", "10"),
    ]

    freq_cutoff = 4

    # Storage
    all_slt_low = []
    all_slt_high = []
    # Path A: backward WITHOUT SLT
    all_noslt_low_per_iter = [[] for _ in range(8)]
    all_noslt_high_per_iter = [[] for _ in range(8)]
    # Path B: backward WITH SLT
    all_withslt_low_per_iter = [[] for _ in range(8)]
    all_withslt_high_per_iter = [[] for _ in range(8)]
    # Direction alignment
    all_cos_slt_vs_noslt_cum = []
    all_cos_slt_vs_noslt_per_iter = [[] for _ in range(8)]
    all_cos_slt_low_vs_noslt_low = []

    for prompt, sty_id in samples:
        sty_path = f"{CWD}/sty/{sty_id}.png"
        torch.cuda.empty_cache()
        print(f"\n  Processing: {prompt[:35]}... x sty{sty_id}")
        data = capture_freq_data(prompt, sty_path, max_iters=8)
        if data is None:
            print("    SKIPPED"); continue

        slt_field = data["slt_field"]
        bw_noslt_grads = data["bw_noslt_grads"]
        bw_noslt_cum = data["bw_noslt_cumulative"]
        bw_withslt_grads = data["bw_withslt_grads"]
        bw_withslt_cum = data["bw_withslt_cumulative"]

        # Frequency decomposition of SLT
        slt_low_r, slt_high_r, slt_low_f, slt_high_f = freq_energy_split(slt_field, cutoff=freq_cutoff)
        all_slt_low.append(slt_low_r)
        all_slt_high.append(slt_high_r)
        print(f"    SLT: low_freq={slt_low_r*100:.1f}%, high_freq={slt_high_r*100:.1f}%")

        # Path A: backward WITHOUT SLT per iteration (cumulative update)
        cum_noslt = np.zeros_like(slt_field)
        for it, grad in enumerate(bw_noslt_grads):
            cum_noslt = cum_noslt + grad
            bw_low_r, bw_high_r, _, _ = freq_energy_split(cum_noslt, cutoff=freq_cutoff)
            all_noslt_low_per_iter[it].append(bw_low_r)
            all_noslt_high_per_iter[it].append(bw_high_r)
            cos_iter = direction_cosine(slt_field, cum_noslt)
            all_cos_slt_vs_noslt_per_iter[it].append(cos_iter)
            if it in (0, 1, 3, 7):
                print(f"    Bkwd(no SLT) it{it+1}: low={bw_low_r*100:.1f}%, high={bw_high_r*100:.1f}%, cos(SLT)={cos_iter:.3f}")

        # Path B: backward WITH SLT per iteration
        cum_withslt = np.zeros_like(slt_field)
        for it, grad in enumerate(bw_withslt_grads):
            cum_withslt = cum_withslt + grad
            bw_low_r, bw_high_r, _, _ = freq_energy_split(cum_withslt, cutoff=freq_cutoff)
            all_withslt_low_per_iter[it].append(bw_low_r)
            all_withslt_high_per_iter[it].append(bw_high_r)
            if it in (0, 1):
                print(f"    Bkwd(+SLT) it{it+1}: low={bw_low_r*100:.1f}%, high={bw_high_r*100:.1f}%")

        # Direction: does backward-without-SLT converge to SLT direction?
        cos_cum = direction_cosine(slt_field, bw_noslt_cum)
        all_cos_slt_vs_noslt_cum.append(cos_cum)
        print(f"    cos(SLT, bkwd_noslt_8iter_update) = {cos_cum:.3f}")

        # Low-freq alignment
        _, _, noslt_cum_low, _ = freq_energy_split(bw_noslt_cum, cutoff=freq_cutoff)
        cos_low = direction_cosine(slt_low_f, noslt_cum_low)
        all_cos_slt_low_vs_noslt_low.append(cos_low)
        print(f"    cos(SLT_low, bkwd_noslt_low) = {cos_low:.3f}")

    # ============== Aggregate results ==============
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    results = {
        "freq_cutoff": freq_cutoff,
        "n_samples": len(all_slt_low),
        "slt": {
            "low_freq_ratio_mean": float(np.mean(all_slt_low)),
            "low_freq_ratio_std": float(np.std(all_slt_low)),
            "high_freq_ratio_mean": float(np.mean(all_slt_high)),
            "high_freq_ratio_std": float(np.std(all_slt_high)),
        },
        "backward_no_slt_per_iter": {
            "low_freq_ratio": [float(np.mean(v)) if v else 0 for v in all_noslt_low_per_iter],
            "high_freq_ratio": [float(np.mean(v)) if v else 0 for v in all_noslt_high_per_iter],
        },
        "backward_with_slt_per_iter": {
            "low_freq_ratio": [float(np.mean(v)) if v else 0 for v in all_withslt_low_per_iter],
            "high_freq_ratio": [float(np.mean(v)) if v else 0 for v in all_withslt_high_per_iter],
        },
        "direction_alignment": {
            "cos_slt_vs_noslt_cumulative": float(np.mean(all_cos_slt_vs_noslt_cum)),
            "cos_slt_low_vs_noslt_low": float(np.mean(all_cos_slt_low_vs_noslt_low)),
            "cos_slt_vs_noslt_per_iter": [float(np.mean(v)) if v else 0 for v in all_cos_slt_vs_noslt_per_iter],
        },
    }

    print(f"\nSLT energy: {results['slt']['low_freq_ratio_mean']*100:.1f}% low-freq, "
          f"{results['slt']['high_freq_ratio_mean']*100:.1f}% high-freq")

    print(f"\nBackward WITHOUT SLT — cumulative energy per iteration:")
    for it in range(8):
        low = results['backward_no_slt_per_iter']['low_freq_ratio'][it]
        high = results['backward_no_slt_per_iter']['high_freq_ratio'][it]
        cos = results['direction_alignment']['cos_slt_vs_noslt_per_iter'][it]
        print(f"  iter {it+1}: low={low*100:.1f}%, high={high*100:.1f}%, cos(SLT,cum)={cos:.3f}")

    print(f"\nBackward WITH SLT — cumulative energy per iteration:")
    for it in range(8):
        low = results['backward_with_slt_per_iter']['low_freq_ratio'][it]
        high = results['backward_with_slt_per_iter']['high_freq_ratio'][it]
        print(f"  iter {it+1}: low={low*100:.1f}%, high={high*100:.1f}%")

    print(f"\nDirection alignment (SLT vs backward-without-SLT):")
    print(f"  cos(SLT, bkwd_noslt_8iter) = {results['direction_alignment']['cos_slt_vs_noslt_cumulative']:.3f}")
    print(f"  cos(SLT_low, bkwd_noslt_low) = {results['direction_alignment']['cos_slt_low_vs_noslt_low']:.3f}")

    # Save
    out_path = RESULTS_DIR / "freq_decompose.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
