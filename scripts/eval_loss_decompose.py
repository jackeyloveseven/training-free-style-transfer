"""
Loss Decomposition: Why Hybrid works (the REAL mechanism).

Disproved: frequency complementarity (SLT ≠ low-freq, cos(SLT, backward) ≈ 0).
Proved here: SLT reduces the "distribution alignment" part of style loss upfront,
so backward starts from a lower loss and needs fewer iters for texture.

Measures at the first eligible denoising step:
1. style_loss (attention matching): L1(self_attn_out, sinkhorn_target)
2. dist_loss (distribution): |mean(latent) - mean(style)| + |std(latent) - std(style)|
3. Measure these BEFORE any style op, AFTER SLT, AFTER each backward iter (with/without SLT)

Output: results/loss_decompose.json

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/eval_loss_decompose.py
"""
import argparse, csv, glob, os, sys, time, types, json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

DIFFSYNTH_ROOT = "third_party/DiffSynth-Studio"
KLEIN9B_ROOT = "models/klein9b"
CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
RESULTS_DIR = Path(CWD) / "results"

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


# ============== Loss measurement ==============
def measure_losses(latents_state, style_latent_, style_latent, ref_k_list, ref_v_list,
                   image_ids, ts, n_layers, n_ot, _txt_len):
    """Measure style_loss and dist_loss for a given latent state (no grad)."""
    cache.clear(); controller.step()
    with torch.no_grad():
        _ = dit(hidden_states=latents_state.detach(), timestep=ts.to("cuda", torch.bfloat16)/1000,
               guidance=None, encoder_hidden_states=null_embeds, txt_ids=null_text_ids, img_ids=image_ids)
    q_list, _, _, self_out_list = cache.get()

    # Style loss: L1(self_attn_out, sinkhorn_target)
    loss_fn = torch.nn.L1Loss()
    style_loss = 0.0
    for idx2, (q, rk, rv, so) in enumerate(zip(q_list, ref_k_list, ref_v_list, self_out_list)):
        if idx2 >= n_layers - n_ot:
            target = sinkhorn_attn_freq(q, rk, rv, n_iters=6, tau=0.1, hi_boost=2.0)
        else:
            target = F.scaled_dot_product_attention(q, rk, rv)
        style_loss += loss_fn(so, target).item()
    style_loss /= n_layers

    # Distribution loss: mean/std alignment in latent space
    lat = latents_state.detach().float()
    sty = style_latent.float()
    dist_loss = (torch.abs(lat.mean(1) - sty.mean(1)).mean() +
                 torch.abs(lat.std(1) - sty.std(1)).mean()).item()

    return style_loss, dist_loss


# ============== Core experiment ==============
def run_loss_trace(prompt, style_path, max_iters=8):
    """Track style_loss and dist_loss through: baseline → SLT → backward (with/without SLT)."""
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

        # Get ref K/V
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

        # === Checkpoint 0: raw latent (after denoising, before any style op) ===
        sl_raw, dl_raw = measure_losses(latents, style_latent_, style_latent,
                                         ref_k_list, ref_v_list, image_ids, ts, n_layers, n_ot, _txt_len)

        # === AdaIN ===
        with torch.no_grad():
            latents_adain = feature_adain(latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=0.5).squeeze(1)
        latents_adain = latents_adain.detach()
        content_anchor = latents_adain.clone()

        sl_adain, dl_adain = measure_losses(latents_adain, style_latent_, style_latent,
                                             ref_k_list, ref_v_list, image_ids, ts, n_layers, n_ot, _txt_len)

        # === SLT alpha=0.1 ===
        with torch.no_grad():
            latents_slt = sinkhorn_latent_transport(latents_adain.clone(), style_latent_, style_latent,
                alpha=0.1, n_iters=10, tau=0.02, grag_delta=grag_deep)
        latents_slt = latents_slt.detach()

        sl_slt, dl_slt = measure_losses(latents_slt, style_latent_, style_latent,
                                         ref_k_list, ref_v_list, image_ids, ts, n_layers, n_ot, _txt_len)

        # === PATH A: Backward WITHOUT SLT (from adain) ===
        latents_noslt = latents_adain.clone().detach()
        latents_noslt.requires_grad_(True)
        optimizer_noslt = torch.optim.Adam([latents_noslt], lr=lr)
        noslt_trace = []

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
            optimizer_noslt.step()
            # Measure after this iter
            sl_it, dl_it = measure_losses(latents_noslt, style_latent_, style_latent,
                                           ref_k_list, ref_v_list, image_ids, ts, n_layers, n_ot, _txt_len)
            noslt_trace.append((sl_it, dl_it))

        # === PATH B: Backward WITH SLT (from slt output) ===
        latents_withslt = latents_slt.clone().detach()
        latents_withslt.requires_grad_(True)
        optimizer_withslt = torch.optim.Adam([latents_withslt], lr=lr)
        withslt_trace = []

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
            optimizer_withslt.step()
            sl_it, dl_it = measure_losses(latents_withslt, style_latent_, style_latent,
                                           ref_k_list, ref_v_list, image_ids, ts, n_layers, n_ot, _txt_len)
            withslt_trace.append((sl_it, dl_it))

        return {
            "checkpoints": {
                "raw": {"style_loss": sl_raw, "dist_loss": dl_raw},
                "adain": {"style_loss": sl_adain, "dist_loss": dl_adain},
                "slt": {"style_loss": sl_slt, "dist_loss": dl_slt},
            },
            "backward_noslt": [{"style_loss": s, "dist_loss": d} for s, d in noslt_trace],
            "backward_withslt": [{"style_loss": s, "dist_loss": d} for s, d in withslt_trace],
        }

    return None


# ============== Main ==============
def main():
    print("="*70)
    print("LOSS DECOMPOSITION: Why Hybrid Works")
    print("="*70)

    samples = [
        ("A panda eat bamboo", "27"),
        ("A castle on a cliff by the sea", "28"),
        ("A field of sunflowers at sunset", "33"),
        ("A cat sitting on a windowsill", "10"),
    ]

    all_results = []

    for prompt, sty_id in samples:
        sty_path = f"{CWD}/sty/{sty_id}.png"
        torch.cuda.empty_cache()
        print(f"\n{'='*50}")
        print(f"  {prompt} × sty/{sty_id}")
        print(f"{'='*50}")

        data = run_loss_trace(prompt, sty_path, max_iters=8)
        if data is None:
            print("  SKIPPED"); continue

        cp = data["checkpoints"]
        print(f"\n  Checkpoints:")
        print(f"    Raw:   style_loss={cp['raw']['style_loss']:.4f}  dist_loss={cp['raw']['dist_loss']:.4f}")
        print(f"    AdaIN: style_loss={cp['adain']['style_loss']:.4f}  dist_loss={cp['adain']['dist_loss']:.4f}")
        print(f"    SLT:   style_loss={cp['slt']['style_loss']:.4f}  dist_loss={cp['slt']['dist_loss']:.4f}")
        print(f"    SLT reduces style_loss by {(1-cp['slt']['style_loss']/cp['adain']['style_loss'])*100:.1f}%")
        print(f"    SLT reduces dist_loss by {(1-cp['slt']['dist_loss']/cp['adain']['dist_loss'])*100:.1f}%")

        print(f"\n  Backward WITHOUT SLT (style_loss / dist_loss):")
        for it, d in enumerate(data["backward_noslt"]):
            if it < 4 or it == 7:
                print(f"    iter {it+1}: {d['style_loss']:.4f} / {d['dist_loss']:.4f}")

        print(f"\n  Backward WITH SLT (style_loss / dist_loss):")
        for it, d in enumerate(data["backward_withslt"]):
            if it < 4 or it == 7:
                print(f"    iter {it+1}: {d['style_loss']:.4f} / {d['dist_loss']:.4f}")

        # Key comparison: withSLT iter2 vs noslt iter?
        ws2_sl = data["backward_withslt"][1]["style_loss"]
        for it, d in enumerate(data["backward_noslt"]):
            if d["style_loss"] <= ws2_sl:
                print(f"\n  → Backward+SLT@iter2 = noslt@iter{it+1} (style_loss ≈ {ws2_sl:.4f})")
                break
        else:
            print(f"\n  → Backward+SLT@iter2 ({ws2_sl:.4f}) < noslt@iter8 ({data['backward_noslt'][7]['style_loss']:.4f})")

        all_results.append({"prompt": prompt, "style": sty_id, **data})

    # Aggregate
    print("\n" + "="*70)
    print("AGGREGATE SUMMARY")
    print("="*70)

    n = len(all_results)
    avg_slt_style_reduction = np.mean([
        1 - r["checkpoints"]["slt"]["style_loss"] / r["checkpoints"]["adain"]["style_loss"]
        for r in all_results
    ])
    avg_slt_dist_reduction = np.mean([
        1 - r["checkpoints"]["slt"]["dist_loss"] / r["checkpoints"]["adain"]["dist_loss"]
        for r in all_results
    ])
    print(f"\n  SLT reduces style_loss by {avg_slt_style_reduction*100:.1f}% (avg)")
    print(f"  SLT reduces dist_loss by {avg_slt_dist_reduction*100:.1f}% (avg)")

    # Compare convergence: at how many iters does noslt match withslt@2?
    for r in all_results:
        ws2 = r["backward_withslt"][1]["style_loss"]
        matched_iter = 9
        for it, d in enumerate(r["backward_noslt"]):
            if d["style_loss"] <= ws2:
                matched_iter = it + 1; break
        print(f"  {r['prompt'][:25]}... : withSLT@2 ≈ noslt@{matched_iter}")

    # Save full traces
    out_path = RESULTS_DIR / "loss_decompose.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
