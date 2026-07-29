"""
Gradient Source Analysis: WHY does backward improve style transfer?

Key question: Where does the gradient's spatial coherence come from?
- Hypothesis A: From the style reference's spatial structure (loss landscape)
- Hypothesis B: From the network's internal spatial priors (residual/FFN)

Conditions:
1. Normal: full gradient (control, same as jacobian_ablation)
2. Shuffle style K/V: randomize spatial order of style tokens in ref_k/ref_v
   → If cos drops: spatial structure flows from style image through loss
3. Content-only loss: only L1(latent, content_anchor), no style loss
   → If cos drops to ~0: spatial coherence comes from style loss target
   → If cos stays high: network structure alone creates spatial gradients
4. Single-token loss: style loss on ONE token only (center)
   → Tests if spatial coherence comes from per-token independent targets
   → Or from cross-token coupling in the backward pass

Output: results/gradient_source.json
"""
import os, sys, types, glob, json, gc
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
_all_prompts = ["A panda eat bamboo", "A castle on a cliff by the sea",
                "A field of sunflowers at sunset", "A cat sitting on a windowsill"]
for _p in _all_prompts:
    encode_prompt(_p)
if pipe.text_encoder_qwen3 is not None:
    pipe.text_encoder_qwen3.to("cpu")
elif hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
    pipe.text_encoder.to("cpu")
torch.cuda.empty_cache()
print(f"Text encoder offloaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")


def compute_neighbor_cosine(grad_tensor, lat_h=32, lat_w=32):
    g = grad_tensor.detach().float().squeeze(0)
    g_2d = g.view(lat_h, lat_w, -1)
    g_norm = F.normalize(g_2d.reshape(lat_h*lat_w, -1), dim=-1).view(lat_h, lat_w, -1)
    cos_h = (g_norm[:-1,:,:] * g_norm[1:,:,:]).sum(-1).mean().item()
    cos_w = (g_norm[:,:-1,:] * g_norm[:,1:,:]).sum(-1).mean().item()
    return (cos_h + cos_w) / 2


# ============== Core experiment ==============
def run_gradient_source(prompt, style_path, n_iters=4):
    lat_h, lat_w = 32, 32
    start_time_val = 850
    lr = 0.03; content_w = 1.5; style_w = 2.0
    grag_deep = 0.5; grag_shallow = 0.7; seed = 43

    style_pil = Image.open(style_path).convert("RGB").resize((512,512), Image.BICUBIC)

    vae.to("cuda")
    with torch.no_grad():
        sty_pix = pipe.preprocess_image(style_pil)
        style_latent_4d = vae.encode(sty_pix)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")
    vae.to("cpu"); torch.cuda.empty_cache()

    image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)).unsqueeze(0).to("cuda")
    sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
    style_image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)).unsqueeze(0).to("cuda")
    prompt_embeds, text_ids = encode_prompt(prompt)
    dit.to("cuda")

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

        # AdaIN + SLT
        with torch.no_grad():
            latents_adain = feature_adain(latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=0.5).squeeze(1)
            latents_slt = sinkhorn_latent_transport(latents_adain, style_latent_, style_latent,
                alpha=0.1, n_iters=10, tau=0.02, grag_delta=grag_deep)
        content_anchor = latents_adain.clone().detach()

        results = {}

        # Helper: run one condition
        def run_condition(name, ref_k, ref_v, use_style_loss=True, use_content_loss=True, mask_tokens=None):
            lat = latents_slt.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([lat], lr=lr)
            cos_list, loss_list = [], []

            for it in range(n_iters):
                opt.zero_grad()
                cache.clear(); controller.step()
                with torch.enable_grad():
                    _ = dit(hidden_states=lat, timestep=ts.to("cuda", torch.bfloat16).detach()/1000,
                           guidance=None, encoder_hidden_states=null_embeds.detach(),
                           txt_ids=null_text_ids.detach(), img_ids=image_ids.detach())
                del _
                q_list, _, _, self_out_list = cache.get()

                loss = torch.tensor(0.0, device="cuda", requires_grad=True)

                if use_style_loss:
                    ref_outs = []
                    for idx2, (q, rk, rv) in enumerate(zip(q_list, ref_k, ref_v)):
                        if idx2 >= n_layers - n_ot:
                            ref_outs.append(sinkhorn_attn_freq(q.detach(), rk, rv))
                        else:
                            ref_outs.append(F.scaled_dot_product_attention(q.detach(), rk, rv))

                    if mask_tokens is not None:
                        # Only compute loss on specific tokens
                        sl = sum(loss_fn(so[:,:,mask_tokens,:], ro[:,:,mask_tokens,:].detach())
                                 for so, ro in zip(self_out_list, ref_outs))
                    else:
                        sl = sum(loss_fn(so, ro.detach()) for so, ro in zip(self_out_list, ref_outs))
                    loss = loss + style_w * sl

                if use_content_loss:
                    cl = loss_fn(lat, content_anchor)
                    loss = loss + content_w * cl

                loss_list.append(loss.item())
                loss.backward()
                grad = lat.grad.clone()
                cos_list.append(compute_neighbor_cosine(grad))
                opt.step()
                print(f"    {name} iter {it+1}: loss={loss_list[-1]:.4f}, cos@d1={cos_list[-1]:.4f}")
                torch.cuda.empty_cache()

            del lat, opt, grad
            gc.collect(); torch.cuda.empty_cache()
            return {"loss": loss_list, "neighbor_cos": cos_list}

        # === CONDITION 1: Normal (control) ===
        results["normal"] = run_condition("Normal", ref_k_list, ref_v_list)

        # === CONDITION 2: Shuffled style K/V (destroy spatial order) ===
        shuffle_perm = torch.randperm(ref_k_list[0].shape[2] - _txt_len)
        ref_k_shuf = []
        ref_v_shuf = []
        for idx in range(n_layers):
            k, v = ref_k_list[idx], ref_v_list[idx]
            kt, ki = k[:,:,:_txt_len,:], k[:,:,_txt_len:,:]
            vt, vi = v[:,:,:_txt_len,:], v[:,:,_txt_len:,:]
            ref_k_shuf.append(torch.cat([kt, ki[:,:,shuffle_perm,:]], dim=2))
            ref_v_shuf.append(torch.cat([vt, vi[:,:,shuffle_perm,:]], dim=2))
        results["shuffle_style_kv"] = run_condition("Shuffle style K/V", ref_k_shuf, ref_v_shuf)
        del ref_k_shuf, ref_v_shuf
        gc.collect(); torch.cuda.empty_cache()

        # === CONDITION 3: Content-only loss (no style loss) ===
        results["content_only"] = run_condition("Content-only", ref_k_list, ref_v_list,
                                                use_style_loss=False, use_content_loss=True)

        # === CONDITION 4: Style-only loss, single center patch (16 tokens around center) ===
        center_h, center_w = lat_h // 2, lat_w // 2
        patch_indices = []
        for dh in range(-2, 2):
            for dw in range(-2, 2):
                patch_indices.append((center_h + dh) * lat_w + (center_w + dw))
        # Need to add txt_len offset since self_out includes text tokens
        patch_mask = torch.tensor(patch_indices, device="cuda")
        results["single_patch"] = run_condition("Single-patch style", ref_k_list, ref_v_list,
                                                use_style_loss=True, use_content_loss=False,
                                                mask_tokens=patch_mask)

        del ref_k_list, ref_v_list, content_anchor, latents_slt, latents_adain
        del style_latent_, style_latent, style_latent_4d
        gc.collect(); torch.cuda.empty_cache()

        return results

    return None


# ============== Main ==============
def main():
    print("="*70)
    print("GRADIENT SOURCE: Why does backward improve style transfer?")
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
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n{'='*50}")
        print(f"  {prompt} × sty/{sty_id}")
        print(f"{'='*50}")

        data = run_gradient_source(prompt, sty_path, n_iters=4)
        if data is None:
            print("  SKIPPED"); continue
        all_results.append({"prompt": prompt, "style": sty_id, **data})

    # Aggregate
    print("\n" + "="*70)
    print("AGGREGATE SUMMARY")
    print("="*70)
    n = len(all_results)
    for cond in ["normal", "shuffle_style_kv", "content_only", "single_patch"]:
        losses = [r[cond]["loss"][-1] for r in all_results if cond in r]
        coss = [r[cond]["neighbor_cos"][0] for r in all_results if cond in r]
        if coss:
            print(f"\n  {cond}:")
            print(f"    Final loss (iter4): {np.mean(losses):.4f} ± {np.std(losses):.4f}")
            print(f"    Neighbor cos (iter1): {np.mean(coss):.4f} ± {np.std(coss):.4f}")

    out_path = RESULTS_DIR / "gradient_source.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
