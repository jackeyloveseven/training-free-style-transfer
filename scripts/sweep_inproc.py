"""
Single-process sweep: load model once, sweep params, report real inference time.
Run with sensenova python (has DiffSynth).
Metric computation at the end using the generated images.
"""
import os, sys, time, glob, types
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

DIFFSYNTH_ROOT = "/mnt/local/yzh/prog/DiffSynth-Studio"
KLEIN9B_ROOT = "/mnt/local/klein9b"
if DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)

# ModelScope stub
if "modelscope" not in sys.modules:
    module = types.ModuleType("modelscope")
    module.snapshot_download = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no modelscope"))
    sys.modules["modelscope"] = module

from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.diffusion.flow_match import FlowMatchScheduler

def _sorted_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files: raise FileNotFoundError(f"No files: {pattern}")
    return files

# Load model once
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

# === Import same utilities as the script ===
cwd = '/home/ubuntu/chengyanli/items/save/attention_20260426_090948'
sys.path.insert(0, cwd)

# We'll inline the key parts from custom_main_klein9b_t2i.py

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
    def reset_mode(self):
        self.mode = "cache"
        self.inject_ref_k = []
        self.inject_ref_v = []
        self.inject_beta = 0.0
        self.inject_idx = 0

def sinkhorn_attn_freq(Q, K_style, V_style, n_iters=6, tau=0.1, hi_boost=2.0):
    B, H, Lc, D = Q.shape
    Q_lo = Q.mean(dim=2, keepdim=True).expand_as(Q)
    K_lo = K_style.mean(dim=2, keepdim=True).expand_as(K_style)
    Q_hi = Q - Q_lo
    K_hi = K_style - K_lo
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

# Register hooks
def register_hooks(dit, controller, cache):
    from diffsynth.models.flux2_dit import _get_qkv_projections, apply_rotary_emb

    def make_double_hook(original_block_attn):
        def hooked(attn, hidden_states, encoder_hidden_states=None,
                   attention_mask=None, image_rotary_emb=None):
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states)
            query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
            key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
            value = value.unflatten(-1, (attn.heads, -1))
            txt_len = 0
            if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
                encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
                encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
                encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
                txt_len = encoder_hidden_states.shape[1]
                controller.txt_len = txt_len
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            query, key, value = query.to(hidden_states.dtype), key.to(hidden_states.dtype), value.to(hidden_states.dtype)
            q_t, k_t, v_t = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)
            if controller.cur_self_layer in controller.self_layers:
                if controller.mode == "cache":
                    cache.add(q_t, k_t, v_t, attn_out_t)
            controller.cur_self_layer += 1
            attn_output = attn_out_t.transpose(1,2).flatten(2,3).to(query.dtype)
            if encoder_hidden_states is not None:
                encoder_out, img_out = attn_output.split([txt_len, attn_output.shape[1]-txt_len], dim=1)
                return attn.to_out[0](attn.to_out[1](img_out) if False else img_out), attn.to_add_out(encoder_out)
            else:
                return attn.to_out[0](attn.to_out[1](attn_output) if False else attn_output)
        return hooked

    def make_single_hook(original_attn):
        def hooked(attn, hidden_states, attention_mask=None, image_rotary_emb=None):
            from diffsynth.models.flux2_dit import apply_rotary_emb as are
            proj_out = attn.to_qkv_mlp_proj(hidden_states)
            qkv, mlp_hidden_states = torch.split(proj_out, [3*attn.inner_dim, attn.mlp_hidden_dim*attn.mlp_mult_factor], dim=-1)
            query, key, value = qkv.chunk(3, dim=-1)
            query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
            key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
            value = value.unflatten(-1, (attn.heads, -1))
            if image_rotary_emb is not None:
                query = are(query, image_rotary_emb, sequence_dim=1)
                key = are(key, image_rotary_emb, sequence_dim=1)
            query, key, value = query.to(hidden_states.dtype), key.to(hidden_states.dtype), value.to(hidden_states.dtype)
            q_t, k_t, v_t = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)
            if controller.cur_self_layer in controller.self_layers:
                if controller.mode == "cache":
                    cache.add(q_t, k_t, v_t, attn_out_t)
            controller.cur_self_layer += 1
            attn_output = attn_out_t.transpose(1,2).flatten(2,3).to(query.dtype)
            mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
            return attn.to_out(hidden_states)
        return hooked

    for block in dit.transformer_blocks:
        block.attn.processor = make_double_hook(block.attn)
    for block in dit.single_transformer_blocks:
        block.attn.processor = make_single_hook(block.attn)

cache = Klein9BCache()
start_layer, end_layer = 12, 28
controller = Klein9BController(self_layers=(start_layer, end_layer))
register_hooks(dit, controller, cache)

# Encode style
style_image_path = os.path.join(cwd, 'sty/27.png')
pipe.load_models_to_device(["vae"])
with torch.no_grad():
    sty_pix = pipe.preprocess_image(Image.open(style_image_path).convert("RGB").resize((512,512), Image.BICUBIC))
    style_latent_4d = vae.encode(sty_pix)
    style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

# Sizes
width, height = 512, 512
lat_h, lat_w = height // 16, width // 16
image_ids = torch.cartesian_prod(
    torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)
).unsqueeze(0).to("cuda")
sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
style_image_ids = torch.cartesian_prod(
    torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)
).unsqueeze(0).to("cuda")

# Encode prompt
prompt = 'A field of sunflowers at sunset'
from diffsynth.pipelines.flux2_image import Flux2Unit_Qwen3PromptEmbedder, Flux2Unit_PromptEmbedder

if pipe.text_encoder_qwen3 is not None:
    pipe.load_models_to_device(["text_encoder_qwen3"])
    embedder = Flux2Unit_Qwen3PromptEmbedder()
    prompt_embeds, text_ids = embedder.encode_prompt(pipe.text_encoder_qwen3, pipe.tokenizer, prompt, dtype=pipe.torch_dtype, device="cuda")
    null_embeds, null_text_ids = embedder.encode_prompt(pipe.text_encoder_qwen3, pipe.tokenizer, "", dtype=pipe.torch_dtype, device="cuda")
else:
    pipe.load_models_to_device(["text_encoder"])
    embedder = Flux2Unit_PromptEmbedder()
    prompt_embeds, text_ids = embedder.encode_prompt(pipe.text_encoder, pipe.tokenizer, prompt, dtype=pipe.torch_dtype, device="cuda")
    null_embeds, null_text_ids = embedder.encode_prompt(pipe.text_encoder, pipe.tokenizer, "", dtype=pipe.torch_dtype, device="cuda")

pipe.load_models_to_device(["dit"])
dit.to("cuda"); dit.requires_grad_(False)

loss_fn = torch.nn.L1Loss()

def run_once(num_steps, iters, start_time, style_w, adain_alpha, hi_boost, content_w, dist_w=0.15):
    """Run one generation, return (time_seconds, output_path)."""
    scheduler = FlowMatchScheduler("FLUX.2")
    torch.manual_seed(43)
    noise = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    noise = rearrange(noise, "B C H W -> B (H W) C")
    latents = noise.clone()

    scheduler.set_timesteps(num_steps, dynamic_shift_len=lat_h * lat_w)
    timesteps = scheduler.timesteps

    torch.cuda.synchronize()
    t0 = time.time()

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()

        with torch.no_grad():
            noise_pred = dit(hidden_states=latents, timestep=timestep/1000, guidance=None,
                           encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=image_ids)
        latents = scheduler.step(noise_pred, t_val, latents)

        if t_val >= start_time:
            continue

        # Style forward
        if i < num_steps - 1:
            ts_s = timesteps[i+1:i+2]
            sty_noise = torch.randn_like(style_latent)
            style_latent_ = scheduler.add_noise(style_latent, sty_noise, ts_s)
        else:
            ts_s = torch.tensor([0.0])
            style_latent_ = style_latent

        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(hidden_states=style_latent_.to(device="cuda", dtype=torch.bfloat16),
                   timestep=ts_s.to(device="cuda", dtype=torch.bfloat16)/1000, guidance=None,
                   encoder_hidden_states=null_embeds, txt_ids=null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, _ = cache.get()

        # GRAG = off (delta=1.0)
        _txt_len = controller.txt_len

        # AdaIN
        content_anchor = latents.detach().clone()
        with torch.no_grad():
            latents = feature_adain(latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=adain_alpha).squeeze(1).detach()

        # Adam
        _progress = 1.0 - (float(t_val.item()) / max(start_time, 1.0))
        _dynamic_iters = max(1, int(iters * (1.0 - 0.6 * _progress)))

        latents.requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=0.01)
        _ts_cuda = ts_s.to(device="cuda", dtype=torch.bfloat16) / 1000
        _tau_max, _tau_min = 0.15, 0.07
        _tau = _tau_max - (_tau_max - _tau_min) * _progress
        _ot_layer_frac = 0.5

        for it in range(_dynamic_iters):
            cache.clear(); controller.step()
            optimizer.zero_grad()
            with torch.enable_grad():
                _dit_out = dit(hidden_states=latents, timestep=_ts_cuda.detach(), guidance=None,
                             encoder_hidden_states=null_embeds.detach(), txt_ids=null_text_ids.detach(),
                             img_ids=image_ids.detach())
            del _dit_out
            q_list, _, _, self_out_list = cache.get()

            n_layers = len(q_list)
            n_ot_layers = max(1, int(n_layers * _ot_layer_frac))
            ref_self_out_list = []
            for idx, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                q_det = q.detach()
                if idx >= n_layers - n_ot_layers:
                    ref_self_out_list.append(sinkhorn_attn_freq(q_det, ref_k, ref_v, n_iters=6, tau=_tau, hi_boost=hi_boost))
                else:
                    ref_self_out_list.append(F.scaled_dot_product_attention(q_det, ref_k, ref_v))

            style_loss = sum(loss_fn(so, rso.detach()) for so, rso in zip(self_out_list, ref_self_out_list))
            lat4d = latents.unsqueeze(1)
            sty4d = style_latent_.unsqueeze(1).detach()
            dist_loss = loss_fn(lat4d.mean(dim=2,keepdim=True), sty4d.mean(dim=2,keepdim=True)) + \
                       loss_fn(lat4d.std(dim=2,keepdim=True), sty4d.std(dim=2,keepdim=True))
            decay = 1.0 - (it / max(iters, 1)) * 0.5
            content_loss = loss_fn(latents, content_anchor) * decay
            loss = style_w * style_loss + dist_w * dist_loss + content_w * content_loss
            loss.backward()
            optimizer.step()

        latents = latents.detach()

    # Final adain + decode
    latents = adain_1d(latents, style_latent)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    latents_4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
    with torch.no_grad():
        image = vae.decode(latents_4d)
    image = pipe.vae_output_to_image(image)
    out_path = os.path.join(cwd, 'output_sweep_tmp.png')
    image.save(out_path)
    return elapsed, out_path


# === SWEEP ===
# (num_steps, iters, start_time, style_w, adain_alpha, hi_boost, content_w)
configs = [
    # Target: ≤7s inference. iters is the main knob.
    # 25 steps, iters=1 (only adain+1 adam step per styled step)
    (25, 1, 850, 2.0, 0.5, 2.0, 0.2),
    (25, 1, 850, 4.0, 0.8, 2.0, 0.1),
    (25, 1, 700, 4.0, 0.8, 2.0, 0.1),
    (25, 1, 700, 6.0, 1.0, 2.0, 0.0),
    # 25 steps, iters=2
    (25, 2, 850, 4.0, 0.8, 2.0, 0.1),
    (25, 2, 700, 4.0, 0.8, 2.0, 0.1),
    (25, 2, 700, 6.0, 1.0, 2.0, 0.0),
    (25, 2, 850, 6.0, 1.0, 2.0, 0.0),
    # 25 steps, iters=3
    (25, 3, 850, 4.0, 0.8, 2.0, 0.1),
    (25, 3, 700, 6.0, 1.0, 2.0, 0.0),
    # 15 steps (fewer total steps = less overhead)
    (15, 3, 700, 6.0, 1.0, 2.0, 0.0),
    (15, 5, 700, 6.0, 1.0, 2.0, 0.0),
    # High start_time = fewer styled steps
    (25, 3, 950, 6.0, 1.0, 2.0, 0.0),
    (25, 5, 950, 6.0, 1.0, 2.0, 0.0),
    (25, 5, 970, 6.0, 1.0, 2.0, 0.0),
]

print(f"{'NS':>3} {'IT':>3} {'ST':>4} {'SW':>4} {'AA':>4} {'HB':>4} {'CW':>4} | {'Time':>5}")
print("-" * 50)

results = []
for cfg in configs:
    ns, it, st, sw, aa, hb, cw = cfg
    elapsed, out_path = run_once(ns, it, st, sw, aa, hb, cw)
    flag = " ***" if elapsed <= 7.0 else (" <" if elapsed <= 10.0 else "")
    print(f"{ns:>3} {it:>3} {st:>4} {sw:>4.1f} {aa:>4.1f} {hb:>4.1f} {cw:>4.1f} | {elapsed:>5.1f}{flag}")
    results.append((cfg, elapsed, out_path))
    sys.stdout.flush()

print("\n=== Configs under 7s ===")
for cfg, elapsed, _ in results:
    if elapsed <= 7.0:
        print(f"  {cfg} -> {elapsed:.1f}s")
