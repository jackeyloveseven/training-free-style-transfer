"""
Klein 9B (FLUX.2) OT-Attention Style Transfer

Uses the same OT v3 algorithm as custom_main_img.py but adapted for
DiffSynth-Studio's Flux2ImagePipeline / Flux2DiT model.

Architecture mapping:
- Klein 9B: 8 double-stream + 48 single-stream = 56 total layers
- Attention tensor layout: [B, S, H, D] (DiffSynth convention)
- Edit image conditioning: same as Kontext (concat along sequence dim)
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

# --- Parameters (matching our standard config) ---
seed = int(os.environ.get("SEED", "43"))
width = int(os.environ.get("WIDTH", "512"))
height = int(os.environ.get("HEIGHT", "512"))
num_inference_steps = int(os.environ.get("NUM_STEPS", "25"))
start_time = int(os.environ.get("OVERRIDE_START_TIME", "870"))
os.environ["OVERRIDE_START_TIME"] = str(start_time)
embedded_guidance = float(os.environ.get("EMBEDDED_GUIDANCE", "0"))
lr = float(os.environ.get("LR", "0.01"))
iters = int(os.environ.get("ITERS", "3"))
# Klein 9B: 8 double + 24 single = 32 total layers
start_layer, end_layer = 12, 28
style_beta = float(os.environ.get("STYLE_BETA", "0.7"))
prompt = os.environ.get("PROMPT", "masterpiece")

content_image_path = os.environ.get("CNT_IMAGE",
    "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/cnt/00.png")
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


# --- Attention hooking for DiffSynth's Flux2DiT ---
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

    def step(self):
        self.cur_self_layer = 0


def register_klein9b_attn_hooks(dit, controller, cache):
    """
    Monkey-patch Flux2DiT attention layers to cache Q/K/V/output
    for layers in controller.self_layers.

    DiffSynth attention uses [B, S, H, D] layout.
    We transpose to [B, H, S, D] for compatibility with our sinkhorn_attn.
    """
    from diffsynth.core.attention.attention import torch_sdpa

    def make_double_hook(original_block_attn):
        attn_module = original_block_attn

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

            # [B, S, H, D] -> sdpa expects [B, H, S, D]
            q_t = query.transpose(1, 2)
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)

            if controller.cur_self_layer in controller.self_layers:
                cache.add(q_t, k_t, v_t, attn_out_t)

            controller.cur_self_layer += 1

            # Back to [B, S, H, D]
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
        attn_module = original_attn

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

            # [B, S, H, D] -> [B, H, S, D]
            q_t = query.transpose(1, 2)
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)

            if controller.cur_self_layer in controller.self_layers:
                cache.add(q_t, k_t, v_t, attn_out_t)

            controller.cur_self_layer += 1

            # Back to [B, S, H*D]
            attn_output = attn_out_t.transpose(1, 2).flatten(2, 3).to(query.dtype)

            mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
            hidden_states = attn.to_out(hidden_states)
            return hidden_states

        return hooked_processor

    # Hook double-stream blocks: replace the processor callable
    for block in dit.transformer_blocks:
        block.attn.processor = make_double_hook(block.attn)

    # Hook single-stream blocks: replace the processor callable
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
# Disable VRAM management to keep all models on GPU
pipe.vram_management_enabled = False

# Setup hooks
cache = Klein9BCache()
controller = Klein9BController(self_layers=(start_layer, end_layer))
register_klein9b_attn_hooks(dit, controller, cache)

# --- Encode images ---
print(f"Content: {content_image_path}")
print(f"Style: {style_image_path}")

content_img = load_image_tensor(content_image_path, size=(width, height))
style_img = load_image_tensor(style_image_path, size=(512, 512))

# VAE encode
pipe.load_models_to_device(["vae"])
with torch.no_grad():
    cnt_pix = pipe.preprocess_image(Image.open(content_image_path).convert("RGB").resize((width, height), Image.BICUBIC))
    content_latent = vae.encode(cnt_pix)  # [1, C, H/16, W/16]
    content_latent_packed = rearrange(content_latent, "B C H W -> B (H W) C")

    sty_pix = pipe.preprocess_image(Image.open(style_image_path).convert("RGB").resize((512, 512), Image.BICUBIC))
    style_latent_4d = vae.encode(sty_pix)
    style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

# Prepare image IDs
lat_h, lat_w = height // 16, width // 16
image_ids = torch.cartesian_prod(
    torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)
).unsqueeze(0).to("cuda")

# Content image IDs (edit image with t_coord offset)
cnt_lat_h, cnt_lat_w = content_latent.shape[2], content_latent.shape[3]
cnt_image_ids = torch.cartesian_prod(
    torch.tensor([10]), torch.arange(cnt_lat_h), torch.arange(cnt_lat_w), torch.arange(1)
).unsqueeze(0).to("cuda")

# Combined IDs for denoising
combined_img_ids = torch.cat([image_ids, cnt_image_ids], dim=1)

# Style image IDs (same shape as style latent)
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

# --- Load dit onto GPU for the denoising loop ---
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

# --- Denoising loop with OT attention ---
loss_fn = torch.nn.L1Loss()
# Klein 9B has guidance_embeds=false, so guidance=None
guidance_tensor = None
if embedded_guidance > 0:
    guidance_tensor = torch.tensor([embedded_guidance], device="cuda", dtype=torch.bfloat16)

_DIST_W = float(os.environ.get("DIST_W", "0.15"))
_CONTENT_W = float(os.environ.get("CONTENT_W", "0.6"))
_ADAIN_ALPHA = float(os.environ.get("ADAIN_ALPHA", "0.3"))
_GRAG_DEEP = float(os.environ.get("GRAG_DEEP", "0.5"))
_GRAG_SHALLOW = float(os.environ.get("GRAG_SHALLOW", "0.75"))
_OT_TAU_MAX = float(os.environ.get("OT_TAU_MAX", "0.15"))
_OT_TAU_MIN = float(os.environ.get("OT_TAU_MIN", "0.07"))
_OT_ITERS = int(os.environ.get("OT_ITERS", "6"))
_OT_LAYER_FRAC = float(os.environ.get("OT_LAYER_FRAC", "0.5"))
_STYLE_W = float(os.environ.get("STYLE_W", "1.2"))

latent_seq_len = latents.shape[1]

pbar = tqdm(enumerate(timesteps), total=len(timesteps), desc="Klein9B OT-Style")
for i, t_val in pbar:
    timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
    cache.clear(); controller.step()

    # Step 1: Denoising (content image concat)
    with torch.no_grad():
        latent_model_input = torch.cat(
            [latents, content_latent_packed.to(latents.dtype)], dim=1
        )
        noise_pred = dit(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance_tensor,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=combined_img_ids,
        )
        noise_pred = noise_pred[:, :latent_seq_len]
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

    # Step 3: Style forward → ref K/V
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
        latents_warmed = feature_adain(
            latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=_ADAIN_ALPHA
        ).squeeze(1)
    latents = latents_warmed.detach()

    # Step 5: Adam optimization
    latents.requires_grad_(True)
    optimizer = torch.optim.Adam([latents], lr=lr)
    _ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000

    for it in range(iters):
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

        # OT-attention style loss
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
                    sinkhorn_attn(q_det, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau)
                )
            else:
                ref_self_out_list.append(
                    F.scaled_dot_product_attention(q_det, ref_k, ref_v)
                )

        style_loss = sum(
            loss_fn(so, rso.detach())
            for so, rso in zip(self_out_list, ref_self_out_list)
        )

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
        optimizer.step()

        pbar.set_postfix(sl=f"{style_loss.item():.2f}", cl=f"{content_loss.item():.2f}")

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
image.save(out_dir / "output_klein9b.png")
Image.open(content_image_path).convert("RGB").resize((width, height)).save(out_dir / "content_klein9b.png")
Image.open(style_image_path).convert("RGB").resize((512, 512)).save(out_dir / "style_klein9b.png")
print(f"Saved: output_klein9b.png, content_klein9b.png, style_klein9b.png")
