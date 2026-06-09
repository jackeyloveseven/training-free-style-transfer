"""
Unified batch evaluation: T2I ablations + I2I experiments on Klein 9B.
Single process, model loaded once. Covers all P0/P1 experiments from EXPERIMENTS_NEEDED.md.

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_all.py --task t2i_ablation
  /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_all.py --task i2i_ablation
  /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_all.py --task t2i_baselines
  /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_all.py --task i2i_baselines
  /home/ubuntu/miniconda3/envs/sensenova/bin/python eval_all.py --task all
"""
import argparse, csv, glob, os, sys, time, types
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
import open_clip

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

DIFFSYNTH_ROOT = "/mnt/local/yzh/prog/DiffSynth-Studio"
KLEIN9B_ROOT = "/mnt/local/klein9b"
CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
RESULTS_DIR = Path(CWD) / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============== Sample sets ==============
T2I_PROMPTS_30 = [
    ("kp0", "A panda eat bamboo"),
    ("kp1", "A castle on a cliff by the sea"),
    ("kp2", "A cat sitting on a windowsill"),
    ("kp3", "A field of sunflowers at sunset"),
    ("kp4", "A portrait of an old fisherman"),
]

T2I_PROMPTS_FULL = [
    ("p0", "A cat sitting on a wooden table"),
    ("p1", "A sports car on a mountain road"),
    ("p2", "A field of sunflowers at sunset"),
    ("p3", "A dog running in a park"),
    ("p4", "A snowy mountain peak at dawn"),
    ("p5", "Portrait of a young woman"),
]

STYLES_6 = ["00", "10", "14", "20", "27", "28", "33", "38"]
STYLES_ALL = [f"{i:02d}" for i in range(40) if f"{i:02d}" not in ("04",)]

I2I_CNTS_5 = ["00", "04", "08", "12", "17"]
I2I_CNTS_ALL = [f"{i:02d}" for i in range(19)]  # cnt19 excluded per spec but it doesn't exist as 19

EXCLUDE_STY = {"04"}
EXCLUDE_CNT = {"19"}

# ============== DiffSynth setup ==============
def _install_modelscope_stub():
    if "modelscope" in sys.modules:
        return
    module = types.ModuleType("modelscope")
    module.snapshot_download = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("modelscope not installed"))
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

# ============== CSD + CLIP metrics ==============
sys.path.insert(0, '/home/ubuntu/chengyanli/items/CSD_Score')
from model import CSD_CLIP, convert_state_dict

csd_dir = '/home/ubuntu/chengyanli/items/CSD_Score'
csd_model = CSD_CLIP('vit_large', 'default', model_path=f'{csd_dir}/models/ViT-L-14.pt')
ck = torch.load(f'{csd_dir}/models/checkpoint.pth', map_location='cpu', weights_only=False)
csd_model.load_state_dict(convert_state_dict(ck['model_state_dict']), strict=False)
csd_model = csd_model.to('cuda').eval()

clip_model_vit, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model_vit = clip_model_vit.to('cuda').eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1,3,1,1)
_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1,3,1,1)

def _img_to_clip_tensor(path_or_pil):
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert('RGB')
    else:
        img = path_or_pil.convert('RGB')
    t = transforms.ToTensor()(img).unsqueeze(0).to('cuda')
    t = F.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
    return (t - _mean) / _std

def compute_csd(output_path, style_path):
    t_out = _img_to_clip_tensor(output_path)
    t_sty = _img_to_clip_tensor(style_path)
    with torch.no_grad():
        _, _, s_out = csd_model(t_out.float())
        _, _, s_sty = csd_model(t_sty.float())
    return (F.normalize(s_out, dim=-1) * F.normalize(s_sty, dim=-1)).sum().item()

def compute_clip_t(output_path, text):
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
    img = Image.open(output_path).convert('RGB')
    img_t = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])(img).unsqueeze(0).to('cuda')
    text_t = clip_tokenizer([text]).to('cuda')
    with torch.no_grad():
        img_feat = clip_model_vit.encode_image(img_t)
        txt_feat = clip_model_vit.encode_text(text_t)
    return (F.normalize(img_feat, dim=-1) * F.normalize(txt_feat, dim=-1)).sum().item()

def compute_clip_i(output_path, ref_path):
    img1 = Image.open(output_path).convert('RGB')
    img2 = Image.open(ref_path).convert('RGB')
    preproc = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    t1 = preproc(img1).unsqueeze(0).to('cuda')
    t2 = preproc(img2).unsqueeze(0).to('cuda')
    with torch.no_grad():
        f1 = clip_model_vit.encode_image(t1)
        f2 = clip_model_vit.encode_image(t2)
    return (F.normalize(f1, dim=-1) * F.normalize(f2, dim=-1)).sum().item()

# ============== Load Klein 9B ==============
print("Loading Klein 9B model...")
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

# ============== Attention hooks (from T2I script) ==============
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

def register_klein9b_attn_hooks(dit_model, controller, cache):
    from diffsynth.core.attention.attention import torch_sdpa

    def make_double_hook(original_block_attn):
        def hooked_processor(attn, hidden_states, encoder_hidden_states=None,
                             attention_mask=None, image_rotary_emb=None):
            from diffsynth.models.flux2_dit import _get_qkv_projections, apply_rotary_emb
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states)
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
                cache.add(q_t, k_t, v_t, attn_out_t)
            controller.cur_self_layer += 1
            attn_output = attn_out_t.transpose(1, 2).flatten(2, 3).to(query.dtype)
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
        def hooked_processor(attn, hidden_states, attention_mask=None, image_rotary_emb=None):
            from diffsynth.models.flux2_dit import apply_rotary_emb
            proj_out = attn.to_qkv_mlp_proj(hidden_states)
            qkv, mlp_hidden_states = torch.split(
                proj_out, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)
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
                cache.add(q_t, k_t, v_t, attn_out_t)
            controller.cur_self_layer += 1
            attn_output = attn_out_t.transpose(1, 2).flatten(2, 3).to(query.dtype)
            mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
            hidden_states = attn.to_out(hidden_states)
            return hidden_states
        return hooked_processor

    for block in dit_model.transformer_blocks:
        block.attn.processor = make_double_hook(block.attn)
    for block in dit_model.single_transformer_blocks:
        block.attn.processor = make_single_hook(block.attn)

start_layer, end_layer = 12, 28
cache = Klein9BCache()
controller = Klein9BController(self_layers=(start_layer, end_layer))
register_klein9b_attn_hooks(dit, controller, cache)

pipe.load_models_to_device(["dit"])
dit.to("cuda")
dit.requires_grad_(False)

# ============== Utility functions ==============
def sinkhorn_attn(Q, K_style, V_style, n_iters=6, tau=0.1):
    B, H, Lc, D = Q.shape
    logits = torch.matmul(Q, K_style.transpose(-1, -2)) / (D ** 0.5 * tau)
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    P = torch.softmax(log_P, dim=-1)
    return torch.matmul(P, V_style)

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

def sinkhorn_latent_transport(content_lat, style_lat_noisy, style_lat_clean,
                              alpha=0.5, n_iters=10, tau=0.1, grag_delta=1.0, mode="ot_adain"):
    B, N, D = content_lat.shape
    style_noisy_g = style_lat_noisy
    style_clean_g = style_lat_clean
    if grag_delta < 1.0:
        s_mean = style_lat_noisy.mean(dim=1, keepdim=True)
        style_noisy_g = s_mean + grag_delta * (style_lat_noisy - s_mean)
        sc_mean = style_lat_clean.mean(dim=1, keepdim=True)
        style_clean_g = sc_mean + grag_delta * (style_lat_clean - sc_mean)
    c_f = content_lat.float()
    s_f = style_noisy_g.float()
    c_hi = c_f - c_f.mean(dim=1, keepdim=True)
    s_hi = s_f - s_f.mean(dim=1, keepdim=True)
    c_norm = F.normalize(c_hi, dim=-1)
    s_norm = F.normalize(s_hi, dim=-1)
    logits = torch.bmm(c_norm, s_norm.transpose(1, 2)) / tau
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    T = torch.softmax(log_P, dim=-1)
    if mode == "blend":
        transported = torch.bmm(T, style_clean_g.float()).to(content_lat.dtype)
        return (1.0 - alpha) * content_lat + alpha * transported
    else:
        style_f = style_clean_g.float()
        matched_mean = torch.bmm(T, style_f)
        matched_var = torch.bmm(T, style_f ** 2) - matched_mean ** 2
        matched_std = (matched_var.clamp(min=1e-6)).sqrt()
        content_f = content_lat.float()
        c_mean = content_f.mean(dim=1, keepdim=True)
        c_std = content_f.std(dim=1, keepdim=True).clamp(min=1e-6)
        normed = (content_f - c_mean) / c_std
        styled = normed * matched_std + matched_mean
        return ((1.0 - alpha) * content_f + alpha * styled).to(content_lat.dtype)

# ============== Prompt encoding (cached) ==============
from diffsynth.pipelines.flux2_image import Flux2Unit_Qwen3PromptEmbedder, Flux2Unit_PromptEmbedder

_prompt_cache = {}

def encode_prompt(text):
    if text in _prompt_cache:
        return _prompt_cache[text]
    if pipe.text_encoder_qwen3 is not None:
        pipe.load_models_to_device(["text_encoder_qwen3"])
        embedder = Flux2Unit_Qwen3PromptEmbedder()
        embeds, ids = embedder.encode_prompt(
            pipe.text_encoder_qwen3, pipe.tokenizer, text, dtype=pipe.torch_dtype, device="cuda")
    else:
        pipe.load_models_to_device(["text_encoder"])
        embedder = Flux2Unit_PromptEmbedder()
        embeds, ids = embedder.encode_prompt(
            pipe.text_encoder, pipe.tokenizer, text, dtype=pipe.torch_dtype, device="cuda")
    pipe.load_models_to_device(["dit"])
    dit.to("cuda")
    _prompt_cache[text] = (embeds, ids)
    return embeds, ids

# Pre-encode null prompt
null_embeds, null_text_ids = encode_prompt("")

# ============== T2I Generation ==============
def generate_t2i(prompt, style_path, config):
    """
    config keys:
      mode: "hybrid" | "adam" | "slt" | "inject" | "none"
      iters: int (Adam iterations)
      lr: float
      content_w: float
      style_w: float
      slt_alpha: float
      freq_hi_boost: float (0 = standard sinkhorn)
      sinkhorn_mode: "freq" | "softmax" | "standard"
      inject_beta: float
      grag_deep: float
      grag_shallow: float
    """
    mode = config.get("mode", "hybrid")
    adam_iters = config.get("iters", 2)
    lr = config.get("lr", 0.03)
    content_w = config.get("content_w", 1.5)
    style_w = config.get("style_w", 2.0)
    slt_alpha = config.get("slt_alpha", 0.1)
    freq_hi_boost = config.get("freq_hi_boost", 2.0)
    sinkhorn_mode = config.get("sinkhorn_mode", "freq")
    inject_beta = config.get("inject_beta", 0.5)
    grag_deep = config.get("grag_deep", 0.5)
    grag_shallow = config.get("grag_shallow", 0.7)
    adain_alpha = config.get("adain_alpha", 0.5)
    start_time_val = config.get("start_time", 850)

    width, height = 512, 512
    num_inference_steps = 25
    seed = 43

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        sty_pix = pipe.preprocess_image(Image.open(style_path).convert("RGB").resize((512,512), Image.BICUBIC))
        style_latent_4d = vae.encode(sty_pix)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

    lat_h, lat_w = height // 16, width // 16
    image_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)
    ).unsqueeze(0).to("cuda")
    sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
    style_image_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)
    ).unsqueeze(0).to("cuda")

    prompt_embeds, text_ids = encode_prompt(prompt)
    pipe.load_models_to_device(["dit"])
    dit.to("cuda")

    torch.manual_seed(seed)
    noise = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    noise = rearrange(noise, "B C H W -> B (H W) C")
    latents = noise.clone()

    scheduler = FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(num_inference_steps, dynamic_shift_len=lat_h * lat_w)
    timesteps = scheduler.timesteps

    loss_fn = torch.nn.L1Loss()
    _OT_TAU_MAX, _OT_TAU_MIN = 0.15, 0.07
    _OT_ITERS = 6
    _OT_LAYER_FRAC = 0.5

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()

        with torch.no_grad():
            noise_pred = dit(
                hidden_states=latents, timestep=timestep / 1000, guidance=None,
                encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=image_ids)
        latents = scheduler.step(noise_pred, t_val, latents)

        if t_val >= start_time_val:
            continue
        if mode == "none":
            continue

        # Noisy style
        if i < num_inference_steps - 1:
            timestep_style = timesteps[i+1:i+2]
            sty_noise = torch.randn_like(style_latent)
            style_latent_ = scheduler.add_noise(style_latent, sty_noise, timestep_style)
        else:
            timestep_style = torch.tensor([0.0])
            style_latent_ = style_latent

        # Style forward
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(
                hidden_states=style_latent_.to(device="cuda", dtype=torch.bfloat16),
                timestep=timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000,
                guidance=None, encoder_hidden_states=null_embeds,
                txt_ids=null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, _ = cache.get()

        # GRAG
        _txt_len = controller.txt_len
        n_cached = len(ref_k_list)
        for idx in range(n_cached):
            depth_frac = idx / max(n_cached - 1, 1)
            delta_i = grag_shallow + (grag_deep - grag_shallow) * depth_frac
            k, v = ref_k_list[idx], ref_v_list[idx]
            k_txt, k_img = k[:,:,:_txt_len,:], k[:,:,_txt_len:,:]
            v_txt, v_img = v[:,:,:_txt_len,:], v[:,:,_txt_len:,:]
            km, vm = k_img.mean(dim=2, keepdim=True), v_img.mean(dim=2, keepdim=True)
            k_img = km + delta_i * (k_img - km)
            v_img = vm + delta_i * (v_img - vm)
            ref_k_list[idx] = torch.cat([k_txt, k_img], dim=2)
            ref_v_list[idx] = torch.cat([v_txt, v_img], dim=2)

        content_anchor = latents.detach().clone()

        # AdaIN warm-start
        with torch.no_grad():
            latents = feature_adain(
                latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=adain_alpha
            ).squeeze(1)
        latents = latents.detach()

        if mode == "inject":
            # No optimization, just keep adain result
            continue

        # SLT pre-alignment (for hybrid mode)
        if mode == "hybrid" and slt_alpha > 0:
            with torch.no_grad():
                latents = sinkhorn_latent_transport(
                    latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=0.02,
                    grag_delta=grag_deep, mode="ot_adain")

        if mode == "slt":
            # Pure SLT, no Adam
            with torch.no_grad():
                latents = sinkhorn_latent_transport(
                    latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=0.02,
                    grag_delta=grag_deep, mode="ot_adain")
            continue

        # Adam optimization (for hybrid and adam modes)
        actual_iters = adam_iters
        if mode == "adam":
            actual_iters = adam_iters

        if actual_iters > 0:
            latents.requires_grad_(True)
            optimizer = torch.optim.Adam([latents], lr=lr)
            _ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000

            for it in range(actual_iters):
                cache.clear(); controller.step()
                optimizer.zero_grad()
                with torch.enable_grad():
                    _dit_out = dit(
                        hidden_states=latents, timestep=_ts_cuda.detach(), guidance=None,
                        encoder_hidden_states=null_embeds.detach(),
                        txt_ids=null_text_ids.detach(), img_ids=image_ids.detach())
                del _dit_out
                q_list, _, _, self_out_list = cache.get()
                _t_float = float(t_val.item())
                _progress = 1.0 - (_t_float / max(start_time_val, 1.0))
                _tau = _OT_TAU_MAX - (_OT_TAU_MAX - _OT_TAU_MIN) * _progress
                n_layers = len(q_list)
                n_ot_layers = max(1, int(n_layers * _OT_LAYER_FRAC))

                ref_self_out_list = []
                for idx2, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                    q_det = q.detach()
                    if idx2 >= n_layers - n_ot_layers:
                        if sinkhorn_mode == "freq":
                            ref_self_out_list.append(
                                sinkhorn_attn_freq(q_det, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau, hi_boost=freq_hi_boost))
                        elif sinkhorn_mode == "softmax":
                            ref_self_out_list.append(
                                F.scaled_dot_product_attention(q_det, ref_k, ref_v))
                        else:
                            ref_self_out_list.append(
                                sinkhorn_attn(q_det, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau))
                    else:
                        ref_self_out_list.append(F.scaled_dot_product_attention(q_det, ref_k, ref_v))

                style_loss = sum(loss_fn(so, rso.detach()) for so, rso in zip(self_out_list, ref_self_out_list))
                content_loss = loss_fn(latents, content_anchor)
                loss = style_w * style_loss + content_w * content_loss
                loss.backward()
                optimizer.step()
            latents = latents.detach()

    # Final color transfer
    latents = adain_1d(latents, style_latent)

    pipe.load_models_to_device(["vae"])
    latents_4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
    image = vae.decode(latents_4d)
    image = pipe.vae_output_to_image(image)
    pipe.load_models_to_device(["dit"])
    dit.to("cuda")
    return image

# ============== I2I Generation ==============
def generate_i2i(cnt_path, style_path, prompt, config):
    """Same as T2I but with content image concat (like custom_main_klein9b.py)."""
    mode = config.get("mode", "hybrid")
    adam_iters = config.get("iters", 2)
    lr = config.get("lr", 0.03)
    content_w = config.get("content_w", 1.5)
    style_w = config.get("style_w", 2.0)
    slt_alpha = config.get("slt_alpha", 0.1)
    freq_hi_boost = config.get("freq_hi_boost", 2.0)
    sinkhorn_mode = config.get("sinkhorn_mode", "freq")
    grag_deep = config.get("grag_deep", 0.5)
    grag_shallow = config.get("grag_shallow", 0.7)
    adain_alpha = config.get("adain_alpha", 0.3)
    start_time_val = config.get("start_time", 870)

    width, height = 512, 512
    num_inference_steps = 25
    seed = 43

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        cnt_pix = pipe.preprocess_image(Image.open(cnt_path).convert("RGB").resize((width, height), Image.BICUBIC))
        content_latent = vae.encode(cnt_pix)
        content_latent_packed = rearrange(content_latent, "B C H W -> B (H W) C")

        sty_pix = pipe.preprocess_image(Image.open(style_path).convert("RGB").resize((512,512), Image.BICUBIC))
        style_latent_4d = vae.encode(sty_pix)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

    lat_h, lat_w = height // 16, width // 16
    image_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)
    ).unsqueeze(0).to("cuda")

    cnt_lat_h, cnt_lat_w = content_latent.shape[2], content_latent.shape[3]
    cnt_image_ids = torch.cartesian_prod(
        torch.tensor([10]), torch.arange(cnt_lat_h), torch.arange(cnt_lat_w), torch.arange(1)
    ).unsqueeze(0).to("cuda")
    combined_img_ids = torch.cat([image_ids, cnt_image_ids], dim=1)

    sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
    style_image_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)
    ).unsqueeze(0).to("cuda")

    prompt_embeds, text_ids = encode_prompt(prompt)
    pipe.load_models_to_device(["dit"])
    dit.to("cuda")

    torch.manual_seed(seed)
    noise = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    noise = rearrange(noise, "B C H W -> B (H W) C")
    latents = noise.clone()
    latent_seq_len = latents.shape[1]

    scheduler = FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(num_inference_steps, dynamic_shift_len=lat_h * lat_w)
    timesteps = scheduler.timesteps

    loss_fn = torch.nn.L1Loss()
    _OT_TAU_MAX, _OT_TAU_MIN = 0.15, 0.07
    _OT_ITERS = 6
    _OT_LAYER_FRAC = 0.5

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()

        # Denoising with content image concat
        with torch.no_grad():
            latent_model_input = torch.cat(
                [latents, content_latent_packed.to(latents.dtype)], dim=1)
            noise_pred = dit(
                hidden_states=latent_model_input, timestep=timestep / 1000, guidance=None,
                encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=combined_img_ids)
            noise_pred = noise_pred[:, :latent_seq_len]
        latents = scheduler.step(noise_pred, t_val, latents)

        if t_val >= start_time_val:
            continue
        if mode == "none":
            continue

        # Noisy style
        if i < num_inference_steps - 1:
            timestep_style = timesteps[i+1:i+2]
            sty_noise = torch.randn_like(style_latent)
            style_latent_ = scheduler.add_noise(style_latent, sty_noise, timestep_style)
        else:
            timestep_style = torch.tensor([0.0])
            style_latent_ = style_latent

        # Style forward
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(
                hidden_states=style_latent_.to(device="cuda", dtype=torch.bfloat16),
                timestep=timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000,
                guidance=None, encoder_hidden_states=null_embeds,
                txt_ids=null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, _ = cache.get()

        # GRAG
        _txt_len = controller.txt_len
        n_cached = len(ref_k_list)
        for idx in range(n_cached):
            depth_frac = idx / max(n_cached - 1, 1)
            delta_i = grag_shallow + (grag_deep - grag_shallow) * depth_frac
            k, v = ref_k_list[idx], ref_v_list[idx]
            k_txt, k_img = k[:,:,:_txt_len,:], k[:,:,_txt_len:,:]
            v_txt, v_img = v[:,:,:_txt_len,:], v[:,:,_txt_len:,:]
            km, vm = k_img.mean(dim=2, keepdim=True), v_img.mean(dim=2, keepdim=True)
            k_img = km + delta_i * (k_img - km)
            v_img = vm + delta_i * (v_img - vm)
            ref_k_list[idx] = torch.cat([k_txt, k_img], dim=2)
            ref_v_list[idx] = torch.cat([v_txt, v_img], dim=2)

        content_anchor = latents.detach().clone()

        # AdaIN
        with torch.no_grad():
            latents = feature_adain(
                latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=adain_alpha
            ).squeeze(1)
        latents = latents.detach()

        if mode == "inject":
            continue

        if mode == "hybrid" and slt_alpha > 0:
            with torch.no_grad():
                latents = sinkhorn_latent_transport(
                    latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=0.02,
                    grag_delta=grag_deep, mode="ot_adain")

        if mode == "slt":
            with torch.no_grad():
                latents = sinkhorn_latent_transport(
                    latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=0.02,
                    grag_delta=grag_deep, mode="ot_adain")
            continue

        # Adam
        actual_iters = adam_iters
        if actual_iters > 0:
            latents.requires_grad_(True)
            optimizer = torch.optim.Adam([latents], lr=lr)
            _ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000

            for it in range(actual_iters):
                cache.clear(); controller.step()
                optimizer.zero_grad()
                with torch.enable_grad():
                    _dit_out = dit(
                        hidden_states=latents, timestep=_ts_cuda.detach(), guidance=None,
                        encoder_hidden_states=null_embeds.detach(),
                        txt_ids=null_text_ids.detach(), img_ids=image_ids.detach())
                del _dit_out
                q_list, _, _, self_out_list = cache.get()
                _t_float = float(t_val.item())
                _progress = 1.0 - (_t_float / max(start_time_val, 1.0))
                _tau = _OT_TAU_MAX - (_OT_TAU_MAX - _OT_TAU_MIN) * _progress
                n_layers = len(q_list)
                n_ot_layers = max(1, int(n_layers * _OT_LAYER_FRAC))

                ref_self_out_list = []
                for idx2, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                    q_det = q.detach()
                    if idx2 >= n_layers - n_ot_layers:
                        if sinkhorn_mode == "freq":
                            ref_self_out_list.append(
                                sinkhorn_attn_freq(q_det, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau, hi_boost=freq_hi_boost))
                        elif sinkhorn_mode == "softmax":
                            ref_self_out_list.append(
                                F.scaled_dot_product_attention(q_det, ref_k, ref_v))
                        else:
                            ref_self_out_list.append(
                                sinkhorn_attn(q_det, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau))
                    else:
                        ref_self_out_list.append(F.scaled_dot_product_attention(q_det, ref_k, ref_v))

                style_loss = sum(loss_fn(so, rso.detach()) for so, rso in zip(self_out_list, ref_self_out_list))
                content_loss = loss_fn(latents, content_anchor)
                loss = style_w * style_loss + content_w * content_loss
                loss.backward()
                optimizer.step()
            latents = latents.detach()

    latents = adain_1d(latents, style_latent)
    pipe.load_models_to_device(["vae"])
    latents_4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
    image = vae.decode(latents_4d)
    image = pipe.vae_output_to_image(image)
    pipe.load_models_to_device(["dit"])
    dit.to("cuda")
    return image

# ============== Experiment configs ==============
# T2I configs
HYBRID_DEFAULT = {"mode": "hybrid", "iters": 2, "lr": 0.03, "content_w": 1.5, "style_w": 2.0,
                  "slt_alpha": 0.1, "freq_hi_boost": 2.0, "sinkhorn_mode": "freq",
                  "grag_deep": 0.5, "grag_shallow": 0.7}
ADAM8_DEFAULT = {"mode": "adam", "iters": 8, "lr": 0.01, "content_w": 0.2, "style_w": 2.0,
                "slt_alpha": 0.0, "freq_hi_boost": 2.0, "sinkhorn_mode": "freq",
                "grag_deep": 0.5, "grag_shallow": 0.7}

T2I_ABLATIONS = {
    "hybrid2": HYBRID_DEFAULT,
    "adam8": ADAM8_DEFAULT,
    "no_slt": {**HYBRID_DEFAULT, "slt_alpha": 0.0},
    "no_adam": {**HYBRID_DEFAULT, "iters": 0, "mode": "slt"},
    "softmax": {**HYBRID_DEFAULT, "sinkhorn_mode": "softmax"},
    "freq_off": {**HYBRID_DEFAULT, "freq_hi_boost": 0.0, "sinkhorn_mode": "standard"},
}

T2I_BASELINES = {
    "text_only": {"mode": "none"},
    "inject": {"mode": "inject", "inject_beta": 0.5},
}

# I2I configs — optimal found via search_i2i_params.py (2026-05-20):
# Best Pareto: start_time=870, adain_alpha=0.3, slt_alpha=0.1, iters=2, lr=0.03, content_w=1.5
# CSD=0.7512, CLIP-I(cnt)=0.6024, Time=7.9s (vs Adam-8: CSD=0.7073, CLIP-I=0.6028, 21.6s)
I2I_HYBRID = {**HYBRID_DEFAULT, "adain_alpha": 0.3, "start_time": 870}
I2I_ADAM8 = {**ADAM8_DEFAULT, "adain_alpha": 0.3, "start_time": 870, "content_w": 0.6}

I2I_ABLATIONS = {
    "hybrid2": I2I_HYBRID,
    "adam8": I2I_ADAM8,
    "no_slt": {**I2I_HYBRID, "slt_alpha": 0.0},
    "no_adam": {**I2I_HYBRID, "iters": 0, "mode": "slt"},
    "softmax": {**I2I_HYBRID, "sinkhorn_mode": "softmax"},
}

I2I_BASELINES = {
    "text_only": {**I2I_HYBRID, "mode": "none"},
    "inject": {**I2I_HYBRID, "mode": "inject"},
}

# ============== Run tasks ==============
def run_t2i_ablation():
    """T2I P0 ablations: 30 pairs × 6 methods."""
    out_dir = Path(CWD) / "d_grand" / "t2i_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "t2i_ablation.csv"

    rows = []
    print(f"\n{'='*80}")
    print(f"T2I ABLATION: {len(T2I_ABLATIONS)} methods × {len(T2I_PROMPTS_30)} prompts × {len(STYLES_6)} styles = {len(T2I_ABLATIONS)*len(T2I_PROMPTS_30)*len(STYLES_6)} images")
    print(f"{'='*80}")
    print(f"{'Method':<12} {'Prompt':<35} {'Sty':>3} {'CSD':>7} {'CLIP-T':>7} {'Time':>6}")
    print("-" * 75)

    for method_name, config in T2I_ABLATIONS.items():
        for pid, prompt in T2I_PROMPTS_30:
            for sty_id in STYLES_6:
                sty_path = f"{CWD}/sty/{sty_id}.png"
                torch.cuda.empty_cache()
                t0 = time.time()
                img = generate_t2i(prompt, sty_path, config)
                elapsed = time.time() - t0

                fname = f"{method_name}_{pid}_s{sty_id}.png"
                img.save(out_dir / fname)

                csd = compute_csd(str(out_dir / fname), sty_path)
                clip_t = compute_clip_t(str(out_dir / fname), prompt)

                rows.append({"method": method_name, "sty_id": sty_id, "content_id": pid,
                             "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}", "time_s": f"{elapsed:.1f}"})
                print(f"{method_name:<12} {prompt:<35} {sty_id:>3} {csd:>7.4f} {clip_t:>7.4f} {elapsed:>6.1f}")
                sys.stdout.flush()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    # Print averages
    print(f"\n{'='*80}")
    print("T2I ABLATION AVERAGES")
    print(f"{'='*80}")
    for method_name in T2I_ABLATIONS:
        m_rows = [r for r in rows if r["method"] == method_name]
        avg_csd = np.mean([float(r["csd"]) for r in m_rows])
        avg_clip_t = np.mean([float(r["clip_t"]) for r in m_rows])
        avg_time = np.mean([float(r["time_s"]) for r in m_rows])
        print(f"  {method_name:<12} CSD={avg_csd:.4f}  CLIP-T={avg_clip_t:.4f}  Time={avg_time:.1f}s")
    print(f"\nSaved: {csv_path}")


def run_t2i_baselines():
    """T2I P1 baselines: 30 pairs × 2 methods."""
    out_dir = Path(CWD) / "d_grand" / "t2i_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "t2i_baselines.csv"

    rows = []
    print(f"\n{'='*80}")
    print(f"T2I BASELINES: {len(T2I_BASELINES)} methods × 30 pairs")
    print(f"{'='*80}")

    for method_name, config in T2I_BASELINES.items():
        for pid, prompt in T2I_PROMPTS_30:
            for sty_id in STYLES_6:
                sty_path = f"{CWD}/sty/{sty_id}.png"
                torch.cuda.empty_cache()
                t0 = time.time()
                img = generate_t2i(prompt, sty_path, config)
                elapsed = time.time() - t0

                fname = f"{method_name}_{pid}_s{sty_id}.png"
                img.save(out_dir / fname)

                csd = compute_csd(str(out_dir / fname), sty_path)
                clip_t = compute_clip_t(str(out_dir / fname), prompt)

                rows.append({"method": method_name, "sty_id": sty_id, "content_id": pid,
                             "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}", "time_s": f"{elapsed:.1f}"})
                print(f"  {method_name:<12} {pid} s{sty_id} CSD={csd:.4f} CLIP-T={clip_t:.4f} {elapsed:.1f}s")
                sys.stdout.flush()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    for method_name in T2I_BASELINES:
        m_rows = [r for r in rows if r["method"] == method_name]
        avg_csd = np.mean([float(r["csd"]) for r in m_rows])
        avg_clip_t = np.mean([float(r["clip_t"]) for r in m_rows])
        print(f"  {method_name:<12} avg CSD={avg_csd:.4f}  CLIP-T={avg_clip_t:.4f}")
    print(f"\nSaved: {csv_path}")


def run_i2i_ablation():
    """I2I ablations: 30 pairs (5 cnt × 6 sty) × methods."""
    out_dir = Path(CWD) / "d_grand" / "i2i_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "i2i_ablation.csv"

    rows = []
    print(f"\n{'='*80}")
    print(f"I2I ABLATION: {len(I2I_ABLATIONS)} methods × {len(I2I_CNTS_5)} cnt × {len(STYLES_6)} sty = {len(I2I_ABLATIONS)*len(I2I_CNTS_5)*len(STYLES_6)} images")
    print(f"{'='*80}")
    print(f"{'Method':<12} {'Cnt':>3} {'Sty':>3} {'CSD':>7} {'CLIP-T':>7} {'CLIP-I':>7} {'Time':>6}")
    print("-" * 60)

    for method_name, config in I2I_ABLATIONS.items():
        for cnt_id in I2I_CNTS_5:
            cnt_path = f"{CWD}/cnt/{cnt_id}.png"
            for sty_id in STYLES_6:
                sty_path = f"{CWD}/sty/{sty_id}.png"
                torch.cuda.empty_cache()
                t0 = time.time()
                img = generate_i2i(cnt_path, sty_path, "masterpiece", config)
                elapsed = time.time() - t0

                fname = f"{method_name}_cnt{cnt_id}_sty{sty_id}.png"
                img.save(out_dir / fname)

                csd = compute_csd(str(out_dir / fname), sty_path)
                clip_t = compute_clip_t(str(out_dir / fname), "masterpiece")
                clip_i = compute_clip_i(str(out_dir / fname), cnt_path)

                rows.append({"method": method_name, "cnt_id": cnt_id, "sty_id": sty_id,
                             "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}",
                             "clip_i_cnt": f"{clip_i:.4f}", "time_s": f"{elapsed:.1f}"})
                print(f"{method_name:<12} {cnt_id:>3} {sty_id:>3} {csd:>7.4f} {clip_t:>7.4f} {clip_i:>7.4f} {elapsed:>6.1f}")
                sys.stdout.flush()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "cnt_id", "sty_id", "csd", "clip_t", "clip_i_cnt", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*80}")
    print("I2I ABLATION AVERAGES")
    print(f"{'='*80}")
    for method_name in I2I_ABLATIONS:
        m_rows = [r for r in rows if r["method"] == method_name]
        avg_csd = np.mean([float(r["csd"]) for r in m_rows])
        avg_clip_t = np.mean([float(r["clip_t"]) for r in m_rows])
        avg_clip_i = np.mean([float(r["clip_i_cnt"]) for r in m_rows])
        avg_time = np.mean([float(r["time_s"]) for r in m_rows])
        print(f"  {method_name:<12} CSD={avg_csd:.4f}  CLIP-T={avg_clip_t:.4f}  CLIP-I(cnt)={avg_clip_i:.4f}  Time={avg_time:.1f}s")
    print(f"\nSaved: {csv_path}")


def run_i2i_baselines():
    """I2I baselines: 30 pairs × 2 methods."""
    out_dir = Path(CWD) / "d_grand" / "i2i_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "i2i_baselines.csv"

    rows = []
    print(f"\n{'='*80}")
    print(f"I2I BASELINES: {len(I2I_BASELINES)} methods × 30 pairs")
    print(f"{'='*80}")

    for method_name, config in I2I_BASELINES.items():
        for cnt_id in I2I_CNTS_5:
            cnt_path = f"{CWD}/cnt/{cnt_id}.png"
            for sty_id in STYLES_6:
                sty_path = f"{CWD}/sty/{sty_id}.png"
                torch.cuda.empty_cache()
                t0 = time.time()
                img = generate_i2i(cnt_path, sty_path, "masterpiece", config)
                elapsed = time.time() - t0

                fname = f"{method_name}_cnt{cnt_id}_sty{sty_id}.png"
                img.save(out_dir / fname)

                csd = compute_csd(str(out_dir / fname), sty_path)
                clip_t = compute_clip_t(str(out_dir / fname), "masterpiece")
                clip_i = compute_clip_i(str(out_dir / fname), cnt_path)

                rows.append({"method": method_name, "cnt_id": cnt_id, "sty_id": sty_id,
                             "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}",
                             "clip_i_cnt": f"{clip_i:.4f}", "time_s": f"{elapsed:.1f}"})
                print(f"  {method_name:<12} cnt{cnt_id} sty{sty_id} CSD={csd:.4f} CLIP-I={clip_i:.4f} {elapsed:.1f}s")
                sys.stdout.flush()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "cnt_id", "sty_id", "csd", "clip_t", "clip_i_cnt", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    for method_name in I2I_BASELINES:
        m_rows = [r for r in rows if r["method"] == method_name]
        avg_csd = np.mean([float(r["csd"]) for r in m_rows])
        avg_clip_i = np.mean([float(r["clip_i_cnt"]) for r in m_rows])
        print(f"  {method_name:<12} avg CSD={avg_csd:.4f}  CLIP-I(cnt)={avg_clip_i:.4f}")
    print(f"\nSaved: {csv_path}")


def run_t2i_full():
    """T2I full benchmark: 39 sty × 6 prompts = 234 pairs × 2 methods (hybrid + adam8)."""
    out_dir = Path(CWD) / "d_grand" / "t2i_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "t2i_full_benchmark.csv"

    methods = {"hybrid2": HYBRID_DEFAULT, "adam8": ADAM8_DEFAULT}
    total = len(methods) * len(T2I_PROMPTS_FULL) * len(STYLES_ALL)
    rows = []
    print(f"\n{'='*80}")
    print(f"T2I FULL BENCHMARK: {len(methods)} methods × {len(T2I_PROMPTS_FULL)} prompts × {len(STYLES_ALL)} styles = {total} images")
    print(f"{'='*80}")
    print(f"{'Method':<10} {'Prompt':<35} {'Sty':>3} {'CSD':>7} {'CLIP-T':>7} {'Time':>6}")
    print("-" * 75)

    count = 0
    for method_name, config in methods.items():
        for pid, prompt in T2I_PROMPTS_FULL:
            for sty_id in STYLES_ALL:
                sty_path = f"{CWD}/sty/{sty_id}.png"
                fname = f"{method_name}_{pid}_s{sty_id}.png"
                out_path = str(out_dir / fname)

                # Skip if already exists
                if os.path.exists(out_path):
                    count += 1
                    continue

                torch.cuda.empty_cache()
                t0 = time.time()
                img = generate_t2i(prompt, sty_path, config)
                elapsed = time.time() - t0

                img.save(out_path)
                csd = compute_csd(out_path, sty_path)
                clip_t = compute_clip_t(out_path, prompt)

                rows.append({"method": method_name, "sty_id": sty_id, "content_id": pid,
                             "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}", "time_s": f"{elapsed:.1f}"})
                count += 1
                if count % 10 == 0:
                    print(f"{method_name:<10} {prompt:<35} {sty_id:>3} {csd:>7.4f} {clip_t:>7.4f} {elapsed:>6.1f}  [{count}/{total}]")
                sys.stdout.flush()

    # Also compute metrics for any pre-existing images not yet in rows
    for method_name in methods:
        for pid, prompt in T2I_PROMPTS_FULL:
            for sty_id in STYLES_ALL:
                fname = f"{method_name}_{pid}_s{sty_id}.png"
                out_path = str(out_dir / fname)
                if os.path.exists(out_path) and not any(
                    r["method"] == method_name and r["sty_id"] == sty_id and r["content_id"] == pid for r in rows):
                    sty_path = f"{CWD}/sty/{sty_id}.png"
                    csd = compute_csd(out_path, sty_path)
                    clip_t = compute_clip_t(out_path, prompt)
                    rows.append({"method": method_name, "sty_id": sty_id, "content_id": pid,
                                 "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}", "time_s": "0.0"})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*80}")
    print("T2I FULL BENCHMARK AVERAGES")
    print(f"{'='*80}")
    for method_name in methods:
        m_rows = [r for r in rows if r["method"] == method_name]
        if m_rows:
            avg_csd = np.mean([float(r["csd"]) for r in m_rows])
            avg_clip_t = np.mean([float(r["clip_t"]) for r in m_rows])
            avg_time = np.mean([float(r["time_s"]) for r in m_rows])
            print(f"  {method_name:<10} CSD={avg_csd:.4f}  CLIP-T={avg_clip_t:.4f}  Time={avg_time:.1f}s  (n={len(m_rows)})")
    print(f"\nSaved: {csv_path}")


def run_i2i_full():
    """I2I full benchmark: 18 cnt × 39 sty = 702 pairs, hybrid config."""
    out_dir = Path(CWD) / "d_grand" / "i2i_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "i2i_full_benchmark.csv"

    total = len(I2I_CNTS_ALL) * len(STYLES_ALL)
    rows = []
    print(f"\n{'='*80}")
    print(f"I2I FULL BENCHMARK: {len(I2I_CNTS_ALL)} cnt × {len(STYLES_ALL)} sty = {total} images (hybrid)")
    print(f"{'='*80}")
    print(f"{'Cnt':>3} {'Sty':>3} {'CSD':>7} {'CLIP-I':>7} {'Time':>6}")
    print("-" * 40)

    count = 0
    for cnt_id in I2I_CNTS_ALL:
        cnt_path = f"{CWD}/cnt/{cnt_id}.png"
        if not os.path.exists(cnt_path):
            continue
        for sty_id in STYLES_ALL:
            sty_path = f"{CWD}/sty/{sty_id}.png"
            fname = f"cnt{cnt_id}_sty{sty_id}.png"
            out_path = str(out_dir / fname)

            if os.path.exists(out_path):
                count += 1
                continue

            torch.cuda.empty_cache()
            t0 = time.time()
            img = generate_i2i(cnt_path, sty_path, "masterpiece", I2I_HYBRID)
            elapsed = time.time() - t0

            img.save(out_path)
            csd = compute_csd(out_path, sty_path)
            clip_i = compute_clip_i(out_path, cnt_path)

            rows.append({"cnt_id": cnt_id, "sty_id": sty_id,
                         "csd": f"{csd:.4f}", "clip_i_cnt": f"{clip_i:.4f}", "time_s": f"{elapsed:.1f}"})
            count += 1
            if count % 20 == 0:
                print(f"{cnt_id:>3} {sty_id:>3} {csd:>7.4f} {clip_i:>7.4f} {elapsed:>6.1f}  [{count}/{total}]")
            sys.stdout.flush()

    # Compute metrics for pre-existing images
    for cnt_id in I2I_CNTS_ALL:
        cnt_path = f"{CWD}/cnt/{cnt_id}.png"
        if not os.path.exists(cnt_path):
            continue
        for sty_id in STYLES_ALL:
            fname = f"cnt{cnt_id}_sty{sty_id}.png"
            out_path = str(out_dir / fname)
            if os.path.exists(out_path) and not any(
                r["cnt_id"] == cnt_id and r["sty_id"] == sty_id for r in rows):
                sty_path = f"{CWD}/sty/{sty_id}.png"
                csd = compute_csd(out_path, sty_path)
                clip_i = compute_clip_i(out_path, cnt_path)
                rows.append({"cnt_id": cnt_id, "sty_id": sty_id,
                             "csd": f"{csd:.4f}", "clip_i_cnt": f"{clip_i:.4f}", "time_s": "0.0"})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cnt_id", "sty_id", "csd", "clip_i_cnt", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        avg_csd = np.mean([float(r["csd"]) for r in rows])
        avg_clip_i = np.mean([float(r["clip_i_cnt"]) for r in rows])
        avg_time = np.mean([float(r["time_s"]) for r in rows])
        print(f"\n{'='*80}")
        print(f"I2I FULL BENCHMARK AVERAGE: CSD={avg_csd:.4f}  CLIP-I(cnt)={avg_clip_i:.4f}  Time={avg_time:.1f}s  (n={len(rows)})")
        print(f"{'='*80}")
    print(f"\nSaved: {csv_path}")


# ============== Main ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["t2i_ablation", "t2i_baselines", "t2i_full",
                                 "i2i_ablation", "i2i_baselines", "i2i_full", "all"])
    args = parser.parse_args()

    total_start = time.time()

    if args.task in ("t2i_ablation", "all"):
        run_t2i_ablation()
    if args.task in ("t2i_baselines", "all"):
        run_t2i_baselines()
    if args.task in ("t2i_full", "all"):
        run_t2i_full()
    if args.task in ("i2i_ablation", "all"):
        run_i2i_ablation()
    if args.task in ("i2i_baselines", "all"):
        run_i2i_baselines()
    if args.task in ("i2i_full", "all"):
        run_i2i_full()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"TOTAL TIME: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print(f"{'='*80}")
