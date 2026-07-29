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

DIFFSYNTH_ROOT = "third_party/DiffSynth-Studio"
KLEIN9B_ROOT = "models/klein9b"
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
STYLES_NEW6 = os.environ.get("STYLES_OVERRIDE", "01,07,09,22,23,31").split(",")
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
sys.path.insert(0, '../../CSD_Score')
from model import CSD_CLIP, convert_state_dict

csd_dir = '../../CSD_Score'
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
        # "cache" (default): record Q/K/V; "inject": blend attention toward
        # style K/V (real eq:injection baseline, per-image-token output blend)
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
    def set_vshuffle(self, ref_k_list, ref_v_list, beta):
        # V-Shuffle (our MM-DiT adaptation): like inject, but style value tokens
        # are stochastically permuted to sever spatial style-content correspondence.
        self.mode = "vshuffle"
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
                if controller.mode in ("inject", "vshuffle") and controller.inject_idx < len(controller.inject_ref_k):
                    ref_k = controller.inject_ref_k[controller.inject_idx]
                    ref_v = controller.inject_ref_v[controller.inject_idx]
                    if controller.mode == "vshuffle":
                        _perm = torch.randperm(ref_v.shape[2], device=ref_v.device)
                        ref_v = ref_v.index_select(2, _perm)
                    beta = controller.inject_beta
                    _tl = txt_len if txt_len > 0 else controller.txt_len
                    q_img = q_t[:, :, _tl:, :]
                    ref_out = F.scaled_dot_product_attention(
                        q_img, ref_k.to(q_t.dtype), ref_v.to(q_t.dtype))
                    blended = (1.0 - beta) * attn_out_t[:, :, _tl:, :] + beta * ref_out
                    attn_out_t = attn_out_t.clone()
                    attn_out_t[:, :, _tl:, :] = blended
                    controller.inject_idx += 1
                else:
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
                if controller.mode in ("inject", "vshuffle") and controller.inject_idx < len(controller.inject_ref_k):
                    ref_k = controller.inject_ref_k[controller.inject_idx]
                    ref_v = controller.inject_ref_v[controller.inject_idx]
                    if controller.mode == "vshuffle":
                        _perm = torch.randperm(ref_v.shape[2], device=ref_v.device)
                        ref_v = ref_v.index_select(2, _perm)
                    beta = controller.inject_beta
                    _tl = controller.txt_len
                    q_img = q_t[:, :, _tl:, :]
                    ref_out = F.scaled_dot_product_attention(
                        q_img, ref_k.to(q_t.dtype), ref_v.to(q_t.dtype))
                    blended = (1.0 - beta) * attn_out_t[:, :, _tl:, :] + beta * ref_out
                    attn_out_t = attn_out_t.clone()
                    attn_out_t[:, :, _tl:, :] = blended
                    controller.inject_idx += 1
                else:
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
                              alpha=0.5, n_iters=10, tau=0.1, grag_delta=1.0, mode="ot_adain",
                              lat_h=None, lat_w=None, sty_h=None, sty_w=None, smooth_k=5):
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
    if mode == "w2":
        # Closed-form Gaussian-W2 (Monge/Knott-Smith) map: match full mean AND
        # covariance of the style token distribution (vs ot_adain's mean+per-
        # channel-std only). Single global affine -> preserves content high-freq.
        c_f = content_lat.float(); s_f = style_clean_g.float()
        mu_c = c_f.mean(1, keepdim=True); mu_s = s_f.mean(1, keepdim=True)
        Xc = c_f - mu_c; Xs = s_f - mu_s
        eye = torch.eye(D, device=c_f.device, dtype=c_f.dtype).unsqueeze(0)
        Sc = torch.bmm(Xc.transpose(1, 2), Xc) / Xc.shape[1] + 1e-4 * eye
        Ss = torch.bmm(Xs.transpose(1, 2), Xs) / Xs.shape[1] + 1e-4 * eye
        def _mpow(Mx, p):
            w, V = torch.linalg.eigh((Mx + Mx.transpose(-1, -2)) / 2)
            w = w.clamp(min=1e-6).pow(p)
            return V @ torch.diag_embed(w) @ V.transpose(-1, -2)
        Sc_h = _mpow(Sc, 0.5); Sc_ih = _mpow(Sc, -0.5)
        mid_h = _mpow(torch.bmm(Sc_h, torch.bmm(Ss, Sc_h)), 0.5)
        A = torch.bmm(Sc_ih, torch.bmm(mid_h, Sc_ih))         # [B,D,D] symmetric
        transported = torch.bmm(Xc, A) + mu_s
        out = (1.0 - alpha) * c_f + alpha * transported
        if torch.isnan(out).any() or out.abs().max() > 1e4:
            out = c_f
        return out.to(content_lat.dtype)
    if mode == "sw":
        # Sliced-Wasserstein barycentric map: 1D optimal transport along random
        # projections -> matches the FULL distribution (all moments), so it
        # handles non-Gaussian / multimodal colour palettes ot_adain/w2 cannot.
        c_f = content_lat.float(); s_f = style_clean_g.float()
        Pn = 64
        theta = F.normalize(torch.randn(Pn, D, device=c_f.device, dtype=c_f.dtype), dim=1)
        cp = torch.einsum('bnd,pd->bnp', c_f, theta)          # [B,N,P]
        sp = torch.einsum('bmd,pd->bmp', s_f, theta)          # [B,M,P]
        sp_sorted, _ = torch.sort(sp, dim=1)
        Nn = cp.shape[1]; Mm = sp.shape[1]
        ranks = cp.argsort(dim=1).argsort(dim=1)              # [B,N,P] in [0,N-1]
        idx = (ranks.float() / max(Nn - 1, 1) * (Mm - 1)).round().long().clamp(0, Mm - 1)
        mapped = torch.gather(sp_sorted, 1, idx)              # [B,N,P]
        delta = torch.einsum('bnp,pd->bnd', mapped - cp, theta) / Pn
        return (c_f + alpha * delta).to(content_lat.dtype)
    if mode in ("copy", "coherent"):
        # Texture-transport probe: copy REAL style high-freq (not just stats),
        # with a sharp (near-permutation) plan so style brush-strokes survive.
        style_f = style_clean_g.float()                       # [B, M, D]
        M = style_f.shape[1]
        P_sharp = torch.softmax(logits, dim=-1)               # row-only → peaked at tau=0.02
        content_f = content_lat.float()
        if mode == "copy":
            # Sharp barycentric copy — texture present but spatially scrambled
            # (per-token independent: neighbours grab far-apart style tokens).
            transported = torch.bmm(P_sharp, style_f)
        else:
            # Spatially-coherent flow: smooth the *assignment field* (not the
            # values), so neighbouring content tokens sample neighbouring style
            # tokens → style high-freq is retained AND spatially coherent.
            Hs = sty_h or int(round(M ** 0.5)); Ws = sty_w or (M // Hs)
            Hc = lat_h or int(round(N ** 0.5)); Wc = lat_w or (N // Hc)
            rr = torch.arange(Hs, device=content_lat.device, dtype=torch.float32)
            cc = torch.arange(Ws, device=content_lat.device, dtype=torch.float32)
            gy = (2.0 * rr / max(Hs - 1, 1) - 1.0).view(Hs, 1).expand(Hs, Ws).reshape(M)
            gx = (2.0 * cc / max(Ws - 1, 1) - 1.0).view(1, Ws).expand(Hs, Ws).reshape(M)
            sty_xy = torch.stack([gx, gy], dim=1)             # [M, 2] (x,y) in [-1,1]
            flow = torch.bmm(P_sharp, sty_xy.unsqueeze(0).expand(B, M, 2))  # [B, N, 2]
            flow = flow.view(B, Hc, Wc, 2).permute(0, 3, 1, 2)            # [B,2,Hc,Wc]
            pad = smooth_k // 2
            flow = F.avg_pool2d(flow, smooth_k, stride=1, padding=pad)     # spatial smooth
            flow = flow.permute(0, 2, 3, 1)                                # [B,Hc,Wc,2]
            sty_img = style_f.view(B, Hs, Ws, D).permute(0, 3, 1, 2)       # [B,D,Hs,Ws]
            sampled = F.grid_sample(sty_img, flow, mode="bilinear",
                                    padding_mode="border", align_corners=True)  # [B,D,Hc,Wc]
            transported = sampled.permute(0, 2, 3, 1).reshape(B, N, D)
        return ((1.0 - alpha) * content_f + alpha * transported).to(content_lat.dtype)
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

def sliced_wasserstein_feat(x, y, n_proj=48):
    """Differentiable sliced-Wasserstein distance between two token sets.
    x: [B, Lx, D] (grad), y: [B, Ly, D] (target). 1D OT per random projection.
    Used for feature-space OT pre-align in the Adam loop."""
    B, Lx, D = x.shape
    xf = x.float(); yf = y.float()
    theta = F.normalize(torch.randn(n_proj, D, device=x.device, dtype=torch.float32), dim=1)
    xp = torch.matmul(xf, theta.T)            # [B, Lx, P]
    yp = torch.matmul(yf, theta.T)            # [B, Ly, P]
    xs = torch.sort(xp, dim=1).values
    ys = torch.sort(yp, dim=1).values
    if xs.shape[1] != ys.shape[1]:            # resample target to Lx via interp
        ys = F.interpolate(ys.transpose(1, 2), size=xs.shape[1],
                           mode="linear", align_corners=True).transpose(1, 2)
    return (xs - ys).abs().mean()

# ============== CLIP-T-in-loop (plan A) ==============
# Separate CLIP (ViT-L/14) from the eval model (ViT-B-32) to avoid metric-hacking:
# a CLIP-T(ViT-B-32) gain then reflects genuine prompt-alignment, not eval gaming.
_il = {}
_il_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1, 3, 1, 1)
_il_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1, 3, 1, 1)

def _get_il_clip():
    if "m" not in _il:
        from transformers import CLIPModel, CLIPTokenizer
        p = "models/clip-vit-large-patch14"
        _il["m"] = CLIPModel.from_pretrained(p).to("cuda").eval()
        _il["tok"] = CLIPTokenizer.from_pretrained(p)
        for pr in _il["m"].parameters():
            pr.requires_grad_(False)
    return _il["m"], _il["tok"]

def il_text_feat(prompt):
    m, tok = _get_il_clip()
    toks = tok([prompt], return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        return F.normalize(m.get_text_features(**toks).float(), dim=-1)  # [1, D]

def clip_t_inloop_loss(latents, lat_h, lat_w, text_feat):
    """Differentiable CLIP-T loss: decode latents -> image -> ViT-L image feat ->
    1 - cos(img, prompt-text). Backprops to latents through VAE + CLIP."""
    lat4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
    img = vae.decode(lat4d.to(vae.dtype if hasattr(vae, "dtype") else latents.dtype))
    img = img.float()
    img = (img - img.amin()) / (img.amax() - img.amin() + 1e-6)  # robust to VAE output range
    img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
    img = (img - _il_mean) / _il_std
    m, _ = _get_il_clip()
    feat = F.normalize(m.get_image_features(pixel_values=img).float(), dim=-1)
    return (1.0 - (feat * text_feat).sum(-1)).mean()

# ============== T2I Generation ==============
def generate_t2i(prompt, style_path, config):
    """
    config keys:
      mode: "hybrid" | "adam" | "slt" | "inject" | "adain_only" | "none"
        inject     = real attention K/V injection (paper eq:injection),
                     blended during the denoise forward; no AdaIN, no final
                     color transfer
        adain_only = per-step AdaIN warm-start only (the old mislabeled
                     "inject"); color-mechanism decomposition control
      final_adain: bool — override the final adain_1d color transfer
                   (defaults: True except for "none"/"inject")
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
    seed = int(os.environ.get("EVAL_SEED", 43))

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
    _OT_LAYER_FRAC = float(os.environ.get("OT_LAYER_FRAC", "0.5"))

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()

        # Real attention K/V injection (paper eq:injection): extract style K/V
        # at the CURRENT timestep, blend during this step's single denoise
        # forward. One scheduler.step only; no AdaIN anywhere.
        if mode in ("inject", "vshuffle") and t_val < start_time_val:
            if i < num_inference_steps - 1:
                sty_noise_inj = torch.randn_like(style_latent)
                style_latent_inj = scheduler.add_noise(
                    style_latent, sty_noise_inj, timesteps[i:i+1])
            else:
                style_latent_inj = style_latent
            with torch.no_grad():
                _ = dit(
                    hidden_states=style_latent_inj.to(device="cuda", dtype=torch.bfloat16),
                    timestep=timestep / 1000,
                    guidance=None, encoder_hidden_states=null_embeds,
                    txt_ids=null_text_ids, img_ids=style_image_ids)
            _, inj_k_list, inj_v_list, _ = cache.get()
            (controller.set_vshuffle if mode == "vshuffle" else controller.set_inject)(inj_k_list, inj_v_list, inject_beta)
            cache.clear(); controller.step()

        with torch.no_grad():
            noise_pred = dit(
                hidden_states=latents, timestep=timestep / 1000, guidance=None,
                encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=image_ids)
        latents = scheduler.step(noise_pred, t_val, latents)
        if controller.mode in ("inject", "vshuffle"):
            controller.reset_mode()

        if t_val >= start_time_val:
            continue
        if mode in ("none", "inject", "vshuffle"):
            continue

        # Noisy style
        if i < num_inference_steps - 1:
            timestep_style = timesteps[i+1:i+2]
            sty_noise = torch.randn_like(style_latent)
            style_latent_ = scheduler.add_noise(style_latent, sty_noise, timestep_style)
        else:
            timestep_style = torch.tensor([0.0])
            style_latent_ = style_latent

        # Style forward (KSHOT env var enables multi-shot K/V averaging)
        _K = int(os.environ.get("KSHOT", "1"))
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(
                hidden_states=style_latent_.to(device="cuda", dtype=torch.bfloat16),
                timestep=timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000,
                guidance=None, encoder_hidden_states=null_embeds,
                txt_ids=null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, ref_out_list = cache.get()  # ref_out = style's own attn-output distribution (feature-OT target)
        if _K > 1 and i < num_inference_steps - 1:
            for _sh in range(1, _K):
                _sl_aug = scheduler.add_noise(style_latent, torch.randn_like(style_latent), timestep_style)
                cache.clear(); controller.step()
                with torch.no_grad():
                    _ = dit(
                        hidden_states=_sl_aug.to(device="cuda", dtype=torch.bfloat16),
                        timestep=timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000,
                        guidance=None, encoder_hidden_states=null_embeds,
                        txt_ids=null_text_ids, img_ids=style_image_ids)
                _, _k_aug, _v_aug, _ = cache.get()
                for _idx in range(len(ref_k_list)):
                    ref_k_list[_idx] = ref_k_list[_idx] + _k_aug[_idx]
                    ref_v_list[_idx] = ref_v_list[_idx] + _v_aug[_idx]
            for _idx in range(len(ref_k_list)):
                ref_k_list[_idx] = ref_k_list[_idx] / _K
                ref_v_list[_idx] = ref_v_list[_idx] / _K

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

        # AdaIN warm-start — skipped for the texture-transport probes, and
        # toggleable via config["warm_adain"] for the remove-AdaIN ablation.
        if mode not in ("slt_copy", "slt_coherent") and config.get("warm_adain", True):
            with torch.no_grad():
                latents = feature_adain(
                    latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=adain_alpha
                ).squeeze(1)
            latents = latents.detach()

        if mode == "adain_only":
            # Per-step AdaIN warm-start only (the old mislabeled "inject"
            # baseline) — kept as the color-mechanism decomposition control.
            continue

        # SLT pre-alignment (for hybrid mode). slt_mode selects the OT operator:
        # ot_adain (default/legacy) | w2 (full-cov Gaussian) | sw (sliced-W) | blend.
        _slt_mode = config.get("slt_mode", "ot_adain")
        if mode == "hybrid" and slt_alpha > 0:
            with torch.no_grad():
                latents = sinkhorn_latent_transport(
                    latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=config.get("slt_tau", 0.02),
                    grag_delta=grag_deep, mode=_slt_mode)

        if mode == "slt":
            # Pure SLT, no Adam
            with torch.no_grad():
                latents = sinkhorn_latent_transport(
                    latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=config.get("slt_tau", 0.02),
                    grag_delta=grag_deep, mode=_slt_mode)
            continue

        if mode in ("slt_copy", "slt_coherent"):
            # Texture-transport probe: can OT alone carry style texture+colour
            # in one shot? copy=sharp scrambled, coherent=spatially-smoothed flow.
            _tex_mode = "copy" if mode == "slt_copy" else "coherent"
            with torch.no_grad():
                latents = sinkhorn_latent_transport(
                    latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=config.get("slt_tau", 0.02),
                    grag_delta=grag_deep, mode=_tex_mode,
                    lat_h=lat_h, lat_w=lat_w, sty_h=sty_lat_h, sty_w=sty_lat_w,
                    smooth_k=config.get("smooth_k", 5))
            continue

        # Adam optimization (for hybrid and adam modes)
        actual_iters = adam_iters
        if mode == "adam":
            actual_iters = adam_iters

        # feat_pre: feature-space OT pre-align — first feat_pre Adam steps match
        # the content attention-output DISTRIBUTION to the style's (differentiable
        # sliced-Wasserstein, in the space Adam actually optimises) instead of the
        # spatial-L1 target. Tests whether aligning in feature space (vs latent SLT)
        # helps the hybrid end-to-end / reduces the colour-gradient budget (kappa).
        feat_pre = config.get("feat_pre", 0)
        if actual_iters + feat_pre > 0:
            latents.requires_grad_(True)
            optimizer = torch.optim.Adam([latents], lr=lr)
            _ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000
            clip_t_w = config.get("clip_t_w", 0.0)   # plan A: CLIP-T-in-loop (ViT-L, != eval ViT-B-32)
            _txt_feat = il_text_feat(prompt) if clip_t_w > 0 else None
            if clip_t_w > 0:
                pipe.load_models_to_device(["dit", "vae"]); dit.to("cuda")

            for it in range(feat_pre + actual_iters):
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

                if it < feat_pre and ref_out_list:
                    # feature-space OT pre-align (deep/OT layers): match attn-output distribution
                    feat_loss = 0.0
                    for idx2, (so, ro) in enumerate(zip(self_out_list, ref_out_list)):
                        if idx2 < n_layers - n_ot_layers:
                            continue
                        B_, H_, L_, D_ = so.shape
                        feat_loss = feat_loss + sliced_wasserstein_feat(
                            so.reshape(B_, H_ * L_, D_), ro.reshape(1, -1, D_).detach())
                    content_loss = loss_fn(latents, content_anchor)
                    loss = style_w * feat_loss + content_w * content_loss
                else:
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
                if clip_t_w > 0:
                    loss = loss + clip_t_w * clip_t_inloop_loss(latents, lat_h, lat_w, _txt_feat)
                loss.backward()
                optimizer.step()
            latents = latents.detach()

    # Final color transfer — part of OUR pipeline; baselines ("none" pure
    # generation, "inject" faithful eq:injection) must not be color-corrected.
    # Override per-config with "final_adain": True/False (decomposition exps).
    if config.get("final_adain", mode not in ("none", "inject")):
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
    inject_beta = config.get("inject_beta", 0.5)
    start_time_val = config.get("start_time", 870)

    width, height = 512, 512
    num_inference_steps = 25
    seed = int(os.environ.get("EVAL_SEED", 43))

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
    _OT_LAYER_FRAC = float(os.environ.get("OT_LAYER_FRAC", "0.5"))

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()

        # Real attention K/V injection (paper eq:injection), I2I variant:
        # style K/V at the CURRENT timestep, blended during this step's single
        # denoise forward. One scheduler.step only; no AdaIN anywhere.
        if mode in ("inject", "vshuffle") and t_val < start_time_val:
            if i < num_inference_steps - 1:
                sty_noise_inj = torch.randn_like(style_latent)
                style_latent_inj = scheduler.add_noise(
                    style_latent, sty_noise_inj, timesteps[i:i+1])
            else:
                style_latent_inj = style_latent
            with torch.no_grad():
                _ = dit(
                    hidden_states=style_latent_inj.to(device="cuda", dtype=torch.bfloat16),
                    timestep=timestep / 1000,
                    guidance=None, encoder_hidden_states=null_embeds,
                    txt_ids=null_text_ids, img_ids=style_image_ids)
            _, inj_k_list, inj_v_list, _ = cache.get()
            (controller.set_vshuffle if mode == "vshuffle" else controller.set_inject)(inj_k_list, inj_v_list, inject_beta)
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
        if controller.mode in ("inject", "vshuffle"):
            controller.reset_mode()

        if t_val >= start_time_val:
            continue
        if mode in ("none", "inject", "vshuffle"):
            continue

        # Noisy style
        if i < num_inference_steps - 1:
            timestep_style = timesteps[i+1:i+2]
            sty_noise = torch.randn_like(style_latent)
            style_latent_ = scheduler.add_noise(style_latent, sty_noise, timestep_style)
        else:
            timestep_style = torch.tensor([0.0])
            style_latent_ = style_latent

        # Style forward (KSHOT env var enables multi-shot K/V averaging)
        _K = int(os.environ.get("KSHOT", "1"))
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(
                hidden_states=style_latent_.to(device="cuda", dtype=torch.bfloat16),
                timestep=timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000,
                guidance=None, encoder_hidden_states=null_embeds,
                txt_ids=null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, ref_out_list = cache.get()  # ref_out = style's own attn-output distribution (feature-OT target)

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
        if _K > 1 and i < num_inference_steps - 1:
            for _sh in range(1, _K):
                _sl_aug = scheduler.add_noise(style_latent, torch.randn_like(style_latent), timestep_style)
                cache.clear(); controller.step()
                with torch.no_grad():
                    _ = dit(
                        hidden_states=_sl_aug.to(device="cuda", dtype=torch.bfloat16),
                        timestep=timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000,
                        guidance=None, encoder_hidden_states=null_embeds,
                        txt_ids=null_text_ids, img_ids=style_image_ids)
                _, _k_aug, _v_aug, _ = cache.get()
                for _idx in range(len(ref_k_list)):
                    ref_k_list[_idx] = ref_k_list[_idx] + _k_aug[_idx]
                    ref_v_list[_idx] = ref_v_list[_idx] + _v_aug[_idx]
            for _idx in range(len(ref_k_list)):
                ref_k_list[_idx] = ref_k_list[_idx] / _K
                ref_v_list[_idx] = ref_v_list[_idx] / _K

        # AdaIN
        with torch.no_grad():
            latents = feature_adain(
                latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=adain_alpha
            ).squeeze(1)
        latents = latents.detach()

        if mode == "adain_only":
            # Per-step AdaIN warm-start only (old mislabeled "inject").
            continue

        if mode == "hybrid" and slt_alpha > 0:
            with torch.no_grad():
                latents = sinkhorn_latent_transport(
                    latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=config.get("slt_tau", 0.02),
                    grag_delta=grag_deep, mode=config.get("slt_mode", "ot_adain"))

        if mode == "slt":
            with torch.no_grad():
                latents = sinkhorn_latent_transport(
                    latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=config.get("slt_tau", 0.02),
                    grag_delta=grag_deep, mode=config.get("slt_mode", "ot_adain"))
            continue

        # Adam
        actual_iters = adam_iters
        # feat_pre: feature-space OT pre-align — first feat_pre Adam steps match
        # the content attention-output DISTRIBUTION to the style's (differentiable
        # sliced-Wasserstein, in the space Adam actually optimises) instead of the
        # spatial-L1 target. Tests whether aligning in feature space (vs latent SLT)
        # helps the hybrid end-to-end / reduces the colour-gradient budget (kappa).
        feat_pre = config.get("feat_pre", 0)
        if actual_iters + feat_pre > 0:
            latents.requires_grad_(True)
            optimizer = torch.optim.Adam([latents], lr=lr)
            _ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000
            clip_t_w = config.get("clip_t_w", 0.0)   # plan A: CLIP-T-in-loop (ViT-L, != eval ViT-B-32)
            _txt_feat = il_text_feat(prompt) if clip_t_w > 0 else None
            if clip_t_w > 0:
                pipe.load_models_to_device(["dit", "vae"]); dit.to("cuda")

            for it in range(feat_pre + actual_iters):
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

                if it < feat_pre and ref_out_list:
                    # feature-space OT pre-align (deep/OT layers): match attn-output distribution
                    feat_loss = 0.0
                    for idx2, (so, ro) in enumerate(zip(self_out_list, ref_out_list)):
                        if idx2 < n_layers - n_ot_layers:
                            continue
                        B_, H_, L_, D_ = so.shape
                        feat_loss = feat_loss + sliced_wasserstein_feat(
                            so.reshape(B_, H_ * L_, D_), ro.reshape(1, -1, D_).detach())
                    content_loss = loss_fn(latents, content_anchor)
                    loss = style_w * feat_loss + content_w * content_loss
                else:
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
                if clip_t_w > 0:
                    loss = loss + clip_t_w * clip_t_inloop_loss(latents, lat_h, lat_w, _txt_feat)
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
    "inject_b10": {"mode": "inject", "inject_beta": 1.0},
    "inject_b05": {"mode": "inject", "inject_beta": 0.5},
    "adain_only": {"mode": "adain_only"},
    # V-Shuffle (our MM-DiT adaptation): inject with permuted style value tokens.
    "vshuffle": {"mode": "vshuffle", "inject_beta": 0.5},
}

# Color-mechanism decomposition (EXPERIMENTS_NEEDED.md 2026-06-12 #16):
# isolate each color-alignment layer's contribution on the 30-pair subset.
T2I_DECOMP = {
    "bare":        {"mode": "none", "final_adain": False},  # pure generation
    "final_only":  {"mode": "none", "final_adain": True},   # + final adain_1d
    "adain_step":  {"mode": "adain_only"},                  # + per-step AdaIN
    "slt_only":    {**HYBRID_DEFAULT, "mode": "slt", "iters": 0},  # + OT-AdaIN
    "no_slt":      {**HYBRID_DEFAULT, "slt_alpha": 0.0},    # AdaIN + Adam-2
    "hybrid":      HYBRID_DEFAULT,                          # full pipeline
}

# Texture-transport probe: can OT alone carry style texture+colour in one shot
# (no Adam, no AdaIN warm-start)? copy=sharp scrambled, coherent=smoothed flow.
_TEX_BASE = {"mode": "slt", "iters": 0, "grag_deep": 0.5, "grag_shallow": 0.7,
             "final_adain": False}
T2I_TEXTURE = {
    "tex_copy_a07":     {**_TEX_BASE, "mode": "slt_copy",     "slt_alpha": 0.7},
    "tex_coherent_a07": {**_TEX_BASE, "mode": "slt_coherent", "slt_alpha": 0.7, "smooth_k": 5},
    "tex_coherent_a10": {**_TEX_BASE, "mode": "slt_coherent", "slt_alpha": 1.0, "smooth_k": 5},
    "tex_coherent_pol": {**_TEX_BASE, "mode": "slt_coherent", "slt_alpha": 0.7, "smooth_k": 5,
                         "final_adain": True},   # OT texture + final colour polish
}

# SLT-operator optimisation: compare colour-pre-align operators head-to-head.
# All share warm-start + final AdaIN (real pipeline) so the delta isolates the
# OT operator. ot_adain = current/legacy; w2 = full-cov Gaussian-W2 (Monge);
# sw = sliced-Wasserstein (full distribution); blend = paper eq:slt direct transport.
T2I_SLTOPT = {
    # standalone SLT (no Adam) — colour-alignment quality
    "slt_otadain":   {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "ot_adain"},
    "slt_blend":     {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "blend"},
    "slt_w2":        {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "w2"},
    "slt_sw":        {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "sw"},
    "slt_w2_a04":    {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "w2", "slt_alpha": 0.4},
    "slt_sw_a04":    {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "sw", "slt_alpha": 0.4},
    # hybrid (SLT pre-align + Adam-2) — end-to-end quality, the money test
    "hyb_otadain":   {**HYBRID_DEFAULT, "slt_mode": "ot_adain"},
    "hyb_w2":        {**HYBRID_DEFAULT, "slt_mode": "w2"},
    "hyb_sw":        {**HYBRID_DEFAULT, "slt_mode": "sw"},
}

# Follow-up: W2 vs ot_adain hybrid pre-align across SLT strength alpha — does a
# stronger (full-covariance) pre-align push end-to-end CSD past the 0.685 baseline?
T2I_SLTOPT2 = {
    "hyb_ota_a02": {**HYBRID_DEFAULT, "slt_mode": "ot_adain", "slt_alpha": 0.2},
    "hyb_ota_a03": {**HYBRID_DEFAULT, "slt_mode": "ot_adain", "slt_alpha": 0.3},
    "hyb_ota_a04": {**HYBRID_DEFAULT, "slt_mode": "ot_adain", "slt_alpha": 0.4},
    "hyb_w2_a02":  {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2},
    "hyb_w2_a03":  {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.3},
    "hyb_w2_a04":  {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.4},
}

# Feature-space OT (B): align attention-output DISTRIBUTION (sliced-W) in the
# space Adam optimises, vs latent SLT. feat_pre steps replace the latent pre-align.
# adam3/adam4 are iso-cost pure-spatial-L1 controls (same #forwards, same params).
T2I_FEATOT = {
    "hyb_base":    HYBRID_DEFAULT,                                              # slt(latent)+adam2 = 2 fwd
    "featot_p1":   {**HYBRID_DEFAULT, "slt_alpha": 0.0, "feat_pre": 1},         # 1 feat-OT + adam2 = 3 fwd
    "featot_p2":   {**HYBRID_DEFAULT, "slt_alpha": 0.0, "feat_pre": 2},         # 2 feat-OT + adam2 = 4 fwd
    "featot_only": {**HYBRID_DEFAULT, "slt_alpha": 0.0, "feat_pre": 2, "iters": 0},  # feat-OT only = 2 fwd
    "adam3":       {**HYBRID_DEFAULT, "mode": "adam", "iters": 3, "slt_alpha": 0.0},  # iso-cost ctrl for p1
    "adam4":       {**HYBRID_DEFAULT, "mode": "adam", "iters": 4, "slt_alpha": 0.0},  # iso-cost ctrl for p2
}

# (1) new SLT(W2)+Adam vs old(ot_adain)+Adam, and (2) remove-AdaIN ablation:
# can the W2 OT operator replace the (undisclosed) AdaIN warm-start + final
# adain_1d colour transfer? warm_adain/final_adain toggle the two AdaIN layers.
T2I_SLTADAM = {
    "ota_a01":     {**HYBRID_DEFAULT, "slt_mode": "ot_adain"},                       # baseline (≈0.685)
    "w2_a01":      {**HYBRID_DEFAULT, "slt_mode": "w2"},                             # new SLT+Adam
    "w2_a03":      {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.3},           # stronger W2 pre-align
    "w2_noWarm":   {**HYBRID_DEFAULT, "slt_mode": "w2", "warm_adain": False},        # drop warm-start AdaIN
    "w2_noFinal":  {**HYBRID_DEFAULT, "slt_mode": "w2", "final_adain": False},       # drop final adain_1d
    "w2_noAdain":  {**HYBRID_DEFAULT, "slt_mode": "w2", "warm_adain": False, "final_adain": False},  # W2 carries all colour
    "ota_noAdain": {**HYBRID_DEFAULT, "slt_mode": "ot_adain", "warm_adain": False, "final_adain": False},  # control
}

# Complete the W2-vs-ot_adain Pareto grid at n=24 (missing cells; others already
# in t2i_sltadam.csv: ota_a01, w2_a01, w2_a03). Answers: is W2 Pareto-dominant?
T2I_SLTPARETO = {
    "ota_a02": {**HYBRID_DEFAULT, "slt_mode": "ot_adain", "slt_alpha": 0.2},
    "ota_a03": {**HYBRID_DEFAULT, "slt_mode": "ot_adain", "slt_alpha": 0.3},
    "w2_a02":  {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2},
}

# Constrained objective: max CSD s.t. CLIP-T >= 0.3. content_w anchors to the
# PROMPT-faithful pre-Adam latent, so raising it pulls CLIP-T back up. Map the
# CLIP-T>=0.3 frontier via content_w / iters / standalone-W2.
T2I_CLIPT = {
    "w2slt_a02":  {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "w2", "slt_alpha": 0.2},
    "w2slt_a03":  {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "w2", "slt_alpha": 0.3},
    "w2_cw3":     {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2, "content_w": 3.0},
    "w2_cw6":     {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2, "content_w": 6.0},
    "w2_it1_cw3": {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2, "iters": 1, "content_w": 3.0},
    "ota_cw3":    {**HYBRID_DEFAULT, "slt_mode": "ot_adain", "slt_alpha": 0.2, "content_w": 3.0},
}

# Plan A: CLIP-T-in-loop. Take the high-CSD point (W2 α0.2 = 0.72/0.25) and add a
# differentiable CLIP-T term (ViT-L, != eval ViT-B-32) to pull CLIP-T back to >=0.3
# without sacrificing CSD. Sweep the weight.
T2I_CLIPTLOOP = {
    "w2_ct0":  {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2, "clip_t_w": 0.0},
    "w2_ct2":  {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2, "clip_t_w": 2.0},
    "w2_ct5":  {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2, "clip_t_w": 5.0},
    "w2_ct10": {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2, "clip_t_w": 10.0},
}

# Relaxed target CLIP-T >= 0.28: exploit that dropping AdaIN raises CLIP-T
# (w2_noAdain 0.647@0.267) — W2 + Adam, no AdaIN, nudge to 0.28 via lighter Adam /
# light CLIP-T-in-loop, to keep CSD ~0.6 (vs 0.48 SLT-only at the 0.28 band).
_W2NA = {**HYBRID_DEFAULT, "slt_mode": "w2", "warm_adain": False, "final_adain": False}
T2I_RELAX = {
    "w2nA_a01":     {**_W2NA},                                         # α0.1, adam2 (≈0.647@0.267)
    "w2nA_a01_it1": {**_W2NA, "iters": 1},                             # lighter Adam → +CLIP-T
    "w2nA_a01_ct1": {**_W2NA, "clip_t_w": 1.0},                        # light CLIP-T nudge
    "w2nA_a02":     {**_W2NA, "slt_alpha": 0.2},                       # stronger pre-align
    "w2nA_a02_it1": {**_W2NA, "slt_alpha": 0.2, "iters": 1},
    "w2nW_a02":     {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.2, "warm_adain": False},  # drop warm only
}

# Full CLIP-T>=0.28 frontier grid (n=40), split across 2 parallel workers.
def _build_relaxbig():
    g = {}
    na = {"slt_mode": "w2", "warm_adain": False, "final_adain": False}
    for a in (0.1, 0.2, 0.3):
        for it in (1, 2):
            g[f"nA_a{int(a*10)}_it{it}"] = {**HYBRID_DEFAULT, **na, "slt_alpha": a, "iters": it}
    g["nA_a2_it2_ct1"] = {**HYBRID_DEFAULT, **na, "slt_alpha": 0.2, "clip_t_w": 1.0}
    g["nA_a2_it2_ct2"] = {**HYBRID_DEFAULT, **na, "slt_alpha": 0.2, "clip_t_w": 2.0}
    g["nW_a2"] = {**HYBRID_DEFAULT, "slt_mode": "w2", "warm_adain": False, "slt_alpha": 0.2}
    g["nW_a3"] = {**HYBRID_DEFAULT, "slt_mode": "w2", "warm_adain": False, "slt_alpha": 0.3}
    return g
T2I_RELAXBIG = _build_relaxbig()

# Cheap P1 param-sweeps (all existing config keys): #8 LRS-off, #10 SLT τ-sweep,
# #11 t_start sensitivity, #17 blend α-sweep. All on hybrid (ot_adain) defaults.
T2I_P1CHEAP = {
    # #8 LRS/GRAG ablation
    "lrs_off":   {**HYBRID_DEFAULT, "grag_deep": 1.0, "grag_shallow": 1.0},
    # #10 SLT tau sweep (0.02 is default/main)
    "tau_001":   {**HYBRID_DEFAULT, "slt_tau": 0.01},
    "tau_005":   {**HYBRID_DEFAULT, "slt_tau": 0.05},
    "tau_015":   {**HYBRID_DEFAULT, "slt_tau": 0.15},
    # #11 t_start sensitivity (850 is default)
    "tstart_750": {**HYBRID_DEFAULT, "start_time": 750},
    "tstart_800": {**HYBRID_DEFAULT, "start_time": 800},
    "tstart_900": {**HYBRID_DEFAULT, "start_time": 900},
    # #17 blend (eq:slt original) α-sweep (vs ot_adain)
    "blend_a005": {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "blend", "slt_alpha": 0.05},
    "blend_a02":  {**HYBRID_DEFAULT, "mode": "slt", "iters": 0, "slt_mode": "blend", "slt_alpha": 0.2},
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
    "inject_b10": {**I2I_HYBRID, "mode": "inject", "inject_beta": 1.0},
    "inject_b05": {**I2I_HYBRID, "mode": "inject", "inject_beta": 0.5},
    "adain_only": {**I2I_HYBRID, "mode": "adain_only"},
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


def run_t2i_new6styles():
    """Supplementary evaluation: hybrid2 on 6 new diverse styles."""
    out_dir = Path(CWD) / "d_grand" / "t2i_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "t2i_new6styles.csv"

    rows = []
    total = len(T2I_PROMPTS_30) * len(STYLES_NEW6)
    print(f"\n{'='*80}")
    print(f"T2I NEW6STYLES: hybrid2 × {len(T2I_PROMPTS_30)} prompts × {len(STYLES_NEW6)} styles = {total} images")
    print(f"{'='*80}")
    print(f"{'Prompt':<35} {'Sty':>3} {'CSD':>7} {'CLIP-T':>7} {'Time':>6}")
    print("-" * 65)

    for pid, prompt in T2I_PROMPTS_30:
        for sty_id in STYLES_NEW6:
            sty_path = f"{CWD}/sty/{sty_id}.png"
            torch.cuda.empty_cache()
            t0 = time.time()
            img = generate_t2i(prompt, sty_path, HYBRID_DEFAULT)
            elapsed = time.time() - t0

            fname = f"hybrid2_{pid}_s{sty_id}.png"
            img.save(out_dir / fname)

            csd = compute_csd(str(out_dir / fname), sty_path)
            clip_t = compute_clip_t(str(out_dir / fname), prompt)

            rows.append({"method": "hybrid2", "sty_id": sty_id, "content_id": pid,
                         "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}", "time_s": f"{elapsed:.1f}"})
            print(f"{prompt:<35} {sty_id:>3} {csd:>7.4f} {clip_t:>7.4f} {elapsed:>6.1f}")
            sys.stdout.flush()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    avg_csd = np.mean([float(r["csd"]) for r in rows])
    avg_clip_t = np.mean([float(r["clip_t"]) for r in rows])
    print(f"\nhybrid2 new6: n={len(rows)}  CSD={avg_csd:.4f}  CLIP-T={avg_clip_t:.4f}")
    print(f"(reference: original 8-style CSD=0.6810  CLIP-T=0.2737)")
    print(f"Saved: {csv_path}")


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


def run_t2i_decomp():
    """Color-mechanism decomposition (#16): 30 pairs × 6 configs."""
    out_dir = Path(CWD) / "d_grand" / "t2i_decomp"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "t2i_decomp.csv"

    rows = []
    print(f"\n{'='*80}")
    print(f"T2I COLOR DECOMPOSITION: {len(T2I_DECOMP)} configs × 30 pairs")
    print(f"{'='*80}")

    for method_name, config in T2I_DECOMP.items():
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

    for method_name in T2I_DECOMP:
        m_rows = [r for r in rows if r["method"] == method_name]
        avg_csd = np.mean([float(r["csd"]) for r in m_rows])
        avg_clip_t = np.mean([float(r["clip_t"]) for r in m_rows])
        print(f"  {method_name:<12} avg CSD={avg_csd:.4f}  CLIP-T={avg_clip_t:.4f}")
    print(f"\nSaved: {csv_path}")


# Ultimate-method full-benchmark validation (separate CSVs, original baseline kept).
# ULT configs are the W2 / no-AdaIN candidates to verify on the full 234-pair set.
ULT_T2I = {
    "ult_w2_keepA": {**HYBRID_DEFAULT, "slt_mode": "w2"},                                       # operator swap only
    "ult_w2nA_a1":  {**HYBRID_DEFAULT, "slt_mode": "w2", "warm_adain": False, "final_adain": False},            # W2, no AdaIN, α0.1, adam2
    "ult_w2nA_a2":  {**HYBRID_DEFAULT, "slt_mode": "w2", "warm_adain": False, "final_adain": False, "slt_alpha": 0.2},
    "ult_w2nW_a2":  {**HYBRID_DEFAULT, "slt_mode": "w2", "warm_adain": False, "slt_alpha": 0.2},  # drop warm only
}

# Micro-tune around ult_w2_keepA (best non-regression: +0.8% CSD, -0.6% CLIP-T on
# full 234) — trade a sliver of the CSD headroom back to recover CLIP-T to >=0.2756
# for a TRUE uniform (both-axis) non-regression vs original 0.6405/0.2756.
T2I_MICROTUNE = {
    "mt_base":  {**HYBRID_DEFAULT, "slt_mode": "w2"},                          # = ult_w2_keepA ref
    "mt_sw18":  {**HYBRID_DEFAULT, "slt_mode": "w2", "style_w": 1.8},
    "mt_sw16":  {**HYBRID_DEFAULT, "slt_mode": "w2", "style_w": 1.6},
    "mt_a008":  {**HYBRID_DEFAULT, "slt_mode": "w2", "slt_alpha": 0.08},
    "mt_cw20":  {**HYBRID_DEFAULT, "slt_mode": "w2", "content_w": 2.0},
}

def run_t2i_fullcfg(configs, tag):
    """Run given configs on the FULL T2I benchmark (STYLES_ALL × T2I_PROMPTS_FULL),
    writing to a separate CSV so the original baseline (t2i_full_benchmark.csv) is kept."""
    out_dir = Path(CWD) / "d_grand" / f"t2i_full_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"t2i_full_{tag}.csv"
    total = len(configs) * len(T2I_PROMPTS_FULL) * len(STYLES_ALL)
    rows = []
    print(f"\n{'='*80}\nT2I FULL [{tag}]: {len(configs)} configs × {len(T2I_PROMPTS_FULL)}p × {len(STYLES_ALL)}s = {total}\n{'='*80}", flush=True)
    count = 0
    for method_name, config in configs.items():
        for pid, prompt in T2I_PROMPTS_FULL:
            for sty_id in STYLES_ALL:
                sty_path = f"{CWD}/sty/{sty_id}.png"
                out_path = str(out_dir / f"{method_name}_{pid}_s{sty_id}.png")
                if os.path.exists(out_path):
                    count += 1; continue
                torch.cuda.empty_cache()
                t0 = time.time(); img = generate_t2i(prompt, sty_path, config); elapsed = time.time() - t0
                img.save(out_path)
                csd = compute_csd(out_path, sty_path); clip_t = compute_clip_t(out_path, prompt)
                rows.append({"method": method_name, "sty_id": sty_id, "content_id": pid,
                             "csd": f"{csd:.4f}", "clip_t": f"{clip_t:.4f}", "time_s": f"{elapsed:.1f}"})
                count += 1
                if count % 10 == 0:
                    print(f"  {method_name:<14} {pid} s{sty_id} CSD={csd:.4f} T={clip_t:.4f} [{count}/{total}]", flush=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t", "time_s"]); w.writeheader(); w.writerows(rows)
    print(f"\n  {'config':<14}{'CSD':>8}{'CLIP-T':>8}")
    for mn in configs:
        mr = [r for r in rows if r["method"] == mn]
        if mr:
            print(f"  {mn:<14}{np.mean([float(r['csd']) for r in mr]):>8.4f}{np.mean([float(r['clip_t']) for r in mr]):>8.4f}")
    print(f"Saved: {csv_path}", flush=True)


def run_i2i_fullcfg(configs, tag):
    """Run given configs on the FULL I2I benchmark (I2I_CNTS_ALL × STYLES_ALL),
    separate CSV so original i2i_full_benchmark.csv is kept. CSD + CLIP-I(content)."""
    out_dir = Path(CWD) / "d_grand" / f"i2i_full_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"i2i_full_{tag}.csv"
    rows = []
    print(f"\n{'='*80}\nI2I FULL [{tag}]: {len(configs)} cfg × {len(I2I_CNTS_ALL)}cnt × {len(STYLES_ALL)}sty\n{'='*80}", flush=True)
    count = 0
    for method_name, config in configs.items():
        for cnt_id in I2I_CNTS_ALL:
            cnt_path = f"{CWD}/cnt/{cnt_id}.png"
            if not os.path.exists(cnt_path):
                continue
            for sty_id in STYLES_ALL:
                sty_path = f"{CWD}/sty/{sty_id}.png"
                out_path = str(out_dir / f"{method_name}_c{cnt_id}_s{sty_id}.png")
                if os.path.exists(out_path):
                    count += 1; continue
                torch.cuda.empty_cache()
                t0 = time.time(); img = generate_i2i(cnt_path, sty_path, "masterpiece", config); elapsed = time.time() - t0
                img.save(out_path)
                csd = compute_csd(out_path, sty_path); clip_i = compute_clip_i(out_path, cnt_path)
                rows.append({"method": method_name, "cnt_id": cnt_id, "sty_id": sty_id,
                             "csd": f"{csd:.4f}", "clip_i_cnt": f"{clip_i:.4f}", "time_s": f"{elapsed:.1f}"})
                count += 1
                if count % 20 == 0:
                    print(f"  {method_name:<12} c{cnt_id} s{sty_id} CSD={csd:.4f} CLIP-I={clip_i:.4f} [{count}]", flush=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "cnt_id", "sty_id", "csd", "clip_i_cnt", "time_s"]); w.writeheader(); w.writerows(rows)
    print(f"\n  {'config':<12}{'CSD':>8}{'CLIP-I':>8}")
    for mn in configs:
        mr = [r for r in rows if r["method"] == mn]
        if mr:
            print(f"  {mn:<12}{np.mean([float(r['csd']) for r in mr]):>8.4f}{np.mean([float(r['clip_i_cnt']) for r in mr]):>8.4f}")
    print(f"Saved: {csv_path}", flush=True)

ULT_I2I = {"ult_w2_keepA": {**I2I_HYBRID, "slt_mode": "w2"}}

def run_t2i_texture():
    """Texture-transport probe: can OT carry texture+colour in one shot?
    3 prompts x 8 styles x 4 configs. Compares against decomp's slt_only."""
    out_dir = Path(CWD) / "d_grand" / "t2i_texture"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "t2i_texture.csv"
    prompts = T2I_PROMPTS_30[:3]

    rows = []
    print(f"\n{'='*80}")
    print(f"T2I TEXTURE PROBE: {len(T2I_TEXTURE)} configs x {len(prompts)} prompts x {len(STYLES_6)} styles")
    print(f"{'='*80}")

    for method_name, config in T2I_TEXTURE.items():
        for pid, prompt in prompts:
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
                print(f"  {method_name:<18} {pid} s{sty_id} CSD={csd:.4f} CLIP-T={clip_t:.4f} {elapsed:.1f}s")
                sys.stdout.flush()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  {'config':<18} {'CSD':>8} {'CLIP-T':>8}")
    for method_name in T2I_TEXTURE:
        m_rows = [r for r in rows if r["method"] == method_name]
        avg_csd = np.mean([float(r["csd"]) for r in m_rows])
        avg_clip_t = np.mean([float(r["clip_t"]) for r in m_rows])
        print(f"  {method_name:<18} {avg_csd:>8.4f} {avg_clip_t:>8.4f}")
    print(f"\nSaved: {csv_path}")


def run_t2i_sltopt(configs=T2I_SLTOPT, tag="sltopt"):
    """SLT-operator optimisation: ot_adain vs w2 vs sw vs blend, standalone and
    as hybrid pre-align. 8 styles × 5 prompts = 40/config. Target: 提点 over ot_adain."""
    out_dir = Path(CWD) / "d_grand" / f"t2i_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"t2i_{tag}.csv"

    # Quick subset for eyeballing: SLTOPT_QUICK=1 → 2 styles × 2 prompts.
    _ns = int(os.environ.get("SLTOPT_NS", "2" if os.environ.get("SLTOPT_QUICK") else str(len(STYLES_6))))
    _np = int(os.environ.get("SLTOPT_NP", "2" if os.environ.get("SLTOPT_QUICK") else str(len(T2I_PROMPTS_30))))
    _styles = STYLES_6[:_ns]; _prompts = T2I_PROMPTS_30[:_np]

    rows = []
    print(f"\n{'='*80}")
    print(f"T2I {tag.upper()}: {len(configs)} configs × {len(_prompts)} prompts × {len(_styles)} styles")
    print(f"{'='*80}")

    for method_name, config in configs.items():
        for pid, prompt in _prompts:
            for sty_id in _styles:
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
                print(f"  {method_name:<14} {pid} s{sty_id} CSD={csd:.4f} CLIP-T={clip_t:.4f} {elapsed:.1f}s")
                sys.stdout.flush()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sty_id", "content_id", "csd", "clip_t", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  {'config':<14} {'CSD':>8} {'CLIP-T':>8}")
    for method_name in configs:
        m_rows = [r for r in rows if r["method"] == method_name]
        avg_csd = np.mean([float(r["csd"]) for r in m_rows])
        avg_clip_t = np.mean([float(r["clip_t"]) for r in m_rows])
        print(f"  {method_name:<14} {avg_csd:>8.4f} {avg_clip_t:>8.4f}")
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
    # Accepts a single task, "all", or a comma-separated ordered list
    # (e.g. "t2i_ablation,i2i_ablation,t2i_full") so one process can run a
    # chosen pipeline of tasks with a single model load (parallel-worker use).
    parser.add_argument("--task", default="all")
    args = parser.parse_args()

    total_start = time.time()

    _ALL = ["t2i_baselines", "t2i_decomp", "t2i_ablation",
            "i2i_baselines", "i2i_ablation", "t2i_full", "i2i_full"]
    _DISPATCH = {
        "t2i_baselines": run_t2i_baselines,
        "t2i_decomp": run_t2i_decomp,
        "t2i_texture": run_t2i_texture,
        "t2i_sltopt": run_t2i_sltopt,
        "t2i_sltopt2": lambda: run_t2i_sltopt(T2I_SLTOPT2, "sltopt2"),
        "t2i_featot": lambda: run_t2i_sltopt(T2I_FEATOT, "featot"),
        "t2i_sltadam": lambda: run_t2i_sltopt(T2I_SLTADAM, "sltadam"),
        "t2i_sltpareto": lambda: run_t2i_sltopt(T2I_SLTPARETO, "sltpareto"),
        "t2i_clipt": lambda: run_t2i_sltopt(T2I_CLIPT, "clipt"),
        "t2i_cliptloop": lambda: run_t2i_sltopt(T2I_CLIPTLOOP, "cliptloop"),
        "t2i_relax": lambda: run_t2i_sltopt(T2I_RELAX, "relax"),
        "t2i_relaxA": lambda: run_t2i_sltopt({k: v for i, (k, v) in enumerate(T2I_RELAXBIG.items()) if i % 2 == 0}, "relaxA"),
        "t2i_relaxB": lambda: run_t2i_sltopt({k: v for i, (k, v) in enumerate(T2I_RELAXBIG.items()) if i % 2 == 1}, "relaxB"),
        "t2i_fullA": lambda: run_t2i_fullcfg({"ult_w2_keepA": ULT_T2I["ult_w2_keepA"], "ult_w2nA_a2": ULT_T2I["ult_w2nA_a2"]}, "ultA"),
        "t2i_fullB": lambda: run_t2i_fullcfg({"ult_w2nA_a1": ULT_T2I["ult_w2nA_a1"], "ult_w2nW_a2": ULT_T2I["ult_w2nW_a2"]}, "ultB"),
        "t2i_p1cheap": lambda: run_t2i_sltopt(T2I_P1CHEAP, "p1cheap"),
        "t2i_microtune": lambda: run_t2i_sltopt(T2I_MICROTUNE, "microtune"),
        "t2i_microtune_full": lambda: run_t2i_fullcfg(T2I_MICROTUNE, "microtune_full"),
        "t2i_confirm": lambda: run_t2i_fullcfg({"mt_a008": T2I_MICROTUNE["mt_a008"], "mt_sw18": T2I_MICROTUNE["mt_sw18"]}, "confirm"),
        "i2i_fullult": lambda: run_i2i_fullcfg(ULT_I2I, "ult"),
        "t2i_ablation": run_t2i_ablation,
        "t2i_new6styles": run_t2i_new6styles,
        "t2i_kshot2": lambda: run_t2i_sltopt({"kshot2": HYBRID_DEFAULT}, "kshot2"),
        "i2i_baselines": run_i2i_baselines,
        "i2i_ablation": run_i2i_ablation,
        "t2i_full": run_t2i_full,
        "i2i_full": run_i2i_full,
    }
    if args.task == "all":
        task_list = _ALL
    else:
        task_list = [t.strip() for t in args.task.split(",") if t.strip()]
    for _t in task_list:
        if _t not in _DISPATCH:
            raise SystemExit(f"unknown task: {_t}  (valid: {', '.join(_ALL)}, all)")
    for _t in task_list:
        print(f"\n########## TASK: {_t} ##########", flush=True)
        _DISPATCH[_t]()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"TOTAL TIME: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print(f"{'='*80}")
