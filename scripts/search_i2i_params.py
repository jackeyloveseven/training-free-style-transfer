"""
Quick I2I parameter search on Klein 9B.
Uses cnt/00 × sty/27 as test pair.
Sweeps: start_time, adain_alpha, content_w, slt_alpha, hybrid_iters
"""
import glob, os, sys, time, types
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

# ============== Load eval script functions ==============
# We reuse eval_all.py's infrastructure
sys.path.insert(0, CWD)

# Minimal DiffSynth setup
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

# ============== CSD ==============
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

def compute_csd(output_path, style_path):
    t_out = transforms.ToTensor()(Image.open(output_path).convert('RGB')).unsqueeze(0).to('cuda')
    t_out = F.interpolate(t_out, size=(224,224), mode='bilinear', align_corners=False)
    t_out = (t_out - _mean) / _std
    t_sty = transforms.ToTensor()(Image.open(style_path).convert('RGB')).unsqueeze(0).to('cuda')
    t_sty = F.interpolate(t_sty, size=(224,224), mode='bilinear', align_corners=False)
    t_sty = (t_sty - _mean) / _std
    with torch.no_grad():
        _, _, s_out = csd_model(t_out.float())
        _, _, s_sty = csd_model(t_sty.float())
    return (F.normalize(s_out, dim=-1) * F.normalize(s_sty, dim=-1)).sum().item()

def compute_clip_i(output_path, ref_path):
    preproc = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    t1 = preproc(Image.open(output_path).convert('RGB')).unsqueeze(0).to('cuda')
    t2 = preproc(Image.open(ref_path).convert('RGB')).unsqueeze(0).to('cuda')
    with torch.no_grad():
        f1 = clip_model_vit.encode_image(t1)
        f2 = clip_model_vit.encode_image(t2)
    return (F.normalize(f1, dim=-1) * F.normalize(f2, dim=-1)).sum().item()

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
    def make_double_hook(original_block_attn):
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

    def make_single_hook(original_attn):
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

# ============== Utilities ==============
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

def adain_1d(source, target, eps=1e-6):
    s_mean, s_std = source.mean(dim=1, keepdim=True), source.std(dim=1, keepdim=True)
    t_mean, t_std = target.mean(dim=(0,1), keepdim=True), target.std(dim=(0,1), keepdim=True)
    return (source - s_mean) / (s_std + eps) * t_std + t_mean

def sinkhorn_latent_transport(content_lat, style_lat_noisy, style_lat_clean,
                              alpha=0.5, n_iters=10, tau=0.1, grag_delta=1.0, mode="ot_adain"):
    B, N, D = content_lat.shape
    if grag_delta < 1.0:
        sm = style_lat_noisy.mean(dim=1, keepdim=True)
        style_lat_noisy = sm + grag_delta*(style_lat_noisy - sm)
        scm = style_lat_clean.mean(dim=1, keepdim=True)
        style_lat_clean = scm + grag_delta*(style_lat_clean - scm)
    c_f, s_f = content_lat.float(), style_lat_noisy.float()
    c_hi, s_hi = c_f - c_f.mean(1, keepdim=True), s_f - s_f.mean(1, keepdim=True)
    logits = torch.bmm(F.normalize(c_hi, dim=-1), F.normalize(s_hi, dim=-1).transpose(1,2)) / tau
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    T = torch.softmax(log_P, dim=-1)
    style_f = style_lat_clean.float()
    matched_mean = torch.bmm(T, style_f)
    matched_std = (torch.bmm(T, style_f**2) - matched_mean**2).clamp(min=1e-6).sqrt()
    c_mean = c_f.mean(1, keepdim=True)
    c_std = c_f.std(1, keepdim=True).clamp(min=1e-6)
    styled = (c_f - c_mean)/c_std * matched_std + matched_mean
    return ((1-alpha)*c_f + alpha*styled).to(content_lat.dtype)

# ============== Encode prompts ==============
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
prompt_embeds, text_ids = encode_prompt("masterpiece")

# ============== I2I Generate ==============
def generate_i2i(cnt_path, sty_path, config):
    start_time_val = config.get("start_time", 870)
    adain_alpha = config.get("adain_alpha", 0.3)
    slt_alpha = config.get("slt_alpha", 0.1)
    hybrid_iters = config.get("iters", 2)
    lr = config.get("lr", 0.03)
    content_w = config.get("content_w", 1.5)
    style_w = config.get("style_w", 2.0)
    grag_deep = config.get("grag_deep", 0.5)
    grag_shallow = config.get("grag_shallow", 0.7)
    freq_hi_boost = config.get("freq_hi_boost", 2.0)

    width, height = 512, 512
    num_inference_steps = 25
    seed = 43

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        cnt_pix = pipe.preprocess_image(Image.open(cnt_path).convert("RGB").resize((width,height), Image.BICUBIC))
        content_latent = vae.encode(cnt_pix)
        content_latent_packed = rearrange(content_latent, "B C H W -> B (H W) C")
        sty_pix = pipe.preprocess_image(Image.open(sty_path).convert("RGB").resize((512,512), Image.BICUBIC))
        style_latent_4d = vae.encode(sty_pix)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

    lat_h, lat_w = height//16, width//16
    image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)).unsqueeze(0).to("cuda")
    cnt_lat_h, cnt_lat_w = content_latent.shape[2], content_latent.shape[3]
    cnt_image_ids = torch.cartesian_prod(torch.tensor([10]), torch.arange(cnt_lat_h), torch.arange(cnt_lat_w), torch.arange(1)).unsqueeze(0).to("cuda")
    combined_img_ids = torch.cat([image_ids, cnt_image_ids], dim=1)
    sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
    style_image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)).unsqueeze(0).to("cuda")

    pipe.load_models_to_device(["dit"]); dit.to("cuda")

    torch.manual_seed(seed)
    latents = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    latents = rearrange(latents, "B C H W -> B (H W) C")
    latent_seq_len = latents.shape[1]

    scheduler = FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(num_inference_steps, dynamic_shift_len=lat_h*lat_w)
    timesteps = scheduler.timesteps

    loss_fn = torch.nn.L1Loss()

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()

        with torch.no_grad():
            latent_input = torch.cat([latents, content_latent_packed.to(latents.dtype)], dim=1)
            noise_pred = dit(hidden_states=latent_input, timestep=timestep/1000, guidance=None,
                           encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=combined_img_ids)
            noise_pred = noise_pred[:, :latent_seq_len]
        latents = scheduler.step(noise_pred, t_val, latents)

        if t_val >= start_time_val:
            continue

        # Noisy style
        if i < num_inference_steps - 1:
            ts = timesteps[i+1:i+2]
            style_latent_ = scheduler.add_noise(style_latent, torch.randn_like(style_latent), ts)
        else:
            ts = torch.tensor([0.0])
            style_latent_ = style_latent

        # Style forward
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(hidden_states=style_latent_.to("cuda", torch.bfloat16),
                   timestep=ts.to("cuda", torch.bfloat16)/1000, guidance=None,
                   encoder_hidden_states=null_embeds, txt_ids=null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, _ = cache.get()

        # GRAG
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

        content_anchor = latents.detach().clone()

        # AdaIN
        with torch.no_grad():
            latents = feature_adain(latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=adain_alpha).squeeze(1)
        latents = latents.detach()

        # SLT pre-align
        if slt_alpha > 0:
            with torch.no_grad():
                latents = sinkhorn_latent_transport(latents, style_latent_, style_latent,
                    alpha=slt_alpha, n_iters=10, tau=0.02, grag_delta=grag_deep, mode="ot_adain")

        # Adam
        if hybrid_iters > 0:
            latents.requires_grad_(True)
            optimizer = torch.optim.Adam([latents], lr=lr)
            _ts = ts.to("cuda", torch.bfloat16)/1000

            for it in range(hybrid_iters):
                cache.clear(); controller.step()
                optimizer.zero_grad()
                with torch.enable_grad():
                    _ = dit(hidden_states=latents, timestep=_ts.detach(), guidance=None,
                           encoder_hidden_states=null_embeds.detach(), txt_ids=null_text_ids.detach(),
                           img_ids=image_ids.detach())
                del _
                q_list, _, _, self_out_list = cache.get()
                _t = float(t_val.item())
                _prog = 1.0 - (_t / max(start_time_val, 1.0))
                _tau = 0.15 - (0.15-0.07)*_prog
                n_layers = len(q_list)
                n_ot = max(1, int(n_layers*0.5))

                ref_outs = []
                for idx2, (q, rk, rv) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                    if idx2 >= n_layers - n_ot:
                        ref_outs.append(sinkhorn_attn_freq(q.detach(), rk, rv, n_iters=6, tau=_tau, hi_boost=freq_hi_boost))
                    else:
                        ref_outs.append(F.scaled_dot_product_attention(q.detach(), rk, rv))

                sl = sum(loss_fn(so, ro.detach()) for so, ro in zip(self_out_list, ref_outs))
                cl = loss_fn(latents, content_anchor)
                loss = style_w*sl + content_w*cl
                loss.backward()
                optimizer.step()
            latents = latents.detach()

    latents = adain_1d(latents, style_latent)
    pipe.load_models_to_device(["vae"])
    lat4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
    image = vae.decode(lat4d)
    image = pipe.vae_output_to_image(image)
    pipe.load_models_to_device(["dit"]); dit.to("cuda")
    return image


# ============== Parameter search ==============
CNT_PATH = f"{CWD}/cnt/00.png"
STY_PATH = f"{CWD}/sty/27.png"
OUT_DIR = Path(CWD) / "i2i_param_search"
OUT_DIR.mkdir(exist_ok=True)

# Grid search configs
configs = [
    # Baseline: T2I hybrid params directly (may not be optimal for I2I)
    ("t2i_direct", {"start_time": 850, "adain_alpha": 0.5, "slt_alpha": 0.1, "iters": 2, "lr": 0.03, "content_w": 1.5}),
    # Lower adain (protect content)
    ("adain03_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.1, "iters": 2, "lr": 0.03, "content_w": 1.5}),
    ("adain02_st870", {"start_time": 870, "adain_alpha": 0.2, "slt_alpha": 0.1, "iters": 2, "lr": 0.03, "content_w": 1.5}),
    # Higher start_time (fewer style steps)
    ("adain03_st900", {"start_time": 900, "adain_alpha": 0.3, "slt_alpha": 0.1, "iters": 2, "lr": 0.03, "content_w": 1.5}),
    ("adain03_st850", {"start_time": 850, "adain_alpha": 0.3, "slt_alpha": 0.1, "iters": 2, "lr": 0.03, "content_w": 1.5}),
    # Higher content_w
    ("cw2_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.1, "iters": 2, "lr": 0.03, "content_w": 2.0}),
    ("cw3_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.1, "iters": 2, "lr": 0.03, "content_w": 3.0}),
    # More iters
    ("iter3_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.1, "iters": 3, "lr": 0.03, "content_w": 1.5}),
    # Lower LR
    ("lr01_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.1, "iters": 2, "lr": 0.01, "content_w": 1.5}),
    # Higher SLT alpha
    ("slt02_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.2, "iters": 2, "lr": 0.03, "content_w": 1.5}),
    ("slt005_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.05, "iters": 2, "lr": 0.03, "content_w": 1.5}),
    # Lower GRAG (more layout from style)
    ("grag03_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.1, "iters": 2, "lr": 0.03, "content_w": 1.5, "grag_deep": 0.3}),
    # More aggressive style
    ("sw3_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.1, "iters": 2, "lr": 0.03, "content_w": 1.5, "style_w": 3.0}),
    # Adam 8 baseline for comparison
    ("adam8_st870", {"start_time": 870, "adain_alpha": 0.3, "slt_alpha": 0.0, "iters": 8, "lr": 0.01, "content_w": 0.6}),
]

print(f"\n{'='*80}")
print(f"I2I PARAMETER SEARCH: {len(configs)} configs × cnt/00 × sty/27")
print(f"{'='*80}")
print(f"{'Config':<18} {'CSD':>7} {'CLIP-I':>7} {'Time':>6}  params")
print("-" * 70)

results = []
for name, cfg in configs:
    full_cfg = {"style_w": 2.0, "grag_deep": 0.5, "grag_shallow": 0.7, "freq_hi_boost": 2.0}
    full_cfg.update(cfg)

    torch.cuda.empty_cache()
    t0 = time.time()
    img = generate_i2i(CNT_PATH, STY_PATH, full_cfg)
    elapsed = time.time() - t0

    fname = f"{name}.png"
    img.save(OUT_DIR / fname)

    csd = compute_csd(str(OUT_DIR / fname), STY_PATH)
    clip_i = compute_clip_i(str(OUT_DIR / fname), CNT_PATH)

    results.append({"name": name, "csd": csd, "clip_i": clip_i, "time": elapsed, **cfg})
    print(f"{name:<18} {csd:>7.4f} {clip_i:>7.4f} {elapsed:>6.1f}  st={cfg.get('start_time',870)} ada={cfg.get('adain_alpha',0.3)} slt={cfg.get('slt_alpha',0.1)} it={cfg.get('iters',2)} lr={cfg.get('lr',0.03)} cw={cfg.get('content_w',1.5)}")
    sys.stdout.flush()

print(f"\n{'='*80}")
print("SORTED BY CSD (style fidelity):")
for r in sorted(results, key=lambda x: -x["csd"]):
    print(f"  {r['name']:<18} CSD={r['csd']:.4f}  CLIP-I(cnt)={r['clip_i']:.4f}  Time={r['time']:.1f}s")

print(f"\nSONTED BY CLIP-I (content preservation):")
for r in sorted(results, key=lambda x: -x["clip_i"]):
    print(f"  {r['name']:<18} CSD={r['csd']:.4f}  CLIP-I(cnt)={r['clip_i']:.4f}  Time={r['time']:.1f}s")

print(f"\nImages saved to: {OUT_DIR}/")
