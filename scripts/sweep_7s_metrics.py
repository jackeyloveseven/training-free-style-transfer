"""Run 7s-budget configs and compute all 4 gold metrics."""
import os, sys, time, glob, types
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

DIFFSYNTH_ROOT = "third_party/DiffSynth-Studio"
KLEIN9B_ROOT = "models/klein9b"
if DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)

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
dit = pipe.dit; vae = pipe.vae
pipe.vram_management_enabled = False

cwd = '/home/ubuntu/chengyanli/items/save/attention_20260426_090948'

# === Metric models ===
sys.path.insert(0, '../../CSD_Score')
from model import CSD_CLIP, convert_state_dict

csd_dir = '../../CSD_Score'
csd_model = CSD_CLIP('vit_large', 'default', model_path=f'{csd_dir}/models/ViT-L-14.pt')
ck = torch.load(f'{csd_dir}/models/checkpoint.pth', map_location='cpu', weights_only=False)
csd_model.load_state_dict(convert_state_dict(ck['model_state_dict']), strict=False)
csd_model = csd_model.to('cuda').eval()
csd_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1,3,1,1)
csd_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1,3,1,1)
clip_model_m = torch.jit.load(f'{csd_dir}/models/ViT-L-14.pt', map_location='cuda').eval()

from transformers import AutoModel, AutoImageProcessor
dino_proc = AutoImageProcessor.from_pretrained('models/dinov2')
dino_model_m = AutoModel.from_pretrained('models/dinov2').to('cuda').eval()
import open_clip
tokenizer_m = open_clip.get_tokenizer('ViT-L-14')

from torchvision import transforms as T

def img_t(path):
    img = Image.open(path).convert('RGB')
    t = T.ToTensor()(img).unsqueeze(0).to('cuda')
    t = F.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
    return (t - csd_mean) / csd_std

def csd_s(path):
    t = img_t(path); _, _, s = csd_model(t.float()); return F.normalize(s, dim=-1)
def clip_i(path):
    t = img_t(path); f = clip_model_m.encode_image(t.float()); return F.normalize(f.float(), dim=-1)
def clip_txt(text):
    tokens = tokenizer_m([text]).to('cuda'); f = clip_model_m.encode_text(tokens); return F.normalize(f.float(), dim=-1)
def dino_c(path):
    img = Image.open(path).convert('RGB')
    inputs = dino_proc(images=img, return_tensors='pt').to('cuda')
    with torch.no_grad(): out = dino_model_m(**inputs)
    return F.normalize(out.last_hidden_state[:, 0].float(), dim=-1)

sty_path = os.path.join(cwd, 'sty/27.png')
prompt = 'A field of sunflowers at sunset'
sty_csd = csd_s(sty_path); sty_clip = clip_i(sty_path); sty_dino = dino_c(sty_path); txt_emb = clip_txt(prompt)

# === Pipeline utilities (inlined) ===
class Klein9BCache:
    def __init__(self): self.q, self.k, self.v, self.out = [], [], [], []
    def clear(self): self.q.clear(); self.k.clear(); self.v.clear(); self.out.clear()
    def add(self, q, k, v, out): self.q.append(q); self.k.append(k); self.v.append(v); self.out.append(out)
    def get(self): return self.q.copy(), self.k.copy(), self.v.copy(), self.out.copy()

class Klein9BController:
    def __init__(self, self_layers=(0, 16)):
        self.cur_self_layer = 0; self.self_layers = list(range(*self_layers)); self.txt_len = 0; self.mode = "cache"
    def step(self): self.cur_self_layer = 0

def sinkhorn_attn_freq(Q, K_style, V_style, n_iters=6, tau=0.1, hi_boost=2.0):
    B, H, Lc, D = Q.shape
    Q_lo = Q.mean(dim=2, keepdim=True).expand_as(Q); K_lo = K_style.mean(dim=2, keepdim=True).expand_as(K_style)
    Q_hi = Q - Q_lo; K_hi = K_style - K_lo
    logits = torch.matmul(Q_lo, K_lo.transpose(-1,-2))/(D**0.5*tau) + hi_boost*torch.matmul(Q_hi, K_hi.transpose(-1,-2))/(D**0.5*tau)
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    return torch.matmul(torch.softmax(log_P, dim=-1), V_style)

def feature_adain(x, y, alpha=1.0, eps=1e-6):
    m_x, s_x = x.mean(dim=2, keepdim=True), x.std(dim=2, keepdim=True) + eps
    m_y, s_y = y.mean(dim=2, keepdim=True), y.std(dim=2, keepdim=True) + eps
    return alpha * ((x - m_x) / s_x * s_y + m_y) + (1 - alpha) * x

def adain_1d(source, target, eps=1e-6):
    s_mean, s_std = source.mean(dim=1, keepdim=True), source.std(dim=1, keepdim=True)
    t_mean, t_std = target.mean(dim=(0,1), keepdim=True), target.std(dim=(0,1), keepdim=True)
    return (source - s_mean) / (s_std + eps) * t_std + t_mean

# Register hooks
def register_hooks(dit, controller, cache):
    from diffsynth.models.flux2_dit import _get_qkv_projections, apply_rotary_emb
    def make_double_hook(attn_module):
        def hooked(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
            query, key, value, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
            query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
            key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
            value = value.unflatten(-1, (attn.heads, -1))
            txt_len = 0
            if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
                eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
                ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
                ev = ev.unflatten(-1, (attn.heads, -1))
                txt_len = encoder_hidden_states.shape[1]; controller.txt_len = txt_len
                query = torch.cat([eq, query], dim=1); key = torch.cat([ek, key], dim=1); value = torch.cat([ev, value], dim=1)
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
                return attn.to_out[0](img_out), attn.to_add_out(enc_out)
            return attn.to_out[0](attn_output)
        return hooked
    def make_single_hook(attn_module):
        def hooked(attn, hidden_states, attention_mask=None, image_rotary_emb=None):
            from diffsynth.models.flux2_dit import apply_rotary_emb as are
            proj_out = attn.to_qkv_mlp_proj(hidden_states)
            qkv, mlp = torch.split(proj_out, [3*attn.inner_dim, attn.mlp_hidden_dim*attn.mlp_mult_factor], dim=-1)
            query, key, value = qkv.chunk(3, dim=-1)
            query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
            key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
            value = value.unflatten(-1, (attn.heads, -1))
            if image_rotary_emb is not None:
                query = are(query, image_rotary_emb, sequence_dim=1); key = are(key, image_rotary_emb, sequence_dim=1)
            query, key, value = query.to(hidden_states.dtype), key.to(hidden_states.dtype), value.to(hidden_states.dtype)
            q_t, k_t, v_t = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)
            if controller.cur_self_layer in controller.self_layers:
                cache.add(q_t, k_t, v_t, attn_out_t)
            controller.cur_self_layer += 1
            attn_output = attn_out_t.transpose(1,2).flatten(2,3).to(query.dtype)
            mlp = attn.mlp_act_fn(mlp)
            return attn.to_out(torch.cat([attn_output, mlp], dim=-1))
        return hooked
    for block in dit.transformer_blocks: block.attn.processor = make_double_hook(block.attn)
    for block in dit.single_transformer_blocks: block.attn.processor = make_single_hook(block.attn)

cache = Klein9BCache()
controller = Klein9BController(self_layers=(12, 28))
register_hooks(dit, controller, cache)

# Encode style
pipe.load_models_to_device(["vae"])
with torch.no_grad():
    sty_pix = pipe.preprocess_image(Image.open(sty_path).convert("RGB").resize((512,512), Image.BICUBIC))
    style_latent_4d = vae.encode(sty_pix)
    style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

width, height = 512, 512; lat_h, lat_w = height//16, width//16
image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)).unsqueeze(0).to("cuda")
sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
style_image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)).unsqueeze(0).to("cuda")

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

pipe.load_models_to_device(["dit"]); dit.to("cuda"); dit.requires_grad_(False)
loss_fn = torch.nn.L1Loss()

def run_once(num_steps, iters, start_time, style_w, adain_alpha, hi_boost, content_w, dist_w=0.15):
    scheduler = FlowMatchScheduler("FLUX.2")
    torch.manual_seed(43)
    noise = rearrange(torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16), "B C H W -> B (H W) C")
    latents = noise.clone()
    scheduler.set_timesteps(num_steps, dynamic_shift_len=lat_h*lat_w)
    timesteps = scheduler.timesteps
    torch.cuda.synchronize(); t0 = time.time()
    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()
        with torch.no_grad():
            noise_pred = dit(hidden_states=latents, timestep=timestep/1000, guidance=None,
                           encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=image_ids)
        latents = scheduler.step(noise_pred, t_val, latents)
        if t_val >= start_time: continue
        if i < num_steps - 1:
            ts_s = timesteps[i+1:i+2]; style_latent_ = scheduler.add_noise(style_latent, torch.randn_like(style_latent), ts_s)
        else:
            ts_s = torch.tensor([0.0]); style_latent_ = style_latent
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = dit(hidden_states=style_latent_.to(device="cuda", dtype=torch.bfloat16), timestep=ts_s.to(device="cuda", dtype=torch.bfloat16)/1000,
                   guidance=None, encoder_hidden_states=null_embeds, txt_ids=null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, _ = cache.get()
        content_anchor = latents.detach().clone()
        with torch.no_grad():
            latents = feature_adain(latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=adain_alpha).squeeze(1).detach()
        _progress = 1.0 - (float(t_val.item()) / max(start_time, 1.0))
        _dynamic_iters = max(1, int(iters * (1.0 - 0.6 * _progress)))
        latents.requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=0.01)
        _tau = 0.15 - 0.08 * _progress
        for it in range(_dynamic_iters):
            cache.clear(); controller.step(); optimizer.zero_grad()
            with torch.enable_grad():
                _dit_out = dit(hidden_states=latents, timestep=ts_s.to(device="cuda", dtype=torch.bfloat16).detach()/1000, guidance=None,
                             encoder_hidden_states=null_embeds.detach(), txt_ids=null_text_ids.detach(), img_ids=image_ids.detach())
            del _dit_out
            q_list, _, _, self_out_list = cache.get()
            n_layers = len(q_list); n_ot = max(1, int(n_layers * 0.5))
            ref_outs = []
            for idx, (q, rk, rv) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
                qd = q.detach()
                if idx >= n_layers - n_ot: ref_outs.append(sinkhorn_attn_freq(qd, rk, rv, n_iters=6, tau=_tau, hi_boost=hi_boost))
                else: ref_outs.append(F.scaled_dot_product_attention(qd, rk, rv))
            sloss = sum(loss_fn(so, ro.detach()) for so, ro in zip(self_out_list, ref_outs))
            lat4d, sty4d = latents.unsqueeze(1), style_latent_.unsqueeze(1).detach()
            dloss = loss_fn(lat4d.mean(2,keepdim=True), sty4d.mean(2,keepdim=True)) + loss_fn(lat4d.std(2,keepdim=True), sty4d.std(2,keepdim=True))
            closs = loss_fn(latents, content_anchor) * (1.0 - (it/max(iters,1))*0.5)
            loss = style_w*sloss + dist_w*dloss + content_w*closs
            loss.backward(); optimizer.step()
        latents = latents.detach()
    latents = adain_1d(latents, style_latent)
    torch.cuda.synchronize(); elapsed = time.time() - t0
    latents_4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
    with torch.no_grad(): image = vae.decode(latents_4d)
    image = pipe.vae_output_to_image(image)
    out_path = os.path.join(cwd, 'output_sweep_tmp.png')
    image.save(out_path)
    return elapsed, out_path

# Configs targeting ≤7s
configs = [
    # (num_steps, iters, start_time, style_w, adain_alpha, hi_boost, content_w)
    (25, 1, 850, 2.0, 0.5, 2.0, 0.2),
    (25, 1, 850, 4.0, 0.8, 2.0, 0.1),
    (25, 1, 850, 6.0, 1.0, 2.0, 0.0),
    (25, 1, 700, 4.0, 0.8, 2.0, 0.1),
    (25, 1, 700, 6.0, 1.0, 2.0, 0.0),
    (25, 2, 850, 4.0, 0.8, 2.0, 0.1),
    (25, 2, 850, 6.0, 1.0, 2.0, 0.0),
    (25, 2, 700, 6.0, 1.0, 2.0, 0.0),
    (25, 3, 850, 4.0, 0.8, 2.0, 0.1),
    (25, 3, 850, 6.0, 1.0, 2.0, 0.0),
    (25, 3, 700, 6.0, 1.0, 2.0, 0.0),
]

print(f"\n{'NS':>3} {'IT':>3} {'ST':>4} {'SW':>4} {'AA':>4} {'HB':>4} {'CW':>4} | {'CSD':>6} {'CLIP-T':>7} {'CI-S':>6} {'DINO':>6} {'Time':>5}")
print("-" * 78)

for cfg in configs:
    ns, it, st, sw, aa, hb, cw = cfg
    elapsed, out_path = run_once(ns, it, st, sw, aa, hb, cw)
    with torch.no_grad():
        o_csd = csd_s(out_path); csd_v = (o_csd * sty_csd).sum().item()
        o_ci = clip_i(out_path); clip_t_v = (o_ci * txt_emb).sum().item(); ci_sty_v = (o_ci * sty_clip).sum().item()
        o_d = dino_c(out_path); d_sty_v = (o_d * sty_dino).sum().item()
    print(f"{ns:>3} {it:>3} {st:>4} {sw:>4.1f} {aa:>4.1f} {hb:>4.1f} {cw:>4.1f} | {csd_v:>6.4f} {clip_t_v:>7.4f} {ci_sty_v:>6.4f} {d_sty_v:>6.4f} {elapsed:>5.1f}")
    sys.stdout.flush()
