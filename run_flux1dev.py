import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from accelerate.utils import set_seed
from mypipeline import AttDistPipeline
from utils import *


model_name = "models/FLUX.1-dev"
lr = 0.01
iters = 2
seed = 43
width = 512
height = 512
mixed_precision = "bf16"
num_inference_steps = 25
guidance_scale = 3.5
enable_gradient_checkpoint = True
start_layer, end_layer = 20, 45
start_time = int(os.environ.get("OVERRIDE_START_TIME", "800"))
os.environ["OVERRIDE_START_TIME"] = str(start_time)
style_beta = 0.7
core_type = os.environ.get("CORE_TYPE", "core_ot_attn")
prompt=os.environ.get("PROMPT", "A panda eat bamboo")


pipe = AttDistPipeline.from_pretrained(
    model_name, torch_dtype=torch.float16)


if float(os.environ.get("CSD_W", "0.5")) > 0:
    # CSD-in-loop needs all modules resident on GPU (cpu_offload breaks
    # grad-checkpointing recompute when extra backward paths are present).
    pipe.freeze()
    try: pipe.enable_vae_slicing()
    except AttributeError: pass
    try: pipe.enable_vae_tiling()
    except AttributeError: pass
    pipe.to("cuda")
else:
    memory_efficient(pipe)
set_seed(seed)
loss_fn = torch.nn.L1Loss()

style_image_raw = [os.environ.get("STY_IMAGE", "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty/27.png")]
style_image = torch.cat([load_image(path, size=(512, 512)) for path in style_image_raw])

# --- Optional style-image preprocessing ---
# PREPROCESS={none, dino, siglip, clip}
# PREPROCESS_INVERT=1 blurs LOW-saliency (background) instead of HIGH-saliency (objects)
_PP = os.environ.get("PREPROCESS", "none").lower()
_PP_INV = os.environ.get("PREPROCESS_INVERT", "0") == "1"
if _PP != "none":
    from style_preprocess import suppress_objects
    style_image = suppress_objects(style_image, method=_PP, invert=_PP_INV)
    print(f"[Preprocess] applied {_PP} "
          f"{'bg-blur (invert)' if _PP_INV else 'object-blur'}")


# --- Pre-compute style latent rank_95 to drive layer range selection ---
# (core_hybrid does its own rank caching for OT alpha etc; here we compute it
#  early so we can set self_layers accordingly before Controller init.)
# Low-rank sty → shallow layers (0-20); mid → (10-30); high → default (20-45).
# This matches baseline grid-search winners.
import torch as _torch
with _torch.no_grad():
    _dt_vae = next(pipe.vae.parameters()).dtype
    _img_vae = (style_image.to("cuda", dtype=_dt_vae) * 2.0 - 1.0)
    _lat = pipe.vae.encode(_img_vae).latent_dist.sample()
    _lat = (_lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    _num_ch = pipe.transformer.config.in_channels // 4
    _packed = pipe._pack_latents(_lat, 1, _num_ch, _lat.shape[2], _lat.shape[3])
    _sl = _packed[0].float()
    _sl_c = _sl - _sl.mean(dim=0, keepdim=True)
    _L, _D = _sl_c.shape
    _C = _sl_c.T @ _sl_c / _L
    _C = (_C + _C.T) / 2
    _S = _torch.linalg.eigvalsh(_C + 1e-8 * _torch.eye(_D, device=_sl.device))
    _S = _torch.clamp(_S, min=0).flip(0)
    _p = (_S / (_S.sum() + 1e-8)).cpu().numpy()
    import numpy as _np
    _cum = _np.cumsum(_p)
    _rank95 = int(_np.searchsorted(_cum, 0.95) + 1)

# Layer range + iters (rank-adaptive, calibrated to avoid Goodhart)
# Ablation: FIXED_LAYERS=1 disables rank-adaptive layer/iter selection
if os.environ.get("FIXED_LAYERS", "0") == "1":
    start_layer, end_layer = 20, 45
    iters = 2
    print(f"[ablation] FIXED_LAYERS=1 → layers=(20,45) iters=2 (rank={_rank95})")
else:
    _LO = int(os.environ.get("LO_LAYER", "30"))
    _HI = int(os.environ.get("HI_LAYER", "40"))
    if _rank95 <= _LO:
        start_layer, end_layer = 0, 20
        iters = 3
    elif _rank95 <= _HI:
        start_layer, end_layer = 10, 30
        iters = 2
    else:
        start_layer, end_layer = 20, 45
        iters = 2
    print(f"[layer-adaptive] rank_95={_rank95} (LO={_LO},HI={_HI}) → layers=({start_layer},{end_layer}) iters={iters}")

controller = Controller(self_layers=(start_layer, end_layer))

# --- CSD-in-loop: cache CSD model + precomputed ref embedding on controller ---
# Activated by CSD_W > 0. Adam loop will add:
#   csd_loss = L1(csd_style(decoded Tweedie x0), controller.csd_ref_embed)
_CSD_W = float(os.environ.get("CSD_W", "0.5"))
if _CSD_W > 0:
    import sys, torch
    sys.path.insert(0, "../../CSD_Score")
    from model import CSD_CLIP, convert_state_dict
    from torchvision import transforms as _tfs
    import torchvision.transforms.functional as _TF
    _csd_dir = "../../CSD_Score"
    _csd = CSD_CLIP("vit_large", "default", model_path=f"{_csd_dir}/models/ViT-L-14.pt")
    _ck = torch.load(f"{_csd_dir}/models/checkpoint.pth", map_location="cpu", weights_only=False)
    _csd.load_state_dict(convert_state_dict(_ck["model_state_dict"]), strict=False)
    _csd = _csd.to("cuda").eval()
    # Freeze CSD params; we only backprop into latents through it
    for _p in _csd.parameters():
        _p.requires_grad_(False)

    _csd_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1, 3, 1, 1)
    _csd_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1, 3, 1, 1)

    def _csd_embed(img01):
        """img01: [B,3,H,W] in [0,1], returns style embedding [B, Dc] (normalized)."""
        img = F.interpolate(img01, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - _csd_mean) / _csd_std
        _, _, s = _csd(img.to(torch.float32))
        return torch.nn.functional.normalize(s, dim=-1)

    import torch.nn.functional as F
    controller.csd_embed_fn = _csd_embed
    # Compute reference embedding from the style image (it's already in [0,1] form)
    with torch.no_grad():
        ref_emb = _csd_embed(style_image.to("cuda"))
    controller.csd_ref_embed = ref_emb.detach()
    controller.csd_w = _CSD_W
    controller.pipe_for_decode = pipe    # core_hybrid will use this to decode
    print(f"[CSD-in-loop] ref embedding cached, CSD_W={_CSD_W}")

# --- Multi-style group bank (#4 from roadmap) ---
# STY_GROUP="14,18,21"  comma-separated sty ids (filename without .png)
# Each is VAE-encoded once; core_hybrid averages ref K/V across them.
# Also triggers NOISE_MODE=B so core picks from the bank.
_STY_GROUP = os.environ.get("STY_GROUP", "").strip()
if _STY_GROUP:
    import torch
    _group_ids = [s.strip() for s in _STY_GROUP.split(",") if s.strip()]
    controller.style_latent_bank = []
    _dev = "cuda"
    _dt = next(pipe.vae.parameters()).dtype
    _num_ch = pipe.transformer.config.in_channels // 4
    for _sid in _group_ids:
        _img_t = load_image(f"/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty/{_sid}.png",
                             size=(512, 512))
        _img_in = (_img_t.to(_dev, dtype=_dt) * 2.0 - 1.0)
        with torch.no_grad():
            lat = pipe.vae.encode(_img_in).latent_dist.sample()
            lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            lat_packed = pipe._pack_latents(lat, 1, _num_ch, lat.shape[2], lat.shape[3])
            controller.style_latent_bank.append(lat_packed)
    os.environ["KSHOT"] = str(len(_group_ids))
    os.environ["NOISE_MODE"] = "B"
    print(f"[GroupBank] loaded {len(_group_ids)} style latents from {_group_ids}")
else:
    _CROP_K = int(os.environ.get("CROP_K", "0"))
    if _CROP_K > 1:
        # --- Multi-crop K-shot of SAME style image (#1) ---
        import torch, random
        from torchvision.transforms import functional as _TF
        from PIL import Image as _PIL
        _raw = _PIL.open(style_image_raw[0]).convert("RGB")
        _W0, _H0 = _raw.size
        controller.style_latent_bank = []
        _dev = "cuda"
        _dt = next(pipe.vae.parameters()).dtype
        _num_ch = pipe.transformer.config.in_channels // 4
        random.seed(0)
        crops = [_raw.resize((512, 512), _PIL.BICUBIC)]
        for _i in range(_CROP_K - 1):
            scale = random.uniform(0.6, 0.95)
            side = int(min(_W0, _H0) * scale)
            left = random.randint(0, _W0 - side)
            top  = random.randint(0, _H0 - side)
            c = _raw.crop((left, top, left + side, top + side)).resize((512, 512), _PIL.BICUBIC)
            crops.append(c)
        for _c in crops:
            _t = _TF.to_tensor(_c).unsqueeze(0)
            _img_in = (_t.to(_dev, dtype=_dt) * 2.0 - 1.0)
            with torch.no_grad():
                lat = pipe.vae.encode(_img_in).latent_dist.sample()
                lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                lat_packed = pipe._pack_latents(lat, 1, _num_ch, lat.shape[2], lat.shape[3])
                controller.style_latent_bank.append(lat_packed)
        os.environ["KSHOT"] = str(len(crops))
        os.environ["NOISE_MODE"] = "B"
        print(f"[MultiCrop] built bank of {len(crops)} crops")
    else:
        # Original Scheme B posterior-resample path (K=1 default)
        _K_BANK = int(os.environ.get("KSHOT", "1"))
        _NMODE  = os.environ.get("NOISE_MODE", "A")
        if _K_BANK > 1 and _NMODE == "B":
            import torch
            controller.style_latent_bank = []
            _dev = "cuda"
            _dt = next(pipe.vae.parameters()).dtype
            _img = (style_image.to(_dev, dtype=_dt) * 2.0 - 1.0)
            _num_ch = pipe.transformer.config.in_channels // 4
            with torch.no_grad():
                for _ in range(_K_BANK):
                    lat = pipe.vae.encode(_img).latent_dist.sample()
                    lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                    lat_packed = pipe._pack_latents(lat, 1, _num_ch, lat.shape[2], lat.shape[3])
                    controller.style_latent_bank.append(lat_packed)
            print(f"[SchemeB] built VAE posterior bank of size {_K_BANK}")

set_seed(seed)
result = pipe.sample(
    lr=lr,
    prompt=prompt,
    loss_fn=loss_fn,
    iters=iters,
    width=width,
    height=height,
    start_time=start_time,
    core_type=core_type,
    controller=controller,
    style_image=style_image,
    guidance_scale=guidance_scale,
    mixed_precision=mixed_precision,
    num_inference_steps=num_inference_steps,
    enable_gradient_checkpoint=enable_gradient_checkpoint,
    style_beta=style_beta,
)

_out_path = os.environ.get("OUT_PATH", "output.png")
os.makedirs(os.path.dirname(_out_path) or ".", exist_ok=True)
save_image(result, _out_path)
print(f"[saved] {_out_path}")
