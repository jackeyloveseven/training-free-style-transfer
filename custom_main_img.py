import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from accelerate.utils import set_seed
from mypipeline_img import AttDistImgPipeline
from utils import *


model_name = "models/FLUX.1-Kontext-dev"
lr = 0.01
iters = 2
seed = 43
width = 512
height = 512
mixed_precision = "bf16"
num_inference_steps = int(os.environ.get("NUM_STEPS", "25"))
guidance_scale = 3.5
enable_gradient_checkpoint = True
start_layer, end_layer = 20, 45
start_time = int(os.environ.get("OVERRIDE_START_TIME", "870"))
os.environ["OVERRIDE_START_TIME"] = str(start_time)
style_beta = 0.7
core_type = os.environ.get("CORE_TYPE", "core_ot_attn_img")
prompt = os.environ.get("PROMPT", "masterpiece")

# Content image path (the image whose structure/content we want to preserve)
content_image_path = os.environ.get("CNT_IMAGE",
    "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/cnt/00.png")
# Style image path
style_image_path = os.environ.get("STY_IMAGE",
    "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty/27.png")


pipe = AttDistImgPipeline.from_pretrained(
    model_name, torch_dtype=torch.float16)


if float(os.environ.get("CSD_W", "0.5")) > 0:
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

# Load content image
content_image = torch.cat([load_image(content_image_path, size=(height, width))])
print(f"Content image: {content_image_path}")

# Load style image
style_image = torch.cat([load_image(style_image_path, size=(512, 512))])
print(f"Style image: {style_image_path}")


# --- Pre-compute style latent rank_95 to drive layer range selection ---
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

# Layer range + iters (rank-adaptive)
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

# --- CSD-in-loop ---
_CSD_W = float(os.environ.get("CSD_W", "0.5"))
if _CSD_W > 0:
    import sys, torch
    sys.path.insert(0, "../../CSD_Score")
    from model import CSD_CLIP, convert_state_dict
    import torch.nn.functional as F
    _csd_dir = "../../CSD_Score"
    _csd = CSD_CLIP("vit_large", "default", model_path=f"{_csd_dir}/models/ViT-L-14.pt")
    _ck = torch.load(f"{_csd_dir}/models/checkpoint.pth", map_location="cpu", weights_only=False)
    _csd.load_state_dict(convert_state_dict(_ck["model_state_dict"]), strict=False)
    _csd = _csd.to("cuda").eval()
    for _p in _csd.parameters():
        _p.requires_grad_(False)

    _csd_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1, 3, 1, 1)
    _csd_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1, 3, 1, 1)

    def _csd_embed(img01):
        img = F.interpolate(img01, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - _csd_mean) / _csd_std
        _, _, s = _csd(img.to(torch.float32))
        return torch.nn.functional.normalize(s, dim=-1)

    controller.csd_embed_fn = _csd_embed
    with torch.no_grad():
        ref_emb = _csd_embed(style_image.to("cuda"))
    controller.csd_ref_embed = ref_emb.detach()
    controller.csd_w = _CSD_W
    controller.pipe_for_decode = pipe
    print(f"[CSD-in-loop] ref embedding cached, CSD_W={_CSD_W}")


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
    content_image=content_image,
    style_image=style_image,
    guidance_scale=guidance_scale,
    mixed_precision=mixed_precision,
    num_inference_steps=num_inference_steps,
    enable_gradient_checkpoint=enable_gradient_checkpoint,
    style_beta=style_beta,
)

save_image(content_image, "content.png")
save_image(style_image, "style.png")
save_image(result, "output_img.png")
show_image("content.png", title="content image")
show_image("style.png", title="style image")
show_image("output_img.png", title=prompt)
