"""KSHOT=2 experiment: does averaging 2 style forwards improve CSD without hurting CLIP-T?

Runs hybrid2 config with KSHOT=2 on the 40-sample ablation subset.
Compare against existing t2i_ablation.csv (hybrid2, KSHOT=1).

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/eval_kshot2.py
"""
import csv, os, sys, time, types
from pathlib import Path

# ── KSHOT must be set before the core module is imported ──────────────────
os.environ["KSHOT"] = "2"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision import transforms
import open_clip

DIFFSYNTH_ROOT = "third_party/DiffSynth-Studio"
KLEIN9B_ROOT   = "models/klein9b"
CWD            = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
OUT_DIR        = Path(CWD) / "d_grand" / "t2i_kshot2"
CSV_OUT        = Path(CWD) / "results" / "t2i_kshot2.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
PROMPTS = [
    ("kp0", "A panda eat bamboo"),
    ("kp1", "A castle on a cliff by the sea"),
    ("kp2", "A cat sitting on a windowsill"),
    ("kp3", "A field of sunflowers at sunset"),
    ("kp4", "A portrait of an old fisherman"),
]

HYBRID_DEFAULT = {
    "mode": "hybrid", "iters": 2, "lr": 0.03, "content_w": 1.5, "style_w": 2.0,
    "slt_alpha": 0.1, "freq_hi_boost": 2.0, "sinkhorn_mode": "freq",
    "grag_deep": 0.5, "grag_shallow": 0.7,
}

# ── DiffSynth + Klein 9B setup ────────────────────────────────────────────
def _install_modelscope_stub():
    if "modelscope" in sys.modules:
        return
    m = types.ModuleType("modelscope")
    m.snapshot_download = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no modelscope"))
    sys.modules["modelscope"] = m

if DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)
_install_modelscope_stub()

from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.diffusion.flow_match import FlowMatchScheduler

sys.path.insert(0, CWD)
from utils import sinkhorn_attn, AdaIN1d
import importlib, cores.core_ot_attn as _core_mod
importlib.reload(_core_mod)
from cores.core_ot_attn import OTAttnController, OTAttnCache, style_transfer_step

print("Loading Klein 9B …")
pipe = Flux2ImagePipeline.from_pretrained(
    model_path=KLEIN9B_ROOT,
    torch_dtype=torch.bfloat16,
)
pipe.device = torch.device("cuda:0")
pipe.load_models_to_device(["text_encoder", "text_encoder_2", "transformer", "vae"])

transformer = pipe.transformer
vae = pipe.vae
scheduler = FlowMatchScheduler(shift=3, sigma_min=0.0, extra_one_step=True)
scheduler.set_timesteps(25, training=False)
timesteps = scheduler.timesteps

# ── Metrics ──────────────────────────────────────────────────────────────
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

_mn = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1,3,1,1)
_sd = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1,3,1,1)

def _to_clip(path):
    img = Image.open(path).convert('RGB')
    t = transforms.ToTensor()(img).unsqueeze(0).to('cuda')
    t = F.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
    return (t - _mn) / _sd

def compute_csd(out_path, sty_path):
    with torch.no_grad():
        _, _, s_out = csd_model(_to_clip(out_path).float())
        _, _, s_sty = csd_model(_to_clip(sty_path).float())
    return (F.normalize(s_out,-1) * F.normalize(s_sty,-1)).sum().item()

def compute_clip_t(out_path, text):
    img = Image.open(out_path).convert('RGB')
    t = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711)),
    ])(img).unsqueeze(0).to('cuda')
    tok = clip_tokenizer([text]).to('cuda')
    with torch.no_grad():
        i = clip_model_vit.encode_image(t)
        tx = clip_model_vit.encode_text(tok)
    return (F.normalize(i,-1) * F.normalize(tx,-1)).sum().item()

# ── Import generate_t2i from eval_all ────────────────────────────────────
# We need the full generate function; easiest is to import it from eval_all
# after the model is already loaded (monkey-patch the globals it needs)
import scripts.eval_all as _ea
# eval_all.py sets up pipe/transformer/etc at module level during import;
# but we already loaded them, so we bind ours into eval_all's namespace
_ea.pipe        = pipe
_ea.transformer = transformer
_ea.vae         = vae
_ea.scheduler   = scheduler
_ea.timesteps   = timesteps
_ea.csd_model   = csd_model
_ea.clip_model_vit = clip_model_vit
_ea.clip_tokenizer = clip_tokenizer
generate_t2i  = _ea.generate_t2i
compute_csd_fn = _ea.compute_csd
compute_clip_t_fn = _ea.compute_clip_t

# ── Run experiment ────────────────────────────────────────────────────────
print(f"\nKSHOT=2 experiment: {len(STYLES)} styles × {len(PROMPTS)} prompts = {len(STYLES)*len(PROMPTS)} images")
print(f"{'Sty':<4} {'Prompt':<10} {'CSD':>7} {'CLIP-T':>7} {'Time':>6}")
print("-"*40)

rows, csds, clipts = [], [], []
for sty_id in STYLES:
    sty_path = f"{CWD}/sty/{sty_id}.png"
    for pid, prompt in PROMPTS:
        torch.cuda.empty_cache()
        t0 = time.time()
        img = generate_t2i(prompt, sty_path, HYBRID_DEFAULT)
        elapsed = time.time() - t0

        fname = f"kshot2_{pid}_s{sty_id}.png"
        fpath = OUT_DIR / fname
        img.save(fpath)

        csd    = compute_csd_fn(str(fpath), sty_path)
        clip_t = compute_clip_t_fn(str(fpath), prompt)
        rows.append({"method":"kshot2","sty_id":sty_id,"content_id":pid,
                     "csd":f"{csd:.4f}","clip_t":f"{clip_t:.4f}","time_s":f"{elapsed:.1f}"})
        csds.append(csd); clipts.append(clip_t)
        print(f"s{sty_id}  {pid:<10} {csd:>7.4f} {clip_t:>7.4f} {elapsed:>6.1f}s")
        sys.stdout.flush()

with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method","sty_id","content_id","csd","clip_t","time_s"])
    w.writeheader(); w.writerows(rows)

print("="*40)
print(f"n={len(csds)}")
print(f"mean CSD    = {np.mean(csds):.4f}")
print(f"mean CLIP-T = {np.mean(clipts):.4f}")
print(f"(baseline hybrid2: CSD=0.6810  CLIP-T=0.2737)")
print(f"CSV -> {CSV_OUT}")
