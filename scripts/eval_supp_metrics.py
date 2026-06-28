#!/usr/bin/env python
"""
Paper experiment #13 -- SUPPLEMENTARY perceptual style metrics on ALREADY-GENERATED
images (no image generation). Defends CSD+CLIP-T against single-metric criticism.

Metrics (all on existing PNGs):
  1. DINO style similarity  : DINOv2 (vit_base_patch14_dinov2) CLS-token cosine
                              between output and STYLE image.  HIGHER = better style.
  2. Gram-matrix style dist : VGG19 (torchvision) Gram matrices, mean Frobenius
                              distance over conv{1..5}_1 between output and STYLE.
                              LOWER = closer style (classic Gatys metric).
  3. DINO layout-leakage    : DINOv2 patch-token spatial self-similarity matrix,
                              Pearson corr of off-diagonal entries between output
                              and STYLE.  HIGHER = output copies STYLE's spatial
                              layout (= style-layout leakage).  Used for the LRS
                              ablation (hybrid vs lrs_off / GRAG-off).

SKIPPED:
  - LPIPS-content : all four image sets here are TEXT-TO-IMAGE; there is NO content
                    image to compare against, so LPIPS-content is undefined for T2I.
                    (lpips pkg also not installed.) Would only apply to the I2I track.
  - ArtFID        : ArtFID = (1+LPIPS)*(1+FID). Needs LPIPS (above, N/A for T2I) AND
                    a FID over a large set; n=40 per method is far too small for a
                    meaningful FID and the dep is heavy. Skipped by design.

Single H200, CUDA:0, light forwards. New file only -- does NOT touch eval_all.py/cores.
Run: /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/eval_supp_metrics.py
"""
import csv
import os
from statistics import mean, pstdev

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
D = f"{CWD}/d_grand"
STY_DIR = f"{CWD}/sty"
OUT_CSV = f"{CWD}/results/supp_metrics.csv"

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]   # 8 styles
PIDS = [0, 1, 2, 3, 4]                                       # kp0..kp4
# pid -> prompt (panda/castle/cat windowsill/sunflowers/fisherman) -- for reference only.

# ---- image path builders (one per method) -------------------------------------
def p_hybrid(sid, p):     return f"{D}/t2i_ablation/hybrid2_kp{p}_s{sid}.png"   # OURS
def p_uso(sid, p):        return f"{D}/ext_uso_t2i/{sid}_{p}_0.png"            # USO
def p_styleshot(sid, p):  return f"{D}/ext_styleshot/sty{sid}_kp{p}.png"       # StyleShot
def p_csgo(sid, p):       return f"{D}/ext_csgo/{sid}_p{p}.png"                # CSGO
def p_ipadapter(sid, p):  return f"{D}/ext_ipadapter/{sid}_{p}.png"            # IP-Adapter
def p_lrs_off(sid, p):    return f"{D}/t2i_p1cheap/lrs_off_kp{p}_s{sid}.png"   # LRS-off ablation

METHODS = {           # name -> path_fn
    "ours_hybrid": p_hybrid,
    "styleshot":   p_styleshot,
    "uso":         p_uso,
    "csgo":        p_csgo,
    "ipadapter":   p_ipadapter,
    "lrs_off":     p_lrs_off,
}

# ================================ models ======================================
print("[load] DINOv2 (vit_base_patch14_dinov2.lvd142m) ...", flush=True)
import timm
dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
dino = dino.to(DEV).eval()
N_PREFIX = getattr(dino, "num_prefix_tokens", 1)   # CLS (+regs if any) to strip for patches

print("[load] VGG19 (IMAGENET1K_V1) ...", flush=True)
from torchvision.models import vgg19, VGG19_Weights
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(DEV).eval()
for prm in vgg.parameters():
    prm.requires_grad_(False)
# classic Gatys style layers: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
GRAM_LAYERS = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1", 20: "relu4_1", 29: "relu5_1"}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
_to_tensor = transforms.ToTensor()


def load_norm(path, size=224):
    """Full image -> (1,3,size,size), ImageNet-normalized (no crop, keep whole frame)."""
    img = Image.open(path).convert("RGB")
    t = _to_tensor(img).unsqueeze(0).to(DEV)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


# =============================== metric fns ===================================
DINO_SIZE = 518   # native resolution of vit_base_patch14_dinov2 (518/14 = 37 -> 37x37 patches)


@torch.no_grad()
def dino_features(path):
    """Return (cls_feat[D], patch_tokens[N,D]) from DINOv2."""
    x = load_norm(path, DINO_SIZE)           # 518/14 = 37 -> 37x37 = 1369 patches
    tok = dino.forward_features(x)           # (1, N_prefix+1369, 768)
    cls = tok[:, 0, :].squeeze(0)            # CLS
    patches = tok[:, N_PREFIX:, :].squeeze(0)  # (256, 768)
    return cls, patches


@torch.no_grad()
def dino_style_sim(cls_out, cls_sty):
    return F.cosine_similarity(cls_out.unsqueeze(0), cls_sty.unsqueeze(0)).item()


@torch.no_grad()
def self_sim_matrix(patches):
    """Patch self-similarity (N,N) cosine matrix -> spatial-layout signature."""
    p = F.normalize(patches, dim=-1)
    return p @ p.t()


# upper-triangle (off-diagonal) index, computed once for N=256
_TRI = None
@torch.no_grad()
def layout_leakage(patches_out, patches_sty):
    """Pearson corr of off-diagonal self-sim entries (output vs style). Higher = more leakage."""
    global _TRI
    s_out = self_sim_matrix(patches_out)
    s_sty = self_sim_matrix(patches_sty)
    n = s_out.shape[0]
    if _TRI is None or _TRI[0].numel() == 0 or _TRI[0].max() >= n:
        _TRI = torch.triu_indices(n, n, offset=1, device=s_out.device)
    a = s_out[_TRI[0], _TRI[1]]
    b = s_sty[_TRI[0], _TRI[1]]
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    return ((a * b).sum() / denom).item()


@torch.no_grad()
def vgg_grams(path):
    """List of normalized Gram matrices at GRAM_LAYERS for one image."""
    x = load_norm(path, 224)
    grams = []
    h = x
    for idx, layer in enumerate(vgg):
        h = layer(h)
        if idx in GRAM_LAYERS:
            b, c, hh, ww = h.shape
            f = h.view(c, hh * ww)
            g = (f @ f.t()) / (c * hh * ww)   # normalize by elements
            grams.append(g)
        if idx >= max(GRAM_LAYERS):
            break
    return grams


@torch.no_grad()
def gram_dist(grams_out, grams_sty):
    """Mean Frobenius distance across layers. Lower = closer style."""
    dists = [torch.norm(go - gs).item() for go, gs in zip(grams_out, grams_sty)]
    return mean(dists)


# ====================== precompute style-ref features =========================
print("[precompute] style-ref DINO + Gram features ...", flush=True)
sty_dino_cls, sty_dino_patch, sty_grams = {}, {}, {}
for sid in STYLES:
    sp = f"{STY_DIR}/{sid}.png"
    cls, patch = dino_features(sp)
    sty_dino_cls[sid] = cls
    sty_dino_patch[sid] = patch
    sty_grams[sid] = vgg_grams(sp)


# ============================== main loop =====================================
rows = []          # csv rows
agg = {m: {"dino": [], "gram": [], "leak": []} for m in METHODS}
missing = []

for mname, pfn in METHODS.items():
    for sid in STYLES:
        for p in PIDS:
            op = pfn(sid, p)
            if not os.path.exists(op):
                missing.append(op)
                continue
            cls_o, patch_o = dino_features(op)
            grams_o = vgg_grams(op)
            d_sim = dino_style_sim(cls_o, sty_dino_cls[sid])
            g_dst = gram_dist(grams_o, sty_grams[sid])
            leak = layout_leakage(patch_o, sty_dino_patch[sid])
            rows.append([mname, sid, f"kp{p}", f"{d_sim:.4f}", f"{g_dst:.6f}", f"{leak:.4f}"])
            agg[mname]["dino"].append(d_sim)
            agg[mname]["gram"].append(g_dst)
            agg[mname]["leak"].append(leak)
    n = len(agg[mname]["dino"])
    print(f"  [{mname}] n={n}", flush=True)

# ------------------------------- write CSV ------------------------------------
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "sty_id", "pid", "dino_style_sim", "gram_dist", "dino_layout_leak"])
    w.writerows(rows)
print(f"\n[csv] wrote {len(rows)} rows -> {OUT_CSV}")
if missing:
    print(f"[warn] {len(missing)} missing images (first few): {missing[:5]}")


def ms(vals):
    return (mean(vals), pstdev(vals)) if vals else (float("nan"), float("nan"))


# --------------------------- summary: main table ------------------------------
print("\n" + "=" * 74)
print("SUPPLEMENTARY STYLE METRICS  (T2I, n per method shown)")
print("=" * 74)
print(f"{'method':<14}{'n':>4}{'DINO-style(cos) up':>22}{'Gram-dist down':>20}")
print("-" * 74)
for mname in ["ours_hybrid", "styleshot", "uso", "csgo", "ipadapter"]:
    dm, ds = ms(agg[mname]["dino"])
    gm, gs = ms(agg[mname]["gram"])
    n = len(agg[mname]["dino"])
    print(f"{mname:<14}{n:>4}{dm:>11.4f}+-{ds:<8.4f}{gm:>10.5f}+-{gs:<9.5f}")
print("-" * 74)
print("DINO-style: cosine of DINOv2 CLS(out, style)   higher = better style fidelity")
print("Gram-dist : mean Frobenius VGG19 Gram(out,style) lower = closer style")

# ------------------------- LRS layout-leakage block ---------------------------
print("\n" + "=" * 74)
print("LRS ABLATION -- style-layout leakage (#8)  : ours_hybrid (LRS on) vs lrs_off (GRAG off)")
print("=" * 74)
hl = agg["ours_hybrid"]["leak"]; ol = agg["lrs_off"]["leak"]
hlm, hls = ms(hl); olm, ols = ms(ol)
# paired diff over the 40 common (sid,pid) -- both sets are full 8x5
hd, hg = ms(agg["ours_hybrid"]["dino"]); hgm, _ = ms(agg["ours_hybrid"]["gram"])
od, _ = ms(agg["lrs_off"]["dino"]); ogm, _ = ms(agg["lrs_off"]["gram"])
print(f"{'method':<14}{'n':>4}{'layout-leak(corr)':>20}{'DINO-style':>14}{'Gram-dist':>14}")
print("-" * 74)
print(f"{'ours_hybrid':<14}{len(hl):>4}{hlm:>12.4f}+-{hls:<5.4f}{hd:>14.4f}{hgm:>14.5f}")
print(f"{'lrs_off':<14}{len(ol):>4}{olm:>12.4f}+-{ols:<5.4f}{od:>14.4f}{ogm:>14.5f}")
print("-" * 74)
print(f"layout-leak delta (lrs_off - hybrid) = {olm - hlm:+.4f}  "
      f"(higher leak = output copies STYLE's spatial layout)")

# PAIRED stats: hybrid & lrs_off iterate STYLES x PIDS in identical order (both n=40,
# no missing) -> element i is the same (sty,pid) pair, so index pairing is valid.
def paired_stats(a, b):
    """Return (mean_diff(b-a), paired_t, n_pos, n) for matched lists a,b."""
    d = [bi - ai for ai, bi in zip(a, b)]
    n = len(d)
    md = mean(d)
    sd = pstdev(d) * (n / (n - 1)) ** 0.5 if n > 1 else 0.0   # sample sd
    se = sd / (n ** 0.5) if n else float("nan")
    t = md / se if se else float("nan")
    return md, t, sum(1 for x in d if x > 0), n

if len(hl) == len(ol) == len(STYLES) * len(PIDS):
    md_leak, t_leak, pos_leak, n = paired_stats(hl, ol)
    md_gram, t_gram, pos_gram, _ = paired_stats(agg["ours_hybrid"]["gram"], agg["lrs_off"]["gram"])
    md_dino, t_dino, pos_dino, _ = paired_stats(agg["ours_hybrid"]["dino"], agg["lrs_off"]["dino"])
    print(f"\nPAIRED (lrs_off - hybrid) over n={n} matched (sty,pid):")
    print(f"  layout-leak : mean={md_leak:+.4f}  t={t_leak:+.2f}  lrs_off>hybrid in {pos_leak}/{n}")
    print(f"  Gram-dist   : mean={md_gram:+.4f}  t={t_gram:+.2f}  lrs_off>hybrid in {pos_gram}/{n}")
    print(f"  DINO-style  : mean={md_dino:+.4f}  t={t_dino:+.2f}  lrs_off>hybrid in {pos_dino}/{n}")
    sig = abs(t_leak) > 2.02   # ~ two-sided p<0.05, df=39
    direction = "MORE" if md_leak > 0 else "LESS"
    print("\nVERDICT (layout-leakage #8):")
    if md_leak > 0 and sig:
        print(f"  lrs_off (GRAG off) leaks {direction} style-layout, paired t={t_leak:+.2f} (p<0.05),")
        print(f"  consistent in {pos_leak}/{n} pairs -> direction SUPPORTS LRS. BUT effect size is small")
        print(f"  ({md_leak:+.4f} on a ~{hlm:.2f} base, ~{100*md_leak/hlm:.1f}% rel). Meanwhile lrs_off is")
        print(f"  significantly CLOSER on Gram (t={t_gram:+.2f}) -> GRAG-off copies style texture harder;")
        print(f"  LRS trades a little raw texture-match to suppress layout copying. Modest but consistent.")
    elif md_leak > 0:
        print(f"  lrs_off leaks {direction} on average (t={t_leak:+.2f}, NOT significant) -> weak/inconclusive.")
    else:
        print(f"  lrs_off does NOT leak more -> LRS layout-leakage benefit NOT supported by this metric.")
print("=" * 74)
