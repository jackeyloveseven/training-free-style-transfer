#!/usr/bin/env python3
"""Unified I2I evaluation: for a method's output tiles over the 40-pair benchmark
(8 styles x 5 content), compute CSD, CLIP-I(content), LPIPS(content), FID_style,
ArtFID, texture-energy, orientation-coherence. Writes results/i2i_eval_<tag>.csv
+ prints a summary row.

Usage:
  flux/bin/python3 i2i_eval_all.py <tag> <out_dir> <fname_template>
  fname_template uses {cnt} and {sty}, e.g. "csgo_cnt{cnt}_sty{sty}.png"
  (multi-seed ours: pass a dir with one seed; run once per seed, then average.)
"""
import sys, os, csv, glob
import numpy as np
import torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
STY, CNT = f"{CWD}/sty", f"{CWD}/cnt"
RES = f"{CWD}/results"
STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
CNTS = ["00", "04", "08", "12", "17"]
DEV = "cuda"

tag, out_dir, tmpl = sys.argv[1], sys.argv[2], sys.argv[3]

# ---- CSD + CLIP ----
sys.path.insert(0, "/home/ubuntu/chengyanli/items/CSD_Score")
from model import CSD_CLIP, convert_state_dict
import open_clip
csd_dir = "/home/ubuntu/chengyanli/items/CSD_Score"
csd = CSD_CLIP("vit_large", "default", model_path=f"{csd_dir}/models/ViT-L-14.pt")
ck = torch.load(f"{csd_dir}/models/checkpoint.pth", map_location="cpu", weights_only=False)
csd.load_state_dict(convert_state_dict(ck["model_state_dict"]), strict=False)
csd = csd.to(DEV).eval()
clipm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clipm = clipm.to(DEV).eval()
import lpips as lpips_pkg
lpips_fn = lpips_pkg.LPIPS(net="alex").to(DEV).eval()
from pytorch_fid.inception import InceptionV3
# n=40 << 2048 makes the 2048-d FID covariance rank-deficient (inflated, hundreds);
# the 64-d Inception block is the small-sample-appropriate choice.
fid_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[64]]).to(DEV).eval()

def _csd_t(p):
    im = Image.open(p).convert("RGB").resize((224, 224))
    return TF.normalize(TF.to_tensor(im), [0.48145466,0.4578275,0.40821073],
                        [0.26862954,0.26130258,0.27577711]).unsqueeze(0).to(DEV)
def compute_csd(o, s):
    with torch.no_grad():
        _,_,a = csd(_csd_t(o).float()); _,_,b = csd(_csd_t(s).float())
    return (F.normalize(a,dim=-1)*F.normalize(b,dim=-1)).sum().item()
def _clip_t(p):
    im = Image.open(p).convert("RGB").resize((224,224), Image.BICUBIC)
    return TF.normalize(TF.to_tensor(im),[0.48145466,0.4578275,0.40821073],
                        [0.26862954,0.26130258,0.27577711]).unsqueeze(0).to(DEV)
def compute_clip_i(o, c):
    with torch.no_grad():
        f1 = clipm.encode_image(_clip_t(o)); f2 = clipm.encode_image(_clip_t(c))
    return (F.normalize(f1,dim=-1)*F.normalize(f2,dim=-1)).sum().item()
def load_lpips(p):
    im = Image.open(p).convert("RGB").resize((512,512))
    return (TF.to_tensor(im)*2-1).unsqueeze(0).to(DEV)
def compute_lpips(o, c):
    with torch.no_grad():
        return lpips_fn(load_lpips(o), load_lpips(c)).item()
def fid_acts(paths):
    acts=[]
    with torch.no_grad():
        for p in paths:
            im = Image.open(p).convert("RGB").resize((299,299))
            t = TF.to_tensor(im).unsqueeze(0).to(DEV)
            a = torch.nn.functional.adaptive_avg_pool2d(fid_model(t)[0],(1,1)).squeeze(-1).squeeze(-1).cpu().numpy()
            acts.append(a[0])
    return np.array(acts)
def fid_from(a1, a2):
    from scipy import linalg
    m1,m2 = a1.mean(0), a2.mean(0); c1,c2 = np.cov(a1,rowvar=False), np.cov(a2,rowvar=False)
    diff = m1-m2; cov,_ = linalg.sqrtm(c1@c2, disp=False)
    if np.iscomplexobj(cov): cov = cov.real
    return float(diff@diff + np.trace(c1+c2-2*cov))
def gray(p,sz=256): return np.asarray(Image.open(p).convert("L").resize((sz,sz)),float)/255.
def blur(g,k=9):
    import scipy.ndimage as ndi; return ndi.uniform_filter(g,k)
def texture_energy(g): return float(np.abs(g-blur(g)).mean())
def orient_coh(g):
    import scipy.ndimage as ndi
    hp = g-blur(g); gx=ndi.sobel(hp,1); gy=ndi.sobel(hp,0)
    P=16; H=g.shape[0]//P; vals=[]; w=[]
    for i in range(H):
        for j in range(H):
            sx=gx[i*P:(i+1)*P,j*P:(j+1)*P]; sy=gy[i*P:(i+1)*P,j*P:(j+1)*P]
            Jxx=(sx*sx).sum(); Jyy=(sy*sy).sum(); Jxy=(sx*sy).sum()
            tr=Jxx+Jyy; det=Jxx*Jyy-Jxy*Jxy
            l1=tr/2+np.sqrt(max(tr*tr/4-det,0)); l2=tr/2-np.sqrt(max(tr*tr/4-det,0))
            c=(l1-l2)/(l1+l2+1e-8); vals.append(c); w.append(tr)
    w=np.array(w); vals=np.array(vals); return float((vals*w).sum()/(w.sum()+1e-8))

rows=[]; outs=[]; stys=[]
for cid in CNTS:
    for sid in STYLES:
        op = os.path.join(out_dir, tmpl.format(cnt=cid, sty=sid))
        if not os.path.exists(op):
            print("MISSING", op); continue
        sp=f"{STY}/{sid}.png"; cp=f"{CNT}/{cid}.png"
        rows.append({"method":tag,"cnt_id":cid,"sty_id":sid,
                     "csd":compute_csd(op,sp),"clip_i":compute_clip_i(op,cp),
                     "lpips":compute_lpips(op,cp),
                     "tex_energy":texture_energy(gray(op)),"orient_coh":orient_coh(gray(op))})
        outs.append(op); stys.append(sp)
# FID_style: outputs vs style refs (n small -> indicative only)
fid_style = fid_from(fid_acts(outs), fid_acts(stys)) if len(outs)>=2 else float("nan")
mean = lambda k: np.mean([r[k] for r in rows])
artfid = (1+mean("lpips"))*(1+fid_style)
with open(f"{RES}/i2i_eval_{tag}.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(
        [{k:(f"{v:.4f}" if isinstance(v,float) else v) for k,v in r.items()} for r in rows])
print(f"\n=== {tag} (n={len(rows)}) ===")
print(f"CSD {mean('csd'):.4f}  CLIP-I {mean('clip_i'):.4f}  LPIPS {mean('lpips'):.4f}  "
      f"FID_style {fid_style:.1f}  ArtFID {artfid:.1f}  TexEnergy {mean('tex_energy'):.4f}  "
      f"OrientCoh {mean('orient_coh'):.4f}")
