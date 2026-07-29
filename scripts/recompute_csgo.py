"""Recompute CSGO CSD/CLIP-T after the cc=0.0 gray-bug fix; OVERWRITE ext_csgo.csv.
Run: /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/recompute_csgo.py"""
import csv, sys
from pathlib import Path
import torch, torch.nn.functional as F, numpy as np
from PIL import Image
import open_clip
ROOT=Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948")
sys.path.insert(0,'../../CSD_Score')
from model import CSD_CLIP, convert_state_dict
CSD_DIR='../../CSD_Score'
csd=CSD_CLIP('vit_large','default',model_path=f'{CSD_DIR}/models/ViT-L-14.pt')
ck=torch.load(f'{CSD_DIR}/models/checkpoint.pth',map_location='cpu',weights_only=False)
csd.load_state_dict(convert_state_dict(ck['model_state_dict']),strict=False); csd=csd.to('cuda').eval()
clip_model,_,_=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai')
clip_model=clip_model.to('cuda').eval(); clip_tok=open_clip.get_tokenizer('ViT-B-32')
_mn=torch.tensor([0.48145466,0.4578275,0.40821073],device='cuda').view(1,3,1,1)
_sd=torch.tensor([0.26862954,0.26130258,0.27577711],device='cuda').view(1,3,1,1)
def _clip(p):
    im=Image.open(p).convert('RGB').resize((224,224),Image.BICUBIC)
    t=torch.from_numpy(np.asarray(im)).permute(2,0,1).unsqueeze(0).float().to('cuda')/255.
    return (t-_mn)/_sd
def csd_sim(o,s):
    with torch.no_grad():
        _,_,a=csd(_clip(o).float()); _,_,b=csd(_clip(s).float())
    return F.cosine_similarity(a,b).item()
def clipt(o,txt):
    with torch.no_grad():
        i=clip_model.encode_image(_clip(o)); tx=clip_model.encode_text(clip_tok([txt]).to('cuda'))
    return (F.normalize(i,dim=-1)*F.normalize(tx,dim=-1)).sum().item()
STYLES=["00","01","02","07","08","09","10","12","14","19","20","22","23","27","28","31","33","34","35","37","38"]
PROMPTS=[(0,"A panda eat bamboo"),(1,"A castle on a cliff by the sea"),(2,"A cat sitting on a windowsill"),(3,"A field of sunflowers at sunset"),(4,"A portrait of an old fisherman")]
rows=[]
for sid in STYLES:
    for idx,prompt in PROMPTS:
        p=ROOT/"d_grand"/"ext_csgo"/f"{sid}_p{idx}.png"
        if not p.exists(): print("MISSING",p.name); continue
        c=csd_sim(str(p),str(ROOT/"sty"/f"{sid}.png")); t=clipt(str(p),prompt)
        rows.append({"method":"csgo","sty_id":sid,"content_id":f"p{idx}","csd":f"{c:.4f}","clip_t":f"{t:.4f}"})
with open(ROOT/"results"/"ext_csgo.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["method","sty_id","content_id","csd","clip_t"]); w.writeheader(); w.writerows(rows)
print(f"OVERWROTE ext_csgo.csv n={len(rows)} | new mean CSD={np.mean([float(r['csd']) for r in rows]):.4f} CLIP-T={np.mean([float(r['clip_t']) for r in rows]):.4f}")
print("CSGO_RECOMPUTE_DONE")
