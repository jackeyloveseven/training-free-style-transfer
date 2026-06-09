"""Sweep within 7s budget (iters=2-3) for best 4 gold metrics."""
import subprocess, sys, os, time
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, '/home/ubuntu/chengyanli/items/CSD_Score')
from model import CSD_CLIP, convert_state_dict

csd_dir = '/home/ubuntu/chengyanli/items/CSD_Score'
csd_model = CSD_CLIP('vit_large', 'default', model_path=f'{csd_dir}/models/ViT-L-14.pt')
ck = torch.load(f'{csd_dir}/models/checkpoint.pth', map_location='cpu', weights_only=False)
csd_model.load_state_dict(convert_state_dict(ck['model_state_dict']), strict=False)
csd_model = csd_model.to('cuda').eval()

csd_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1,3,1,1)
csd_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1,3,1,1)

clip_model = torch.jit.load(f'{csd_dir}/models/ViT-L-14.pt', map_location='cuda').eval()

from transformers import AutoModel, AutoImageProcessor
dino_proc = AutoImageProcessor.from_pretrained('/home/ubuntu/chengyanli/hfmodels/dinov2')
dino_model = AutoModel.from_pretrained('/home/ubuntu/chengyanli/hfmodels/dinov2').to('cuda').eval()

import open_clip
tokenizer = open_clip.get_tokenizer('ViT-L-14')

def img_t(path):
    img = Image.open(path).convert('RGB')
    t = transforms.ToTensor()(img).unsqueeze(0).to('cuda')
    t = F.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
    return (t - csd_mean) / csd_std

def csd_s(path):
    t = img_t(path)
    _, _, s = csd_model(t.float())
    return F.normalize(s, dim=-1)

def clip_i(path):
    t = img_t(path)
    f = clip_model.encode_image(t.float())
    return F.normalize(f.float(), dim=-1)

def clip_txt(text):
    tokens = tokenizer([text]).to('cuda')
    f = clip_model.encode_text(tokens)
    return F.normalize(f.float(), dim=-1)

def dino_c(path):
    img = Image.open(path).convert('RGB')
    inputs = dino_proc(images=img, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = dino_model(**inputs)
    return F.normalize(out.last_hidden_state[:, 0].float(), dim=-1)

cwd = '/home/ubuntu/chengyanli/items/save/attention_20260426_090948'
sty_path = os.path.join(cwd, 'sty/27.png')
prompt = 'A field of sunflowers at sunset'

sty_csd = csd_s(sty_path)
sty_clip = clip_i(sty_path)
sty_dino = dino_c(sty_path)
txt_emb = clip_txt(prompt)

python = '/home/ubuntu/miniconda3/envs/sensenova/bin/python'
script = 'custom_main_klein9b_t2i.py'
out_img = os.path.join(cwd, 'output_klein9b_t2i.png')

# Focus on iters=2-3 to stay under 7s
configs = [
    # (iters, start_time, style_w, adain_alpha, hi_boost, content_w)
    # iters=3, varying other params
    (3, 850, 2.0, 0.5, 2.0, 0.2),   # baseline 3-iter
    (3, 850, 4.0, 0.8, 2.0, 0.1),   # aggressive style
    (3, 750, 4.0, 0.8, 2.0, 0.1),   # lower start_time
    (3, 700, 4.0, 1.0, 2.0, 0.0),   # max style, no content
    (3, 700, 6.0, 1.0, 2.0, 0.0),   # even higher style_w
    (3, 600, 6.0, 1.0, 2.0, 0.0),   # earliest start
    (3, 700, 4.0, 0.8, 4.0, 0.0),   # high hi_boost
    (3, 850, 6.0, 1.0, 4.0, 0.0),   # late start but extreme style
    # iters=2
    (2, 700, 6.0, 1.0, 2.0, 0.0),
    (2, 600, 6.0, 1.0, 2.0, 0.0),
    (2, 850, 8.0, 1.0, 2.0, 0.0),
    (2, 700, 8.0, 1.0, 4.0, 0.0),
    # iters=4 (might still be ~10s)
    (4, 700, 4.0, 0.8, 2.0, 0.1),
    (4, 700, 6.0, 1.0, 2.0, 0.0),
    (4, 600, 4.0, 1.0, 2.0, 0.0),
]

print(f"{'IT':>3} {'ST':>4} {'SW':>4} {'AA':>4} {'HB':>4} {'CW':>4} | {'CSD':>6} {'CLIP-T':>7} {'CI-S':>6} {'DINO':>6} {'Time':>5}")
print("-" * 72)

for iters, st, sw, aa, hb, cw in configs:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['PROMPT'] = prompt
    env['STY_IMAGE'] = 'sty/27.png'
    env['ITERS'] = str(iters)
    env['START_TIME'] = str(st)
    env['STYLE_W'] = str(sw)
    env['ADAIN_ALPHA'] = str(aa)
    env['FREQ_HI_BOOST'] = str(hb)
    env['CONTENT_W'] = str(cw)

    t0 = time.time()
    result = subprocess.run(
        [python, script], env=env, capture_output=True, text=True,
        cwd=cwd, timeout=60
    )
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"{iters:>3} {st:>4} {sw:>4.1f} {aa:>4.1f} {hb:>4.1f} {cw:>4.2f} | {'FAIL':>6}")
        continue

    with torch.no_grad():
        out_csd = csd_s(out_img)
        csd = (out_csd * sty_csd).sum().item()
        out_ci = clip_i(out_img)
        clip_t = (out_ci * txt_emb).sum().item()
        ci_sty = (out_ci * sty_clip).sum().item()
        out_d = dino_c(out_img)
        d_sty = (out_d * sty_dino).sum().item()

    print(f"{iters:>3} {st:>4} {sw:>4.1f} {aa:>4.1f} {hb:>4.1f} {cw:>4.2f} | {csd:>6.4f} {clip_t:>7.4f} {ci_sty:>6.4f} {d_sty:>6.4f} {wall:>5.1f}")
    sys.stdout.flush()
