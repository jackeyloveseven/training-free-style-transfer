"""Run Klein 9B T2I once and compute all 4 gold metrics + time."""
import subprocess, sys, os, time
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# === CSD ===
sys.path.insert(0, '/home/ubuntu/chengyanli/items/CSD_Score')
from model import CSD_CLIP, convert_state_dict

csd_dir = '/home/ubuntu/chengyanli/items/CSD_Score'
csd_model = CSD_CLIP('vit_large', 'default', model_path=f'{csd_dir}/models/ViT-L-14.pt')
ck = torch.load(f'{csd_dir}/models/checkpoint.pth', map_location='cpu', weights_only=False)
csd_model.load_state_dict(convert_state_dict(ck['model_state_dict']), strict=False)
csd_model = csd_model.to('cuda').eval()

csd_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1,3,1,1)
csd_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1,3,1,1)

# === CLIP (from CSD's ViT-L-14) ===
clip_model = torch.jit.load(f'{csd_dir}/models/ViT-L-14.pt', map_location='cuda').eval()

# === DINOv2 ===
from transformers import AutoModel, AutoImageProcessor
dino_proc = AutoImageProcessor.from_pretrained('/home/ubuntu/chengyanli/hfmodels/dinov2')
dino_model = AutoModel.from_pretrained('/home/ubuntu/chengyanli/hfmodels/dinov2').to('cuda').eval()

def img_tensor_224(path):
    img = Image.open(path).convert('RGB')
    t = transforms.ToTensor()(img).unsqueeze(0).to('cuda')
    t = F.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
    return (t - csd_mean) / csd_std

def csd_style(path):
    t = img_tensor_224(path)
    _, _, s = csd_model(t.float())
    return F.normalize(s, dim=-1)

def clip_img(path):
    t = img_tensor_224(path)
    f = clip_model.encode_image(t.float())
    return F.normalize(f.float(), dim=-1)

def clip_text(text):
    import open_clip
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    tokens = tokenizer([text]).to('cuda')
    f = clip_model.encode_text(tokens)
    return F.normalize(f.float(), dim=-1)

def dino_cls(path):
    img = Image.open(path).convert('RGB')
    inputs = dino_proc(images=img, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = dino_model(**inputs)
    return F.normalize(out.last_hidden_state[:, 0].float(), dim=-1)

# Config
sty_path = 'sty/27.png'
prompt = 'A field of sunflowers at sunset'
python = '/home/ubuntu/miniconda3/envs/sensenova/bin/python'
script = 'custom_main_klein9b_t2i.py'
out_img = 'output_klein9b_t2i.png'
cwd = '/home/ubuntu/chengyanli/items/save/attention_20260426_090948'

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'
env['PROMPT'] = prompt
env['STY_IMAGE'] = sty_path

print(f"Running Klein 9B T2I: prompt='{prompt}', sty=27")
print(f"Config: ITERS=5, START_TIME=850, STYLE_W=2.0, SINKHORN_MODE=freq, HI_BOOST=2.0")
print("-" * 60)

t0 = time.time()
result = subprocess.run(
    [python, script], env=env, capture_output=True, text=True,
    cwd=cwd, timeout=120
)
wall_time = time.time() - t0

if result.returncode != 0:
    print("FAILED!")
    print(result.stderr[-2000:])
    sys.exit(1)

# Find output
out_path = os.path.join(cwd, out_img)
if not os.path.exists(out_path):
    candidates = [f for f in os.listdir(cwd) if f.endswith('.png') and 'klein' in f.lower()]
    if not candidates:
        candidates = [f for f in os.listdir(cwd) if f.startswith('output') and f.endswith('.png')]
    if candidates:
        out_path = os.path.join(cwd, sorted(candidates)[-1])
    else:
        print("No output image found!")
        print("stdout:", result.stdout[-500:])
        sys.exit(1)

print(f"\nOutput: {os.path.basename(out_path)}")
print(f"Time: {wall_time:.1f}s")
print()

# Compute metrics
with torch.no_grad():
    # CSD
    out_s = csd_style(out_path)
    sty_s = csd_style(os.path.join(cwd, sty_path))
    csd = (out_s * sty_s).sum().item()

    # CLIP-T
    out_clip = clip_img(out_path)
    txt_clip = clip_text(prompt)
    clip_t = (out_clip * txt_clip).sum().item()

    # CLIP-I(sty)
    sty_clip = clip_img(os.path.join(cwd, sty_path))
    clip_i_sty = (out_clip * sty_clip).sum().item()

    # DINO(sty)
    out_dino = dino_cls(out_path)
    sty_dino = dino_cls(os.path.join(cwd, sty_path))
    dino_sty = (out_dino * sty_dino).sum().item()

print(f"{'Metric':<12} {'Value':>8}")
print("-" * 22)
print(f"{'CSD':<12} {csd:>8.4f}")
print(f"{'CLIP-T':<12} {clip_t:>8.4f}")
print(f"{'CLIP-I(sty)':<12} {clip_i_sty:>8.4f}")
print(f"{'DINO(sty)':<12} {dino_sty:>8.4f}")
print(f"{'Time(s)':<12} {wall_time:>8.1f}")
