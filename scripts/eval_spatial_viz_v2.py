#!/usr/bin/env python3
"""eval_spatial_viz_v2.py — 2026-07-03 re-capture for two supp-figure redos
(items/sec Fig.15 + Fig.19a), saving RAW tensors this time so future restyles
never need the GPU again.

1. Update fields (Fig.15 redo): for three (prompt, style) pairs that MATCH
   existing t2i_ablation tiles (the old fig8 rows paired sample2's PCA —
   captured with sty/00 — with sty/33 thumbnails; v2 keeps capture and
   display pairs identical), rerun capture_update_fields and save the raw
   [32,32,D] fields (slt_update / backward_grad / adam_update_1 / random)
   as results/spatial_fields_v2/field_{kp}_s{sid}.npz (fp16).

2. LRS attention capture (Fig.19a redo): style forward on sty/20 caches the
   per-layer style K; a content forward at the same captured timestep caches
   the content Q. Save image-token Q/K for a shallow and a deep self-attn
   layer -> results/spatial_fields_v2/lrs_qk_s20.npz. The renderer computes
   softmax(q·(mu+gamma R)^T) maps offline for gamma in {1, 0.5, 0}.

Run (upstream repo, sensenova env, idle H200):
  cd /home/ubuntu/chengyanli/items/save/attention_20260426_090948 && \
  /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/eval_spatial_viz_v2.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent))
import eval_spatial_viz as base  # noqa: E402  (loads Klein 9B at import)

OUT = Path(base.RESULTS_DIR) / "spatial_fields_v2"
OUT.mkdir(exist_ok=True)

SAMPLES = [  # (kp id, canonical prompt, style id) — tiles hybrid2_{kp}_s{sid} exist
    ("kp0", "A panda eat bamboo", "27"),
    ("kp3", "A field of sunflowers at sunset", "33"),
    ("kp1", "A castle on a cliff by the sea", "28"),
]


def capture_fields():
    for kp, prompt, sid in SAMPLES:
        print(f"\n[fields] {kp} '{prompt}' x sty/{sid}", flush=True)
        torch.cuda.empty_cache()
        res = base.capture_update_fields(prompt, f"{base.CWD}/sty/{sid}.png")
        cos = {m: base.neighbor_cosine_at_distance(res[m], 1)
               for m in ("backward_grad", "slt_update", "adam_update_1", "random")}
        print(f"  cos@d1: {cos}", flush=True)
        np.savez_compressed(
            OUT / f"field_{kp}_s{sid}.npz",
            slt_update=res["slt_update"].astype(np.float16),
            backward_grad=res["backward_grad"].astype(np.float16),
            adam_update_1=res["adam_update_1"].astype(np.float16),
            random=res["random"].astype(np.float16),
            cos_d1=np.array([cos["backward_grad"], cos["slt_update"],
                             cos["adam_update_1"], cos["random"]]),
            prompt=np.array(prompt), sid=np.array(sid))


@torch.no_grad()
def capture_inject_fields():
    """Inject (K/V replacement) update field for the same three SAMPLES —
    faithful replica of eval_spatial_extended.capture_all_methods' INJECT block
    (the source of CANONICAL rho Inject=0.349): style K/V get the LRS delta
    (grag 0.7 shallow -> 0.5 deep), top-50% layers use sinkhorn OT attention,
    field = mean over layers+heads of (target - self_out) on image tokens.
    Saves results/spatial_fields_v2/inject_{kp}_s{sid}.npz."""
    import torch.nn.functional as F
    pipe, dit, vae = base.pipe, base.dit, base.vae
    cache, controller = base.cache, base.controller
    width = height = 512
    lat_h, lat_w = height // 16, width // 16
    grag_shallow, grag_deep = 0.7, 0.5

    for kp, prompt, sid in SAMPLES:
        print(f"\n[inject] {kp} '{prompt}' x sty/{sid}", flush=True)
        torch.cuda.empty_cache()

        pipe.load_models_to_device(["vae"])
        sty_pix = pipe.preprocess_image(
            Image.open(f"{base.CWD}/sty/{sid}.png").convert("RGB")
                 .resize((512, 512), Image.BICUBIC))
        style_latent_4d = vae.encode(sty_pix)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")
        sty_h, sty_w = style_latent_4d.shape[2], style_latent_4d.shape[3]

        image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lat_h),
                                         torch.arange(lat_w), torch.arange(1)
                                         ).unsqueeze(0).to("cuda")
        style_image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(sty_h),
                                               torch.arange(sty_w), torch.arange(1)
                                               ).unsqueeze(0).to("cuda")
        prompt_embeds, text_ids = base.encode_prompt(prompt)
        pipe.load_models_to_device(["dit"]); dit.to("cuda")

        torch.manual_seed(43)
        latents = torch.randn(1, 128, lat_h, lat_w, device="cuda",
                              dtype=torch.bfloat16)
        latents = rearrange(latents, "B C H W -> B (H W) C")
        scheduler = base.FlowMatchScheduler("FLUX.2")
        scheduler.set_timesteps(25, dynamic_shift_len=lat_h * lat_w)

        for i, t_val in enumerate(scheduler.timesteps):
            timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
            cache.clear(); controller.step()
            noise_pred = dit(hidden_states=latents, timestep=timestep / 1000,
                             guidance=None, encoder_hidden_states=prompt_embeds,
                             txt_ids=text_ids, img_ids=image_ids)
            latents = scheduler.step(noise_pred, t_val, latents)
            if t_val >= 850:
                continue

            # style forward -> per-layer ref K/V (with LRS delta, as deployed)
            ts = scheduler.timesteps[i + 1:i + 2]
            style_noisy = scheduler.add_noise(style_latent,
                                              torch.randn_like(style_latent), ts)
            cache.clear(); controller.step()
            _ = dit(hidden_states=style_noisy.to("cuda", torch.bfloat16),
                    timestep=ts.to("cuda", torch.bfloat16) / 1000, guidance=None,
                    encoder_hidden_states=base.null_embeds,
                    txt_ids=base.null_text_ids, img_ids=style_image_ids)
            _, ref_k_list, ref_v_list, _ = cache.get()

            txt_len = controller.txt_len
            n_layers = len(ref_k_list)
            for idx in range(n_layers):
                df = idx / max(n_layers - 1, 1)
                delta = grag_shallow + (grag_deep - grag_shallow) * df
                k, v = ref_k_list[idx], ref_v_list[idx]
                kt, ki = k[:, :, :txt_len, :], k[:, :, txt_len:, :]
                vt, vi = v[:, :, :txt_len, :], v[:, :, txt_len:, :]
                km, vm = ki.mean(2, keepdim=True), vi.mean(2, keepdim=True)
                ref_k_list[idx] = torch.cat([kt, km + delta * (ki - km)], dim=2)
                ref_v_list[idx] = torch.cat([vt, vm + delta * (vi - vm)], dim=2)

            # content forward -> q + current attention outputs
            cache.clear(); controller.step()
            _ = dit(hidden_states=latents,
                    timestep=ts.to("cuda", torch.bfloat16) / 1000, guidance=None,
                    encoder_hidden_states=base.null_embeds,
                    txt_ids=base.null_text_ids, img_ids=image_ids)
            q_list, _, _, self_out_list = cache.get()

            n_ot = max(1, int(n_layers * 0.5))
            deltas = []
            for idx2, (q, rk, rv, so) in enumerate(
                    zip(q_list, ref_k_list, ref_v_list, self_out_list)):
                if idx2 >= n_layers - n_ot:
                    target = base.sinkhorn_attn_freq(q.detach(), rk, rv,
                                                     n_iters=6, tau=0.1,
                                                     hi_boost=2.0)
                else:
                    target = F.scaled_dot_product_attention(q.detach(), rk, rv)
                deltas.append((target - so)[:, :, txt_len:, :])
            inject_avg = torch.stack([d.mean(dim=1) for d in deltas]).mean(dim=0)
            field = inject_avg[0].float().cpu().numpy().reshape(lat_h, lat_w, -1)
            cos = base.neighbor_cosine_at_distance(field, 1)
            print(f"  inject cos@d1: {cos:.4f}", flush=True)
            np.savez_compressed(OUT / f"inject_{kp}_s{sid}.npz",
                                inject=field.astype(np.float16),
                                cos_d1=np.array([cos]),
                                prompt=np.array(prompt), sid=np.array(sid))
            break


@torch.no_grad()
def capture_lrs_qk(style_sid="20", prompt="A castle on a cliff by the sea"):
    """Minimal replica of capture_update_fields' first captured step, keeping
    the per-layer style K (style forward) and content Q (content forward)."""
    print(f"\n[lrs] q/k capture: '{prompt}' x sty/{style_sid}", flush=True)
    pipe, dit, vae = base.pipe, base.dit, base.vae
    cache, controller = base.cache, base.controller
    width = height = 512
    lat_h, lat_w = height // 16, width // 16

    pipe.load_models_to_device(["vae"])
    sty_pix = pipe.preprocess_image(
        Image.open(f"{base.CWD}/sty/{style_sid}.png").convert("RGB")
             .resize((512, 512), Image.BICUBIC))
    style_latent_4d = vae.encode(sty_pix)
    style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")
    sty_h, sty_w = style_latent_4d.shape[2], style_latent_4d.shape[3]

    image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lat_h),
                                     torch.arange(lat_w), torch.arange(1)
                                     ).unsqueeze(0).to("cuda")
    style_image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(sty_h),
                                           torch.arange(sty_w), torch.arange(1)
                                           ).unsqueeze(0).to("cuda")
    prompt_embeds, text_ids = base.encode_prompt(prompt)
    pipe.load_models_to_device(["dit"]); dit.to("cuda")

    torch.manual_seed(43)
    latents = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    latents = rearrange(latents, "B C H W -> B (H W) C")
    scheduler = base.FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(25, dynamic_shift_len=lat_h * lat_w)

    for i, t_val in enumerate(scheduler.timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        cache.clear(); controller.step()
        noise_pred = dit(hidden_states=latents, timestep=timestep / 1000,
                         guidance=None, encoder_hidden_states=prompt_embeds,
                         txt_ids=text_ids, img_ids=image_ids)
        latents = scheduler.step(noise_pred, t_val, latents)
        if t_val >= 850:
            continue

        # style forward -> per-layer K
        ts = scheduler.timesteps[i + 1:i + 2]
        style_noisy = scheduler.add_noise(style_latent,
                                          torch.randn_like(style_latent), ts)
        cache.clear(); controller.step()
        _ = dit(hidden_states=style_noisy.to("cuda", torch.bfloat16),
                timestep=ts.to("cuda", torch.bfloat16) / 1000, guidance=None,
                encoder_hidden_states=base.null_embeds,
                txt_ids=base.null_text_ids, img_ids=style_image_ids)
        _, k_list, _, _ = cache.get()

        # content forward at the same timestep -> per-layer Q
        cache.clear(); controller.step()
        _ = dit(hidden_states=latents,
                timestep=ts.to("cuda", torch.bfloat16) / 1000, guidance=None,
                encoder_hidden_states=base.null_embeds,
                txt_ids=base.null_text_ids, img_ids=image_ids)
        q_list, _, _, _ = cache.get()

        txt_len = controller.txt_len
        n = len(k_list)
        layers = {"shallow": 1, "deep": n - 2}
        payload = {"txt_len": np.array(txt_len), "n_layers": np.array(n),
                   "lat_h": np.array(lat_h), "lat_w": np.array(lat_w),
                   "sty_h": np.array(sty_h), "sty_w": np.array(sty_w)}
        for tag, li in layers.items():
            k_img = k_list[li][0, :, txt_len:, :].float().cpu().numpy()
            q_img = q_list[li][0, :, txt_len:, :].float().cpu().numpy()
            payload[f"k_{tag}"] = k_img.astype(np.float16)
            payload[f"q_{tag}"] = q_img.astype(np.float16)
            print(f"  layer {li} ({tag}): k{k_img.shape} q{q_img.shape}", flush=True)
        np.savez_compressed(OUT / f"lrs_qk_s{style_sid}.npz", **payload)
        return
    raise RuntimeError("no eligible step reached")


if __name__ == "__main__":
    t0 = time.time()
    if "--inject" in sys.argv:      # 2026-07-03 addendum: inject column only
        capture_inject_fields()
    else:
        capture_fields()
        capture_lrs_qk()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min → {OUT}", flush=True)
