"""
TASK #4 — Theory assumption profiling (sec/supp.tex Theorem 1 & Theorem 2).

Replicates eval_all.generate_t2i's Adam loop (no edits to eval_all beyond the
seed line) with hooks that dump per-iter gradient norms and content-key tensors,
then measures the theorem constants empirically on 30 (style,prompt) pairs.

Exact definitions (sec/supp.tex + sec/03_method.tex sec:method:freq):
  * epsilon  (Thm 1, eq:supp_smooth): ||K_{c,i} - K_{c,i+1}||_2 / sqrt(d) for
      spatially-adjacent tokens on the 32x32 latent grid; content keys (with
      RoPE) in the self-attention layers. Report mean / p50 / p95.
  * The style loss is decomposed by FREQUENCY (Eq.~eq:freq,
      L_style = L_lo + gamma_hi*L_hi; Thm2 L = L_c + L_t):
        L_c = L_lo = low-freq (color) band,  L_t = L_hi = high-freq (texture) band,
      via the per-channel mean/residual split used by sinkhorn_attn_freq
      (lo = mean over tokens, hi = residual). g_c = grad L_lo, g_t = grad L_hi,
      g = g_c + g_t.
  * kappa  (Assumption A1): kappa = ||g_c|| / ||g||, profiled on PURE ADAM at the
      denoising start point z^(0) (paper: ~0.85 over first iters).
  * delta  (Assumption A3): ||g_t(z^SLT) - g_t(z^(0))|| / ||g_t(z^(0))|| with
      g_t = high-freq band gradient (paper ~0.04).
  * cos(g_c, g_t)  (Thm2 Step 3): near-orthogonality of color/texture bands
      (paper 0.07 +/- 0.03).

Secondary diagnostic (the task's literal "content-loss term vs style-loss term"
gloss): kappa_term = ||w_c grad L_cnt|| / ||w_c grad L_cnt + w_s grad L_style||,
reported alongside so the two readings are explicit.

Saves results/theory_profiling.json and prints aggregate stats.

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/profile_theory.py
"""
import os, sys, json, time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import eval_all as E  # single Klein-9B load

torch = E.torch
F = E.F
rearrange = E.rearrange

# per-iter kappa is profiled on PURE ADAM (the paper's "pure Adam" baseline,
# 8 iters, no SLT) -> ADAM8_DEFAULT; delta uses the hybrid SLT step.
ADAM_CFG = E.ADAM8_DEFAULT
SLT_ALPHA = E.HYBRID_DEFAULT["slt_alpha"]   # 0.1
PAPER = {"kappa": 0.85, "cos_gc_gt": [0.07, 0.03], "delta": 0.04}
OUT_JSON = E.RESULTS_DIR / "theory_profiling.json"

_OT_TAU_MAX, _OT_TAU_MIN, _OT_ITERS, _OT_LAYER_FRAC = 0.15, 0.07, 6, 0.5
loss_fn = torch.nn.L1Loss()


def _eps_from_keys(k_list, txt_len):
    """epsilon samples: ||K_i - K_{i+1}||_2 / sqrt(d), adjacent tokens on 32x32."""
    out = []
    for k in k_list:                        # k: [B, H, txt_len+L, D]
        img = k[:, :, txt_len:, :].float()  # content image keys (with RoPE)
        B, H, L, D = img.shape
        side = int(round(L ** 0.5))
        if side * side != L:
            continue
        grid = img.reshape(B, H, side, side, D)
        dh = (grid[:, :, :, 1:, :] - grid[:, :, :, :-1, :]).norm(dim=-1) / (D ** 0.5)
        dv = (grid[:, :, 1:, :, :] - grid[:, :, :-1, :, :]).norm(dim=-1) / (D ** 0.5)
        out.append(torch.cat([dh.reshape(-1), dv.reshape(-1)]).detach().cpu().numpy())
    if not out:
        return np.array([])
    allv = np.concatenate(out)
    if allv.size > 4000:
        allv = np.random.default_rng(0).choice(allv, 4000, replace=False)
    return allv


def _grads(latents, ref_k_list, ref_v_list, content_anchor, ts_cuda, image_ids,
           t_val, start_time_val):
    """Forward on live leaf `latents`; return frequency-band + term gradients.

    Returns dict with g_lo, g_hi (style low/high-freq band grads),
    g_style, g_cnt (full style / content term grads), and eps samples.
    """
    E.cache.clear(); E.controller.step()
    with torch.enable_grad():
        _ = E.dit(hidden_states=latents, timestep=ts_cuda.detach(), guidance=None,
                  encoder_hidden_states=E.null_embeds.detach(),
                  txt_ids=E.null_text_ids.detach(), img_ids=image_ids.detach())
    q_list, k_list, _, self_out_list = E.cache.get()
    _t_float = float(t_val.item())
    _progress = 1.0 - (_t_float / max(start_time_val, 1.0))
    _tau = _OT_TAU_MAX - (_OT_TAU_MAX - _OT_TAU_MIN) * _progress
    n_layers = len(q_list)
    n_ot = max(1, int(n_layers * _OT_LAYER_FRAC))
    ref = []
    for idx2, (q, rk, rv) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
        qd = q.detach()
        if idx2 >= n_layers - n_ot:
            ref.append(E.sinkhorn_attn_freq(qd, rk, rv, n_iters=_OT_ITERS, tau=_tau,
                                            hi_boost=ADAM_CFG["freq_hi_boost"]))
        else:
            ref.append(F.scaled_dot_product_attention(qd, rk, rv))

    style_loss = sum(loss_fn(so, r.detach()) for so, r in zip(self_out_list, ref))
    L_lo = 0.0; L_hi = 0.0
    for so, r in zip(self_out_list, ref):
        r = r.detach()
        so_lo = so.mean(dim=2, keepdim=True); so_hi = so - so_lo      # per-channel freq split
        r_lo = r.mean(dim=2, keepdim=True);  r_hi = r - r_lo
        L_lo = L_lo + loss_fn(so_lo, r_lo)
        L_hi = L_hi + loss_fn(so_hi, r_hi)
    content_loss = loss_fn(latents, content_anchor)

    g_lo = torch.autograd.grad(L_lo, latents, retain_graph=True)[0].detach()
    g_hi = torch.autograd.grad(L_hi, latents, retain_graph=True)[0].detach()
    g_style = torch.autograd.grad(style_loss, latents, retain_graph=False)[0].detach()
    g_cnt = torch.autograd.grad(content_loss, latents, retain_graph=False)[0].detach()
    return {"g_lo": g_lo, "g_hi": g_hi, "g_style": g_style, "g_cnt": g_cnt,
            "eps": _eps_from_keys(k_list, E.controller.txt_len)}


def _n(x):
    return float(x.float().norm().item())


def _cos(a, b):
    na, nb = _n(a), _n(b)
    return float((a.float() * b.float()).sum().item()) / max(na * nb, 1e-12)


def _metrics(gr, w_c, w_s):
    """kappa/cos under the frequency decomposition (primary) and term decomposition."""
    g_lo, g_hi = gr["g_lo"], gr["g_hi"]
    g_f = g_lo + g_hi
    kap_f = _n(g_lo) / max(_n(g_f), 1e-12)
    cos_f = _cos(g_lo, g_hi)
    g_ct = w_c * gr["g_cnt"]; g_tt = w_s * gr["g_style"]
    g_t = g_ct + g_tt
    kap_t = _n(g_ct) / max(_n(g_t), 1e-12)
    cos_t = _cos(g_ct, g_tt)
    return kap_f, cos_f, kap_t, cos_t


def profile_pair(prompt, style_path):
    cfg = ADAM_CFG
    lr = cfg["lr"]; w_c = cfg["content_w"]; w_s = cfg["style_w"]
    adam_iters = cfg["iters"]
    adain_alpha = E.HYBRID_DEFAULT.get("adain_alpha", 0.5)
    start_time_val = cfg.get("start_time", 850)
    grag_deep, grag_shallow = cfg["grag_deep"], cfg["grag_shallow"]

    width = height = 512
    num_inference_steps = 25
    seed = int(os.environ.get("EVAL_SEED", 43))

    E.pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        sty_pix = E.pipe.preprocess_image(
            E.Image.open(style_path).convert("RGB").resize((512, 512), E.Image.BICUBIC))
        style_latent_4d = E.vae.encode(sty_pix)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

    lat_h, lat_w = height // 16, width // 16
    image_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)
    ).unsqueeze(0).to("cuda")
    sty_lat_h, sty_lat_w = style_latent_4d.shape[2], style_latent_4d.shape[3]
    style_image_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(sty_lat_h), torch.arange(sty_lat_w), torch.arange(1)
    ).unsqueeze(0).to("cuda")

    prompt_embeds, text_ids = E.encode_prompt(prompt)
    E.pipe.load_models_to_device(["dit"]); E.dit.to("cuda")

    torch.manual_seed(seed)
    noise = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    noise = rearrange(noise, "B C H W -> B (H W) C")
    latents = noise.clone()

    scheduler = E.FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(num_inference_steps, dynamic_shift_len=lat_h * lat_w)
    timesteps = scheduler.timesteps

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        E.cache.clear(); E.controller.step()
        with torch.no_grad():
            noise_pred = E.dit(hidden_states=latents, timestep=timestep / 1000, guidance=None,
                               encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=image_ids)
        latents = scheduler.step(noise_pred, t_val, latents)
        if t_val >= start_time_val:
            continue

        # ===== first optimization-active step: instrument =====
        if i < num_inference_steps - 1:
            timestep_style = timesteps[i + 1:i + 2]
            sty_noise = torch.randn_like(style_latent)
            style_latent_ = scheduler.add_noise(style_latent, sty_noise, timestep_style)
        else:
            timestep_style = torch.tensor([0.0]); style_latent_ = style_latent

        # style forward -> ref K/V
        E.cache.clear(); E.controller.step()
        with torch.no_grad():
            _ = E.dit(hidden_states=style_latent_.to(device="cuda", dtype=torch.bfloat16),
                      timestep=timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000,
                      guidance=None, encoder_hidden_states=E.null_embeds,
                      txt_ids=E.null_text_ids, img_ids=style_image_ids)
        _, ref_k_list, ref_v_list, _ = E.cache.get()

        # GRAG (LRS)
        _txt_len = E.controller.txt_len
        n_cached = len(ref_k_list)
        for idx in range(n_cached):
            depth_frac = idx / max(n_cached - 1, 1)
            delta_i = grag_shallow + (grag_deep - grag_shallow) * depth_frac
            k, v = ref_k_list[idx], ref_v_list[idx]
            k_txt, k_img = k[:, :, :_txt_len, :], k[:, :, _txt_len:, :]
            v_txt, v_img = v[:, :, :_txt_len, :], v[:, :, _txt_len:, :]
            km, vm = k_img.mean(dim=2, keepdim=True), v_img.mean(dim=2, keepdim=True)
            ref_k_list[idx] = torch.cat([k_txt, km + delta_i * (k_img - km)], dim=2)
            ref_v_list[idx] = torch.cat([v_txt, vm + delta_i * (v_img - vm)], dim=2)

        content_anchor = latents.detach().clone()
        ts_cuda = timestep_style.to(device="cuda", dtype=torch.bfloat16) / 1000

        # AdaIN warm-start -> z^(0)
        with torch.no_grad():
            latents = E.feature_adain(latents.unsqueeze(1), style_latent_.unsqueeze(1),
                                      alpha=adain_alpha).squeeze(1)
        z0 = latents.detach()

        def grads_at(z):
            leaf = z.clone().requires_grad_(True)
            return _grads(leaf, ref_k_list, ref_v_list, content_anchor, ts_cuda,
                          image_ids, t_val, start_time_val)

        # --- z^(0) diagnostics (A1 pure-Adam start) ---
        gr0 = grads_at(z0)
        kap_f0, cos_f0, kap_t0, cos_t0 = _metrics(gr0, w_c, w_s)
        g_hi0 = gr0["g_hi"]

        # --- delta: SLT step (hybrid) ---
        with torch.no_grad():
            z_slt = E.sinkhorn_latent_transport(z0, style_latent_, style_latent,
                                                alpha=SLT_ALPHA, n_iters=10, tau=0.02,
                                                grag_delta=grag_deep, mode="ot_adain")
        gr_slt = grads_at(z_slt)
        delta = _n(gr_slt["g_hi"] - g_hi0) / max(_n(g_hi0), 1e-12)
        kap_f_slt = _metrics(gr_slt, w_c, w_s)[0]   # A2: should drop vs kap_f0

        # --- per-iter kappa/cos on PURE ADAM from z^(0) ---
        leaf = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([leaf], lr=lr)
        kf, cf, kt, ct = [], [], [], []
        eps_iter1 = np.array([])
        for it in range(adam_iters):
            opt.zero_grad()
            gr = _grads(leaf, ref_k_list, ref_v_list, content_anchor, ts_cuda,
                        image_ids, t_val, start_time_val)
            a, b, c, d = _metrics(gr, w_c, w_s)
            kf.append(a); cf.append(b); kt.append(c); ct.append(d)
            if it == 0:
                eps_iter1 = gr["eps"]
            leaf.grad = w_s * gr["g_style"] + w_c * gr["g_cnt"]   # real total-loss step
            opt.step()

        return {"kappa_freq": kf, "cos_freq": cf, "kappa_term": kt, "cos_term": ct,
                "kappa_freq_z0": kap_f0, "cos_freq_z0": cos_f0,
                "kappa_term_z0": kap_t0, "cos_term_z0": cos_t0,
                "kappa_freq_zslt": kap_f_slt, "delta": delta,
                "eps": eps_iter1, "active_t": float(t_val.item()), "step_i": i}
    return None


def main():
    styles = E.STYLES_6[:6]
    pairs = [(pid, prompt, s) for s in styles for (pid, prompt) in E.T2I_PROMPTS_30]
    print(f"Theory profiling on {len(pairs)} (style,prompt) pairs "
          f"(pure-Adam iters={ADAM_CFG['iters']}, slt_alpha={SLT_ALPHA})", flush=True)

    per_pair, eps_chunks = [], []
    for n, (pid, prompt, sty) in enumerate(pairs):
        t0 = time.time()
        r = profile_pair(prompt, f"{E.CWD}/sty/{sty}.png")
        if r is None:
            print(f"[warn] no active step for {sty}/{pid}"); continue
        per_pair.append({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in r.items() if k != "eps"})
        per_pair[-1]["eps_mean"] = float(np.mean(r["eps"])) if r["eps"].size else None
        if r["eps"].size:
            eps_chunks.append(r["eps"])
        print(f"[{n+1:>2}/{len(pairs)}] {sty}/{pid}  kF(z0)={r['kappa_freq_z0']:.3f} "
              f"kF1={r['kappa_freq'][0]:.3f} cosF_mean={np.mean(r['cos_freq']):+.3f} "
              f"delta={r['delta']:.4f} eps_p50={np.median(r['eps']) if r['eps'].size else float('nan'):.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    KF = np.array([p["kappa_freq"] for p in per_pair])
    CF = np.array([p["cos_freq"] for p in per_pair])
    KT = np.array([p["kappa_term"] for p in per_pair])
    CT = np.array([p["cos_term"] for p in per_pair])
    D = np.array([p["delta"] for p in per_pair])
    KF0 = np.array([p["kappa_freq_z0"] for p in per_pair])
    CF0 = np.array([p["cos_freq_z0"] for p in per_pair])
    KFS = np.array([p["kappa_freq_zslt"] for p in per_pair])
    eps_all = np.concatenate(eps_chunks) if eps_chunks else np.array([])

    def ms(a):
        a = np.asarray(a, float)
        return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0}

    summary = {
        "definitions": {
            "g_c": "grad of low-freq (color) band of style loss (primary, supp.tex L_c=L_lo)",
            "g_t": "grad of high-freq (texture) band of style loss (primary, supp.tex L_t=L_hi)",
            "kappa": "||g_c||/||g||, g=g_c+g_t; profiled on pure Adam at z^(0)",
            "delta": "||g_t(z^SLT)-g_t(z^(0))||/||g_t(z^(0))||",
            "epsilon": "||K_i-K_{i+1}||_2/sqrt(d), adjacent 32x32 self-attn content keys (RoPE)",
            "secondary_term": "kappa_term/cos_term use content-loss-term vs style-loss-term grads",
        },
        "adam_config": {k: ADAM_CFG[k] for k in ("mode", "iters", "lr", "content_w", "style_w",
                                                 "freq_hi_boost", "sinkhorn_mode", "grag_deep", "grag_shallow")},
        "slt_alpha": SLT_ALPHA,
        "n_pairs": len(per_pair),
        "epsilon": {
            "mean": float(np.mean(eps_all)) if eps_all.size else None,
            "p50": float(np.percentile(eps_all, 50)) if eps_all.size else None,
            "p95": float(np.percentile(eps_all, 95)) if eps_all.size else None,
            "n_samples": int(eps_all.size),
        },
        "kappa_freq_z0": {**ms(KF0), "note": "pure-Adam start, paper assumes ~0.85"},
        "kappa_freq_per_iter": {str(it + 1): ms(KF[:, it]) for it in range(KF.shape[1])},
        "kappa_freq_overall": ms(KF.reshape(-1)),
        "kappa_freq_zslt": {**ms(KFS), "note": "A2: color resolved after SLT -> should drop"},
        "cos_gc_gt_freq_z0": ms(CF0),
        "cos_gc_gt_freq_per_iter": {str(it + 1): ms(CF[:, it]) for it in range(CF.shape[1])},
        "cos_gc_gt_freq_overall": ms(CF.reshape(-1)),
        "delta": {**ms(D), "p50": float(np.percentile(D, 50))},
        "secondary_term_decomposition": {
            "kappa_term_z0": ms(KT[:, 0] if KT.ndim == 2 else KT),
            "kappa_term_overall": ms(KT.reshape(-1)),
            "cos_term_overall": ms(CT.reshape(-1)),
        },
        "paper_assumed": PAPER,
        "per_pair": per_pair,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- print ----
    e = summary["epsilon"]
    print(f"\n{'='*82}\nTHEORY PROFILING  (n_pairs={summary['n_pairs']})\n{'='*82}")
    print(f"epsilon (||dK||/sqrt(d), adjacent 32x32):  mean={e['mean']:.4f}  "
          f"p50={e['p50']:.4f}  p95={e['p95']:.4f}  (n={e['n_samples']})")
    print(f"\n[PRIMARY: frequency (color/texture) decomposition of the style loss]")
    print(f"kappa = ||g_lo||/||g||  (paper assumes ~{PAPER['kappa']}):")
    print(f"  z^(0) pure-Adam start: {summary['kappa_freq_z0']['mean']:.3f} +/- {summary['kappa_freq_z0']['std']:.3f}")
    for it in range(KF.shape[1]):
        k = summary["kappa_freq_per_iter"][str(it + 1)]
        print(f"    iter {it+1}: {k['mean']:.3f} +/- {k['std']:.3f}")
    print(f"  z^SLT (A2 check): {summary['kappa_freq_zslt']['mean']:.3f} +/- {summary['kappa_freq_zslt']['std']:.3f}")
    co = summary["cos_gc_gt_freq_overall"]; co0 = summary["cos_gc_gt_freq_z0"]
    print(f"cos(g_c,g_t): z0={co0['mean']:+.3f}+/-{co0['std']:.3f}  overall={co['mean']:+.3f}+/-{co['std']:.3f}  "
          f"(paper {PAPER['cos_gc_gt'][0]}+/-{PAPER['cos_gc_gt'][1]})")
    print(f"delta = ||g_t(z^SLT)-g_t(z0)||/||g_t(z0)||: {summary['delta']['mean']:.4f} +/- "
          f"{summary['delta']['std']:.4f}  p50={summary['delta']['p50']:.4f}  (paper ~{PAPER['delta']})")
    st = summary["secondary_term_decomposition"]
    print(f"\n[SECONDARY: content-term vs style-term gloss]")
    print(f"  kappa_term overall={st['kappa_term_overall']['mean']:.3f}  cos_term overall={st['cos_term_overall']['mean']:+.3f}")
    print(f"\nSaved: {OUT_JSON}")
    return summary


if __name__ == "__main__":
    main()
