import os
import torch
import torch.nn.functional as F
import utils

# Weights controllable via env vars
_DIST_W    = float(os.environ.get("DIST_W",    "0.15"))   # distribution pull weight
_CONTENT_W = float(os.environ.get("CONTENT_W", "0.6"))    # content structure anchor (V4: doubled)
_OT_ALPHA  = float(os.environ.get("OT_ALPHA",  "0.5"))    # warm-start OT strength (conservative)

def step(**kwargs):
    i                    = kwargs["i"]
    t                    = kwargs["t"]
    timestep             = kwargs["timestep"]
    latents              = kwargs["latents"]
    guidance             = kwargs["guidance"]
    pooled_prompt_embeds = kwargs["pooled_prompt_embeds"]
    prompt_embeds        = kwargs["prompt_embeds"]
    text_ids             = kwargs["text_ids"]
    latent_image_ids     = kwargs["latent_image_ids"]
    num_inference_steps  = kwargs["num_inference_steps"]
    timesteps            = kwargs["timesteps"]
    style_latent         = kwargs["style_latent"]
    scheduler            = kwargs["scheduler"]
    cache                = kwargs["cache"]
    controller           = kwargs["controller"]
    transformer          = kwargs["transformer"]
    null_guidance        = kwargs["null_guidance"]
    null_pooled_embeds   = kwargs["null_pooled_embeds"]
    null_embeds          = kwargs["null_embeds"]
    null_text_ids        = kwargs["null_text_ids"]
    null_image_id        = kwargs["null_image_id"]
    lr                   = kwargs["lr"]
    accelerator          = kwargs["accelerator"]
    iters                = max(kwargs["iters"], 3)   # V8: min 3 iters
    loss_fn              = kwargs["loss_fn"]
    pbar                 = kwargs["pbar"]
    start_time           = kwargs["start_time"]
    style_beta           = kwargs["style_beta"]

    # ── 0. One-time rank detection on style latent (cache on controller) ──
    # Drives BOTH adaptive OT warm-start and adaptive start_time.
    # Low-rank styles (sparse, flat) need earlier style intervention AND no OT;
    # high-rank painterly styles follow current Gold Standard.
    if not hasattr(controller, "_rank_cached"):
        with torch.no_grad():
            sl = style_latent[0].float()                  # [L, D]
            sl_c = sl - sl.mean(dim=0, keepdim=True)
            L_toks, D_ = sl_c.shape
            C = sl_c.T @ sl_c / L_toks
            C = (C + C.T) / 2
            S = torch.linalg.eigvalsh(C + 1e-8 * torch.eye(D_, device=sl.device))
            S = torch.clamp(S, min=0).flip(0)             # descending
            p = (S / (S.sum() + 1e-8)).cpu().numpy()
            import numpy as _np
            cum = _np.cumsum(p)
            r95 = int(_np.searchsorted(cum, 0.95) + 1)

        # V8: disable OT covariance warp (causes stripe artifacts).
        controller._cached_ot_alpha = 0.0

        # style_loss weight: low-rank → boosted for style fidelity;
        # high-rank → reduced to preserve text semantics (CLIP-T).
        LO_SW, HI_SW = 24.0, 48.0
        frac_sw = max(0.0, min(1.0, (r95 - LO_SW) / (HI_SW - LO_SW)))
        _style_w_base = 1.5 - 0.5 * frac_sw                         # V8: 1.5..1.0
        _style_w_scale = float(os.environ.get("STYLE_W_SCALE", "1.0"))
        controller._cached_style_w = _style_w_base * _style_w_scale

        # mu_blend weight for OT warm-start: high-rank → lower (OT handles color);
        # low-rank doesn't hit this path anyway (alpha=0), but keep consistent.
        # rank=38 → 0.9 ; rank=46 → 0.7
        LO_MU, HI_MU = 38.0, 46.0
        frac_mu = max(0.0, min(1.0, (r95 - LO_MU) / (HI_MU - LO_MU)))
        controller._cached_mu_weight = 0.9 - 0.2 * frac_mu          # 0.9..0.7

        # CSD_W rank-adaptive: V4 tapers further to avoid Goodhart.
        # low-rank → 0.5x, high-rank → 0.15x  (was 0.8x..0.2x)
        _base_csd_w = getattr(controller, "csd_w", 0.0)
        if _base_csd_w > 0:
            LO_CW, HI_CW = 24.0, 44.0
            frac_cw = max(0.0, min(1.0, (r95 - LO_CW) / (HI_CW - LO_CW)))
            adapted_cw = _base_csd_w * (0.5 - 0.35 * frac_cw)       # 0.5x..0.15x
            controller.csd_w = max(0.03, adapted_cw)
            print(f"[CSD-adaptive] base_w={_base_csd_w} → {controller.csd_w:.3f}")

        # start_time: V5 default=650, mild rank-adaptive nudge.
        LO_ST, HI_ST = 24.0, 44.0
        frac_st = max(0.0, min(1.0, (r95 - LO_ST) / (HI_ST - LO_ST)))
        adjusted_start_time = int(660 - 20 * frac_st)     # 660..640
        # OVERRIDE_START_TIME env var takes precedence (ablation / i2i methods)
        if "OVERRIDE_START_TIME" in os.environ:
            adjusted_start_time = int(os.environ["OVERRIDE_START_TIME"])
        controller._cached_start_time = adjusted_start_time

        controller._cached_rank95 = r95
        controller._rank_cached = True
        print(f"[adaptive] rank_95={r95} → OT_ALPHA={controller._cached_ot_alpha:.3f} "
              f"start_time={adjusted_start_time} style_w={controller._cached_style_w:.2f} "
              f"mu_w={controller._cached_mu_weight:.2f}")

    start_time = controller._cached_start_time

    # ── 1. Standard denoising step ────────────────────────────────────────
    with torch.no_grad():
        noise_pred = transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    if t >= start_time:
        return latents

    # ── 2. Noisy style latent at next timestep ────────────────────────────
    if i < num_inference_steps - 1:
        timestep_style = timesteps[i + 1 : i + 2]
        has_noise = True
    else:
        timestep_style = torch.tensor([0], device=style_latent.device)
        has_noise = False

    # ── 3. Style forward → ref K/V  (K-shot manifold expectation) ─────────
    K          = int(os.environ.get("KSHOT", "1"))
    NOISE_MODE = os.environ.get("NOISE_MODE", "std").upper()

    def _style_forward(sl):
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = transformer(
                hidden_states=sl,
                timestep=timestep_style / 1000,
                guidance=null_guidance,
                pooled_projections=null_pooled_embeds,
                encoder_hidden_states=null_embeds,
                txt_ids=null_text_ids,
                img_ids=null_image_id,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
        _, k_list, v_list, _ = cache.get()
        return k_list, v_list

    def _sample_style_latent(shot_idx):
        if not has_noise:
            return style_latent
        if NOISE_MODE == "B" and hasattr(controller, "style_latent_bank"):
            base = controller.style_latent_bank[shot_idx % len(controller.style_latent_bank)]
            noise = torch.randn_like(base)
            return scheduler.scale_noise(base, timestep_style, noise)
        if NOISE_MODE == "A":
            mu = style_latent.mean(dim=1, keepdim=True)
            sd = style_latent.std(dim=1,  keepdim=True) + 1e-6
            z  = torch.randn_like(style_latent)
            noise = mu + sd * z
            return scheduler.scale_noise(style_latent, timestep_style, noise)
        noise = torch.randn_like(style_latent)
        return scheduler.scale_noise(style_latent, timestep_style, noise)

    style_latent_ = _sample_style_latent(0)
    ref_k_list, ref_v_list = _style_forward(style_latent_)

    if K > 1 and has_noise:
        for sh in range(1, K):
            sl_aug = _sample_style_latent(sh)
            k_aug, v_aug = _style_forward(sl_aug)
            for idx in range(len(ref_k_list)):
                ref_k_list[idx] = ref_k_list[idx] + k_aug[idx]
                ref_v_list[idx] = ref_v_list[idx] + v_aug[idx]
        for idx in range(len(ref_k_list)):
            ref_k_list[idx] = ref_k_list[idx] / K
            ref_v_list[idx] = ref_v_list[idx] / K

    # ── 3.5. GRAG: compress ref K/V spatial variance to kill layout leakage ─
    r95 = controller._cached_rank95
    if os.environ.get("GRAG_OFF", "0") == "1":
        grag_delta_deep = 1.0
        grag_delta_shallow = 1.0
    elif "GRAG_FORCE" in os.environ:
        grag_delta_deep = float(os.environ["GRAG_FORCE"])
        grag_delta_shallow = min(0.65, grag_delta_deep + 0.3)
    else:
        grag_delta_deep = 0.15
        _LO_G = int(os.environ.get("LO_GRAG", "30"))
        _HI_G = int(os.environ.get("HI_GRAG", "48"))
        if r95 <= _LO_G:
            grag_delta_shallow = 0.75
        elif r95 <= 44:
            grag_delta_shallow = 0.55
        else:
            grag_delta_shallow = 0.45

    if not hasattr(controller, "_grag_logged"):
        controller._grag_logged = True
        print(f"[GRAG] rank_95={r95} → shallow={grag_delta_shallow:.2f} deep={grag_delta_deep:.2f}")

    _txt_len = getattr(controller, 'txt_len', 0)
    n_cached = len(ref_k_list)
    for idx in range(n_cached):
        depth_frac = idx / max(n_cached - 1, 1)
        delta_i = grag_delta_shallow + (grag_delta_deep - grag_delta_shallow) * depth_frac
        k = ref_k_list[idx]
        v = ref_v_list[idx]
        k_txt, k_img = k[:, :, :_txt_len, :], k[:, :, _txt_len:, :]
        v_txt, v_img = v[:, :, :_txt_len, :], v[:, :, _txt_len:, :]
        km = k_img.mean(dim=2, keepdim=True)
        vm = v_img.mean(dim=2, keepdim=True)
        k_img = km + delta_i * (k_img - km)
        v_img = vm + delta_i * (v_img - vm)
        ref_k_list[idx] = torch.cat([k_txt, k_img], dim=2)
        ref_v_list[idx] = torch.cat([v_txt, v_img], dim=2)

    # ── 4. Warm-start: AdaIN
    _adain_alpha = controller._cached_mu_weight
    with torch.no_grad():
        latents_warmed = utils.feature_adain(
            latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=_adain_alpha
        ).squeeze(1)

    # Content anchor (legacy drift-control only; no semantic info)
    content_anchor = latents_warmed.detach().clone()
    latents = latents_warmed.detach()

    # ── CONTENT BLEND (no loss, no gradient) ──────────────────────────────
    # SDEdit-style mixing: lerp current latent with noised clean content_latent
    # only at high-t ("coarse structure" phase). Adam then reclaims style with
    # boosted iteration budget. Knobs:
    #   CNT_BLEND_W       — max blend weight (0..1), default 0
    #   CNT_BLEND_MIN_T   — only blend when t >= this (default 700; 0 = always)
    #   CNT_BLEND_BOOST   — Adam iters override during blend steps (default 5)
    _CNT_BLEND_W     = float(os.environ.get("CNT_BLEND_W",      "0.0"))
    _CNT_BLEND_MIN_T = float(os.environ.get("CNT_BLEND_MIN_T",  "700"))
    _CNT_BLEND_BOOST = int(  os.environ.get("CNT_BLEND_BOOST",  "5"))
    _blend_active = (_CNT_BLEND_W > 0 and hasattr(controller, "content_latent")
                     and has_noise and float(t.item()) >= _CNT_BLEND_MIN_T)
    if _blend_active:
        with torch.no_grad():
            c_lat_b = controller.content_latent.to(latents.dtype)
            n_fixed = controller.content_noise.to(latents.dtype)
            c_noisy = scheduler.scale_noise(c_lat_b, timestep_style, n_fixed)
            # Linear decay from full strength at start_time to 0 at MIN_T
            span = max(start_time - _CNT_BLEND_MIN_T, 1.0)
            prog = max(0.0, (float(t.item()) - _CNT_BLEND_MIN_T) / span)
            w_eff = _CNT_BLEND_W * prog
            latents = ((1.0 - w_eff) * latents + w_eff * c_noisy).detach()

    _iters_this_step = iters
    if _blend_active and _CNT_BLEND_BOOST > iters:
        _iters_this_step = _CNT_BLEND_BOOST

    # ── 4.5. OPTIONAL: content K/V forward (Method D) ────────────────────
    # If controller.content_latent is set (i2i mode with content injection),
    # run a parallel forward on noisy content latent at the same timestep,
    # cache its K/V to pull Adam back toward content-consistent attention.
    cnt_k_list = cnt_v_list = None
    _CNT_KV_W = float(os.environ.get("CNT_KV_W", "0.0"))
    if _CNT_KV_W > 0 and hasattr(controller, "content_latent"):
        c_lat = controller.content_latent
        if has_noise:
            c_noise = torch.randn_like(c_lat)
            c_lat_t = scheduler.scale_noise(c_lat, timestep_style, c_noise)
        else:
            c_lat_t = c_lat
        cache.clear(); controller.step()
        with torch.no_grad():
            _ = transformer(
                hidden_states=c_lat_t,
                timestep=timestep_style / 1000,
                guidance=null_guidance,
                pooled_projections=null_pooled_embeds,
                encoder_hidden_states=null_embeds,
                txt_ids=null_text_ids,
                img_ids=null_image_id,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
        _, cnt_k_list, cnt_v_list, _ = cache.get()

    # ── 5. Adam: style K/V loss + distribution pull + content anchor ──────
    optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
    optimizer = accelerator.prepare(optimizer)

    for it in range(_iters_this_step):
        cache.clear(); controller.step()
        optimizer.zero_grad()
        _ = transformer(
            hidden_states=latents,
            timestep=timestep_style / 1000,
            guidance=null_guidance,
            pooled_projections=null_pooled_embeds,
            encoder_hidden_states=null_embeds,
            txt_ids=null_text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        q_list, _, _, self_out_list = cache.get()

        # Style loss: OT-guided attention matching
        # Timestep-adaptive tau: early=soft (preserve structure), late=sharp (push style)
        _OT_TAU_MAX = float(os.environ.get("OT_TAU_MAX", "0.15"))
        _OT_TAU_MIN = float(os.environ.get("OT_TAU_MIN", "0.07"))
        _OT_ITERS = int(os.environ.get("OT_ITERS", "6"))
        # Progress: t goes from start_time→0, so lower t = later in denoising
        _t_val = float(t.item()) if hasattr(t, 'item') else float(t)
        _progress = 1.0 - (_t_val / max(start_time, 1.0))  # 0→1 as denoising progresses
        _tau = _OT_TAU_MAX - (_OT_TAU_MAX - _OT_TAU_MIN) * _progress

        # Layer-adaptive: shallow layers get standard attn (structure), deep get OT (style)
        _OT_LAYER_FRAC = float(os.environ.get("OT_LAYER_FRAC", "0.5"))  # bottom fraction uses OT
        n_layers = len(q_list)
        n_ot_layers = max(1, int(n_layers * _OT_LAYER_FRAC))

        ref_self_out_list = []
        for idx, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
            if idx >= n_layers - n_ot_layers:
                # Deep layers: Sinkhorn OT attention
                ref_self_out_list.append(
                    utils.sinkhorn_attn(q, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau)
                )
            else:
                # Shallow layers: standard attention (preserves structure)
                ref_self_out_list.append(
                    F.scaled_dot_product_attention(q, ref_k, ref_v)
                )

        style_loss = sum(
            loss_fn(so, rso.detach())
            for so, rso in zip(self_out_list, ref_self_out_list)
        )

        # Optional: content K/V loss (Method D, CNT_KV_W > 0)
        cnt_kv_loss = torch.tensor(0.0, device=latents.device)
        if cnt_k_list is not None:
            cnt_self_out_list = [
                F.scaled_dot_product_attention(q, ck, cv)
                for q, ck, cv in zip(q_list, cnt_k_list, cnt_v_list)
            ]
            cnt_kv_loss = sum(
                loss_fn(so, cso.detach())
                for so, cso in zip(self_out_list, cnt_self_out_list)
            )

        # Distribution loss: pull latent stats toward style
        lat4d = latents.unsqueeze(1)
        sty4d = style_latent_.unsqueeze(1).detach()
        dist_loss = (
            loss_fn(lat4d.mean(dim=2, keepdim=True), sty4d.mean(dim=2, keepdim=True)) +
            loss_fn(lat4d.std(dim=2, keepdim=True),  sty4d.std(dim=2, keepdim=True))
        )

        # Content anchor loss: prevent structure collapse
        # Decays over iters so later steps can push harder on style
        decay = 1.0 - (it / max(_iters_this_step, 1)) * 0.5   # 1.0 → 0.5
        content_loss = loss_fn(latents, content_anchor) * decay

        # CSD-in-loop loss (if activated): decode Tweedie x0 → CSD embedding → L1 vs ref
        csd_loss_val = 0.0
        csd_loss = torch.tensor(0.0, device=latents.device)
        if getattr(controller, "csd_w", 0.0) > 0 and hasattr(controller, "pipe_for_decode"):
            _pipe = controller.pipe_for_decode
            # Unpack + un-scale latent (same as pipe.latent2image but gradient-safe)
            _vae = _pipe.vae
            _H = int((latents.shape[1] * 4) ** 0.5 * _pipe.vae_scale_factor)   # fallback
            # Use kwargs we have: height/width from outer scope not available; use stored shape
            # latents are packed; need pipe._unpack_latents(latents, H, W, vae_scale)
            # We assume square latent and infer: L_packed = (H/16)^2 × 1 for flux
            L_packed = latents.shape[1]
            grid = int(L_packed ** 0.5)
            H_img = grid * _pipe.vae_scale_factor * 2   # 2 = patch factor
            W_img = H_img
            try:
                lat_unpacked = _pipe._unpack_latents(latents, H_img, W_img, _pipe.vae_scale_factor)
                lat_unpacked = (lat_unpacked / _vae.config.scaling_factor) + _vae.config.shift_factor
                _dec = _vae.decode(lat_unpacked.to(dtype=next(_vae.parameters()).dtype), return_dict=False)[0]
                img01 = (_dec * 0.5 + 0.5).clamp(0, 1).float()
                emb_gen = controller.csd_embed_fn(img01)
                # cosine distance = 1 - cos_sim
                csd_loss = 1.0 - (emb_gen * controller.csd_ref_embed).sum(dim=-1).mean()
                csd_loss_val = csd_loss.item()
            except Exception as _e:
                print(f"[CSD-loop] decode failed: {_e}")
                csd_loss = torch.tensor(0.0, device=latents.device)

        # Content-in-loop (CNT_LOOP_W > 0): pull output toward content image
        # in CLIP image-embedding space, structurally symmetric to CSD-in-loop.
        # Requires controller.clip_cnt_embed_fn and controller.clip_cnt_ref_embed
        # to be set in i2i mode.
        cnt_loop_loss = torch.tensor(0.0, device=latents.device)
        _CNT_LOOP_W = float(os.environ.get("CNT_LOOP_W", "0.0"))
        if _CNT_LOOP_W > 0 and hasattr(controller, "clip_cnt_embed_fn") \
                and hasattr(controller, "pipe_for_decode"):
            try:
                _pipe = controller.pipe_for_decode
                _vae = _pipe.vae
                L_packed = latents.shape[1]
                grid = int(L_packed ** 0.5)
                H_img = grid * _pipe.vae_scale_factor * 2
                lat_u = _pipe._unpack_latents(latents, H_img, H_img, _pipe.vae_scale_factor)
                lat_u = (lat_u / _vae.config.scaling_factor) + _vae.config.shift_factor
                _dec_c = _vae.decode(lat_u.to(dtype=next(_vae.parameters()).dtype),
                                      return_dict=False)[0]
                img01_c = (_dec_c * 0.5 + 0.5).clamp(0, 1).float()
                clip_g = controller.clip_cnt_embed_fn(img01_c)
                cnt_loop_loss = 1.0 - (clip_g * controller.clip_cnt_ref_embed).sum(dim=-1).mean()
            except Exception as _e:
                print(f"[Cnt-loop] decode failed: {_e}")

        loss = (controller._cached_style_w * style_loss
                + _DIST_W * dist_loss
                + _CONTENT_W * content_loss
                + getattr(controller, "csd_w", 0.0) * csd_loss
                + _CNT_KV_W * cnt_kv_loss
                + _CNT_LOOP_W * cnt_loop_loss)
        accelerator.backward(loss)
        optimizer.step()

        pbar.set_postfix(
            sl=f"{style_loss.item():.2f}",
            cl=f"{content_loss.item():.2f}",
            t=t.item()
        )

    return latents
