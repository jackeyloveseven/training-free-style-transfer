"""
OT-Attention v3 for Kontext image-to-image.

Same as core_ot_attn.py but the denoising forward (step 1) concatenates
content_latent_packed to hidden_states (Kontext-style conditioning).
Style forward and Adam forward are unchanged — they operate on latents only.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import utils

_DIST_W    = float(os.environ.get("DIST_W",    "0.15"))
_CONTENT_W = float(os.environ.get("CONTENT_W", "0.6"))
_OT_ALPHA  = float(os.environ.get("OT_ALPHA",  "0.5"))


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
    iters                = max(kwargs["iters"], 3)
    loss_fn              = kwargs["loss_fn"]
    pbar                 = kwargs["pbar"]
    start_time           = kwargs["start_time"]
    style_beta           = kwargs["style_beta"]

    # Kontext-specific
    content_latent_packed = kwargs["content_latent_packed"]
    combined_img_ids      = kwargs["combined_img_ids"]

    # ── 0. Rank detection (identical to v3) ──
    # Cache is keyed to the style latent: batch evaluation over multiple
    # styles re-adapts per style. The old boolean guard leaked the first
    # style's cached params (and an already-adapted csd_w) to every
    # subsequent image in the same process.
    _style_key = (tuple(style_latent.shape),
                  float(style_latent.float().mean()),
                  float(style_latent.float().std()))
    if getattr(controller, "_rank_cache_key", None) != _style_key:
        controller._rank_cache_key = _style_key
        with torch.no_grad():
            sl = style_latent[0].float()
            sl_c = sl - sl.mean(dim=0, keepdim=True)
            L_toks, D_ = sl_c.shape
            C = sl_c.T @ sl_c / L_toks
            C = (C + C.T) / 2
            S = torch.linalg.eigvalsh(C + 1e-8 * torch.eye(D_, device=sl.device))
            S = torch.clamp(S, min=0).flip(0)
            p = (S / (S.sum() + 1e-8)).cpu().numpy()
            cum = np.cumsum(p)
            r95 = int(np.searchsorted(cum, 0.95) + 1)

        controller._cached_ot_alpha = 0.0

        LO_SW, HI_SW = 24.0, 48.0
        frac_sw = max(0.0, min(1.0, (r95 - LO_SW) / (HI_SW - LO_SW)))
        _style_w_base = 1.5 - 0.5 * frac_sw
        _style_w_scale = float(os.environ.get("STYLE_W_SCALE", "1.0"))
        controller._cached_style_w = _style_w_base * _style_w_scale

        LO_MU, HI_MU = 38.0, 46.0
        frac_mu = max(0.0, min(1.0, (r95 - LO_MU) / (HI_MU - LO_MU)))
        controller._cached_mu_weight = 0.9 - 0.2 * frac_mu

        # Adapt from the ORIGINAL csd_w, not the previously adapted value
        # (controller.csd_w is overwritten in place below).
        if not hasattr(controller, "_base_csd_w"):
            controller._base_csd_w = getattr(controller, "csd_w", 0.0)
        _base_csd_w = controller._base_csd_w
        if _base_csd_w > 0:
            LO_CW, HI_CW = 24.0, 44.0
            frac_cw = max(0.0, min(1.0, (r95 - LO_CW) / (HI_CW - LO_CW)))
            adapted_cw = _base_csd_w * (0.5 - 0.35 * frac_cw)
            controller.csd_w = max(0.03, adapted_cw)
            print(f"[CSD-adaptive] base_w={_base_csd_w} → {controller.csd_w:.3f}")

        LO_ST, HI_ST = 24.0, 44.0
        frac_st = max(0.0, min(1.0, (r95 - LO_ST) / (HI_ST - LO_ST)))
        adjusted_start_time = int(660 - 20 * frac_st)
        if "OVERRIDE_START_TIME" in os.environ:
            adjusted_start_time = int(os.environ["OVERRIDE_START_TIME"])
        controller._cached_start_time = adjusted_start_time

        controller._cached_rank95 = r95
        controller._rank_cached = True
        print(f"[adaptive] rank_95={r95} → OT_ALPHA={controller._cached_ot_alpha:.3f} "
              f"start_time={adjusted_start_time} style_w={controller._cached_style_w:.2f} "
              f"mu_w={controller._cached_mu_weight:.2f}")

    start_time = controller._cached_start_time

    # ── 1. Denoising step: Kontext concat content to hidden_states ──
    latent_seq_len = latents.shape[1]
    with torch.no_grad():
        latent_model_input = torch.cat(
            [latents, content_latent_packed.to(latents.dtype)], dim=1
        )
        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=combined_img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, :latent_seq_len]
    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    if t >= start_time:
        return latents

    # ── 2. Noisy style latent ──
    if i < num_inference_steps - 1:
        timestep_style = timesteps[i + 1 : i + 2]
        has_noise = True
    else:
        timestep_style = torch.tensor([0], device=style_latent.device)
        has_noise = False

    # ── 3. Style forward → ref K/V ──
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

    # ── 3.5. GRAG (i2i: relaxed to preserve brushwork texture) ──
    r95 = controller._cached_rank95
    if os.environ.get("GRAG_OFF", "0") == "1":
        grag_delta_deep = 1.0
        grag_delta_shallow = 1.0
    elif "GRAG_FORCE" in os.environ:
        grag_delta_deep = float(os.environ["GRAG_FORCE"])
        grag_delta_shallow = min(0.85, grag_delta_deep + 0.25)
    else:
        grag_delta_deep = float(os.environ.get("GRAG_DEEP", "0.5"))
        grag_delta_shallow = float(os.environ.get("GRAG_SHALLOW", "0.75"))

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

    # ── 4. Warm-start: AdaIN (i2i: weaker to preserve Kontext content structure) ──
    _adain_alpha = float(os.environ.get("ADAIN_ALPHA", "0.3"))
    content_anchor = latents.detach().clone()
    with torch.no_grad():
        latents_warmed = utils.feature_adain(
            latents.unsqueeze(1), style_latent_.unsqueeze(1), alpha=_adain_alpha
        ).squeeze(1)
    latents = latents_warmed.detach()

    # ── 5. Adam: style K/V loss + distribution pull + content anchor ──
    optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
    optimizer = accelerator.prepare(optimizer)

    for it in range(iters):
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

        # Style loss: OT in deep layers, standard attn in shallow
        _OT_TAU_MAX = float(os.environ.get("OT_TAU_MAX", "0.15"))
        _OT_TAU_MIN = float(os.environ.get("OT_TAU_MIN", "0.07"))
        _OT_ITERS = int(os.environ.get("OT_ITERS", "6"))
        _t_val = float(t.item()) if hasattr(t, 'item') else float(t)
        _progress = 1.0 - (_t_val / max(start_time, 1.0))
        _tau = _OT_TAU_MAX - (_OT_TAU_MAX - _OT_TAU_MIN) * _progress

        _OT_LAYER_FRAC = float(os.environ.get("OT_LAYER_FRAC", "0.5"))
        n_layers = len(q_list)
        n_ot_layers = max(1, int(n_layers * _OT_LAYER_FRAC))

        ref_self_out_list = []
        for idx, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
            if idx >= n_layers - n_ot_layers:
                ref_self_out_list.append(
                    utils.sinkhorn_attn(q, ref_k, ref_v, n_iters=_OT_ITERS, tau=_tau)
                )
            else:
                ref_self_out_list.append(
                    F.scaled_dot_product_attention(q, ref_k, ref_v)
                )

        style_loss = sum(
            loss_fn(so, rso.detach())
            for so, rso in zip(self_out_list, ref_self_out_list)
        )

        lat4d = latents.unsqueeze(1)
        sty4d = style_latent_.unsqueeze(1).detach()
        dist_loss = (
            loss_fn(lat4d.mean(dim=2, keepdim=True), sty4d.mean(dim=2, keepdim=True)) +
            loss_fn(lat4d.std(dim=2, keepdim=True),  sty4d.std(dim=2, keepdim=True))
        )

        decay = 1.0 - (it / max(iters, 1)) * 0.5
        content_loss = loss_fn(latents, content_anchor) * decay

        # CSD-in-loop
        csd_loss = torch.tensor(0.0, device=latents.device)
        if getattr(controller, "csd_w", 0.0) > 0 and hasattr(controller, "pipe_for_decode"):
            _pipe = controller.pipe_for_decode
            _vae = _pipe.vae
            L_packed = latents.shape[1]
            grid = int(L_packed ** 0.5)
            H_img = grid * _pipe.vae_scale_factor * 2
            try:
                lat_unpacked = _pipe._unpack_latents(latents, H_img, H_img, _pipe.vae_scale_factor)
                lat_unpacked = (lat_unpacked / _vae.config.scaling_factor) + _vae.config.shift_factor
                _dec = _vae.decode(lat_unpacked.to(dtype=next(_vae.parameters()).dtype), return_dict=False)[0]
                img01 = (_dec * 0.5 + 0.5).clamp(0, 1).float()
                emb_gen = controller.csd_embed_fn(img01)
                csd_loss = 1.0 - (emb_gen * controller.csd_ref_embed).sum(dim=-1).mean()
            except Exception:
                pass

        loss = (controller._cached_style_w * style_loss
                + _DIST_W * dist_loss
                + _CONTENT_W * content_loss
                + getattr(controller, "csd_w", 0.0) * csd_loss)
        accelerator.backward(loss)
        optimizer.step()

        pbar.set_postfix(
            sl=f"{style_loss.item():.2f}",
            cl=f"{content_loss.item():.2f}",
            t=t.item()
        )

    return latents
