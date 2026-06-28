import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def register_attn_control(unet, controller, cache=None):
    def attn_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, *args, **kwargs):
            # 获取原始计算结果
            batch_size, _, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
            query = self.to_q(hidden_states); key = self.to_k(hidden_states); value = self.to_v(hidden_states)
            inner_dim = key.shape[-1]; head_dim = inner_dim // self.heads
            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            if self.norm_q is not None: query = self.norm_q(query)
            if self.norm_k is not None: key = self.norm_k(key)
            
            txt_len = 0
            if encoder_hidden_states is not None:
                txt_len = encoder_hidden_states.shape[1]
                controller.txt_len = txt_len
                e_q = self.add_q_proj(encoder_hidden_states).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                e_k = self.add_k_proj(encoder_hidden_states).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                e_v = self.add_v_proj(encoder_hidden_states).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                if self.norm_added_q is not None: e_q = self.norm_added_q(e_q)
                if self.norm_added_k is not None: e_k = self.norm_added_k(e_k)
                query, key, value = torch.cat([e_q, query], dim=2), torch.cat([e_k, key], dim=2), torch.cat([e_v, value], dim=2)
            else:
                txt_len = getattr(controller, 'txt_len', 512)
            
            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query, key = apply_rotary_emb(query, image_rotary_emb), apply_rotary_emb(key, image_rotary_emb)

            attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            
            # --- Passive Cache Mode ---
            if controller.cur_self_layer in controller.self_layers:
                cache.add(query, key, value, attn_output)

                # --- OT Align Injection Mode ---
                if controller.mode == "ot_align" and controller.inject_idx < len(controller.out_inject):
                    ref_out = controller.out_inject[controller.inject_idx]  # [B, H, L+txt, D]
                    # attn_output shape: [B, H, L_total, D]
                    # Only align image tokens (skip text prefix)
                    _txt_len = txt_len if txt_len > 0 else getattr(controller, "txt_len", 0)
                    img_out = attn_output[:, :, _txt_len:, :]      # [B, H, L_img, D]
                    ref_img_out = ref_out[:, :, _txt_len:, :]       # [B, H, L_img, D]
                    aligned = batch_manifold_align(img_out, ref_img_out, alpha=getattr(controller, "style_beta", 0.7))
                    # Write back aligned image tokens
                    attn_output = attn_output.clone()
                    attn_output[:, :, _txt_len:, :] = aligned
                    controller.inject_idx += 1

            controller.cur_self_layer += 1
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim).to(query.dtype)
            
            if encoder_hidden_states is not None:
                # 遵循 DoubleStream 返回协议 (图像, 文本)
                img_out = attn_output[:, txt_len:, :]
                txt_out = attn_output[:, :txt_len, :]
                img_out = self.to_out[0](img_out)
                img_out = self.to_out[1](img_out)
                txt_out = self.to_add_out(txt_out)
                return img_out, txt_out
            return attn_output
        return forward

    def modify_forward(net, count):
        if net.__class__.__name__ == "FluxAttention":
            net.forward = attn_forward(net)
            return count + 1
        for name, subnet in net.named_children():
            count = modify_forward(subnet, count)
        return count
    modify_forward(unet, 0)

def adain(source, target, eps=1e-6):
    source_mean, source_std = torch.mean(source, dim=1, keepdim=True), torch.std(
        source, dim=1, keepdim=True
    )
    target_mean, target_std = torch.mean(
        target, dim=(0, 1), keepdim=True
    ), torch.std(target, dim=(0, 1), keepdim=True)
    normalized_source = (source - source_mean) / (source_std + eps)
    transferred_source = normalized_source * target_std + target_mean
    return transferred_source

def feature_adain(x, y, alpha=1.0, eps=1e-6):
    m_x, s_x = x.mean(dim=2, keepdim=True), x.std(dim=2, keepdim=True) + eps
    m_y, s_y = y.mean(dim=2, keepdim=True), y.std(dim=2, keepdim=True) + eps
    res = (x - m_x) / s_x * s_y + m_y
    return alpha * res + (1 - alpha) * x

def batch_manifold_align(x, y, alpha=1.0, eps=1e-4, mu_weight=None):
    """
    Block-Diagonal Optimal Transport for style transfer warm-start.

    Splits D channels into G groups and computes independent OT within each group.
    This eliminates cross-group coupling (root cause of spatial stripe artifacts)
    while transferring higher-order statistics beyond mean+std (unlike AdaIN).

    Each block computes: T_g = Σ_x^{-1/2} (Σ_x^{1/2} Σ_y Σ_x^{1/2})^{1/2} Σ_x^{-1/2}
    which is the closed-form Wasserstein-2 optimal transport map for Gaussians.
    """
    import os as _os
    if mu_weight is None:
        mu_weight = float(_os.environ.get("MU_BLEND", "0.9"))

    B, H, L, D = x.shape
    _G = int(_os.environ.get("OT_GROUPS", "8"))
    assert D % _G == 0, f"D={D} must be divisible by OT_GROUPS={_G}"
    d = D // _G  # channels per group

    x_f = x.reshape(B * H, L, D).float()
    y_f = y.reshape(B * H, L, D).float()
    N = B * H

    # Global mean blend
    mu_x = x_f.mean(dim=1, keepdim=True)  # [N, 1, D]
    mu_y = y_f.mean(dim=1, keepdim=True)
    mu_blend = mu_weight * mu_y + (1.0 - mu_weight) * mu_x

    x_c = x_f - mu_x  # centered
    y_c = y_f - mu_y

    I_d = torch.eye(d, device=x.device, dtype=torch.float32)

    try:
        results = []
        for g in range(_G):
            s, e = g * d, (g + 1) * d
            xg = x_c[:, :, s:e]  # [N, L, d]
            yg = y_c[:, :, s:e]

            # Covariance matrices [N, d, d]
            Cx = torch.bmm(xg.transpose(1, 2), xg) / L
            Cy = torch.bmm(yg.transpose(1, 2), yg) / L
            Cx = (Cx + Cx.transpose(1, 2)) / 2
            Cy = (Cy + Cy.transpose(1, 2)) / 2

            # Cx^{1/2} and Cx^{-1/2}
            S_x, U_x = torch.linalg.eigh(Cx + eps * I_d)
            S_x = torch.clamp(S_x, min=eps)
            Cx_h = torch.bmm(U_x, torch.bmm(torch.diag_embed(torch.sqrt(S_x)), U_x.transpose(1, 2)))
            Cx_ih = torch.bmm(U_x, torch.bmm(torch.diag_embed(torch.clamp(1.0 / torch.sqrt(S_x), max=10.0)), U_x.transpose(1, 2)))

            # Core = Cx^{1/2} Cy Cx^{1/2}
            core = torch.bmm(Cx_h, torch.bmm(Cy, Cx_h))
            core = (core + core.transpose(1, 2)) / 2

            # Core^{1/2}
            S_c, U_c = torch.linalg.eigh(core + eps * I_d)
            S_c = torch.sqrt(torch.clamp(S_c, min=eps))
            S_c = torch.clamp(S_c, max=4.0)  # prevent extreme scaling
            core_h = torch.bmm(U_c, torch.bmm(torch.diag_embed(S_c), U_c.transpose(1, 2)))

            # Transport map A_g = Cx^{-1/2} Core^{1/2} Cx^{-1/2}
            Ag = torch.bmm(Cx_ih, torch.bmm(core_h, Cx_ih))

            if alpha < 1.0:
                Ag = alpha * Ag + (1 - alpha) * I_d

            # Apply: x_g' = x_g @ A_g
            results.append(torch.bmm(xg, Ag))

        # Concatenate groups and add blended mean
        res = torch.cat(results, dim=2)  # [N, L, D]
        res = (res + mu_blend).reshape(B, H, L, D)

        # Variance amplitude transfer
        res_std = res.std(dim=2, keepdim=True) + eps
        orig_std = x_f.reshape(B, H, L, D).std(dim=2, keepdim=True) + eps
        style_std = y_f.reshape(B, H, L, D).std(dim=2, keepdim=True) + eps
        target_std = alpha * style_std + (1 - alpha) * orig_std
        target_std = torch.clamp(target_std, max=orig_std * 5.0)
        res = res * (target_std / res_std)

        res = res.to(x.dtype)

        if torch.isnan(res).any() or res.abs().max() > 100:
            print("Warning: Numeric instability in Block-Diag OT, falling back to AdaIN")
            res = feature_adain(x, y, alpha=alpha)

        return res
    except Exception as e:
        print(f"Warning: Block-Diag OT crash ({e}), falling back to AdaIN")
        return feature_adain(x, y, alpha=alpha)

def sinkhorn_attn(Q, K_style, V_style, n_iters=3, tau=0.1, eps=1e-8):
    """
    OT-guided attention: Sinkhorn-balanced attention between content Q and style K/V.

    Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
      → All content tokens can collapse onto the same style hotspot.

    OT attention: Sinkhorn(Q @ K^T / tau) @ V
      → Transport plan is doubly-stochastic: each style token gets utilized,
        each content token gets a balanced mix. Prevents mode collapse.

    Args:
        Q: [B, H, Lc, D] content queries
        K_style: [B, H, Ls, D] style keys
        V_style: [B, H, Ls, D] style values
        n_iters: Sinkhorn iterations (3-5 is enough)
        tau: temperature (lower = sharper assignment, higher = softer)

    Returns: [B, H, Lc, D] — OT-weighted style values
    """
    B, H, Lc, D = Q.shape
    Ls = K_style.shape[2]

    # Cost matrix = negative similarity (we want to maximize Q·K)
    # [B, H, Lc, Ls]
    logits = torch.matmul(Q, K_style.transpose(-1, -2)) / (D ** 0.5 * tau)

    # Sinkhorn iterations for doubly-stochastic normalization
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)  # row normalize
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)  # col normalize

    # Final row normalization to get proper transport plan (rows sum to 1)
    P = torch.softmax(log_P, dim=-1)  # [B, H, Lc, Ls]

    # Apply transport plan to style values
    out = torch.matmul(P, V_style)  # [B, H, Lc, D]
    return out


def sinkhorn_attn_freq(Q, K_style, V_style, n_iters=6, tau=0.1, hi_boost=2.0):
    """
    Frequency-aware Sinkhorn: boost high-freq components in cost matrix.

    Subtracts spatial mean from Q/K before computing cost, so the OT plan
    is driven by texture variation (high-freq) rather than absolute position (low-freq).
    The plan is then applied to full V_style (preserving all info in transport).

    hi_boost: weight for the high-freq cost relative to low-freq (>1 = prefer texture alignment)
    Ported verbatim from custom_main_klein9b_t2i.py (Klein9B reference).
    """
    B, H, Lc, D = Q.shape
    Ls = K_style.shape[2]

    # Decompose Q and K into low-freq (spatial mean) and high-freq (residual)
    Q_lo = Q.mean(dim=2, keepdim=True).expand_as(Q)  # [B, H, Lc, D]
    K_lo = K_style.mean(dim=2, keepdim=True).expand_as(K_style)
    Q_hi = Q - Q_lo
    K_hi = K_style - K_lo

    # Cost = blend of low-freq and high-freq similarity
    logits_hi = torch.matmul(Q_hi, K_hi.transpose(-1, -2)) / (D ** 0.5 * tau)
    logits_lo = torch.matmul(Q_lo, K_lo.transpose(-1, -2)) / (D ** 0.5 * tau)
    logits = logits_lo + hi_boost * logits_hi

    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    P = torch.softmax(log_P, dim=-1)
    return torch.matmul(P, V_style)


def sinkhorn_latent_transport(content_lat, style_lat_noisy, style_lat_clean,
                              alpha=0.5, n_iters=10, tau=0.1, grag_delta=1.0,
                              mode="ot_adain"):
    """
    OT-based latent-space style transport (no forward/backward; replaces Adam in-loop).

    Modes:
      - "blend": linear interpolation toward transported style (fast but kills texture)
      - "ot_adain": OT-matched per-token AdaIN (preserves content structure, transfers style statistics)

    Cost: ~microseconds (two bmm + n_iters row/col normalize on N x M).
    Ported verbatim from custom_main_klein9b_t2i.py (Klein9B reference).
    """
    B, N, D = content_lat.shape

    # GRAG analog: compress style tokens toward mean to reduce layout leakage
    style_noisy_g = style_lat_noisy
    style_clean_g = style_lat_clean
    if grag_delta < 1.0:
        s_mean = style_lat_noisy.mean(dim=1, keepdim=True)
        style_noisy_g = s_mean + grag_delta * (style_lat_noisy - s_mean)
        sc_mean = style_lat_clean.mean(dim=1, keepdim=True)
        style_clean_g = sc_mean + grag_delta * (style_lat_clean - sc_mean)

    # Cost matrix: frequency-aware cosine similarity (high-freq residuals → match by texture, not position)
    c_f = content_lat.float()
    s_f = style_noisy_g.float()
    c_hi = c_f - c_f.mean(dim=1, keepdim=True)
    s_hi = s_f - s_f.mean(dim=1, keepdim=True)
    c_norm = F.normalize(c_hi, dim=-1)
    s_norm = F.normalize(s_hi, dim=-1)
    logits = torch.bmm(c_norm, s_norm.transpose(1, 2)) / tau  # [B, N, M]

    # Sinkhorn normalization (doubly-stochastic)
    log_P = logits
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    T = torch.softmax(log_P, dim=-1)  # [B, N, M]

    if mode == "blend":
        transported = torch.bmm(T, style_clean_g.float()).to(content_lat.dtype)
        return (1.0 - alpha) * content_lat + alpha * transported
    else:
        # OT-AdaIN: per-content-token OT-matched style mean/std; transfer stats, keep content high-freq.
        style_f = style_clean_g.float()  # [B, M, D]
        matched_mean = torch.bmm(T, style_f)  # [B, N, D]
        matched_var = torch.bmm(T, style_f ** 2) - matched_mean ** 2
        matched_std = (matched_var.clamp(min=1e-6)).sqrt()
        content_f = content_lat.float()
        c_mean = content_f.mean(dim=1, keepdim=True)
        c_std = content_f.std(dim=1, keepdim=True).clamp(min=1e-6)
        normed = (content_f - c_mean) / c_std
        styled = normed * matched_std + matched_mean
        return ((1.0 - alpha) * content_f + alpha * styled).to(content_lat.dtype)


def sliced_wasserstein(x, y, n_projections=64):
    """
    Sliced Wasserstein Distance between two token distributions.
    x, y: [B, L, D] — B batches of L tokens in D-dim space.
    Returns scalar loss (mean over batches and projections).
    Differentiable w.r.t. x.
    """
    B, L, D = x.shape
    # Work in float32 for stability
    x_f = x.float()
    y_f = y.float()

    # Random projection directions (unit vectors on sphere)
    theta = torch.randn(n_projections, D, device=x.device, dtype=torch.float32)
    theta = theta / (theta.norm(dim=1, keepdim=True) + 1e-8)

    # Project: [B, L, D] @ [D, P] -> [B, L, P]
    x_proj = torch.matmul(x_f, theta.T)
    y_proj = torch.matmul(y_f, theta.T)

    # Sort along token dimension (1D OT = sorted assignment)
    x_sorted = torch.sort(x_proj, dim=1).values
    y_sorted = torch.sort(y_proj, dim=1).values

    # W1 distance per projection = mean |sorted_x - sorted_y|
    return (x_sorted - y_sorted).abs().mean()


def load_image(image_path, size=None, mode="RGB"):
    img = Image.open(image_path).convert(mode)
    if size is None: w, h = img.size; size = ((w // 64) * 64, (h // 64) * 64)
    return ToTensor()(img.resize(size, Image.BICUBIC)).unsqueeze(0)

class Controller:
    def step(self): self.cur_self_layer, self.inject_idx = 0, 0
    def __init__(self, self_layers=(0, 16)):
        self.cur_self_layer, self.self_layers = 0, list(range(*self_layers))
        self.mode, self.k_inject, self.v_inject, self.out_inject, self.inject_idx = "read", [], [], [], 0
    def set_inject(self, k_list, v_list, out_list=None, mode="replace"):
        self.k_inject, self.v_inject, self.out_inject, self.mode, self.inject_idx = k_list, v_list, out_list or [], mode, 0
    def reset_mode(self): self.mode, self.k_inject, self.v_inject, self.out_inject, self.inject_idx = "read", [], [], [], 0

class DataCache:
    def __init__(self): self.q, self.k, self.v, self.out = [], [], [], []
    def clear(self): self.q.clear(), self.k.clear(), self.v.clear(), self.out.clear()
    def add(self, q, k, v, out): self.q.append(q), self.k.append(k), self.v.append(v), self.out.append(out)
    def get(self): return self.q.copy(), self.k.copy(), self.v.copy(), self.out.copy()

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if X.size(0) > X.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + eps) 

    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton-schulz
    Specifically adapted for N-dimensional Latent spaces by reshaping gradients.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')

                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                
                # Reshape to 2D for orthogonalization (Flatten spatial/sequence dims)
                original_shape = grad.shape
                if len(original_shape) > 2:
                    grad_2d = grad.view(original_shape[0] * original_shape[1], -1) if len(original_shape) == 3 else grad.view(original_shape[0], -1)
                elif len(original_shape) == 1:
                    grad_2d = grad.unsqueeze(0)
                else:
                    grad_2d = grad

                # Newton-Schulz orthogonalization
                update_2d = zeropower_via_newtonschulz5(grad_2d, steps=group['ns_steps'])
                update = update_2d.view(original_shape)
                
                # Momentum
                buf.mul_(group['momentum']).add_(update)
                
                if group['nesterov']:
                    update = update.add(buf, alpha=group['momentum'])
                else:
                    update = buf
                
                # Update
                p.add_(update, alpha=-group['lr'])

        return loss

def memory_efficient(model):
    model.freeze()
    for m in ["enable_model_cpu_offload", "enable_vae_slicing", "enable_vae_tiling"]:
        try: getattr(model, m)()
        except AttributeError: pass

def show_image(path, title, display_height=3, title_fontsize=12):
    img = Image.open(path); aspect = img.size[0] / img.size[1]
    plt.figure(figsize=(display_height * aspect, display_height))
    plt.imshow(img); plt.title(title, fontsize=title_fontsize, fontweight='bold', pad=20); plt.axis('off'); plt.tight_layout(); plt.show()
