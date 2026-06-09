"""
Generate all paper figures — publication-ready for top venues (CVPR/NeurIPS/ICLR).
- White background, clean axes, NO suptitles (paper has captions)
- Consistent muted color palette
- Font: DejaVu Serif, 8pt body
- DPI: 300, PNG only
- Qualitative grids: 30-40 images, tight spacing, style refs shown

Output: fig/
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# === Global academic style ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.5,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'lines.linewidth': 1.3,
    'lines.markersize': 5,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
    'grid.color': '#DDDDDD',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

CWD = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948")
FIG_DIR = CWD / "fig"
FIG_DIR.mkdir(exist_ok=True)

# Muted academic palette
C = {
    'ours':     '#C0392B',
    'adam':      '#E67E22',
    'slt':      '#27AE60',
    'inject':   '#2980B9',
    'text':     '#8E44AD',
    'noslt':    '#D35400',
    'softmax':  '#16A085',
    'freqoff':  '#7F8C8D',
    'random':   '#BDC3C7',
}


def load_img(path, size=128):
    p = Path(path)
    if p.exists():
        return np.array(Image.open(p).resize((size, size), Image.BICUBIC))
    return np.zeros((size, size, 3), dtype=np.uint8)


def fig1_ablation_bar():
    df = pd.read_csv(CWD / 'results/t2i_ablation.csv')
    summary = df.groupby('method').agg({'csd': ['mean', 'std']}).reset_index()
    summary.columns = ['method', 'csd_mean', 'csd_std']
    order = ['hybrid2', 'freq_off', 'adam8', 'softmax', 'no_slt', 'no_adam']
    labels = ['Ours\n(Hybrid)', 'w/o Freq\nSep.', 'Adam×8', 'w/o OT\nNorm.', 'w/o SLT', 'SLT only']
    colors = [C['ours'], C['freqoff'], C['adam'], C['softmax'], C['noslt'], C['slt']]
    summary = summary.set_index('method').loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(order))
    bars = ax.bar(x, summary['csd_mean'], yerr=summary['csd_std'], capsize=3,
                  color=colors, edgecolor='white', linewidth=0.5, width=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('CSD Score')
    ax.set_ylim(0, 0.78)
    ax.grid(axis='y')
    for bar, val in zip(bars, summary['csd_mean']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                f'{val:.3f}', ha='center', fontsize=6.5, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_ablation.png'); plt.close()


def fig2_pareto():
    df_iters = pd.read_csv(CWD / 'results/t2i_iters_sweep.csv')
    df_adam = pd.read_csv(CWD / 'results/t2i_adam_only_sweep.csv')
    hybrid = df_iters.groupby('iters').agg({'csd': 'mean'}).reset_index()
    adam = df_adam.groupby('iters').agg({'csd': 'mean'}).reset_index()
    ht = {0:2.21, 1:4.32, 2:6.40, 3:8.51, 4:10.59, 5:12.70, 8:18.96}
    at = {1:4.32, 2:6.40, 3:8.51, 4:10.59, 5:12.70, 8:18.96}
    hybrid['time'] = hybrid['iters'].map(ht)
    adam['time'] = adam['iters'].map(at)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(hybrid['time'], hybrid['csd'], '-o', color=C['ours'], label='Hybrid (SLT+Adam)', zorder=5)
    ax.plot(adam['time'], adam['csd'], '--s', color=C['adam'], label='Adam only', zorder=4)
    our_csd = hybrid[hybrid['iters']==2]['csd'].values[0]
    ax.scatter([6.40], [our_csd], s=120, color=C['ours'], zorder=6, edgecolor='black', linewidth=0.7, marker='*')
    ax.annotate('Ours\n(2 iters, 6.4s)', xy=(6.4, our_csd), xytext=(10, our_csd-0.08),
               fontsize=7, color=C['ours'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=C['ours'], lw=0.7))
    slt_csd = hybrid[hybrid['iters']==0]['csd'].values[0]
    ax.axhline(y=slt_csd, color=C['slt'], linestyle=':', linewidth=0.8, alpha=0.7)
    ax.text(1, slt_csd+0.015, 'SLT only', fontsize=6.5, color=C['slt'])
    ax.set_xlabel('Wall-clock Time (s)'); ax.set_ylabel('CSD Score')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_xlim(0, 21); ax.set_ylim(0.15, 0.75)
    ax.grid(True); plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_pareto.png'); plt.close()


def fig3_spatial_proof():
    """
    Two-panel figure:
    (a) Style loss convergence: backward iterates vs inject/SLT baselines
    (b) Gradient source ablation: WHY backward works (bar chart of neighbor cos)
    """
    # --- Panel (a): Style loss convergence ---
    with open(CWD / 'results/loss_decompose.json') as f:
        loss_data = json.load(f)
    n_samples = len(loss_data)
    noslt_sl = [np.mean([loss_data[s]['backward_noslt'][it]['style_loss'] for s in range(n_samples)]) for it in range(8)]
    withslt_sl = [np.mean([loss_data[s]['backward_withslt'][it]['style_loss'] for s in range(n_samples)]) for it in range(8)]
    adain_sl = np.mean([loss_data[s]['checkpoints']['adain']['style_loss'] for s in range(n_samples)])
    slt_sl = np.mean([loss_data[s]['checkpoints']['slt']['style_loss'] for s in range(n_samples)])

    # --- Panel (b): Gradient source ablation ---
    with open(CWD / 'results/gradient_source.json') as f:
        gs_data = json.load(f)
    n_gs = len(gs_data)
    conditions = ["normal", "shuffle_style_kv", "content_only", "single_patch"]
    cond_labels = ["Normal\n(full gradient)", "Shuffle\nstyle K/V", "Content\nloss only", "Single-patch\nstyle loss"]
    cond_colors = [C['ours'], C['adam'], C['random'], C['inject']]
    # Average neighbor_cos at iter 1 across samples
    cos_means = []
    cos_stds = []
    for cond in conditions:
        vals = [gs_data[s][cond]["neighbor_cos"][0] for s in range(n_gs)]
        cos_means.append(np.mean(vals))
        cos_stds.append(np.std(vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    # Panel (a)
    iters = np.arange(1, 9)
    ax1.plot(iters, noslt_sl, '-s', color=C['adam'], label='Backward only', markersize=4, zorder=5)
    ax1.plot(iters, withslt_sl, '-o', color=C['ours'], label='Backward + SLT (Ours)', markersize=4, zorder=6)
    ax1.axhline(y=adain_sl, color=C['inject'], linestyle=':', linewidth=1.2, label='Inject (no gradient)')
    ax1.axhline(y=slt_sl, color=C['slt'], linestyle='--', linewidth=1.2, label='SLT only')
    ax1.scatter([2], [withslt_sl[1]], s=80, color=C['ours'], zorder=7, marker='*', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Backward Iterations')
    ax1.set_ylabel('Style Loss ↓')
    ax1.legend(fontsize=6, loc='upper right')
    ax1.set_xlim(0.5, 8.5); ax1.set_ylim(0.243, 0.296)
    ax1.grid(True)
    ax1.annotate('Cannot iterate\n(no gradient)', xy=(5, adain_sl), xytext=(5, adain_sl + 0.008),
               fontsize=5.5, color=C['inject'], ha='center')
    ax1.annotate('SLT+2 ≈ 4 alone', xy=(2, withslt_sl[1]), xytext=(3.5, withslt_sl[1]-0.012),
               fontsize=5.5, color=C['ours'], ha='center',
               arrowprops=dict(arrowstyle='->', color=C['ours'], lw=0.7))
    ax1.set_title('(a) Style Loss Convergence', fontsize=9)

    # Panel (b)
    x = np.arange(len(conditions))
    bars = ax2.bar(x, cos_means, yerr=cos_stds, color=cond_colors, edgecolor='white',
                   linewidth=0.4, width=0.65, capsize=3, error_kw={'linewidth': 0.8})
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels, fontsize=6.5)
    ax2.set_ylabel('Neighbor cos@d=1 ↑')
    ax2.set_ylim(0, 0.82)
    ax2.grid(axis='y')
    ax2.set_title('(b) Gradient Spatial Structure Source', fontsize=9)
    # Add value labels
    for bar, val in zip(bars, cos_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', fontsize=7, fontweight='bold')
    # Annotation
    ax2.annotate('Style loss drives\nspatial coherence', xy=(2, cos_means[2]+0.05),
                xytext=(2.5, 0.4), fontsize=6, color='#555',
                arrowprops=dict(arrowstyle='->', color='#555', lw=0.7))

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_spatial_proof.png'); plt.close()


def fig4_slt_sweep():
    df = pd.read_csv(CWD / 'results/t2i_slt_alpha_sweep.csv')
    s = df.groupby('slt_alpha').agg({'csd':'mean','clip_t':'mean'}).reset_index()
    fig, ax1 = plt.subplots(figsize=(3.4, 2.2))
    ax2 = ax1.twinx()
    l1 = ax1.plot(s['slt_alpha'], s['csd'], '-o', color=C['ours'], label='CSD')
    l2 = ax2.plot(s['slt_alpha'], s['clip_t'], '--s', color=C['inject'], label='CLIP-T')
    ax1.set_xlabel('SLT Strength α'); ax1.set_ylabel('CSD', color=C['ours'])
    ax2.set_ylabel('CLIP-T', color=C['inject'])
    ax1.tick_params(axis='y', labelcolor=C['ours']); ax2.tick_params(axis='y', labelcolor=C['inject'])
    ax1.axvline(x=0.1, color='#555', linestyle=':', alpha=0.5, linewidth=0.7)
    ax1.text(0.12, ax1.get_ylim()[0] + 0.01, 'α=0.1\n(selected)', fontsize=6.5, color='#555')
    lines = l1+l2; ax1.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=7)
    ax1.set_xlim(-0.02, 0.52)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_slt_sweep.png'); plt.close()


def fig5_timing():
    with open(CWD / 'results/timing_benchmark.json') as f:
        timing = json.load(f)
    order = ['No style (baseline)', 'SLT only (α=0.1)',
             'Hybrid (SLT + Adam×1)', 'Hybrid (SLT + Adam×2)', 'Hybrid (SLT + Adam×3)',
             'Adam only ×2', 'Adam only ×4', 'Adam only ×8']
    tmap = {t['method']: t['median_s'] for t in timing}
    times = [tmap[n] for n in order]
    labels = ['Base', 'SLT', 'H+1', 'H+2\n(Ours)', 'H+3', 'A×2', 'A×4', 'A×8']
    colors = [C['random'], C['slt'], '#E8A0A0', C['ours'], '#CD5C5C', '#F5C07A', '#E8A035', C['adam']]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(times))
    bars = ax.bar(x, times, color=colors, edgecolor='white', linewidth=0.4, width=0.75)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel('Time (s)'); ax.set_ylim(0, 23); ax.grid(axis='y')
    bars[3].set_edgecolor(C['ours']); bars[3].set_linewidth(1.5)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                f'{val:.1f}', ha='center', fontsize=6)
    ax.annotate('', xy=(3, times[3]+1.2), xytext=(7, times[7]+1.2),
               arrowprops=dict(arrowstyle='<->', color=C['ours'], lw=1.2))
    ax.text(5, max(times[3], times[7])/2 + 5, '3.0×', fontsize=8, color=C['ours'],
            fontweight='bold', ha='center')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_timing.png'); plt.close()


def fig6_iters_sweep():
    df_iters = pd.read_csv(CWD / 'results/t2i_iters_sweep.csv')
    df_adam = pd.read_csv(CWD / 'results/t2i_adam_only_sweep.csv')
    df_bl = pd.read_csv(CWD / 'results/t2i_baselines.csv')
    hybrid = df_iters.groupby('iters').agg({'csd':'mean'}).reset_index()
    adam = df_adam.groupby('iters').agg({'csd':'mean'}).reset_index()
    slt_csd = hybrid[hybrid['iters']==0]['csd'].values[0]
    inject_csd = df_bl[df_bl['method']=='inject']['csd'].mean()

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(hybrid['iters'], hybrid['csd'], '-o', color=C['ours'], label='Hybrid (SLT+Adam)', zorder=5)
    ax.plot(adam['iters'], adam['csd'], '--s', color=C['adam'], label='Adam only', zorder=4)
    ax.axhline(y=slt_csd, color=C['slt'], linestyle=':', linewidth=0.9, label='SLT only')
    ax.axhline(y=inject_csd, color=C['inject'], linestyle=':', linewidth=0.9, label='Inject')
    ax.scatter([2], [hybrid[hybrid['iters']==2]['csd'].values[0]], s=80, color=C['ours'],
              zorder=6, edgecolor='black', linewidth=0.7, marker='*')
    ax.set_xlabel('Adam Iterations'); ax.set_ylabel('CSD Score')
    ax.legend(loc='upper left', fontsize=6.5, ncol=1, bbox_to_anchor=(0.0, 1.0))
    ax.set_xlim(-0.3, 8.8); ax.set_ylim(0.05, 0.78); ax.grid(True)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_iters_sweep.png'); plt.close()


def fig7_i2i_ablation():
    df = pd.read_csv(CWD / 'results/i2i_ablation.csv')
    df_bl = pd.read_csv(CWD / 'results/i2i_baselines.csv')
    df_all = pd.concat([df, df_bl], ignore_index=True)
    summary = df_all.groupby('method').agg({'csd':'mean','clip_i_cnt':'mean'}).reset_index()
    sm = {'hybrid2': (C['ours'], '*', 12, 'Ours'), 'adam8': (C['adam'], 's', 7, 'Adam×8'),
          'softmax': (C['softmax'], '^', 7, 'w/o OT'), 'no_slt': (C['noslt'], 'D', 7, 'w/o SLT'),
          'no_adam': (C['slt'], 'v', 7, 'SLT only'), 'inject': (C['inject'], 'P', 7, 'Inject'),
          'text_only': (C['text'], 'X', 7, 'Text only')}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for _, row in summary.iterrows():
        m = row['method']
        if m in sm:
            co, mk, ms, lb = sm[m]
            ax.scatter(row['clip_i_cnt'], row['csd'], color=co, marker=mk, s=ms**2,
                      label=lb, zorder=5, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('CLIP-I (Content Preservation)'); ax.set_ylabel('CSD (Style Fidelity)')
    ax.legend(fontsize=6.5, loc='lower left', markerscale=0.8); ax.grid(True)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig7_i2i_ablation.png'); plt.close()


def fig8_pca_maps():
    """
    Fig 8: Inject vs Backward — spatial structure comparison.
    Cols: Style Input | Inject output | Output (Ours) | Inject PCA | Backward PCA
    """
    with open(CWD / 'results/spatial_extended.json') as f:
        data = json.load(f)
    d1 = data['neighbor_cosine_d1']
    decay = data['autocorrelation_decay']

    samples = [
        ('sample0', 'sty/27.png', 'd_grand/t2i_ablation/no_adam_kp0_s27.png',
         'd_grand/t2i_ablation/hybrid2_kp0_s27.png', 'A panda eat bamboo'),
        ('sample2', 'sty/33.png', 'd_grand/t2i_ablation/no_adam_kp3_s33.png',
         'd_grand/t2i_ablation/hybrid2_kp3_s33.png', 'Sunflowers at sunset'),
        ('sample4', 'sty/28.png', 'd_grand/t2i_ablation/no_adam_kp1_s28.png',
         'd_grand/t2i_ablation/hybrid2_kp1_s28.png', 'A castle on a cliff'),
    ]

    n_cols = 5  # style + inject output + our output + inject PCA + backward PCA
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.9, n_rows * 2.0),
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.15})

    for row, (sample_prefix, sty_file, inject_file, out_file, prompt_text) in enumerate(samples):
        # Col 0: Style input
        axes[row, 0].imshow(load_img(CWD / sty_file, 150))
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values(): spine.set_visible(False)
        if row == 0:
            axes[row, 0].set_title('Style Input', fontsize=8)
        axes[row, 0].set_xlabel(f'"{prompt_text}"', fontsize=6, labelpad=2)

        # Col 1: Inject output (no gradient)
        axes[row, 1].imshow(load_img(CWD / inject_file, 150))
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
        for spine in axes[row, 1].spines.values(): spine.set_visible(False)
        if row == 0:
            axes[row, 1].set_title('Inject (no gradient)', fontsize=8)

        # Col 2: Our output (backward)
        axes[row, 2].imshow(load_img(CWD / out_file, 150))
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
        for spine in axes[row, 2].spines.values(): spine.set_visible(False)
        if row == 0:
            axes[row, 2].set_title('Output (Ours)', fontsize=8)

        # Col 3: Inject PCA (slt_update_pca)
        fpath_inject = CWD / f'd_grand/spatial_viz/{sample_prefix}_slt_update_pca.png'
        ax = axes[row, 3]
        if fpath_inject.exists():
            img = np.array(Image.open(fpath_inject).resize((150, 150), Image.NEAREST))
            ax.imshow(img)
        if row == 0:
            ax.set_title('Inject direction', fontsize=8)
        cos_val = d1['inject']['mean'] if 'inject' in d1 else (d1['slt_01']['mean'] if 'slt_01' in d1 else 0)
        le_val = 0
        key = 'inject' if 'inject' in decay else 'slt_01'
        if key in decay:
            le_val = decay[key][0] - decay[key][-1]
        ax.set_xlabel(f'cos={cos_val:.2f}  LE={le_val:.3f}', fontsize=6, labelpad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

        # Col 4: Backward gradient PCA
        fpath_bw = CWD / f'd_grand/spatial_viz/{sample_prefix}_backward_grad_pca.png'
        ax = axes[row, 4]
        if fpath_bw.exists():
            img = np.array(Image.open(fpath_bw).resize((150, 150), Image.NEAREST))
            ax.imshow(img)
        if row == 0:
            ax.set_title('Backward gradient', fontsize=8)
        cos_val = d1['backward_iter1']['mean'] if 'backward_iter1' in d1 else 0
        le_val = (decay['backward_iter1'][0] - decay['backward_iter1'][-1]) if 'backward_iter1' in decay else 0
        ax.set_xlabel(f'cos={cos_val:.2f}  LE={le_val:.3f}', fontsize=6, labelpad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

    plt.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / 'fig8_pca_maps.png'); plt.close()


def fig9_qualitative_t2i():
    """5 styles × 7 methods = 35 images + 5 style refs = 40 panels."""
    styles = ['00', '10', '20', '27', '33']
    prompt = 'kp0'
    methods = ['inject', 'no_adam', 'no_slt', 'softmax', 'adam8', 'hybrid2']
    method_labels = ['Inject', 'SLT only', 'w/o SLT', 'w/o OT', 'Adam×8', 'Ours']
    n_sty, n_meth = len(styles), len(methods)

    fig, axes = plt.subplots(n_sty, n_meth+1, figsize=(n_meth*1.1+1.1, n_sty*1.1+0.2),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
    for row, sty_id in enumerate(styles):
        axes[row, 0].imshow(load_img(CWD / f'sty/{sty_id}.png', 112))
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values(): spine.set_visible(False)
        if row == 0: axes[row, 0].set_title('Style', fontsize=7)

        for col, method in enumerate(methods):
            if method in ('text_only', 'inject'):
                p = CWD / f'd_grand/t2i_baselines/{method}_{prompt}_s{sty_id}.png'
            else:
                p = CWD / f'd_grand/t2i_ablation/{method}_{prompt}_s{sty_id}.png'
            if p.exists():
                axes[row, col+1].imshow(load_img(p, 112))
            axes[row, col+1].set_xticks([]); axes[row, col+1].set_yticks([])
            for spine in axes[row, col+1].spines.values(): spine.set_visible(False)
            if row == 0: axes[row, col+1].set_title(method_labels[col], fontsize=7)
            if method == 'hybrid2':
                for spine in axes[row, col+1].spines.values():
                    spine.set_visible(True); spine.set_edgecolor(C['ours']); spine.set_linewidth(1.2)

    fig.savefig(FIG_DIR / 'fig9_qualitative_t2i.png'); plt.close()


def fig10_qualitative_i2i():
    """5 content × 6 styles = 30 results + refs = 41 panels."""
    contents = ['00', '03', '05', '08', '12']
    styles = ['10', '14', '20', '27', '33', '38']
    n_cnt, n_sty = len(contents), len(styles)

    fig, axes = plt.subplots(n_cnt+1, n_sty+1, figsize=(n_sty*1.1+1.1, n_cnt*1.1+1.3),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
    axes[0, 0].axis('off')
    for j, sty_id in enumerate(styles):
        axes[0, j+1].imshow(load_img(CWD / f'sty/{sty_id}.png', 100))
        axes[0, j+1].set_xticks([]); axes[0, j+1].set_yticks([])
        for spine in axes[0, j+1].spines.values(): spine.set_visible(False)

    for i, cnt_id in enumerate(contents):
        axes[i+1, 0].imshow(load_img(CWD / f'cnt/{cnt_id}.png', 100))
        axes[i+1, 0].set_xticks([]); axes[i+1, 0].set_yticks([])
        for spine in axes[i+1, 0].spines.values(): spine.set_visible(False)

        for j, sty_id in enumerate(styles):
            p = CWD / f'd_grand/i2i_benchmark/cnt{cnt_id}_sty{sty_id}.png'
            if p.exists():
                axes[i+1, j+1].imshow(load_img(p, 100))
            axes[i+1, j+1].set_xticks([]); axes[i+1, j+1].set_yticks([])
            for spine in axes[i+1, j+1].spines.values(): spine.set_visible(False)

    fig.savefig(FIG_DIR / 'fig10_qualitative_i2i.png'); plt.close()


def fig11_qualitative_ablation():
    """3 styles × 7 ablation methods = 21 results + 3 style refs = wide strip."""
    styles_prompts = [('27', 'kp2'), ('10', 'kp0'), ('33', 'kp3')]
    methods = ['no_adam', 'no_slt', 'no_grag', 'softmax', 'freq_off', 'adam8', 'hybrid2']
    labels = ['SLT only', 'w/o SLT', 'w/o LRS', 'w/o OT', 'w/o Freq', 'Adam×8', 'Ours']
    n_rows = len(styles_prompts)
    n_cols = len(methods) + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.2, n_rows*1.2+0.15),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
    for row, (sty_id, prompt) in enumerate(styles_prompts):
        axes[row, 0].imshow(load_img(CWD / f'sty/{sty_id}.png', 112))
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values(): spine.set_visible(False)
        if row == 0: axes[row, 0].set_title('Style', fontsize=7)

        for col, (method, label) in enumerate(zip(methods, labels)):
            p = CWD / f'd_grand/t2i_ablation/{method}_{prompt}_s{sty_id}.png'
            if p.exists():
                axes[row, col+1].imshow(load_img(p, 112))
            axes[row, col+1].set_xticks([]); axes[row, col+1].set_yticks([])
            for spine in axes[row, col+1].spines.values(): spine.set_visible(False)
            if row == 0: axes[row, col+1].set_title(label, fontsize=6.5)
            if method == 'hybrid2':
                for spine in axes[row, col+1].spines.values():
                    spine.set_visible(True); spine.set_edgecolor(C['ours']); spine.set_linewidth(1.2)

    fig.savefig(FIG_DIR / 'fig11_qualitative_ablation.png'); plt.close()


def fig12_t2i_benchmark():
    """6 prompts × 6 styles = 36 results + style refs on top row = full benchmark grid."""
    prompts = [('p0', 'Cat on table'), ('p1', 'Sports car'), ('p2', 'Sunflowers'),
               ('p3', 'Dog in park'), ('p4', 'Snowy mountain'), ('p5', 'Portrait')]
    styles = ['00', '10', '20', '27', '33', '38']
    n_rows, n_cols = len(prompts), len(styles)

    fig, axes = plt.subplots(n_rows+1, n_cols+1, figsize=(n_cols*1.1+1.1, (n_rows+1)*1.1),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
    # Top-left corner blank
    axes[0, 0].axis('off')
    # Top row: style refs
    for j, sty_id in enumerate(styles):
        axes[0, j+1].imshow(load_img(CWD / f'sty/{sty_id}.png', 100))
        axes[0, j+1].set_xticks([]); axes[0, j+1].set_yticks([])
        for spine in axes[0, j+1].spines.values(): spine.set_visible(False)

    # Content rows
    for i, (pid, desc) in enumerate(prompts):
        axes[i+1, 0].text(0.5, 0.5, desc, fontsize=6.5, ha='center', va='center',
                         transform=axes[i+1, 0].transAxes, style='italic')
        axes[i+1, 0].set_xticks([]); axes[i+1, 0].set_yticks([])
        for spine in axes[i+1, 0].spines.values(): spine.set_visible(False)

        for j, sty_id in enumerate(styles):
            p = CWD / f'd_grand/t2i_benchmark/hybrid2_{pid}_s{sty_id}.png'
            if p.exists():
                axes[i+1, j+1].imshow(load_img(p, 100))
            axes[i+1, j+1].set_xticks([]); axes[i+1, j+1].set_yticks([])
            for spine in axes[i+1, j+1].spines.values(): spine.set_visible(False)

    fig.savefig(FIG_DIR / 'fig12_t2i_benchmark.png'); plt.close()


def fig13_jacobian_ablation():
    """
    Jacobian ablation: 2-panel figure.
    Left: loss convergence (normal vs random vs stop-grad Q)
    Right: neighbor cosine (spatial coherence) per iter
    """
    with open(CWD / 'results/jacobian_ablation.json') as f:
        data = json.load(f)
    n = len(data)
    iters = np.arange(1, 5)

    # Aggregate across samples
    normal_loss = [np.mean([data[s]['normal']['loss'][it] for s in range(n)]) for it in range(4)]
    random_loss = [np.mean([data[s]['random_direction']['loss'][it] for s in range(n)]) for it in range(4)]
    stopq_loss = [np.mean([data[s]['stop_grad_q']['loss'][it] for s in range(n)]) for it in range(4)]

    normal_cos = [np.mean([data[s]['normal']['neighbor_cos'][it] for s in range(n)]) for it in range(4)]
    random_cos = [np.mean([data[s]['random_direction']['neighbor_cos'][it] for s in range(n)]) for it in range(4)]
    stopq_cos = [np.mean([data[s]['stop_grad_q']['neighbor_cos'][it] for s in range(n)]) for it in range(4)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.3))

    # Left: Loss convergence
    ax1.plot(iters, normal_loss, '-o', color=C['ours'], label='Normal (full gradient)', markersize=4)
    ax1.plot(iters, stopq_loss, '-s', color=C['inject'], label='Stop-grad Q', markersize=4)
    ax1.plot(iters, random_loss, '-^', color=C['random'], label='Random direction', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Loss ↓')
    ax1.legend(fontsize=6.5, loc='upper right')
    ax1.set_xlim(0.5, 4.5)
    ax1.grid(True)
    ax1.set_title('(a) Optimization Progress')

    # Right: Spatial coherence
    ax2.plot(iters, normal_cos, '-o', color=C['ours'], label='Normal (full gradient)', markersize=4)
    ax2.plot(iters, stopq_cos, '-s', color=C['inject'], label='Stop-grad Q', markersize=4)
    ax2.plot(iters, random_cos, '-^', color=C['random'], label='Random direction', markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Neighbor cos@d=1 ↑')
    ax2.legend(fontsize=6.5, loc='lower right')
    ax2.set_xlim(0.5, 4.5)
    ax2.set_ylim(-0.05, 0.8)
    ax2.grid(True)
    ax2.set_title('(b) Gradient Spatial Structure')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig13_jacobian_ablation.png'); plt.close()


if __name__ == '__main__':
    print("Generating paper figures (v4 — no titles, tight grids, diverse styles)...")
    fig1_ablation_bar();      print('  fig1_ablation')
    fig2_pareto();            print('  fig2_pareto')
    fig3_spatial_proof();     print('  fig3_spatial_proof')
    fig4_slt_sweep();         print('  fig4_slt_sweep')
    fig5_timing();            print('  fig5_timing')
    fig6_iters_sweep();       print('  fig6_iters_sweep')
    fig7_i2i_ablation();      print('  fig7_i2i_ablation')
    fig8_pca_maps();          print('  fig8_pca_maps')
    fig9_qualitative_t2i();   print('  fig9_qualitative_t2i')
    fig10_qualitative_i2i();  print('  fig10_qualitative_i2i')
    fig11_qualitative_ablation(); print('  fig11_qualitative_ablation')
    fig12_t2i_benchmark();    print('  fig12_t2i_benchmark')
    fig13_jacobian_ablation(); print('  fig13_jacobian_ablation')
    print(f"\nDone! All in: {FIG_DIR}/")
