"""
scAgeNet Plotting
=================
21 publication-ready plot functions.

All functions:
  - Accept results dict from scagenet.profile()
  - Accept save_path=None (show) or str (save to disk)
  - Return the matplotlib Figure for further customization
  - Respect set_style("paper"/"screen")
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from typing import Optional, Union, List, Dict

from .utils import (
    SCAGENET_PALETTE, order_cell_types_by_family, get_family,
    family_color, tier_color, save_figure, get_universal_categories,
)


# ---------------------------------------------------------------------------
# ── CATEGORY 1: CORE AGING ANALYSIS (7 plots) ───────────────────────────
# ---------------------------------------------------------------------------

def plot_aging_heatmap(
    results: dict,
    top_n: int = 50,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 1: Aging program activation heatmap (genes × cell types)."""
    imp = results.get('importance')
    gene_tiers = results.get('gene_tiers')
    if imp is None or gene_tiers is None:
        raise ValueError("results must contain 'importance' and 'gene_tiers'.")

    top_genes = (
        gene_tiers.sort_values('mean_importance', ascending=False)
        .head(top_n)['gene'].tolist()
    )
    # Keep only genes present in importance matrix
    top_genes = [g for g in top_genes if g in imp.index][:top_n]

    ct_order = order_cell_types_by_family(list(imp.columns))
    plot_mat = imp.loc[top_genes, [c for c in ct_order if c in imp.columns]]

    tier_map = dict(zip(gene_tiers['gene'], gene_tiers['tier']))
    tier_colors = [tier_color(tier_map.get(g, 'Non-Driver')) for g in top_genes]

    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={'width_ratios': [0.04, 1]},
    )

    # Left color bar: gene tiers
    ax_tier = axes[0]
    for i, color in enumerate(tier_colors):
        ax_tier.add_patch(
            mpatches.Rectangle((0, len(top_genes) - i - 1), 1, 1, color=color)
        )
    ax_tier.set_xlim(0, 1)
    ax_tier.set_ylim(0, len(top_genes))
    ax_tier.axis('off')

    # Main heatmap
    ax = axes[1]
    sns.heatmap(
        plot_mat,
        ax=ax,
        cmap='magma',
        xticklabels=True,
        yticklabels=True,
        cbar_kws={'label': 'Importance score', 'shrink': 0.5},
        **{k: v for k, v in kwargs.items() if k in ['vmin', 'vmax', 'linewidths']},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    ax.set_xlabel('Cell type')
    ax.set_ylabel(f'Top {len(top_genes)} aging genes')
    ax.set_title(kwargs.get('title', 'Aging Program Activation Heatmap'))

    # Legend for tiers
    handles = [
        mpatches.Patch(color=SCAGENET_PALETTE['universal'], label='Universal Driver'),
        mpatches.Patch(color=SCAGENET_PALETTE['cell_type'], label='Cell-Type Driver'),
        mpatches.Patch(color=SCAGENET_PALETTE['non_driver'], label='Non-Driver'),
    ]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1.05),
              fontsize=7, frameon=False)

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_cell_ranking(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 2: Cell-type aging signal ranking (horizontal bars)."""
    rankings = results.get('cell_rankings')
    if rankings is None:
        raise ValueError("results must contain 'cell_rankings'.")

    df = rankings.sort_values('mean_score', ascending=True)
    colors = [family_color(ct) for ct in df['cell_type']]
    overall_mean = df['mean_score'].mean()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(df)), df['mean_score'], color=colors, alpha=0.85)
    ax.axvline(overall_mean, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

    if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
        for i, row in enumerate(df.itertuples()):
            ax.errorbar(
                row.mean_score, i,
                xerr=[[row.mean_score - row.ci_lower], [row.ci_upper - row.mean_score]],
                fmt='none', color='grey', linewidth=1.0, capsize=2,
            )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['cell_type'], fontsize=7)
    ax.set_xlabel('Mean aging score (relative)')
    ax.set_title(kwargs.get('title', 'Cell-Type Aging Signal Ranking'))

    # Family legend
    handles = [
        mpatches.Patch(color=SCAGENET_PALETTE['immune'], label='Immune'),
        mpatches.Patch(color=SCAGENET_PALETTE['structural'], label='Structural'),
        mpatches.Patch(color=SCAGENET_PALETTE['endothelial'], label='Endothelial'),
        mpatches.Patch(color=SCAGENET_PALETTE['epithelial'], label='Epithelial'),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_architecture(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 7),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 3: Aging architecture classification (stacked bars)."""
    arch = results.get('architecture')
    if arch is None:
        raise ValueError("results must contain 'architecture'.")

    df = arch.sort_values('pct_universal', ascending=True)
    pct_univ = df['pct_universal'].values
    pct_priv = 1.0 - pct_univ
    labels = df['cell_type'].values
    arch_class = df['architecture_class'].values

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(df)), pct_univ,
            color=SCAGENET_PALETTE['centralized'], alpha=0.8, label='Universal drivers')
    ax.barh(range(len(df)), pct_priv, left=pct_univ,
            color=SCAGENET_PALETTE['cell_type'], alpha=0.8, label='Private genes')

    ax.axvline(0.40, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(0.60, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Fraction of top genes')
    ax.set_xlim(0, 1)
    ax.set_title(kwargs.get('title', 'Aging Architecture Classification'))

    # Right-side labels
    for i, cls in enumerate(arch_class):
        ax.text(1.02, i, cls, va='center', ha='left', fontsize=6,
                color=SCAGENET_PALETTE.get(cls, '#555'))

    ax.legend(fontsize=7, frameon=False, loc='lower right')
    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_reference_comparison(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 4: Gene program comparison to reference (scatter)."""
    ref = results.get('reference_comparison', {})
    imp = results.get('importance')
    gene_tiers = results.get('gene_tiers')

    from .utils import get_reference_importance
    try:
        ref_imp = get_reference_importance()
    except FileNotFoundError:
        print("[Plot] reference_importance.csv not found. Cannot draw reference scatter.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Reference data not available', ha='center', va='center',
                transform=ax.transAxes)
        save_figure(fig, save_path, dpi)
        return fig

    shared = list(set(imp.index) & set(ref_imp.index))
    user_mean = imp.loc[shared].mean(axis=1)
    ref_mean = ref_imp.loc[shared].mean(axis=1)

    tier_map = dict(zip(gene_tiers['gene'], gene_tiers['tier'])) if gene_tiers is not None else {}
    colors = [tier_color(tier_map.get(g, 'Non-Driver')) for g in shared]
    sizes = [10 if tier_map.get(g, 'Non-Driver') == 'Non-Driver' else 20 for g in shared]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(ref_mean, user_mean, c=colors, s=sizes, alpha=0.6, linewidths=0)

    # Diagonal
    lim = max(ref_mean.max(), user_mean.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5)

    # Label top 10
    top10 = user_mean.nlargest(10).index
    for g in top10:
        ax.annotate(g, (ref_mean[g], user_mean[g]), fontsize=6,
                    xytext=(3, 3), textcoords='offset points')

    # Annotation box
    rho_val = ref.get('overall_rho')
    j50 = ref.get('jaccard', {}).get(50)
    ann = []
    if rho_val is not None:
        ann.append(f"Spearman ρ = {rho_val:.3f}")
    if j50 is not None:
        ann.append(f"Jaccard@50 = {j50:.3f}")
    if ann:
        ax.text(0.05, 0.92, '\n'.join(ann), transform=ax.transAxes,
                fontsize=7, va='top', bbox=dict(boxstyle='round', alpha=0.1))

    ax.set_xlabel('Reference importance (mean)')
    ax.set_ylabel('User importance (mean)')
    ax.set_title(kwargs.get('title', 'Gene Program Comparison to Reference'))

    handles = [
        mpatches.Patch(color=SCAGENET_PALETTE['universal'], label='Universal Driver'),
        mpatches.Patch(color=SCAGENET_PALETTE['cell_type'], label='Cell-Type Driver'),
        mpatches.Patch(color=SCAGENET_PALETTE['non_driver'], label='Non-Driver'),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_jaccard_curve(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 5),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 5: Jaccard similarity at multiple gene-count thresholds."""
    ref = results.get('reference_comparison', {})
    jaccard = ref.get('jaccard', {})

    if not jaccard:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No reference comparison data', ha='center', va='center',
                transform=ax.transAxes)
        save_figure(fig, save_path, dpi)
        return fig

    ks = sorted(jaccard.keys())
    vals = [jaccard[k] for k in ks]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, vals, 'o-', color='black', linewidth=1.5, markersize=4, label='All genes')
    ax.axhline(0.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Top-N genes considered')
    ax.set_ylabel('Jaccard similarity (user ∩ reference)')
    ax.set_title(kwargs.get('title', 'Gene Program Overlap with Reference'))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_go_enrichment(
    results: dict,
    tier: str = "all",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 6: GO enrichment bubble plot."""
    go = results.get('go_enrichment', {})
    if not go:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No GO enrichment data\n(run with run_go=True)',
                ha='center', va='center', transform=ax.transAxes)
        save_figure(fig, save_path, dpi)
        return fig

    def _bubble_panel(ax, df, title, color):
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, 'No significant terms', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            return
        df = df.head(10).copy()
        df['-log10p'] = -np.log10(df['p_value'].clip(lower=1e-20))
        y_pos = range(len(df))
        sc = ax.scatter(df['gene_ratio'], y_pos,
                       s=df['intersection_size'] * 5,
                       c=df['-log10p'], cmap='YlOrRd', alpha=0.8,
                       edgecolors='grey', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['name'], fontsize=7)
        ax.set_xlabel('Gene ratio')
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label='-log10(p)', shrink=0.6)

    if tier == "all":
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        _bubble_panel(axes[0], go.get('universal'), 'Universal Drivers (GO)', '#E8853F')
        _bubble_panel(axes[1], go.get('cell_type'), 'Cell-Type Drivers (GO)', '#8B6DAF')
    else:
        fig, ax = plt.subplots(figsize=(figsize[0] // 2 + 2, figsize[1]))
        df = go.get(tier, pd.DataFrame())
        title = f"{'Universal' if tier=='universal' else 'Cell-Type'} Drivers (GO)"
        _bubble_panel(ax, df, title, SCAGENET_PALETTE.get(tier, '#999'))

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_summary_dashboard(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 7: Single-page summary dashboard (3×2 grid)."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35)

    # ── Panel 1: Heatmap mini ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    imp = results.get('importance')
    gene_tiers = results.get('gene_tiers')
    if imp is not None and gene_tiers is not None:
        top20 = (
            gene_tiers.sort_values('mean_importance', ascending=False)
            .head(20)['gene'].tolist()
        )
        top20 = [g for g in top20 if g in imp.index]
        if top20:
            ct_order = order_cell_types_by_family(list(imp.columns))
            mat = imp.loc[top20, [c for c in ct_order if c in imp.columns]]
            sns.heatmap(mat, ax=ax1, cmap='magma', xticklabels=False,
                        yticklabels=True, cbar=False)
            ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=5)
            ax1.set_title('Aging Heatmap (top 20)', fontsize=8)

    # ── Panel 2: Cell ranking mini ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    rankings = results.get('cell_rankings')
    if rankings is not None:
        df = rankings.sort_values('mean_score', ascending=True)
        colors = [family_color(ct) for ct in df['cell_type']]
        ax2.barh(range(len(df)), df['mean_score'], color=colors, alpha=0.8)
        ax2.set_yticks(range(len(df)))
        ax2.set_yticklabels(df['cell_type'], fontsize=5)
        ax2.set_title('Cell Ranking', fontsize=8)

    # ── Panel 3: Architecture mini ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    arch = results.get('architecture')
    if arch is not None:
        df_a = arch.sort_values('pct_universal', ascending=True)
        ax3.barh(range(len(df_a)), df_a['pct_universal'],
                 color=SCAGENET_PALETTE['centralized'], alpha=0.8)
        ax3.barh(range(len(df_a)), 1 - df_a['pct_universal'].values,
                 left=df_a['pct_universal'].values,
                 color=SCAGENET_PALETTE['cell_type'], alpha=0.8)
        ax3.axvline(0.6, color='k', linestyle='--', linewidth=0.6)
        ax3.axvline(0.4, color='k', linestyle='--', linewidth=0.6)
        ax3.set_yticks(range(len(df_a)))
        ax3.set_yticklabels(df_a['cell_type'], fontsize=5)
        ax3.set_title('Architecture', fontsize=8)

    # ── Panel 4: Reference scatter mini ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ref = results.get('reference_comparison', {})
    if ref.get('overall_rho') is not None:
        rho = ref['overall_rho']
        j50 = ref.get('jaccard', {}).get(50, 'N/A')
        ax4.text(0.5, 0.6, f"Reference ρ = {rho:.3f}", ha='center',
                 fontsize=14, transform=ax4.transAxes, fontweight='bold')
        ax4.text(0.5, 0.4, f"Jaccard@50 = {j50:.3f}" if isinstance(j50, float) else f"Jaccard@50 = {j50}",
                 ha='center', fontsize=12, transform=ax4.transAxes)
        ax4.axis('off')
        ax4.set_title('Reference Similarity', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No reference', ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')

    # ── Panel 5: Text summary ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    adata = results.get('adata')
    lines = []
    if adata is not None:
        lines.append(f"Cells: {adata.n_obs:,}")
        ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
        n_ct = adata.obs[ct_col].nunique()
        lines.append(f"Cell types: {n_ct}")
    if gene_tiers is not None:
        top5 = gene_tiers.sort_values('mean_importance', ascending=False)['gene'].head(5).tolist()
        lines.append(f"Top genes: {', '.join(top5)}")
    if arch is not None:
        n_cent = (arch['architecture_class'] == 'centralized').sum()
        n_decent = (arch['architecture_class'] == 'decentralized').sum()
        lines.append(f"Centralized: {n_cent}, Decentralized: {n_decent}")
    summary_text = '\n'.join(lines) if lines else 'No summary available'
    ax5.text(0.05, 0.95, summary_text, va='top', ha='left',
             transform=ax5.transAxes, fontsize=8,
             bbox=dict(boxstyle='round', alpha=0.1))
    ax5.set_title('Summary', fontsize=8)

    # ── Panel 6: GO top 5 ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    go = results.get('go_enrichment', {})
    univ_go = go.get('universal', pd.DataFrame())
    if len(univ_go) > 0:
        top5_go = univ_go.head(5)
        go_text = "Top GO terms (universal):\n" + '\n'.join(
            f"• {row['name'][:35]} (p={row['p_value']:.1e})"
            for _, row in top5_go.iterrows()
        )
        ax6.text(0.05, 0.95, go_text, va='top', ha='left',
                 transform=ax6.transAxes, fontsize=7,
                 bbox=dict(boxstyle='round', alpha=0.1))
    else:
        ax6.text(0.5, 0.5, 'No GO data', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('GO Enrichment', fontsize=8)

    fig.suptitle(kwargs.get('title', 'scAgeNet Analysis Summary'), fontsize=11, y=0.98)
    save_figure(fig, save_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# ── CATEGORY 2: UMAP VISUALIZATIONS (7 plots) ────────────────────────────
# ---------------------------------------------------------------------------

def _get_umap_coords(results):
    adata = results.get('adata')
    if adata is None or 'X_umap' not in adata.obsm:
        raise ValueError("UMAP not found in results['adata'].obsm['X_umap'].")
    return adata.obsm['X_umap']


def plot_umap_celltype(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 8: UMAP colored by cell type."""
    adata = results.get('adata')
    umap = _get_umap_coords(results)
    ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
    labels = adata.obs[ct_col].values

    unique_cts = sorted(set(labels))
    palette = {ct: family_color(ct) for ct in unique_cts}

    fig, ax = plt.subplots(figsize=figsize)
    for ct in unique_cts:
        mask = labels == ct
        ax.scatter(umap[mask, 0], umap[mask, 1], c=palette[ct], s=1.5,
                   alpha=0.6, label=ct, linewidths=0)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(kwargs.get('title', 'UMAP — Cell Type'))
    ax.axis('off')

    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=6,
                    markerscale=3, frameon=False)
    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_umap_aging_score(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (6, 5),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 9: UMAP colored by predicted aging score."""
    adata = results.get('adata')
    umap = _get_umap_coords(results)
    scores = results.get('predictions')

    if scores is None and adata is not None and 'aging_score' in adata.obs:
        scores = adata.obs['aging_score']

    if scores is None:
        raise ValueError("No aging scores found. Run profile() first.")

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(umap[:, 0], umap[:, 1], c=scores, cmap='coolwarm',
                    s=1.5, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=ax, label='Aging score (relative)', shrink=0.7)
    ax.set_title(kwargs.get('title', 'UMAP — Aging Score'))
    ax.axis('off')
    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_umap_condition(
    results: dict,
    condition_col: str,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 10: UMAP colored by condition/group."""
    adata = results.get('adata')
    umap = _get_umap_coords(results)

    if condition_col not in adata.obs.columns:
        raise ValueError(f"Column '{condition_col}' not found in adata.obs.")

    conditions = adata.obs[condition_col].astype(str).values
    unique_conds = sorted(set(conditions))
    cond_palette = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    if len(unique_conds) == 2:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        for i, cond in enumerate(unique_conds):
            mask = conditions == cond
            axes[i].scatter(umap[~mask, 0], umap[~mask, 1], c='#EEEEEE', s=1, alpha=0.3,
                           linewidths=0)
            axes[i].scatter(umap[mask, 0], umap[mask, 1],
                           c=cond_palette[i % len(cond_palette)], s=2, alpha=0.8, linewidths=0)
            axes[i].set_title(cond)
            axes[i].axis('off')
        fig.suptitle(kwargs.get('title', f'UMAP — {condition_col}'))
    else:
        fig, ax = plt.subplots(figsize=(figsize[0] // 2 + 2, figsize[1]))
        for i, cond in enumerate(unique_conds):
            mask = conditions == cond
            ax.scatter(umap[mask, 0], umap[mask, 1],
                      c=cond_palette[i % len(cond_palette)], s=1.5, alpha=0.7,
                      label=cond, linewidths=0)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(kwargs.get('title', f'UMAP — {condition_col}'))
        ax.axis('off')

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_umap_gene(
    results: dict,
    gene: Union[str, List[str]],
    save_path: Optional[str] = None,
    figsize: tuple = (6, 5),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 11: UMAP colored by gene expression."""
    adata = results.get('adata')
    umap = _get_umap_coords(results)

    genes = [gene] if isinstance(gene, str) else gene

    ncols = min(len(genes), 3)
    nrows = (len(genes) + ncols - 1) // ncols
    fig_w = figsize[0] * ncols
    fig_h = figsize[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    from scipy import sparse
    for idx, g in enumerate(genes):
        ax = axes[idx // ncols][idx % ncols]
        if g not in adata.var_names:
            ax.text(0.5, 0.5, f'{g}\nnot found', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            continue

        g_idx = list(adata.var_names).index(g)
        X = adata.X
        expr = X[:, g_idx].toarray().ravel() if sparse.issparse(X) else X[:, g_idx]

        sc_plot = ax.scatter(umap[:, 0], umap[:, 1], c=expr, cmap='YlOrRd',
                            s=1.5, alpha=0.8, linewidths=0)
        plt.colorbar(sc_plot, ax=ax, label='Expression', shrink=0.7)
        ax.set_title(g)
        ax.axis('off')

    # Hide unused axes
    for idx in range(len(genes), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    fig.suptitle(kwargs.get('title', 'Gene Expression — UMAP'))
    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_umap_architecture(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (6, 5),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 12: UMAP colored by aging architecture class per cell."""
    adata = results.get('adata')
    arch = results.get('architecture')
    umap = _get_umap_coords(results)

    if arch is None:
        raise ValueError("results must contain 'architecture'.")

    arch_map = dict(zip(arch['cell_type'], arch['architecture_class']))
    ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
    cell_arch = adata.obs[ct_col].map(arch_map).fillna('unknown').values

    fig, ax = plt.subplots(figsize=figsize)
    for cls in ['centralized', 'intermediate', 'decentralized', 'unknown']:
        mask = cell_arch == cls
        ax.scatter(umap[mask, 0], umap[mask, 1],
                  c=SCAGENET_PALETTE.get(cls, '#CCCCCC'), s=1.5,
                  alpha=0.7, label=cls.capitalize(), linewidths=0)

    ax.legend(fontsize=7, markerscale=3, frameon=False)
    ax.set_title(kwargs.get('title', 'UMAP — Aging Architecture'))
    ax.axis('off')
    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_umap_tier_activation(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 13: UMAP — universal vs private gene program activation."""
    adata = results.get('adata')
    gene_tiers = results.get('gene_tiers')
    umap = _get_umap_coords(results)

    if gene_tiers is None:
        raise ValueError("results must contain 'gene_tiers'.")

    from scipy import sparse

    univ_genes = gene_tiers[gene_tiers['tier'] == 'Universal Driver']['gene'].tolist()
    priv_genes = gene_tiers[gene_tiers['tier'] == 'Cell-Type Driver']['gene'].tolist()

    def _mean_expr(genes_subset):
        genes_in = [g for g in genes_subset if g in adata.var_names]
        if not genes_in:
            return np.zeros(adata.n_obs)
        idx = [list(adata.var_names).index(g) for g in genes_in]
        X = adata.X
        sub = X[:, idx].toarray() if sparse.issparse(X) else X[:, idx]
        return sub.mean(axis=1)

    univ_expr = _mean_expr(univ_genes)
    priv_expr = _mean_expr(priv_genes)

    vmin = min(univ_expr.min(), priv_expr.min())
    vmax = max(univ_expr.max(), priv_expr.max())

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    sc1 = axes[0].scatter(umap[:, 0], umap[:, 1], c=univ_expr, cmap='YlOrRd',
                          s=1.5, alpha=0.8, vmin=vmin, vmax=vmax, linewidths=0)
    plt.colorbar(sc1, ax=axes[0], label='Mean expression', shrink=0.7)
    axes[0].set_title('Universal drivers')
    axes[0].axis('off')

    sc2 = axes[1].scatter(umap[:, 0], umap[:, 1], c=priv_expr, cmap='YlOrRd',
                          s=1.5, alpha=0.8, vmin=vmin, vmax=vmax, linewidths=0)
    plt.colorbar(sc2, ax=axes[1], label='Mean expression', shrink=0.7)
    axes[1].set_title('Cell-type-specific genes')
    axes[1].axis('off')

    fig.suptitle(kwargs.get('title', 'UMAP — Gene Tier Activation'))
    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_umap_panel(
    results: dict,
    condition_col: Optional[str] = None,
    genes: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 14: Convenience 2×2 UMAP overview panel."""
    adata = results.get('adata')
    umap = _get_umap_coords(results)
    ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'

    n_rows = 2
    n_cols = 2
    if genes:
        n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Cell type
    labels = adata.obs[ct_col].values
    unique_cts = sorted(set(labels))
    for ct in unique_cts:
        mask = labels == ct
        axes[0, 0].scatter(umap[mask, 0], umap[mask, 1], c=family_color(ct),
                           s=1, alpha=0.6, linewidths=0)
    axes[0, 0].set_title('Cell type')
    axes[0, 0].axis('off')

    # Aging score
    scores = results.get('predictions')
    if scores is None and 'aging_score' in adata.obs:
        scores = adata.obs['aging_score']
    if scores is not None:
        sc = axes[0, 1].scatter(umap[:, 0], umap[:, 1], c=scores, cmap='coolwarm',
                                s=1, alpha=0.7, linewidths=0)
        plt.colorbar(sc, ax=axes[0, 1], shrink=0.6, label='Score')
    axes[0, 1].set_title('Aging score')
    axes[0, 1].axis('off')

    # Condition or architecture
    if condition_col and condition_col in adata.obs.columns:
        conds = adata.obs[condition_col].astype(str).values
        unique_c = sorted(set(conds))
        colors_c = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
        for i, c in enumerate(unique_c):
            mask = conds == c
            axes[1, 0].scatter(umap[mask, 0], umap[mask, 1],
                               c=colors_c[i % len(colors_c)], s=1, alpha=0.7,
                               label=c, linewidths=0)
        axes[1, 0].legend(fontsize=6, markerscale=3, frameon=False)
        axes[1, 0].set_title(f'Condition: {condition_col}')
    else:
        arch = results.get('architecture')
        if arch is not None:
            arch_map = dict(zip(arch['cell_type'], arch['architecture_class']))
            cell_arch = adata.obs[ct_col].map(arch_map).fillna('unknown').values
            for cls in ['centralized', 'intermediate', 'decentralized']:
                mask = cell_arch == cls
                axes[1, 0].scatter(umap[mask, 0], umap[mask, 1],
                                   c=SCAGENET_PALETTE.get(cls, '#CCC'), s=1,
                                   alpha=0.7, label=cls, linewidths=0)
            axes[1, 0].legend(fontsize=6, markerscale=3, frameon=False)
        axes[1, 0].set_title('Architecture')
    axes[1, 0].axis('off')

    # Architecture or top universal gene
    gene_tiers = results.get('gene_tiers')
    if gene_tiers is not None:
        top_gene = gene_tiers.sort_values('mean_importance', ascending=False)['gene'].iloc[0]
        from scipy import sparse
        if top_gene in adata.var_names:
            g_idx = list(adata.var_names).index(top_gene)
            X = adata.X
            expr = X[:, g_idx].toarray().ravel() if sparse.issparse(X) else X[:, g_idx]
            sc = axes[1, 1].scatter(umap[:, 0], umap[:, 1], c=expr, cmap='YlOrRd',
                                    s=1, alpha=0.8, linewidths=0)
            plt.colorbar(sc, ax=axes[1, 1], shrink=0.6, label='Expression')
            axes[1, 1].set_title(f'Gene: {top_gene}')
    axes[1, 1].axis('off')

    if genes and n_rows == 3:
        for idx, g in enumerate(genes[:n_cols]):
            ax = axes[2, idx]
            from scipy import sparse
            if g in adata.var_names:
                g_idx = list(adata.var_names).index(g)
                X = adata.X
                expr = X[:, g_idx].toarray().ravel() if sparse.issparse(X) else X[:, g_idx]
                sc = ax.scatter(umap[:, 0], umap[:, 1], c=expr, cmap='YlOrRd',
                               s=1, alpha=0.8, linewidths=0)
                plt.colorbar(sc, ax=ax, shrink=0.6)
                ax.set_title(g)
            ax.axis('off')

    fig.suptitle(kwargs.get('title', 'scAgeNet UMAP Overview'), fontsize=11)
    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# ── CATEGORY 3: GROUP COMPARISON PLOTS (5 plots) ─────────────────────────
# ---------------------------------------------------------------------------

def plot_condition_violin(
    results: dict,
    condition_col: str,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 15: Aging score by condition per cell type (violins)."""
    adata = results.get('adata')
    predictions = results.get('predictions')

    if predictions is None and 'aging_score' in adata.obs:
        predictions = adata.obs['aging_score']
    if predictions is None:
        raise ValueError("No aging scores in results.")
    if condition_col not in adata.obs.columns:
        raise ValueError(f"Column '{condition_col}' not in adata.obs.")

    from scipy.stats import mannwhitneyu
    ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'

    df = pd.DataFrame({
        'score': predictions.values,
        'cell_type': adata.obs[ct_col].values,
        'condition': adata.obs[condition_col].values,
    })
    cell_types = sorted(df['cell_type'].unique())
    conditions = sorted(df['condition'].unique())
    colors = ['#2196F3', '#FF5722']

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(cell_types))
    width = 0.35

    for ci, cond in enumerate(conditions[:2]):
        cond_df = df[df['condition'] == cond]
        for pi, ct in enumerate(cell_types):
            ct_data = cond_df[cond_df['cell_type'] == ct]['score'].values
            if len(ct_data) < 5:
                continue
            parts = ax.violinplot([ct_data], positions=[positions[pi] + (ci - 0.5) * width],
                                  widths=width * 0.8, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[ci])
                pc.set_alpha(0.7)

    # P-value stars
    for pi, ct in enumerate(cell_types):
        g1 = df[(df['condition'] == conditions[0]) & (df['cell_type'] == ct)]['score'].values
        g2 = df[(df['condition'] == conditions[1]) & (df['cell_type'] == ct)]['score'].values
        if len(g1) >= 5 and len(g2) >= 5:
            _, p = mannwhitneyu(g1, g2, alternative='two-sided')
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if star:
                ymax = max(g1.max(), g2.max())
                ax.text(positions[pi], ymax * 1.05, star, ha='center', fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(cell_types, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Aging score')
    ax.set_title(kwargs.get('title', f'Aging Score by {condition_col}'))

    handles = [mpatches.Patch(color=colors[i], label=c) for i, c in enumerate(conditions[:2])]
    ax.legend(handles=handles, fontsize=7, frameon=False)

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_condition_auroc(
    results: dict,
    condition_col: str,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 16: AUROC — can aging score distinguish conditions per cell type."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu

    adata = results.get('adata')
    predictions = results.get('predictions')
    if predictions is None and 'aging_score' in adata.obs:
        predictions = adata.obs['aging_score']
    if condition_col not in adata.obs.columns:
        raise ValueError(f"Column '{condition_col}' not in adata.obs.")

    ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
    df = pd.DataFrame({
        'score': predictions.values,
        'cell_type': adata.obs[ct_col].values,
        'condition': adata.obs[condition_col].astype(str).values,
    })
    conditions = sorted(df['condition'].unique())[:2]
    df_sub = df[df['condition'].isin(conditions)]

    records = []
    for ct in sorted(df_sub['cell_type'].unique()):
        ct_df = df_sub[df_sub['cell_type'] == ct]
        g1 = ct_df[ct_df['condition'] == conditions[0]]['score'].values
        g2 = ct_df[ct_df['condition'] == conditions[1]]['score'].values
        if len(g1) < 5 or len(g2) < 5:
            continue
        y_true = np.array([0] * len(g1) + [1] * len(g2))
        y_score = np.concatenate([g1, g2])
        auc = roc_auc_score(y_true, y_score)
        _, p = mannwhitneyu(g1, g2, alternative='two-sided')
        records.append({'cell_type': ct, 'auroc': auc, 'p': p})

    if not records:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Not enough data per cell type', ha='center', va='center')
        save_figure(fig, save_path, dpi)
        return fig

    rdf = pd.DataFrame(records).sort_values('auroc', ascending=True)
    colors = [family_color(ct) for ct in rdf['cell_type']]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(rdf)), rdf['auroc'], color=colors, alpha=0.8)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels(rdf['cell_type'], fontsize=7)
    ax.set_xlabel('AUROC')
    ax.set_title(kwargs.get('title', f'Condition Separability ({conditions[0]} vs {conditions[1]})'))

    for i, row in enumerate(rdf.itertuples()):
        star = '***' if row.p < 0.001 else '**' if row.p < 0.01 else '*' if row.p < 0.05 else ''
        if star:
            ax.text(row.auroc + 0.01, i, star, va='center', fontsize=8)

    mean_auc = rdf['auroc'].mean()
    ax.text(0.05, 0.98, f"Mean AUROC = {mean_auc:.3f}", transform=ax.transAxes,
            va='top', fontsize=8)

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_differential_aging_genes(
    results: dict,
    condition_col: str,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 17: Differential aging genes between conditions (volcano-style)."""
    adata = results.get('adata')
    imp = results.get('importance')
    gene_tiers = results.get('gene_tiers')

    if condition_col not in adata.obs.columns:
        raise ValueError(f"Column '{condition_col}' not in adata.obs.")

    conditions = sorted(adata.obs[condition_col].astype(str).unique())[:2]
    ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'

    # Compute mean importance per condition using stored gene importance per cell
    gene_importance = adata.obsm.get('gene_importance')
    try:
        gene_list = list(imp.index)
    except Exception:
        gene_list = list(adata.var_names)

    if gene_importance is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Per-cell importance not available.\nRun profile() to compute it.",
                ha='center', va='center', transform=ax.transAxes)
        save_figure(fig, save_path, dpi)
        return fig

    from scipy.stats import mannwhitneyu

    mask1 = adata.obs[condition_col].astype(str) == conditions[0]
    mask2 = adata.obs[condition_col].astype(str) == conditions[1]

    mean1 = gene_importance[mask1].mean(axis=0)
    mean2 = gene_importance[mask2].mean(axis=0)
    delta = mean2 - mean1

    pvals = []
    for g_i in range(gene_importance.shape[1]):
        g1 = gene_importance[mask1, g_i]
        g2 = gene_importance[mask2, g_i]
        if len(g1) >= 3 and len(g2) >= 3:
            _, p = mannwhitneyu(g1, g2, alternative='two-sided')
        else:
            p = 1.0
        pvals.append(p)

    pvals = np.array(pvals)
    log_p = -np.log10(np.clip(pvals, 1e-20, 1))

    tier_map = dict(zip(gene_tiers['gene'], gene_tiers['tier'])) if gene_tiers is not None else {}
    colors = [tier_color(tier_map.get(g, 'Non-Driver')) for g in gene_list]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(delta, log_p, c=colors, s=5, alpha=0.6, linewidths=0)

    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.axhline(-np.log10(0.05), color='grey', linestyle=':', linewidth=0.8)

    ax.set_xlabel(f'Δ importance ({conditions[1]} − {conditions[0]})')
    ax.set_ylabel('-log10(p)')
    ax.set_title(kwargs.get('title', 'Differential Aging Genes'))

    # Label top 10 each direction
    top_pos = np.argsort(delta)[-10:]
    top_neg = np.argsort(delta)[:10]
    for idx in list(top_pos) + list(top_neg):
        if idx < len(gene_list):
            ax.annotate(gene_list[idx], (delta[idx], log_p[idx]), fontsize=5,
                       xytext=(3, 3), textcoords='offset points')

    handles = [
        mpatches.Patch(color=SCAGENET_PALETTE['universal'], label='Universal Driver'),
        mpatches.Patch(color=SCAGENET_PALETTE['cell_type'], label='Cell-Type Driver'),
        mpatches.Patch(color=SCAGENET_PALETTE['non_driver'], label='Non-Driver'),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_composition(
    results: dict,
    condition_col: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 18: Cell-type composition (stacked bars)."""
    adata = results.get('adata')
    ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
    cell_types = sorted(adata.obs[ct_col].unique())
    ct_palette = {ct: family_color(ct) for ct in cell_types}

    fig, ax = plt.subplots(figsize=figsize)

    if condition_col and condition_col in adata.obs.columns:
        conditions = sorted(adata.obs[condition_col].astype(str).unique())
        bar_data = {}
        for cond in conditions:
            mask = adata.obs[condition_col].astype(str) == cond
            counts = adata.obs[ct_col][mask].value_counts(normalize=True)
            bar_data[cond] = counts

        x_pos = np.arange(len(conditions))
        bottom = np.zeros(len(conditions))
        for ct in cell_types:
            vals = [bar_data[c].get(ct, 0) for c in conditions]
            ax.bar(x_pos, vals, bottom=bottom, color=ct_palette[ct], label=ct)
            bottom += np.array(vals)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions, rotation=0)
    else:
        counts = adata.obs[ct_col].value_counts(normalize=True)
        bottom = 0
        for ct in cell_types:
            val = counts.get(ct, 0)
            ax.bar(0, val, bottom=bottom, color=ct_palette[ct], label=ct)
            bottom += val
        ax.set_xticks([0])
        ax.set_xticklabels(['All cells'])

    ax.set_ylabel('Proportion')
    ax.set_title(kwargs.get('title', 'Cell-Type Composition'))
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=6, frameon=False)

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


def plot_gene_expression(
    results: dict,
    genes: Union[str, List[str]],
    condition_col: Optional[str] = None,
    cell_type: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 19: Per-gene expression violins."""
    from scipy import sparse

    adata = results.get('adata')
    gene_list = [genes] if isinstance(genes, str) else genes
    ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'

    adata_plot = adata[adata.obs[ct_col] == cell_type].copy() if cell_type else adata

    g = gene_list[0]  # simple version: one gene at a time
    if g not in adata_plot.var_names:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Gene '{g}' not found", ha='center', va='center')
        save_figure(fig, save_path, dpi)
        return fig

    g_idx = list(adata_plot.var_names).index(g)
    X = adata_plot.X
    expr = X[:, g_idx].toarray().ravel() if sparse.issparse(X) else X[:, g_idx]

    fig, ax = plt.subplots(figsize=figsize)

    if condition_col and condition_col in adata_plot.obs.columns:
        conds = adata_plot.obs[condition_col].astype(str).values
        unique_conds = sorted(set(conds))
        data_by_cond = [expr[conds == c] for c in unique_conds]
        ax.violinplot(data_by_cond, positions=range(len(unique_conds)), showmedians=True)
        ax.set_xticks(range(len(unique_conds)))
        ax.set_xticklabels(unique_conds)
    else:
        cts = adata_plot.obs[ct_col].values
        unique_cts = sorted(set(cts))
        data_by_ct = [expr[cts == ct] for ct in unique_cts if (cts == ct).sum() >= 5]
        valid_cts = [ct for ct in unique_cts if (cts == ct).sum() >= 5]
        ax.violinplot(data_by_ct, positions=range(len(valid_cts)), showmedians=True)
        ax.set_xticks(range(len(valid_cts)))
        ax.set_xticklabels(valid_cts, rotation=45, ha='right', fontsize=7)

    title = f'{g} expression'
    if cell_type:
        title += f' — {cell_type}'
    ax.set_ylabel('Expression')
    ax.set_title(kwargs.get('title', title))

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# ── CATEGORY 4: CELL-TYPE DEEP DIVES (2 plots) ───────────────────────────
# ---------------------------------------------------------------------------

def plot_celltype_report(
    results: dict,
    cell_type: str,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 20: Single cell-type deep dive (multi-panel)."""
    adata = results.get('adata')
    imp = results.get('importance')
    gene_tiers = results.get('gene_tiers')
    arch = results.get('architecture')
    go = results.get('go_enrichment', {})
    umap = adata.obsm.get('X_umap')

    if cell_type not in (imp.columns if imp is not None else []):
        available = list(imp.columns) if imp is not None else []
        raise ValueError(f"Cell type '{cell_type}' not in importance matrix. "
                         f"Available: {available[:5]}...")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    tier_map = dict(zip(gene_tiers['gene'], gene_tiers['tier'])) if gene_tiers is not None else {}

    # ── Panel a: Top 20 genes bar chart ──────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    if imp is not None:
        top20 = imp[cell_type].nlargest(20)
        colors = [tier_color(tier_map.get(g, 'Non-Driver')) for g in top20.index]
        ax_a.barh(range(len(top20)), top20.values, color=colors, alpha=0.85)
        ax_a.set_yticks(range(len(top20)))
        ax_a.set_yticklabels(top20.index, fontsize=7)
        ax_a.set_xlabel('Importance')
        ax_a.set_title('Top 20 aging genes')

    # ── Panel b: Architecture pie ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if arch is not None:
        ct_arch = arch[arch['cell_type'] == cell_type]
        if len(ct_arch):
            row = ct_arch.iloc[0]
            ax_b.pie(
                [row['n_universal'], row['n_top_genes'] - row['n_universal']],
                labels=['Universal', 'Private'],
                colors=[SCAGENET_PALETTE['universal'], SCAGENET_PALETTE['cell_type']],
                autopct='%1.0f%%', startangle=90,
            )
            ax_b.set_title(f"Architecture: {row['architecture_class']}")

    # ── Panel c: UMAP highlight ───────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    if umap is not None:
        ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
        labels = adata.obs[ct_col].values
        mask = labels == cell_type
        ax_c.scatter(umap[~mask, 0], umap[~mask, 1], c='#EEEEEE', s=1, alpha=0.3, linewidths=0)
        ax_c.scatter(umap[mask, 0], umap[mask, 1], c=SCAGENET_PALETTE['immune'],
                    s=3, alpha=0.9, linewidths=0)
        ax_c.set_title('UMAP location')
        ax_c.axis('off')

    # ── Panel d: Top 5 GO terms ───────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_d.axis('off')
    univ_go = go.get('universal', pd.DataFrame())
    if len(univ_go) > 0:
        lines = ['Top GO terms:\n'] + [
            f"• {row['name'][:50]} (p={row['p_value']:.1e})"
            for _, row in univ_go.head(5).iterrows()
        ]
        ax_d.text(0.02, 0.95, '\n'.join(lines), va='top', transform=ax_d.transAxes, fontsize=8)

    # ── Panel e: Top 5 gene expression (aging score distribution) ────
    ax_e = fig.add_subplot(gs[1, 2])
    predictions = results.get('predictions')
    if predictions is not None:
        ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
        ct_mask = adata.obs[ct_col].values == cell_type
        ct_scores = predictions.values[ct_mask]
        all_scores = predictions.values
        ax_e.hist(all_scores, bins=30, color='#CCCCCC', alpha=0.6, density=True, label='All cells')
        ax_e.hist(ct_scores, bins=20, color=family_color(cell_type), alpha=0.8,
                 density=True, label=cell_type)
        ax_e.set_xlabel('Aging score')
        ax_e.set_ylabel('Density')
        ax_e.set_title('Score distribution')
        ax_e.legend(fontsize=7, frameon=False)

    fig.suptitle(f'{cell_type} — Aging Profile', fontsize=12, fontweight='bold')
    save_figure(fig, save_path, dpi)
    return fig


def plot_celltype_comparison(
    results: dict,
    cell_type_1: str,
    cell_type_2: str,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Plot 21: Pairwise cell-type importance comparison (scatter)."""
    imp = results.get('importance')
    gene_tiers = results.get('gene_tiers')

    for ct in [cell_type_1, cell_type_2]:
        if ct not in imp.columns:
            raise ValueError(f"'{ct}' not in importance matrix.")

    tier_map = dict(zip(gene_tiers['gene'], gene_tiers['tier'])) if gene_tiers is not None else {}

    x = imp[cell_type_1].values
    y = imp[cell_type_2].values
    genes = list(imp.index)

    top50_1 = set(imp[cell_type_1].nlargest(50).index)
    top50_2 = set(imp[cell_type_2].nlargest(50).index)
    shared = top50_1 & top50_2

    colors = []
    for g in genes:
        if g in shared:
            colors.append('#4CAF50')  # teal
        elif g in top50_1:
            colors.append(SCAGENET_PALETTE['universal'])  # orange
        elif g in top50_2:
            colors.append(SCAGENET_PALETTE['cell_type'])  # purple
        else:
            colors.append('#CCCCCC')

    rho, _ = stats.spearmanr(x, y)
    jaccard = len(shared) / len(top50_1 | top50_2) if (top50_1 | top50_2) else 0

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, c=colors, s=8, alpha=0.7, linewidths=0)

    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5)

    # Label most divergent genes
    diff = np.abs(x - y)
    for idx in np.argsort(diff)[-10:]:
        ax.annotate(genes[idx], (x[idx], y[idx]), fontsize=5,
                   xytext=(3, 3), textcoords='offset points')

    ax.text(0.05, 0.92, f"Spearman ρ = {rho:.3f}\nJaccard@50 = {jaccard:.3f}",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', alpha=0.1))

    ax.set_xlabel(f'{cell_type_1} importance')
    ax.set_ylabel(f'{cell_type_2} importance')
    ax.set_title(kwargs.get('title', f'{cell_type_1} vs {cell_type_2}'))

    handles = [
        mpatches.Patch(color='#4CAF50', label='Shared top-50'),
        mpatches.Patch(color=SCAGENET_PALETTE['universal'], label=f'{cell_type_1} only'),
        mpatches.Patch(color=SCAGENET_PALETTE['cell_type'], label=f'{cell_type_2} only'),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)

    plt.tight_layout()
    save_figure(fig, save_path, dpi)
    return fig
