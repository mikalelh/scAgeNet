"""
scAgeNet Reference Comparison
================================
Compare user importance profiles to the published reference atlas.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional


def compare_to_reference(
    user_importance: pd.DataFrame,
    reference_importance: Optional[pd.DataFrame] = None,
    gene_tiers: Optional[pd.DataFrame] = None,
    thresholds: Optional[List[int]] = None,
) -> Dict:
    """Compare user gene importance profiles to the reference atlas.

    Computes:
    - Overall Spearman ρ (mean importance across cell types)
    - Per-tier Spearman ρ (universal drivers only, cell-type drivers only)
    - Jaccard similarity at multiple gene-count thresholds
    - Shared top gene list
    - Per-cell-type Spearman ρ (for cell types present in both)

    Args:
        user_importance: DataFrame (genes × cell types) from compute_importance().
        reference_importance: Reference importance DataFrame. If None, loads bundled file.
        gene_tiers: DataFrame from classify_genes(). Used for tier-specific ρ.
        thresholds: Gene counts for Jaccard. Default [10, 25, 50, 75, 100, 150, 200, 300, 500].

    Returns:
        Dict with keys:
            overall_rho, universal_rho, celltype_rho,
            jaccard (dict threshold→value), shared_genes, per_celltype_rho
    """
    if thresholds is None:
        thresholds = [10, 25, 50, 75, 100, 150, 200, 300, 500]

    if reference_importance is None:
        from .utils import get_reference_importance
        try:
            reference_importance = get_reference_importance()
        except FileNotFoundError:
            print("[Reference] reference_importance.csv not found. Skipping comparison.")
            return _empty_comparison()

    # Align genes
    shared_genes = list(set(user_importance.index) & set(reference_importance.index))
    if len(shared_genes) < 50:
        print(f"[Reference] Only {len(shared_genes)} shared genes. Comparison may be unreliable.")

    user_sub = user_importance.loc[shared_genes]
    ref_sub = reference_importance.loc[shared_genes]

    # Mean importance across cell types
    user_mean = user_sub.mean(axis=1)
    ref_mean = ref_sub.mean(axis=1)

    # Overall Spearman ρ
    overall_rho, overall_p = stats.spearmanr(user_mean.values, ref_mean.values)

    # Per-tier ρ
    universal_rho, celltype_rho = None, None
    if gene_tiers is not None:
        tier_map = dict(zip(gene_tiers['gene'], gene_tiers['tier']))
        univ_genes = [g for g in shared_genes if tier_map.get(g) == 'Universal Driver']
        ct_genes = [g for g in shared_genes if tier_map.get(g) == 'Cell-Type Driver']

        if len(univ_genes) >= 5:
            u_rho, _ = stats.spearmanr(
                user_mean.loc[univ_genes].values,
                ref_mean.loc[univ_genes].values
            )
            universal_rho = float(u_rho)

        if len(ct_genes) >= 5:
            ct_rho, _ = stats.spearmanr(
                user_mean.loc[ct_genes].values,
                ref_mean.loc[ct_genes].values
            )
            celltype_rho = float(ct_rho)

    # Jaccard at multiple thresholds
    user_ranked = user_mean.sort_values(ascending=False)
    ref_ranked = ref_mean.sort_values(ascending=False)

    jaccard_scores = {}
    shared_top_genes = {}
    for k in thresholds:
        k = min(k, len(user_ranked), len(ref_ranked))
        top_user = set(user_ranked.head(k).index)
        top_ref = set(ref_ranked.head(k).index)
        inter = len(top_user & top_ref)
        union = len(top_user | top_ref)
        jaccard_scores[k] = inter / union if union > 0 else 0.0
        shared_top_genes[k] = sorted(top_user & top_ref)

    # Per-cell-type ρ
    common_cts = list(set(user_importance.columns) & set(reference_importance.columns))
    per_ct_rho = {}
    for ct in common_cts:
        u = user_sub[ct].values
        r = ref_sub[ct].values
        rho, _ = stats.spearmanr(u, r)
        per_ct_rho[ct] = float(rho)

    result = {
        'overall_rho': float(overall_rho),
        'overall_p': float(overall_p),
        'universal_rho': universal_rho,
        'celltype_rho': celltype_rho,
        'jaccard': jaccard_scores,
        'shared_top_genes': shared_top_genes,
        'per_celltype_rho': per_ct_rho,
        'n_shared_genes': len(shared_genes),
        'n_common_celltypes': len(common_cts),
    }

    print(f"[Reference] Overall Spearman ρ = {overall_rho:.3f} (p={overall_p:.2e})")
    if universal_rho is not None:
        print(f"[Reference] Universal driver ρ = {universal_rho:.3f}")
    print(f"[Reference] Jaccard at top-50 genes = {jaccard_scores.get(50, 'N/A'):.3f}")

    return result


def _empty_comparison() -> Dict:
    return {
        'overall_rho': None,
        'overall_p': None,
        'universal_rho': None,
        'celltype_rho': None,
        'jaccard': {},
        'shared_top_genes': {},
        'per_celltype_rho': {},
        'n_shared_genes': 0,
        'n_common_celltypes': 0,
    }
