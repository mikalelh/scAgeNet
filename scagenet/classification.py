"""
scAgeNet Gene Classification
==============================
Classify genes into three tiers using the same percentile thresholds as the reference.

  Universal Driver  — high mean importance, low CV (consistent across cell types)
  Cell-Type Driver  — high peak importance, high CV or specificity (concentrated)
  Non-Driver        — low or negligible importance
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


def classify_genes(
    importance_matrix: pd.DataFrame,
    reference_tiers: Optional[Dict] = None,
    mean_percentile: float = 80.0,
    max_percentile: float = 80.0,
    spec_percentile: float = 60.0,
) -> pd.DataFrame:
    """Classify genes in user data into three tiers.

    Uses the same percentile thresholds as the reference training data.
    If reference_tiers is provided, uses reference thresholds directly;
    otherwise derives thresholds from user data distribution.

    Args:
        importance_matrix: DataFrame (genes × cell types) from compute_importance().
        reference_tiers: Optional dict with keys 'universal', 'cell_type', 'non_driver'
                         from gene_tiers.json. Used to anchor classification.
        mean_percentile: Percentile for universal mean threshold.
        max_percentile:  Percentile for cell-type-driver max threshold.
        spec_percentile: Percentile for specificity threshold.

    Returns:
        DataFrame with columns:
            gene, tier, mean_importance, cv_importance, max_importance,
            specificity, top_cell_type
    """
    mat = importance_matrix.values
    gene_list = list(importance_matrix.index)
    cell_types = list(importance_matrix.columns)

    mean_imp = mat.mean(axis=1)
    std_imp = mat.std(axis=1)
    max_imp = mat.max(axis=1)
    cv_imp = np.where(mean_imp > 1e-10, std_imp / mean_imp, 999.0)
    specificity = max_imp / (mean_imp + 1e-10)
    max_ct_idx = mat.argmax(axis=1)
    max_ct_name = [cell_types[i] if i < len(cell_types) else 'Unknown' for i in max_ct_idx]

    # Thresholds
    mean_thresh = np.percentile(mean_imp, mean_percentile)
    cv_median = np.median(cv_imp)
    max_thresh = np.percentile(max_imp, max_percentile)
    spec_thresh = np.percentile(specificity, spec_percentile)

    print(f"[Classification] Thresholds (data-driven):")
    print(f"  Universal : mean_imp ≥ {mean_thresh:.4f} (p{mean_percentile:.0f}), "
          f"cv < {cv_median:.4f} (median)")
    print(f"  Cell-Type : max_imp ≥ {max_thresh:.4f} (p{max_percentile:.0f}), "
          f"cv ≥ median OR specificity ≥ {spec_thresh:.2f} (p{spec_percentile:.0f})")

    tiers = []
    for i in range(len(gene_list)):
        m = mean_imp[i]
        cv = cv_imp[i]
        mx = max_imp[i]
        sp = specificity[i]

        if m >= mean_thresh and cv < cv_median:
            tiers.append('Universal Driver')
        elif mx >= max_thresh and (cv >= cv_median or sp >= spec_thresh):
            tiers.append('Cell-Type Driver')
        else:
            tiers.append('Non-Driver')

    result = pd.DataFrame({
        'gene': gene_list,
        'tier': tiers,
        'mean_importance': mean_imp,
        'cv_importance': cv_imp,
        'max_importance': max_imp,
        'specificity': specificity,
        'top_cell_type': max_ct_name,
    }).sort_values('mean_importance', ascending=False).reset_index(drop=True)

    summary = result['tier'].value_counts()
    print("[Classification] Summary:")
    for tier in ['Universal Driver', 'Cell-Type Driver', 'Non-Driver']:
        n = summary.get(tier, 0)
        pct = 100 * n / len(result)
        print(f"  {tier:22s}: {n:4d} genes ({pct:.1f}%)")

    return result


def get_top_genes_per_tier(
    gene_tiers: pd.DataFrame,
    importance_matrix: pd.DataFrame,
    n: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Return top N genes for each tier, ranked by mean importance.

    Args:
        gene_tiers: DataFrame from classify_genes().
        importance_matrix: From compute_importance().
        n: Number of top genes per tier.

    Returns:
        Dict {'Universal Driver': df, 'Cell-Type Driver': df, 'Non-Driver': df}
    """
    result = {}
    for tier in ['Universal Driver', 'Cell-Type Driver', 'Non-Driver']:
        subset = gene_tiers[gene_tiers['tier'] == tier].head(n)
        result[tier] = subset
    return result


def get_top_genes_per_cell_type(
    gene_tiers: pd.DataFrame,
    importance_matrix: pd.DataFrame,
    n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """Return top N genes for each cell type (by that cell type's importance column).

    Args:
        gene_tiers: DataFrame from classify_genes().
        importance_matrix: From compute_importance().
        n: Number of top genes per cell type.

    Returns:
        Dict {cell_type: DataFrame with gene, importance, tier}
    """
    result = {}
    tier_map = dict(zip(gene_tiers['gene'], gene_tiers['tier']))

    for ct in importance_matrix.columns:
        ct_imp = importance_matrix[ct].sort_values(ascending=False).head(n)
        df = pd.DataFrame({
            'gene': ct_imp.index,
            'importance': ct_imp.values,
            'tier': [tier_map.get(g, 'Non-Driver') for g in ct_imp.index],
        })
        result[ct] = df

    return result
