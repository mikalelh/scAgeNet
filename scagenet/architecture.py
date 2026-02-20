"""
scAgeNet Aging Architecture Scoring
=====================================
Determine whether each cell type ages via a centralized (shared) or
decentralized (private) program.

  Centralized   — > 60% of top 50 genes are universal drivers (immune-like)
  Decentralized — < 40% of top 50 genes are universal drivers (structural-like)
  Intermediate  — 40–60%
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from .utils import get_family, UNIVERSAL_DRIVER_CATEGORIES


# Thresholds
_CENTRALIZED_THRESHOLD = 0.60
_DECENTRALIZED_THRESHOLD = 0.40
_TOP_N = 50


def score_architecture(
    importance_matrix: pd.DataFrame,
    gene_tiers: pd.DataFrame,
    top_n: int = _TOP_N,
) -> pd.DataFrame:
    """Score aging architecture for each cell type.

    For each cell type:
      1. Identify the top `top_n` genes by importance.
      2. Compute % that are 'Universal Driver'.
      3. Classify as centralized / decentralized / intermediate.

    Args:
        importance_matrix: DataFrame (genes × cell types) from compute_importance().
        gene_tiers: DataFrame from classify_genes() with 'gene' and 'tier' columns.
        top_n: Number of top genes to consider per cell type.

    Returns:
        DataFrame with columns:
            cell_type, family, n_top_genes, n_universal, pct_universal,
            architecture_class
    """
    tier_map = dict(zip(gene_tiers['gene'], gene_tiers['tier']))

    records = []
    for ct in importance_matrix.columns:
        top_genes = (
            importance_matrix[ct]
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        n_total = len(top_genes)
        n_universal = sum(
            1 for g in top_genes if tier_map.get(g, 'Non-Driver') == 'Universal Driver'
        )
        pct = n_universal / n_total if n_total > 0 else 0.0

        if pct >= _CENTRALIZED_THRESHOLD:
            arch = 'centralized'
        elif pct <= _DECENTRALIZED_THRESHOLD:
            arch = 'decentralized'
        else:
            arch = 'intermediate'

        records.append({
            'cell_type': ct,
            'family': get_family(ct),
            'n_top_genes': n_total,
            'n_universal': n_universal,
            'pct_universal': pct,
            'architecture_class': arch,
        })

    result = pd.DataFrame(records).sort_values('pct_universal', ascending=False)

    summary = result['architecture_class'].value_counts()
    print("[Architecture] Summary:")
    for cls in ['centralized', 'intermediate', 'decentralized']:
        n = summary.get(cls, 0)
        print(f"  {cls:15s}: {n} cell types")

    return result.reset_index(drop=True)


def get_universal_driver_genes(
    gene_tiers: pd.DataFrame,
    n: int = 50,
) -> List[str]:
    """Return top N universal driver genes by mean importance."""
    universal = gene_tiers[gene_tiers['tier'] == 'Universal Driver']
    return universal.sort_values('mean_importance', ascending=False)['gene'].head(n).tolist()


def get_private_genes(
    importance_matrix: pd.DataFrame,
    gene_tiers: pd.DataFrame,
    cell_type: str,
    n: int = 30,
) -> List[str]:
    """Return top N cell-type-specific (non-universal) genes for a cell type."""
    if cell_type not in importance_matrix.columns:
        return []

    tier_map = dict(zip(gene_tiers['gene'], gene_tiers['tier']))
    ranked = importance_matrix[cell_type].sort_values(ascending=False)
    private = [g for g in ranked.index if tier_map.get(g, 'Non-Driver') != 'Universal Driver']
    return private[:n]
