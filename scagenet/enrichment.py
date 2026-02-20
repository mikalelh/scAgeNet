"""
scAgeNet GO Enrichment
=======================
Run GO enrichment on active aging programs via gprofiler-official.
Falls back gracefully if gprofiler is not installed.
"""

import pandas as pd
from typing import Dict, List, Optional


def run_go_enrichment(
    gene_tiers: pd.DataFrame,
    importance_matrix: pd.DataFrame,
    organism: str = "mmusculus",
    n_universal: int = 50,
    n_celltype: int = 30,
    condition_genes: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run GO enrichment on active aging gene programs.

    Enrichment is run on:
    1. User's active universal drivers (top n_universal by mean importance).
    2. User's active cell-type-specific drivers (top n_celltype per cell type).
    3. Condition-differential genes (if provided).

    Args:
        gene_tiers: DataFrame from classify_genes().
        importance_matrix: DataFrame (genes × cell types) from compute_importance().
        organism: gprofiler organism code ('mmusculus' for mouse, 'hsapiens' for human).
        n_universal: Top N universal genes to enrich.
        n_celltype: Top N cell-type genes per cell type to enrich.
        condition_genes: Optional list of condition-differential genes.

    Returns:
        Dict with keys 'universal', 'cell_type', optionally 'condition'.
        Values are DataFrames with GO term, p-value, gene list, source.
    """
    try:
        from gprofiler import GProfiler
        gp = GProfiler(return_dataframe=True)
    except ImportError:
        print(
            "[GO] gprofiler-official not installed. Skipping enrichment.\n"
            "    Install with: pip install gprofiler-official"
        )
        return {'universal': pd.DataFrame(), 'cell_type': pd.DataFrame()}

    results = {}

    # --- Universal drivers ---
    univ_genes = (
        gene_tiers[gene_tiers['tier'] == 'Universal Driver']
        .sort_values('mean_importance', ascending=False)['gene']
        .head(n_universal)
        .tolist()
    )

    if univ_genes:
        print(f"[GO] Running enrichment on {len(univ_genes)} universal driver genes ...")
        results['universal'] = _run_gprofiler(gp, univ_genes, organism)
    else:
        results['universal'] = pd.DataFrame()

    # --- Cell-type-specific drivers (top genes across all types) ---
    ct_genes = (
        gene_tiers[gene_tiers['tier'] == 'Cell-Type Driver']
        .sort_values('max_importance', ascending=False)['gene']
        .head(n_celltype)
        .tolist()
    )

    if ct_genes:
        print(f"[GO] Running enrichment on {len(ct_genes)} cell-type driver genes ...")
        results['cell_type'] = _run_gprofiler(gp, ct_genes, organism)
    else:
        results['cell_type'] = pd.DataFrame()

    # --- Condition-differential genes ---
    if condition_genes:
        print(f"[GO] Running enrichment on {len(condition_genes)} differential genes ...")
        results['condition'] = _run_gprofiler(gp, condition_genes, organism)

    return results


def _run_gprofiler(gp, genes: List[str], organism: str) -> pd.DataFrame:
    """Run a single gprofiler query and return cleaned results."""
    try:
        df = gp.profile(
            organism=organism,
            query=genes,
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG'],
        )
        if df is None or len(df) == 0:
            return pd.DataFrame()

        df = df[df['p_value'] < 0.05].copy()
        df = df.sort_values('p_value').head(50)
        df['gene_ratio'] = df['intersection_size'] / df['query_size']
        return df[['name', 'p_value', 'source', 'intersection_size',
                   'query_size', 'gene_ratio', 'intersections']].rename(
            columns={'intersections': 'genes'}
        )
    except Exception as e:
        print(f"  [GO] Warning: {e}")
        return pd.DataFrame()
