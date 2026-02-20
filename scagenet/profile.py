"""
scAgeNet Profile
================
Main entry point: runs the full aging profiling pipeline on a user h5ad file.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict


def profile(
    h5ad_path: str,
    output_dir: str = "scagenet_results/",
    condition_col: Optional[str] = None,
    cell_type_col: str = "cell_type",
    species: str = "mouse",
    n_top_genes: int = 50,
    run_go: bool = True,
    save_plots: bool = True,
    save_tables: bool = True,
    figsize: str = "paper",
    n_cells_per_type: int = 500,
    n_ig_steps: int = 50,
    verbose: bool = True,
) -> Dict:
    """Run the full scAgeNet profiling pipeline.

    Args:
        h5ad_path: Path to input .h5ad file.
        output_dir: Directory for output plots and tables.
        condition_col: Optional column in adata.obs for group comparison
                       (e.g. 'disease', 'treatment').
        cell_type_col: Column in adata.obs with cell type annotations.
        species: 'mouse' (only validated species for now).
        n_top_genes: Number of top genes for heatmap and architecture scoring.
        run_go: Whether to run GO enrichment (requires gprofiler-official).
        save_plots: Whether to save plots to output_dir.
        save_tables: Whether to save result tables as CSV to output_dir.
        figsize: 'paper' (publication) or 'screen' (larger).
        n_cells_per_type: Max cells per cell type for importance computation.
        n_ig_steps: Integration steps for integrated gradients.
        verbose: Print progress.

    Returns:
        results dict with keys:
            adata, importance, gene_tiers, architecture, reference_comparison,
            go_enrichment, cell_rankings, predictions
    """
    t0 = time.time()

    from .utils import set_style
    from .preprocessing import load_and_preprocess, prepare_expression_matrix
    from .model import load_pretrained, TRAINING_CELL_TYPES
    from .importance import compute_importance
    from .classification import classify_genes
    from .architecture import score_architecture
    from .reference import compare_to_reference
    from .enrichment import run_go_enrichment
    from .utils import get_gene_list

    set_style(figsize)

    if save_plots or save_tables:
        os.makedirs(output_dir, exist_ok=True)

    def _log(msg):
        if verbose:
            print(msg)

    _log("\n" + "=" * 60)
    _log("scAgeNet: Aging Program Profiler")
    _log("=" * 60)

    # ── Step 1: Preprocessing ────────────────────────────────────────
    _log("\n[1/9] Preprocessing ...")
    adata = load_and_preprocess(h5ad_path, cell_type_col, species)

    # ── Step 2: Model inference ──────────────────────────────────────
    _log("\n[2/9] Running model inference ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log(f"  Device: {device}")

    try:
        model = load_pretrained(device=device)
    except FileNotFoundError as e:
        warnings.warn(str(e))
        _log("  WARNING: Could not load pretrained model. "
             "Run prepare_data_files.py to generate model_weights.pt.")
        model = None

    predictions = None
    if model is not None:
        try:
            gene_list = get_gene_list()
        except FileNotFoundError:
            gene_list = list(adata.var_names)

        ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
        X_full = prepare_expression_matrix(adata, gene_list)
        cell_types_list = adata.obs[ct_col].values.tolist()

        model.eval()
        all_scores = np.zeros(adata.n_obs)
        batch_size = 512

        with torch.no_grad():
            for start in range(0, adata.n_obs, batch_size):
                end = min(start + batch_size, adata.n_obs)
                x_batch = torch.FloatTensor(X_full[start:end]).to(device)
                ct_batch = cell_types_list[start:end]
                valid_mask = [ct in model.cell_type_to_idx for ct in ct_batch]
                if any(valid_mask):
                    preds = model(x_batch, ct_batch)
                    all_scores[start:end] = preds.cpu().numpy()

        adata.obs['aging_score'] = all_scores
        predictions = pd.Series(all_scores, index=adata.obs_names, name='aging_score')
        _log(f"  Inference complete. Score range: [{all_scores.min():.3f}, {all_scores.max():.3f}]")

    # ── Step 3: Integrated gradients ────────────────────────────────
    _log("\n[3/9] Computing integrated gradients ...")
    importance = None
    if model is not None:
        importance = compute_importance(
            model, adata,
            n_cells_per_type=n_cells_per_type,
            n_steps=n_ig_steps,
            device=device,
        )
    else:
        _log("  Skipping (no model).")

    # ── Step 4: Gene classification ──────────────────────────────────
    _log("\n[4/9] Classifying genes ...")
    gene_tiers = None
    if importance is not None:
        gene_tiers = classify_genes(importance)

    # ── Step 5: Architecture scoring ────────────────────────────────
    _log("\n[5/9] Scoring aging architecture ...")
    architecture = None
    if importance is not None and gene_tiers is not None:
        architecture = score_architecture(importance, gene_tiers, top_n=n_top_genes)

    # ── Step 6: Reference comparison ────────────────────────────────
    _log("\n[6/9] Comparing to reference atlas ...")
    reference_comparison = {}
    if importance is not None:
        reference_comparison = compare_to_reference(importance, gene_tiers=gene_tiers)

    # ── Step 7: GO enrichment ────────────────────────────────────────
    _log("\n[7/9] GO enrichment ...")
    go_enrichment = {}
    if run_go and importance is not None and gene_tiers is not None:
        go_enrichment = run_go_enrichment(gene_tiers, importance)
    elif not run_go:
        _log("  Skipped (run_go=False).")

    # ── Step 8: Cell rankings ────────────────────────────────────────
    _log("\n[8/9] Computing cell type rankings ...")
    cell_rankings = None
    if predictions is not None:
        ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
        df = pd.DataFrame({'score': predictions.values,
                           'cell_type': adata.obs[ct_col].values})
        stats_df = df.groupby('cell_type')['score'].agg(
            mean_score='mean',
            std_score='std',
            n='count',
        ).reset_index()

        # Bootstrap 95% CI
        def _boot_ci(grp):
            data = grp['score'].values
            if len(data) < 5:
                return pd.Series({'ci_lower': data.mean(), 'ci_upper': data.mean()})
            boots = [np.random.choice(data, len(data)).mean() for _ in range(200)]
            return pd.Series({'ci_lower': np.percentile(boots, 2.5),
                              'ci_upper': np.percentile(boots, 97.5)})

        ci_df = df.groupby('cell_type').apply(_boot_ci).reset_index()
        stats_df = stats_df.merge(ci_df, on='cell_type')
        stats_df = stats_df.sort_values('mean_score', ascending=False).reset_index(drop=True)
        stats_df['rank'] = range(1, len(stats_df) + 1)
        cell_rankings = stats_df

    # ── Step 9: Save outputs ─────────────────────────────────────────
    _log("\n[9/9] Saving outputs ...")

    if save_tables:
        tables_dir = os.path.join(output_dir, 'tables')
        os.makedirs(tables_dir, exist_ok=True)

        if importance is not None:
            importance.to_csv(os.path.join(tables_dir, 'importance_matrix.csv'))
        if gene_tiers is not None:
            gene_tiers.to_csv(os.path.join(tables_dir, 'gene_tiers.csv'), index=False)
        if architecture is not None:
            architecture.to_csv(os.path.join(tables_dir, 'architecture_classification.csv'),
                               index=False)
        if cell_rankings is not None:
            cell_rankings.to_csv(os.path.join(tables_dir, 'cell_rankings.csv'), index=False)
        if reference_comparison:
            ref_df = pd.DataFrame([{
                'metric': k, 'value': v
            } for k, v in reference_comparison.items()
                if not isinstance(v, (dict, list))])
            ref_df.to_csv(os.path.join(tables_dir, 'reference_comparison.csv'), index=False)
        if go_enrichment.get('universal') is not None and len(go_enrichment['universal']):
            go_enrichment['universal'].to_csv(
                os.path.join(tables_dir, 'go_enrichment_universal.csv'), index=False)
        if go_enrichment.get('cell_type') is not None and len(go_enrichment['cell_type']):
            go_enrichment['cell_type'].to_csv(
                os.path.join(tables_dir, 'go_enrichment_celltype.csv'), index=False)
        if predictions is not None:
            ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
            preds_df = pd.DataFrame({
                'barcode': adata.obs_names,
                'cell_type': adata.obs[ct_col].values,
                'aging_score': predictions.values,
            })
            preds_df.to_csv(os.path.join(tables_dir, 'predictions.csv'), index=False)
        _log(f"  Tables saved to {tables_dir}/")

    results = {
        'adata': adata,
        'importance': importance,
        'gene_tiers': gene_tiers,
        'architecture': architecture,
        'reference_comparison': reference_comparison,
        'go_enrichment': go_enrichment,
        'cell_rankings': cell_rankings,
        'predictions': predictions,
    }

    if save_plots:
        _save_default_plots(results, output_dir, condition_col, figsize, n_top_genes)

    elapsed = time.time() - t0
    _log(f"\n{'=' * 60}")
    _log(f"scAgeNet profiling complete in {elapsed:.1f}s")
    _log(f"Results saved to: {output_dir}")
    _log("=" * 60)

    _print_summary(results)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_default_plots(results, output_dir, condition_col, figsize_mode, n_top_genes):
    """Save the default set of plots."""
    from . import plotting as plt_mod
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    def _try_save(func, name, **kwargs):
        try:
            fig = func(results, save_path=os.path.join(plots_dir, name), **kwargs)
            return fig
        except Exception as e:
            warnings.warn(f"Plot '{name}' failed: {e}")
            return None

    # Core plots
    _try_save(plt_mod.plot_aging_heatmap, 'aging_heatmap.png', top_n=n_top_genes)
    _try_save(plt_mod.plot_cell_ranking, 'cell_ranking.png')
    _try_save(plt_mod.plot_architecture, 'architecture.png')
    _try_save(plt_mod.plot_reference_comparison, 'reference_comparison.png')
    _try_save(plt_mod.plot_jaccard_curve, 'jaccard_curve.png')
    _try_save(plt_mod.plot_go_enrichment, 'go_enrichment.png')
    _try_save(plt_mod.plot_summary_dashboard, 'summary_dashboard.png')

    # UMAP plots
    _try_save(plt_mod.plot_umap_celltype, 'umap_celltype.png')
    _try_save(plt_mod.plot_umap_aging_score, 'umap_aging_score.png')
    _try_save(plt_mod.plot_umap_architecture, 'umap_architecture.png')
    _try_save(plt_mod.plot_umap_tier_activation, 'umap_tier_activation.png')
    _try_save(plt_mod.plot_umap_panel, 'umap_panel.png', condition_col=condition_col)

    # Condition plots (if condition_col provided)
    if condition_col:
        _try_save(plt_mod.plot_umap_condition, 'umap_condition.png', condition_col=condition_col)
        _try_save(plt_mod.plot_condition_violin, 'condition_violin.png', condition_col=condition_col)
        _try_save(plt_mod.plot_condition_auroc, 'condition_auroc.png', condition_col=condition_col)
        _try_save(plt_mod.plot_composition, 'composition.png', condition_col=condition_col)
        _try_save(plt_mod.plot_differential_aging_genes, 'differential_genes.png',
                  condition_col=condition_col)

    print(f"  Plots saved to {plots_dir}/")


def _print_summary(results):
    """Print a text summary to the console."""
    adata = results.get('adata')
    gene_tiers = results.get('gene_tiers')
    architecture = results.get('architecture')
    cell_rankings = results.get('cell_rankings')
    ref = results.get('reference_comparison', {})

    print("\n── scAgeNet Summary ──────────────────────────────────────")
    if adata is not None:
        ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs else 'cell_type'
        print(f"  Cells:        {adata.n_obs:,}")
        print(f"  Cell types:   {adata.obs[ct_col].nunique()}")

    if gene_tiers is not None:
        for tier in ['Universal Driver', 'Cell-Type Driver', 'Non-Driver']:
            n = (gene_tiers['tier'] == tier).sum()
            print(f"  {tier:22s}: {n} genes")
        top5 = gene_tiers.sort_values('mean_importance', ascending=False)['gene'].head(5).tolist()
        print(f"  Top genes:    {', '.join(top5)}")

    if architecture is not None:
        n_cent = (architecture['architecture_class'] == 'centralized').sum()
        n_decent = (architecture['architecture_class'] == 'decentralized').sum()
        print(f"  Centralized:  {n_cent} cell types")
        print(f"  Decentralized:{n_decent} cell types")

    if cell_rankings is not None and len(cell_rankings):
        top_ct = cell_rankings.iloc[0]['cell_type']
        bot_ct = cell_rankings.iloc[-1]['cell_type']
        print(f"  Most aged:    {top_ct}")
        print(f"  Least aged:   {bot_ct}")

    if ref.get('overall_rho') is not None:
        print(f"  Reference ρ:  {ref['overall_rho']:.3f}")
        j50 = ref.get('jaccard', {}).get(50)
        if j50 is not None:
            print(f"  Jaccard@50:   {j50:.3f}")
    print("──────────────────────────────────────────────────────────")
