"""
scAgeNet quickstart example.

Run with: python examples/quickstart.py /path/to/my_lung_data.h5ad
"""

import sys
import scagenet

# ── One-line profiling ─────────────────────────────────────────────────────

h5ad_path = sys.argv[1] if len(sys.argv) > 1 else "my_lung_data.h5ad"
condition_col = sys.argv[2] if len(sys.argv) > 2 else None

results = scagenet.profile(
    h5ad_path,
    output_dir="scagenet_results/",
    condition_col=condition_col,       # e.g. "disease" or None
    cell_type_col="cell_type",         # column with cell type labels
    species="mouse",
    n_top_genes=50,
    run_go=True,
    save_plots=True,
    save_tables=True,
    figsize="paper",                   # "paper" for publication, "screen" for larger
)

# ── Access results ─────────────────────────────────────────────────────────

print("\nResults keys:", list(results.keys()))
print("Importance matrix shape:", results['importance'].shape)
print("Top 10 universal drivers:")
univ = results['gene_tiers'][results['gene_tiers']['tier'] == 'Universal Driver']
print(univ.head(10)[['gene', 'mean_importance', 'tier']].to_string(index=False))

# ── Individual plots ───────────────────────────────────────────────────────

# Core plots
scagenet.plot_aging_heatmap(results, top_n=30)
scagenet.plot_cell_ranking(results)
scagenet.plot_architecture(results)

# UMAP plots
scagenet.plot_umap_aging_score(results)
scagenet.plot_umap_celltype(results)
scagenet.plot_umap_panel(results)

# Gene expression
scagenet.plot_umap_gene(results, "S100a9")
scagenet.plot_umap_gene(results, ["S100a9", "Cd74", "Sftpc"])

# Cell-type deep dive
scagenet.plot_celltype_report(results, cell_type="alveolar macrophage")

# Condition comparison (requires condition_col)
if condition_col:
    scagenet.plot_condition_violin(results, condition_col=condition_col)
    scagenet.plot_condition_auroc(results, condition_col=condition_col)
    scagenet.plot_differential_aging_genes(results, condition_col=condition_col)

# Pairwise cell-type comparison
scagenet.plot_celltype_comparison(results, "alveolar macrophage", "type II pneumocyte")

# Summary dashboard
scagenet.plot_summary_dashboard(results, save_path="summary.png")
