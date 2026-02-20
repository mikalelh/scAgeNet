"""
scAgeNet: Aging Program Profiler for Single-Cell RNA-Seq
=========================================================
Trained on Tabula Muris Senis mouse lung (29 cell types, 3/18/30 months).

    import scagenet
    results = scagenet.profile("my_data.h5ad", output_dir="results/")

⚠ scAgeNet is an aging program profiler, not a biological age clock.
  It was trained on 3 discrete timepoints and does not predict continuous age.
"""

__version__ = "1.0.0"
__author__ = "Mikal El Hajjar"
__license__ = "MIT"

# ── Core pipeline ──────────────────────────────────────────────────────────
from .profile import profile

# ── Style ──────────────────────────────────────────────────────────────────
from .utils import set_style

# ── Individual plot functions ──────────────────────────────────────────────
from .plotting import (
    # Category 1: Core aging analysis
    plot_aging_heatmap,
    plot_cell_ranking,
    plot_architecture,
    plot_reference_comparison,
    plot_jaccard_curve,
    plot_go_enrichment,
    plot_summary_dashboard,

    # Category 2: UMAP visualizations
    plot_umap_celltype,
    plot_umap_aging_score,
    plot_umap_condition,
    plot_umap_gene,
    plot_umap_architecture,
    plot_umap_tier_activation,
    plot_umap_panel,

    # Category 3: Group comparison
    plot_condition_violin,
    plot_condition_auroc,
    plot_differential_aging_genes,
    plot_composition,
    plot_gene_expression,

    # Category 4: Cell-type deep dives
    plot_celltype_report,
    plot_celltype_comparison,
)

__all__ = [
    "profile",
    "set_style",
    # plots
    "plot_aging_heatmap",
    "plot_cell_ranking",
    "plot_architecture",
    "plot_reference_comparison",
    "plot_jaccard_curve",
    "plot_go_enrichment",
    "plot_summary_dashboard",
    "plot_umap_celltype",
    "plot_umap_aging_score",
    "plot_umap_condition",
    "plot_umap_gene",
    "plot_umap_architecture",
    "plot_umap_tier_activation",
    "plot_umap_panel",
    "plot_condition_violin",
    "plot_condition_auroc",
    "plot_differential_aging_genes",
    "plot_composition",
    "plot_gene_expression",
    "plot_celltype_report",
    "plot_celltype_comparison",
]
