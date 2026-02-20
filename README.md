# scAgeNet 🧬

**Aging program profiler for single-cell RNA-seq**

scAgeNet identifies aging gene programs from scRNA-seq data, classifies them as universal
(shared across cell types) or cell-type-specific, and determines whether each cell type
ages through a centralized or decentralized architecture.

> ⚠️ scAgeNet is an aging **program profiler**, not a biological age clock.
> It was trained on 3 discrete timepoints (3m, 18m, 30m) and does not predict continuous biological age.

## Install

```bash
pip install git+https://github.com/mikalelh/scAgeNet.git
```

For GO enrichment support:
```bash
pip install "git+https://github.com/mikalelh/scAgeNet.git#egg=scagenet[go]"
```

Local development install:
```bash
git clone https://github.com/mikalelh/scAgeNet.git
cd scAgeNet
pip install -e .
```

## Quick Start

```python
import scagenet

# Run full profiling pipeline (generates all plots and tables)
results = scagenet.profile(
    "my_lung_data.h5ad",
    output_dir="results/",
    condition_col="disease",     # optional: group comparison
    cell_type_col="cell_type",
)

# Access results programmatically
results['importance']           # DataFrame: genes × cell types
results['gene_tiers']          # universal / cell-type / non-driver classification
results['architecture']        # centralized vs decentralized per cell type
results['reference_comparison'] # similarity to reference atlas
results['predictions']         # aging score per cell

# Individual plots
scagenet.plot_umap_aging_score(results)
scagenet.plot_aging_heatmap(results, top_n=30)
scagenet.plot_celltype_report(results, cell_type="alveolar macrophage")

# Compare conditions
results = scagenet.profile("data.h5ad", condition_col="disease")
scagenet.plot_condition_violin(results, condition_col="disease")
scagenet.plot_differential_aging_genes(results, condition_col="disease")
```

## CLI

```bash
# Basic
scagenet my_data.h5ad -o results/

# With condition comparison
scagenet my_data.h5ad -o results/ -c disease

# Custom cell type column
scagenet my_data.h5ad -o results/ --cell-type-col cell_ontology_class

# Skip GO enrichment
scagenet my_data.h5ad -o results/ --no-go
```

## What It Outputs

### Plots (21 total)

| # | Plot | Category |
|---|------|----------|
| 1 | Aging program activation heatmap | Core |
| 2 | Cell-type aging signal ranking | Core |
| 3 | Aging architecture classification | Core |
| 4 | Gene program comparison to reference | Core |
| 5 | Jaccard similarity curve | Core |
| 6 | GO enrichment bubble plot | Core |
| 7 | Summary dashboard | Core |
| 8 | UMAP by cell type | UMAP |
| 9 | UMAP by aging score | UMAP |
| 10 | UMAP by condition | UMAP |
| 11 | UMAP by gene expression | UMAP |
| 12 | UMAP by architecture class | UMAP |
| 13 | UMAP by tier activation | UMAP |
| 14 | UMAP overview panel (2×2) | UMAP |
| 15 | Condition violin plots | Group comparison |
| 16 | Condition AUROC bars | Group comparison |
| 17 | Differential aging genes (volcano) | Group comparison |
| 18 | Cell-type composition by group | Group comparison |
| 19 | Per-gene expression violins | Group comparison |
| 20 | Cell-type deep dive report | Deep dive |
| 21 | Pairwise cell-type comparison | Deep dive |

### Tables (CSV)

| File | Contents |
|------|----------|
| `importance_matrix.csv` | Gene × cell type importance scores |
| `gene_tiers.csv` | Universal / cell-type / non-driver classification |
| `architecture_classification.csv` | Centralized / decentralized per cell type |
| `cell_rankings.csv` | Cell types ranked by aging signal |
| `reference_comparison.csv` | Similarity metrics to reference atlas |
| `go_enrichment_universal.csv` | GO terms for universal drivers |
| `go_enrichment_celltype.csv` | GO terms for cell-type drivers |
| `predictions.csv` | Per-cell aging scores |

## Model Architecture

scAgeNet uses a **two-level hierarchical neural network**:

```
Level 1 — Shared Encoder (universal features)
  Input (1,985 genes)
  → Linear(1985 → 512) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(512 → 256)  → BatchNorm → ReLU → Dropout(0.3)
  → Linear(256 → 256)  → BatchNorm → ReLU → Dropout(0.3)
  → 256-dim universal aging representation

Level 2 — Cell-Type Heads (29 heads, one per cell type)
  Universal features (256)
  → Linear(256 → 128) → LayerNorm → ReLU → Dropout(0.2)
  → Linear(128 → 64)  → LayerNorm → ReLU → Dropout(0.2)
  → Linear(64 → 1)
  → Predicted aging score
```

**Training data:** Tabula Muris Senis mouse lung — 18,503 cells, 29 cell types, ages 3/18/30 months.
**Input features:** 1,985 highly variable genes (HVGs).
**Importance method:** Integrated gradients (50 steps, zero baseline).

## Key Findings (from the paper)

- **241 universal aging drivers** (12% of genes): shared across all cell types — dominated by S100a8/a9, Cd74, ribosomal genes
- **309 cell-type-specific drivers** (16%): NK cells have the largest private program (135 genes)
- Immune cells age via a **centralized** shared program (>60% universal drivers)
- Structural cells age via **decentralized** private programs (<40% universal drivers)
- 78–80% of aging genes activate in late life (18m → 30m transition)

## Cell Type Compatibility

Input cell type labels are automatically harmonized to the 29 TMS lung vocabulary.
Supported synonyms include CellTypist, HLCA, Azimuth, and common abbreviations.

Supported cell types:

| Family | Cell Types |
|--------|-----------|
| Immune | B cell, CD4+ T cell, CD8+ T cell, NK cell, regulatory T cell, classical/intermediate/non-classical monocyte, neutrophil, basophil, alveolar macrophage, lung macrophage, myeloid DC, plasmacytoid DC, dendritic cell, plasma cell, mature NK T cell |
| Structural | adventitial cell, pericyte, pulmonary interstitial fibroblast, fibroblast of lung, bronchial smooth muscle cell, smooth muscle cell of the pulmonary artery |
| Endothelial | endothelial cell of lymphatic vessel, vein endothelial cell |
| Epithelial | ciliated columnar cell of tracheobronchial tree, club cell of bronchiole, type II pneumocyte |

## Limitations

- Trained on **mouse lung** only — other tissues/species not validated
- Trained on **3 discrete timepoints** (3m, 18m, 30m) — not a continuous age predictor
- Cell type labels must map to the 29 TMS lung cell types
- Requires **gene symbols** (not Ensembl IDs)
- Species: **mouse only** (human data not validated)

## Citation

[Paper citation placeholder]

## License

MIT
