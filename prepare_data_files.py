"""
prepare_data_files.py
=====================
Run this script ONCE from the scageNet/ directory to generate the bundled
data files from the scAgeNet_pipeline outputs.

Usage:
    python prepare_data_files.py

This copies/converts:
  - model weights (.pth → .pt)
  - gene list (CSV → JSON)
  - importance matrix (CSV → kept as CSV)
  - gene classification → gene_tiers.json
  - cell type mapping (auto-generated from training cell types)
  - universal_driver_categories.json (from utils defaults)
"""

import os
import json
import shutil
import pandas as pd
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SCRIPT_DIR, '..', 'scAgeNet_pipeline')
DATA_OUT = os.path.join(SCRIPT_DIR, 'scagenet', 'data')

os.makedirs(DATA_OUT, exist_ok=True)

print("=" * 60)
print("scAgeNet: Preparing data files")
print("=" * 60)
print(f"Pipeline dir : {os.path.abspath(PIPELINE_DIR)}")
print(f"Data out dir : {DATA_OUT}")


# ── 1. Model weights ────────────────────────────────────────────────────────
src_weights = os.path.join(PIPELINE_DIR, '02_models', 'checkpoints', 'best_model_final.pth')
dst_weights = os.path.join(DATA_OUT, 'model_weights.pt')

if os.path.exists(src_weights):
    shutil.copy2(src_weights, dst_weights)
    print(f"\n[1/6] Copied model weights → {dst_weights}")
else:
    print(f"\n[1/6] WARNING: model weights not found at {src_weights}")


# ── 2. Gene list (CSV → JSON) ───────────────────────────────────────────────
src_genes = os.path.join(PIPELINE_DIR, '01_data', 'processed', 'gene_list.csv')
dst_genes = os.path.join(DATA_OUT, 'gene_list.json')

if os.path.exists(src_genes):
    gene_list = pd.read_csv(src_genes)['gene'].tolist()
    with open(dst_genes, 'w') as f:
        json.dump(gene_list, f, indent=2)
    print(f"[2/6] Gene list: {len(gene_list)} genes → {dst_genes}")
else:
    print(f"[2/6] WARNING: gene list not found at {src_genes}")
    gene_list = []


# ── 3. Reference importance matrix ─────────────────────────────────────────
src_imp = os.path.join(PIPELINE_DIR, '05_results', 'gene_programs',
                        'gene_importance_per_celltype.csv')
dst_imp = os.path.join(DATA_OUT, 'reference_importance.csv')

if os.path.exists(src_imp):
    shutil.copy2(src_imp, dst_imp)
    imp_df = pd.read_csv(src_imp, index_col=0)
    print(f"[3/6] Reference importance: {imp_df.shape} → {dst_imp}")
else:
    print(f"[3/6] WARNING: importance matrix not found at {src_imp}")


# ── 4. Gene tiers (classification CSV → JSON) ───────────────────────────────
src_class = os.path.join(PIPELINE_DIR, '05_results', 'gene_programs',
                          'gene_classification.csv')
dst_tiers = os.path.join(DATA_OUT, 'gene_tiers.json')

if os.path.exists(src_class):
    class_df = pd.read_csv(src_class)
    # Normalize column names (pipeline uses 'category', tool uses 'tier')
    if 'category' in class_df.columns and 'tier' not in class_df.columns:
        class_df = class_df.rename(columns={'category': 'tier'})

    tiers = {
        'universal': class_df[class_df['tier'] == 'Universal Driver']['gene'].tolist(),
        'cell_type': class_df[class_df['tier'] == 'Cell-Type Driver']['gene'].tolist(),
        'non_driver': class_df[class_df['tier'] == 'Non-Driver']['gene'].tolist(),
    }
    with open(dst_tiers, 'w') as f:
        json.dump(tiers, f, indent=2)
    print(f"[4/6] Gene tiers: {len(tiers['universal'])} universal, "
          f"{len(tiers['cell_type'])} cell-type, "
          f"{len(tiers['non_driver'])} non-driver → {dst_tiers}")
else:
    print(f"[4/6] WARNING: gene classification not found at {src_class}")


# ── 5. Cell type mapping ────────────────────────────────────────────────────
# Create a comprehensive synonym map to TMS vocabulary
# Covers common alternate names from CellTypist, HLCA, Azimuth, etc.
dst_mapping = os.path.join(DATA_OUT, 'cell_type_mapping.json')

CELL_TYPE_MAPPING = {
    # Macrophages
    "alveolar macrophage": "alveolar macrophage",
    "AM": "alveolar macrophage",
    "AMs": "alveolar macrophage",
    "lung macrophage": "lung macrophage",
    "interstitial macrophage": "lung macrophage",
    "IM": "lung macrophage",
    "macrophage": "lung macrophage",
    "Macrophages": "lung macrophage",
    # Monocytes
    "classical monocyte": "classical monocyte",
    "CD14+ monocyte": "classical monocyte",
    "CD14+CD16- monocyte": "classical monocyte",
    "non-classical monocyte": "non-classical monocyte",
    "CD16+ monocyte": "non-classical monocyte",
    "CD14-CD16+ monocyte": "non-classical monocyte",
    "intermediate monocyte": "intermediate monocyte",
    "CD14+CD16+ monocyte": "intermediate monocyte",
    # T cells
    "CD4-positive, alpha-beta T cell": "CD4-positive, alpha-beta T cell",
    "CD4+ T cell": "CD4-positive, alpha-beta T cell",
    "CD4 T cell": "CD4-positive, alpha-beta T cell",
    "CD4-positive alpha-beta T cell": "CD4-positive, alpha-beta T cell",
    "T helper cell": "CD4-positive, alpha-beta T cell",
    "CD8-positive, alpha-beta T cell": "CD8-positive, alpha-beta T cell",
    "CD8+ T cell": "CD8-positive, alpha-beta T cell",
    "CD8 T cell": "CD8-positive, alpha-beta T cell",
    "cytotoxic T cell": "CD8-positive, alpha-beta T cell",
    "CTL": "CD8-positive, alpha-beta T cell",
    "regulatory T cell": "regulatory T cell",
    "Treg": "regulatory T cell",
    "T regulatory cell": "regulatory T cell",
    "T cell": "T cell",
    "DN regulatory T cell": "T cell",
    # NK cells
    "NK cell": "NK cell",
    "natural killer cell": "NK cell",
    "natural killer T cell": "mature NK T cell",
    "NKT cell": "mature NK T cell",
    "mature NK T cell": "mature NK T cell",
    # B cells & plasma
    "B cell": "B cell",
    "B lymphocyte": "B cell",
    "plasma cell": "plasma cell",
    "plasmablast": "plasma cell",
    # DCs
    "myeloid dendritic cell": "myeloid dendritic cell",
    "cDC": "myeloid dendritic cell",
    "cDC2": "myeloid dendritic cell",
    "classical dendritic cell": "myeloid dendritic cell",
    "conventional dendritic cell": "myeloid dendritic cell",
    "dendritic cell": "dendritic cell",
    "DC": "dendritic cell",
    "plasmacytoid dendritic cell": "plasmacytoid dendritic cell",
    "pDC": "plasmacytoid dendritic cell",
    # Neutrophil / Basophil
    "neutrophil": "neutrophil",
    "basophil": "basophil",
    "mast cell": "basophil",
    # Epithelial
    "type II pneumocyte": "type II pneumocyte",
    "AT2": "type II pneumocyte",
    "AT2 cell": "type II pneumocyte",
    "alveolar type 2": "type II pneumocyte",
    "ATII": "type II pneumocyte",
    "type 2 pneumocyte": "type II pneumocyte",
    "club cell of bronchiole": "club cell of bronchiole",
    "club cell": "club cell of bronchiole",
    "Clara cell": "club cell of bronchiole",
    "ciliated columnar cell of tracheobronchial tree": "ciliated columnar cell of tracheobronchial tree",
    "ciliated cell": "ciliated columnar cell of tracheobronchial tree",
    "multiciliated cell": "ciliated columnar cell of tracheobronchial tree",
    # Fibroblasts / Structural
    "pulmonary interstitial fibroblast": "pulmonary interstitial fibroblast",
    "fibroblast": "fibroblast of lung",
    "fibroblast of lung": "fibroblast of lung",
    "adventitial cell": "adventitial cell",
    "adventitial fibroblast": "adventitial cell",
    "pericyte cell": "pericyte cell",
    "pericyte": "pericyte cell",
    # Smooth muscle
    "bronchial smooth muscle cell": "bronchial smooth muscle cell",
    "smooth muscle cell of the pulmonary artery": "smooth muscle cell of the pulmonary artery",
    "airway smooth muscle cell": "bronchial smooth muscle cell",
    "vascular smooth muscle cell": "smooth muscle cell of the pulmonary artery",
    # Endothelial
    "endothelial cell of lymphatic vessel": "endothelial cell of lymphatic vessel",
    "lymphatic endothelial cell": "endothelial cell of lymphatic vessel",
    "vein endothelial cell": "vein endothelial cell",
    "venous endothelial cell": "vein endothelial cell",
    "capillary endothelial cell": "vein endothelial cell",
    "endothelial cell": "vein endothelial cell",
}

with open(dst_mapping, 'w') as f:
    json.dump(CELL_TYPE_MAPPING, f, indent=2)
print(f"[5/6] Cell type mapping: {len(CELL_TYPE_MAPPING)} entries → {dst_mapping}")


# ── 6. Universal driver categories ─────────────────────────────────────────
dst_cat = os.path.join(DATA_OUT, 'universal_driver_categories.json')

# Populate from gene classification results if available
categories = {
    "alarmin": ["S100a8", "S100a9", "S100a6", "S100a4"],
    "mhc_ii": ["Cd74", "H2-Aa", "H2-Ab1", "H2-Eb1", "H2-DMa", "H2-Oa"],
    "ribosomal": ["Rps27", "Rpl32", "Rps14", "Rpl13a", "Rps6", "Rpl7", "Rps2",
                  "Rpl13", "Rps3", "Rpl10a", "Rpl11", "Rpl12", "Rps11", "Rps18"],
    "ieg": ["Fos", "Jun", "Egr1", "Atf3", "Klf6", "Junb", "Jund", "Nr4a1"],
    "interferon": ["Ifit1", "Ifit2", "Ifit3", "Mx1", "Oas1a", "Isg15", "Rsad2"],
    "lipid": ["Sftpc", "Sftpb", "Sftpa1", "Abca3", "Lpcat1"],
    "structural": ["Mgp", "Col1a1", "Col3a1", "Fn1", "Vim", "Acta2", "Tagln"],
}

# Enrich from actual top universal genes if available
if os.path.exists(src_class):
    class_df = pd.read_csv(src_class)
    if 'category' in class_df.columns:
        class_df = class_df.rename(columns={'category': 'tier'})
    univ = class_df[class_df['tier'] == 'Universal Driver'].sort_values(
        'mean_importance', ascending=False)['gene'].tolist()

    # Update categories with actual data
    for gene in univ[:200]:
        g_lower = gene.lower()
        if g_lower.startswith('s100'):
            if gene not in categories['alarmin']:
                categories['alarmin'].append(gene)
        elif g_lower.startswith('rps') or g_lower.startswith('rpl'):
            if gene not in categories['ribosomal']:
                categories['ribosomal'].append(gene)
        elif g_lower.startswith('ifit') or g_lower.startswith('mx') or g_lower.startswith('isg'):
            if gene not in categories['interferon']:
                categories['interferon'].append(gene)
        elif g_lower.startswith('fos') or g_lower.startswith('jun') or g_lower.startswith('egr'):
            if gene not in categories['ieg']:
                categories['ieg'].append(gene)
        elif g_lower.startswith('sftp') or g_lower.startswith('abca'):
            if gene not in categories['lipid']:
                categories['lipid'].append(gene)
        elif g_lower.startswith('col') or g_lower.startswith('acta') or g_lower.startswith('mgp'):
            if gene not in categories['structural']:
                categories['structural'].append(gene)

with open(dst_cat, 'w') as f:
    json.dump(categories, f, indent=2)
print(f"[6/6] Universal driver categories → {dst_cat}")

print("\n" + "=" * 60)
print("Data files ready! You can now install the package:")
print("  pip install -e .")
print("  # or:")
print("  pip install git+https://github.com/mikalelh/scAgeNet.git")
print("=" * 60)
