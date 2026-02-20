"""
scAgeNet Preprocessing
======================
Load h5ad, harmonize cell type labels, normalize, subset to HVGs, compute UMAP.
"""

import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from typing import Optional, List, Tuple
from .utils import get_gene_list, get_cell_type_mapping, get_family


def load_and_preprocess(
    h5ad_path: str,
    cell_type_col: str = "cell_type",
    species: str = "mouse",
) -> ad.AnnData:
    """Load and preprocess a single-cell h5ad file for scAgeNet.

    Steps:
    1. Load h5ad with scanpy.
    2. Auto-detect cell type column if not found.
    3. Harmonize cell type labels to TMS vocabulary.
    4. Normalize if raw counts detected (max > 50 heuristic).
    5. Subset to the 1,985 HVGs; fill missing with 0.
    6. Compute neighbor graph + UMAP if not already present.
    7. Add 'cell_type_harmonized' and 'cell_type_family' to adata.obs.

    Args:
        h5ad_path: Path to input h5ad file.
        cell_type_col: Column in adata.obs with cell type annotations.
        species: 'mouse' (only validated species).

    Returns:
        Preprocessed AnnData with harmonized labels and UMAP.
    """
    print(f"[Preprocessing] Loading {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # --- Detect cell type column ---
    cell_type_col = _resolve_cell_type_col(adata, cell_type_col)

    # --- Harmonize labels ---
    harmonized, mapping_report = _harmonize_cell_types(adata, cell_type_col)
    adata.obs['cell_type_harmonized'] = harmonized
    adata.obs['cell_type_family'] = [get_family(ct) for ct in harmonized]

    n_unmapped = (harmonized == '__unmapped__').sum()
    if n_unmapped > 0:
        warnings.warn(
            f"{n_unmapped} cells could not be mapped to TMS vocabulary and will be excluded "
            f"from model inference (kept in AnnData as '__unmapped__')."
        )
    print(f"  Harmonized cell types. Mapped: "
          f"{adata.n_obs - n_unmapped}/{adata.n_obs} cells")

    # --- Normalize ---
    adata = _maybe_normalize(adata)

    # --- Subset to model HVGs ---
    adata = _subset_to_hvgs(adata)

    # --- UMAP ---
    adata = _maybe_compute_umap(adata)

    print(f"[Preprocessing] Done. {adata.n_obs} cells ready.")
    return adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_cell_type_col(adata: ad.AnnData, requested: str) -> str:
    """Find best available cell type column."""
    if requested in adata.obs.columns:
        return requested

    candidates = [
        'cell_type', 'cell_ontology_class', 'celltype', 'CellType',
        'cell_type_harmonized', 'Celltype', 'louvain', 'leiden',
    ]
    for col in candidates:
        if col in adata.obs.columns:
            print(f"  WARNING: Column '{requested}' not found. Using '{col}'.")
            return col

    raise ValueError(
        f"Cell type column '{requested}' not found in adata.obs. "
        f"Available columns: {list(adata.obs.columns)}"
    )


def _harmonize_cell_types(
    adata: ad.AnnData,
    cell_type_col: str,
) -> Tuple[pd.Series, dict]:
    """Map user cell type labels to TMS vocabulary using bundled synonym map."""
    try:
        mapping = get_cell_type_mapping()
    except FileNotFoundError:
        mapping = {}

    raw_labels = adata.obs[cell_type_col].astype(str)

    # Build case-insensitive lookup
    lower_map = {k.lower(): v for k, v in mapping.items()}

    def _map(label: str) -> str:
        # Exact match
        if label in mapping:
            return mapping[label]
        # Case-insensitive
        lbl_lower = label.lower()
        if lbl_lower in lower_map:
            return lower_map[lbl_lower]
        # Partial synonym check
        for syn, tms in lower_map.items():
            if syn in lbl_lower or lbl_lower in syn:
                return tms
        # If label already looks like a TMS term, keep it
        from .model import TRAINING_CELL_TYPES
        if label in TRAINING_CELL_TYPES:
            return label
        return '__unmapped__'

    harmonized = raw_labels.map(_map)

    report = {
        'n_total': len(raw_labels),
        'n_mapped': (harmonized != '__unmapped__').sum(),
        'unmapped_labels': sorted(raw_labels[harmonized == '__unmapped__'].unique()),
    }
    if report['unmapped_labels']:
        print(f"  Unmapped labels: {report['unmapped_labels']}")

    return harmonized, report


def _maybe_normalize(adata: ad.AnnData) -> ad.AnnData:
    """Normalize if raw counts detected (max > 50 heuristic)."""
    X = adata.X
    if sparse.issparse(X):
        max_val = X.max()
    else:
        max_val = np.max(X)

    if max_val > 50:
        print("  Detected raw counts (max > 50). Normalizing...")
        adata.raw = adata
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print("  Normalized to 10,000 counts/cell, log1p applied.")
    else:
        print("  Data appears pre-normalized (max ≤ 50). Skipping normalization.")
    return adata


def _subset_to_hvgs(adata: ad.AnnData) -> ad.AnnData:
    """Subset adata to the 1,985 model HVGs, filling missing genes with 0."""
    try:
        gene_list = get_gene_list()
    except FileNotFoundError:
        warnings.warn("gene_list.json not found. Using all genes as-is.")
        return adata

    present = [g for g in gene_list if g in adata.var_names]
    missing = [g for g in gene_list if g not in adata.var_names]

    if missing:
        print(f"  {len(missing)} model genes missing in dataset → filling with 0.")

    # Subset to present genes
    adata_sub = adata[:, present].copy()

    if missing:
        # Create zero columns for missing genes
        n_cells = adata_sub.n_obs
        zero_matrix = sparse.csr_matrix((n_cells, len(missing)), dtype=np.float32)
        missing_adata = ad.AnnData(
            X=zero_matrix,
            obs=adata_sub.obs.copy(),
            var=pd.DataFrame(index=missing),
        )
        # Concatenate and reorder
        import anndata
        combined = anndata.concat([adata_sub, missing_adata], axis=1)
        combined = combined[:, gene_list].copy()
        # Restore obs columns from original
        combined.obsm = adata_sub.obsm.copy() if hasattr(adata_sub, 'obsm') else {}
        if 'X_umap' in adata.obsm:
            combined.obsm['X_umap'] = adata.obsm['X_umap']
        if 'X_pca' in adata.obsm:
            combined.obsm['X_pca'] = adata.obsm['X_pca']
        print(f"  Subsetted to {combined.n_vars} model genes.")
        return combined

    # All genes present, just reorder
    adata_sub = adata_sub[:, gene_list].copy()
    print(f"  Subsetted to {adata_sub.n_vars} model genes.")
    return adata_sub


def _maybe_compute_umap(adata: ad.AnnData) -> ad.AnnData:
    """Compute neighbor graph + UMAP if not already present."""
    if 'X_umap' in adata.obsm:
        print("  UMAP already present. Skipping computation.")
        return adata

    print("  Computing PCA → neighbors → UMAP ...")
    sc.pp.pca(adata, n_comps=min(50, adata.n_obs - 1, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    print("  UMAP computed.")
    return adata


def prepare_expression_matrix(
    adata: ad.AnnData,
    gene_list: Optional[List[str]] = None,
) -> np.ndarray:
    """Extract and scale expression matrix for model input.

    Args:
        adata: Preprocessed AnnData (already subsetted to HVGs).
        gene_list: Gene order. If None uses adata.var_names.

    Returns:
        Float32 array (n_cells, n_genes), zero-mean unit-variance scaled.
    """
    if gene_list is not None:
        available = [g for g in gene_list if g in adata.var_names]
        X = adata[:, available].X
    else:
        X = adata.X

    if sparse.issparse(X):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds
