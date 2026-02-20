"""
scAgeNet Integrated Gradients
==============================
Gene importance via integrated gradients on user data.
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad
    from .model import ScAgeNet


def compute_importance(
    model: "ScAgeNet",
    adata: "ad.AnnData",
    n_cells_per_type: int = 500,
    n_steps: int = 50,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Compute gene importance via integrated gradients.

    For each cell type present in adata.obs['cell_type_harmonized']:
      - Sample up to n_cells_per_type cells
      - Baseline = zero vector
      - Integrate gradients across n_steps from baseline to input
      - Average |IG| across cells → per-cell-type importance vector

    Args:
        model: Loaded ScAgeNet model (eval mode).
        adata: Preprocessed AnnData (subsetted to 1985 HVGs).
        n_cells_per_type: Max cells to sample per cell type.
        n_steps: Number of integration steps.
        device: Torch device.

    Returns:
        DataFrame (n_genes × n_cell_types) of importance scores, normalized to [0, 1].
        Also stores per-cell importance in adata.obsm['gene_importance'].
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    from .preprocessing import prepare_expression_matrix
    from .utils import get_gene_list

    try:
        gene_list = get_gene_list()
    except FileNotFoundError:
        gene_list = list(adata.var_names)

    X_full = prepare_expression_matrix(adata, gene_list)

    ct_col = 'cell_type_harmonized' if 'cell_type_harmonized' in adata.obs.columns else 'cell_type'
    cell_types = adata.obs[ct_col].values.tolist()

    valid_types = [ct for ct in model.cell_types if ct in set(cell_types)]
    if not valid_types:
        raise ValueError(
            "No cell types in the data match the model's trained cell types. "
            "Check harmonization."
        )

    print(f"[Importance] Computing integrated gradients for {len(valid_types)} cell types ...")

    importance_matrix = np.zeros((len(gene_list), len(valid_types)), dtype=np.float32)
    per_cell_importance = np.zeros((adata.n_obs, len(gene_list)), dtype=np.float32)

    for ct_idx, ct in enumerate(tqdm(valid_types, desc="Cell types")):
        ct_mask_indices = [i for i, c in enumerate(cell_types) if c == ct]
        n_sample = min(n_cells_per_type, len(ct_mask_indices))
        if n_sample < 5:
            continue

        sample_idx = np.random.choice(ct_mask_indices, n_sample, replace=False)
        X_ct = torch.FloatTensor(X_full[sample_idx]).to(device)
        baseline = torch.zeros_like(X_ct)

        # Integrated gradients: sum over n_steps interpolations
        ig = torch.zeros_like(X_ct)
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            interpolated = baseline + alpha * (X_ct - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            pred = model.forward_single_type(interpolated, ct)
            pred.sum().backward()

            if interpolated.grad is not None:
                ig += interpolated.grad.detach()

        ig = ig / n_steps  # Average
        ig_final = (X_ct - baseline) * ig  # IG attribution
        abs_ig = ig_final.abs().cpu().numpy()

        mean_ig = abs_ig.mean(axis=0)
        importance_matrix[:, ct_idx] = mean_ig

        # Store per-cell importance for UMAP overlays
        for local_i, global_i in enumerate(sample_idx):
            per_cell_importance[global_i] = abs_ig[local_i]

    # Normalize to [0, 1]
    max_val = importance_matrix.max()
    if max_val > 0:
        importance_matrix = importance_matrix / max_val

    adata.obsm['gene_importance'] = per_cell_importance

    importance_df = pd.DataFrame(
        importance_matrix,
        index=gene_list[:len(gene_list)],
        columns=valid_types,
    )

    print(f"[Importance] Done. Matrix shape: {importance_df.shape}")
    return importance_df
