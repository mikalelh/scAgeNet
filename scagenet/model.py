"""
scAgeNet Model Architecture
===========================
Hierarchical shared-encoder + cell-type-specific heads.

Architecture:
  SharedEncoder  : 1985 → 512 → 256 → 256  (BatchNorm, ReLU, Dropout 0.3)
  CellTypeHead×N : 256  → 128 → 64  → 1    (LayerNorm, ReLU, Dropout 0.2)

Trained on Tabula Muris Senis lung (18,503 cells, 29 cell types, 3/18/30 months).
"""

import os
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Optional, Dict
import importlib.resources


# 29 cell types from the training set
TRAINING_CELL_TYPES = [
    'B cell',
    'CD4-positive, alpha-beta T cell',
    'CD8-positive, alpha-beta T cell',
    'NK cell',
    'T cell',
    'adventitial cell',
    'alveolar macrophage',
    'basophil',
    'bronchial smooth muscle cell',
    'ciliated columnar cell of tracheobronchial tree',
    'classical monocyte',
    'club cell of bronchiole',
    'dendritic cell',
    'endothelial cell of lymphatic vessel',
    'fibroblast of lung',
    'intermediate monocyte',
    'lung macrophage',
    'mature NK T cell',
    'myeloid dendritic cell',
    'neutrophil',
    'non-classical monocyte',
    'pericyte cell',
    'plasma cell',
    'plasmacytoid dendritic cell',
    'pulmonary interstitial fibroblast',
    'regulatory T cell',
    'smooth muscle cell of the pulmonary artery',
    'type II pneumocyte',
    'vein endothelial cell',
]


class SharedEncoder(nn.Module):
    """Universal aging feature extractor shared across all cell types.

    Architecture: input_dim → 512 → 256 → 256
    Each layer: Linear → BatchNorm → ReLU → Dropout
    """

    def __init__(self, input_dim: int = 1985,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 256]

        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append((f'linear_{i}', nn.Linear(prev_dim, h_dim)))
            layers.append((f'bn_{i}', nn.BatchNorm1d(h_dim)))
            layers.append((f'relu_{i}', nn.ReLU()))
            layers.append((f'dropout_{i}', nn.Dropout(dropout)))
            prev_dim = h_dim

        self.encoder = nn.Sequential(OrderedDict(layers))
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CellTypeHead(nn.Module):
    """Cell-type-specific prediction head.

    Architecture: 256 → 128 → 64 → 1
    """

    def __init__(self, input_dim: int = 256,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append((f'linear_{i}', nn.Linear(prev_dim, h_dim)))
            layers.append((f'ln_{i}', nn.LayerNorm(h_dim)))
            layers.append((f'relu_{i}', nn.ReLU()))
            layers.append((f'dropout_{i}', nn.Dropout(dropout)))
            prev_dim = h_dim

        layers.append(('output', nn.Linear(prev_dim, 1)))
        self.head = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class ScAgeNet(nn.Module):
    """scAgeNet hierarchical aging model.

    One SharedEncoder + N cell-type-specific CellTypeHeads.
    """

    def __init__(self, input_dim: int = 1985,
                 encoder_hidden_dims: Optional[List[int]] = None,
                 head_hidden_dims: Optional[List[int]] = None,
                 encoder_dropout: float = 0.3,
                 head_dropout: float = 0.2,
                 cell_types: Optional[List[str]] = None):
        super().__init__()

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [512, 256, 256]
        if head_hidden_dims is None:
            head_hidden_dims = [128, 64]
        if cell_types is None:
            cell_types = TRAINING_CELL_TYPES

        self.cell_types = list(cell_types)
        self.cell_type_to_idx = {ct: i for i, ct in enumerate(self.cell_types)}

        self.encoder = SharedEncoder(input_dim, encoder_hidden_dims, encoder_dropout)

        self.heads = nn.ModuleDict({
            self._key(ct): CellTypeHead(self.encoder.output_dim, head_hidden_dims, head_dropout)
            for ct in self.cell_types
        })

        self.config = {
            'input_dim': input_dim,
            'encoder_hidden_dims': encoder_hidden_dims,
            'head_hidden_dims': head_hidden_dims,
            'encoder_dropout': encoder_dropout,
            'head_dropout': head_dropout,
            'cell_types': self.cell_types,
        }

    @staticmethod
    def _key(cell_type: str) -> str:
        """Convert cell type name to valid module dict key.
        Must match the original HierarchicalAgingModel._sanitize_name() exactly.
        """
        return cell_type.replace(' ', '_').replace('/', '_').replace('-', '_')

    def forward(self, x: torch.Tensor, cell_type_labels: list) -> torch.Tensor:
        """Forward pass for a mixed batch of cell types."""
        features = self.encoder(x)
        predictions = torch.zeros(x.shape[0], device=x.device)
        for ct in set(cell_type_labels):
            if ct not in self.cell_type_to_idx:
                continue
            key = self._key(ct)
            mask = torch.tensor([lbl == ct for lbl in cell_type_labels],
                                device=x.device, dtype=torch.bool)
            if mask.any():
                predictions[mask] = self.heads[key](features[mask])
        return predictions

    def forward_single_type(self, x: torch.Tensor, cell_type: str) -> torch.Tensor:
        """Forward pass for one cell type (faster, no routing)."""
        features = self.encoder(x)
        return self.heads[self._key(cell_type)](features)

    def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-dim shared representation."""
        return self.encoder(x)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


def load_pretrained(weights_path: Optional[str] = None,
                    device: Optional[torch.device] = None) -> ScAgeNet:
    """Load ScAgeNet with trained weights from data/model_weights.pt.

    Args:
        weights_path: Path to .pt checkpoint. If None, loads bundled weights.
        device: torch device. Defaults to CPU.

    Returns:
        ScAgeNet model in eval mode.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if weights_path is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        weights_path = os.path.join(data_dir, 'model_weights.pt')

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}.\n"
            "Run prepare_data_files.py from the pipeline root to generate them."
        )

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = ScAgeNet(**cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model
