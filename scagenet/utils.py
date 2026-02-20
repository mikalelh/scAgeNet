"""
scAgeNet Utilities
==================
Constants, color palettes, cell-type family mappings, and helpers.
"""

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional, List

# ---------------------------------------------------------------------------
# Cell-type family assignments (TMS lung vocabulary)
# ---------------------------------------------------------------------------

CELL_TYPE_FAMILIES = {
    "immune": [
        "B cell",
        "CD4-positive, alpha-beta T cell",
        "CD8-positive, alpha-beta T cell",
        "NK cell",
        "T cell",
        "mature NK T cell",
        "regulatory T cell",
        "classical monocyte",
        "intermediate monocyte",
        "non-classical monocyte",
        "neutrophil",
        "basophil",
        "myeloid dendritic cell",
        "dendritic cell",
        "plasmacytoid dendritic cell",
        "alveolar macrophage",
        "lung macrophage",
        "plasma cell",
    ],
    "structural": [
        "adventitial cell",
        "pericyte cell",
        "pulmonary interstitial fibroblast",
        "fibroblast of lung",
        "bronchial smooth muscle cell",
        "smooth muscle cell of the pulmonary artery",
    ],
    "endothelial": [
        "endothelial cell of lymphatic vessel",
        "vein endothelial cell",
    ],
    "epithelial": [
        "ciliated columnar cell of tracheobronchial tree",
        "club cell of bronchiole",
        "type II pneumocyte",
    ],
}

# Reverse lookup: cell type → family
_CT_FAMILY_MAP = {ct: fam for fam, cts in CELL_TYPE_FAMILIES.items() for ct in cts}

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

SCAGENET_PALETTE = {
    # Family colors
    "immune": "#E8853F",
    "structural": "#4DAF8B",
    "endothelial": "#5BBCD6",
    "epithelial": "#7CAE00",
    # Tier colors
    "universal": "#E8853F",
    "Universal Driver": "#E8853F",
    "cell_type": "#8B6DAF",
    "Cell-Type Driver": "#8B6DAF",
    "non_driver": "#CCCCCC",
    "Non-Driver": "#CCCCCC",
    # Architecture colors
    "centralized": "#E8853F",
    "decentralized": "#4DAF8B",
    "intermediate": "#AAAAAA",
    # Condition defaults
    "condition_0": "#2196F3",
    "condition_1": "#FF5722",
}

# ---------------------------------------------------------------------------
# Universal driver functional categories
# ---------------------------------------------------------------------------

UNIVERSAL_DRIVER_CATEGORIES = {
    "alarmin": ["S100a8", "S100a9", "S100a6", "S100a4"],
    "mhc_ii": ["Cd74", "H2-Aa", "H2-Ab1", "H2-Eb1", "H2-DMa", "H2-Oa"],
    "ribosomal": ["Rps27", "Rpl32", "Rps14", "Rpl13a", "Rps6", "Rpl7"],
    "ieg": ["Fos", "Jun", "Egr1", "Atf3", "Klf6"],
    "interferon": ["Ifit1", "Ifit2", "Ifit3", "Mx1", "Oas1a"],
    "lipid": ["Sftpc", "Sftpb", "Sftpa1", "Abca3"],
    "structural": ["Mgp", "Col1a1", "Col3a1", "Fn1", "Vim"],
}


# ---------------------------------------------------------------------------
# Matplotlib style helpers
# ---------------------------------------------------------------------------

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

SCREEN_STYLE = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

_current_style = "paper"


def set_style(mode: str = "paper"):
    """Set matplotlib rcParams for publication ('paper') or screen viewing ('screen')."""
    global _current_style
    _current_style = mode
    style = PAPER_STYLE if mode == "paper" else SCREEN_STYLE
    matplotlib.rcParams.update(style)


def get_style() -> str:
    return _current_style


def get_family(cell_type: str) -> str:
    """Return cell-type family ('immune', 'structural', 'endothelial', 'epithelial', 'unknown')."""
    return _CT_FAMILY_MAP.get(cell_type, "unknown")


def family_color(cell_type: str) -> str:
    """Return the palette color for a cell type's family."""
    fam = get_family(cell_type)
    return SCAGENET_PALETTE.get(fam, "#999999")


def tier_color(tier: str) -> str:
    """Return palette color for a gene tier string."""
    return SCAGENET_PALETTE.get(tier, "#999999")


def fuzzy_match_gene(query: str, gene_list: List[str], n: int = 5) -> List[str]:
    """Return up to n closest gene name matches (for typo handling)."""
    from difflib import get_close_matches
    return get_close_matches(query, gene_list, n=n, cutoff=0.6)


def load_data_file(filename: str):
    """Load a JSON or CSV file from the bundled data directory."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file '{filename}' not found at {path}.\n"
            "Run prepare_data_files.py from the pipeline root to generate it."
        )
    if filename.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    elif filename.endswith('.csv'):
        import pandas as pd
        return pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unknown extension for {filename}")


def get_gene_list() -> List[str]:
    """Return the 1985 HVG names used for model input."""
    return load_data_file('gene_list.json')


def get_cell_type_mapping() -> dict:
    """Return the synonym map {user_label: TMS_label}."""
    return load_data_file('cell_type_mapping.json')


def get_reference_importance():
    """Return reference importance matrix (genes × cell types) as DataFrame."""
    return load_data_file('reference_importance.csv')


def get_gene_tiers() -> dict:
    """Return {universal: [...], cell_type: [...], non_driver: [...]}."""
    return load_data_file('gene_tiers.json')


def get_universal_categories() -> dict:
    """Return functional categories for universal drivers."""
    path = os.path.join(os.path.dirname(__file__), 'data', 'universal_driver_categories.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return UNIVERSAL_DRIVER_CATEGORIES


def save_figure(fig, save_path: Optional[str], dpi: int = 300):
    """Save figure to disk if save_path is given, else show it."""
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def order_cell_types_by_family(cell_types: List[str]) -> List[str]:
    """Order cell types: immune first, then structural, endothelial, epithelial, unknown."""
    family_order = ["immune", "structural", "endothelial", "epithelial", "unknown"]
    buckets = {fam: [] for fam in family_order}
    for ct in cell_types:
        buckets[get_family(ct)].append(ct)
    ordered = []
    for fam in family_order:
        ordered.extend(sorted(buckets[fam]))
    return ordered
