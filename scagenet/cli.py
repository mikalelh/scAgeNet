"""
scAgeNet CLI
============
Command-line interface for running the full profiling pipeline.

Usage:
    scagenet my_data.h5ad -o results/ -c disease --cell-type-col cell_ontology_class
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='scagenet',
        description='scAgeNet: Aging program profiler for single-cell RNA-seq',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  scagenet my_data.h5ad -o results/

  # With condition comparison
  scagenet my_data.h5ad -o results/ -c disease

  # Custom cell type column
  scagenet my_data.h5ad -o results/ --cell-type-col cell_ontology_class

  # Skip GO enrichment (faster)
  scagenet my_data.h5ad -o results/ --no-go
        """
    )

    parser.add_argument(
        'h5ad',
        help='Path to input .h5ad file',
    )
    parser.add_argument(
        '-o', '--output',
        default='scagenet_results/',
        metavar='DIR',
        help='Output directory (default: scagenet_results/)',
    )
    parser.add_argument(
        '-c', '--condition',
        default=None,
        metavar='COL',
        help='Column in adata.obs for group comparison (e.g. disease, treatment)',
    )
    parser.add_argument(
        '--cell-type-col',
        default='cell_type',
        metavar='COL',
        help='Column in adata.obs with cell type annotations (default: cell_type)',
    )
    parser.add_argument(
        '--species',
        default='mouse',
        choices=['mouse'],
        help='Species (default: mouse)',
    )
    parser.add_argument(
        '--n-top-genes',
        type=int,
        default=50,
        metavar='N',
        help='Number of top genes for heatmap / architecture (default: 50)',
    )
    parser.add_argument(
        '--no-go',
        action='store_true',
        help='Skip GO enrichment analysis',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation',
    )
    parser.add_argument(
        '--no-tables',
        action='store_true',
        help='Skip CSV table output',
    )
    parser.add_argument(
        '--figsize',
        default='paper',
        choices=['paper', 'screen'],
        help='Figure style: paper (publication) or screen (default: paper)',
    )
    parser.add_argument(
        '--n-cells',
        type=int,
        default=500,
        metavar='N',
        help='Max cells per type for importance computation (default: 500)',
    )
    parser.add_argument(
        '--version',
        action='version',
        version='scAgeNet 1.0.0',
    )

    args = parser.parse_args()

    try:
        from scagenet.profile import profile
    except ImportError:
        print("ERROR: scagenet is not installed. Run: pip install -e .", file=sys.stderr)
        sys.exit(1)

    profile(
        h5ad_path=args.h5ad,
        output_dir=args.output,
        condition_col=args.condition,
        cell_type_col=args.cell_type_col,
        species=args.species,
        n_top_genes=args.n_top_genes,
        run_go=not args.no_go,
        save_plots=not args.no_plots,
        save_tables=not args.no_tables,
        figsize=args.figsize,
        n_cells_per_type=args.n_cells,
    )


if __name__ == '__main__':
    main()
