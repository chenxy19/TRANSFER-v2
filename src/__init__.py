"""
Cell Surface Marker Identification Pipeline

This package contains modules for identifying cell-specific surface markers
for targeted delivery applications.

Modules:
--------
constants
    Configuration constants (immune cell types, non-membrane genes, positive markers from literatures)

io_utils
    Shared I/O utilities for loading and saving data files

data_processing
    Processing and cleaning gene expression data, building expression matrices,
    and generating positive/negative labels from marker databases

controlled_learning
    Penalty matrix optimization algorithm for marker prediction,
    including grid search and marker recommendations

plot_results
    Analysis plots showing ratio distributions (percentage of cell types distinguishable at different thresholds)

Usage:
------
Run the complete pipeline:
    $ python main.py

Or run individual modules:
    $ python -m src.data_processing --data-dir data --output-dir results
    $ python -m src.controlled_learning --data-dir results --output-dir results
    $ python -m src.plot_results --data-dir results --plots-dir plots
"""

__version__ = "1.0.0"

# Expose main functions for programmatic use
from .data_processing import run_data_processing
from .controlled_learning import run_controlled_learning
from .plot_results import run_plotting
