"""
path_config.py — Centralised path definitions for the daily pipeline
=====================================================================
All daily scripts import from here instead of hardcoding paths.

Directory layout (relative to this file):
  silver/code/           ← this file lives here
  silver/data/           ← intermediate .pkl / .npy / .csv files
  silver/results/figures/← output .png figures
  silver/results/tables/ ← output .csv and .json tables
  common_data/           ← shared raw/preprocessed data (root level)
"""

from pathlib import Path

# Root of the project (two levels up: silver/code/ → silver/ → project root)
BASE_DIR = Path(__file__).resolve().parents[2]

# Shared input data
DATA_DIR = BASE_DIR / "common_data"

# Daily-specific directories
_DAILY_DIR    = BASE_DIR / "silver"
PROCESSED_DIR = _DAILY_DIR / "data"
RESULTS_DIR   = _DAILY_DIR / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
TABLES_DIR    = RESULTS_DIR / "tables"
