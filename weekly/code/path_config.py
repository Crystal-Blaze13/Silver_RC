"""
path_config.py — Centralised path definitions for the weekly pipeline
======================================================================
All weekly scripts import from here instead of hardcoding paths.

Directory layout (relative to this file):
  weekly/code/            ← this file lives here
  weekly/processed/       ← intermediate .pkl / .npy / .csv files
  weekly/results/figures/ ← output .png figures
  weekly/results/tables/  ← output .csv and .json tables
  preprocessed_data/processed/         ← shared raw/preprocessed data (two levels up)
"""

from pathlib import Path

# Root of the project (two levels up: weekly/code/ → weekly/ → project root)
BASE_DIR = Path(__file__).resolve().parents[2]

# Shared input data
DATA_DIR = BASE_DIR / "preprocessed_data" / "processed"

# Weekly-specific directories
_WEEKLY_DIR   = BASE_DIR / "weekly"
PROCESSED_DIR = _WEEKLY_DIR / "processed"
RESULTS_DIR   = _WEEKLY_DIR / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
TABLES_DIR    = RESULTS_DIR / "tables"
