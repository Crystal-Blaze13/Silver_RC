"""
path_config.py — Centralised path definitions for the daily pipeline
=====================================================================
All daily scripts import from here instead of hardcoding paths.

Directory layout (relative to this file):
  daily/code/            ← this file lives here
  daily/processed/       ← intermediate .pkl / .npy / .csv files
  daily/results/figures/ ← output .png figures
  daily/results/tables/  ← output .csv and .json tables
  preprocessed_data/processed/        ← shared raw/preprocessed data (two levels up)
"""

from pathlib import Path

# Root of the project (two levels up: daily/code/ → daily/ → project root)
BASE_DIR = Path(__file__).resolve().parents[2]

# Shared input data
DATA_DIR = BASE_DIR / "preprocessed_data" / "processed"

# Daily-specific directories
_DAILY_DIR    = BASE_DIR / "daily"
PROCESSED_DIR = _DAILY_DIR / "processed"
RESULTS_DIR   = _DAILY_DIR / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
TABLES_DIR    = RESULTS_DIR / "tables"
