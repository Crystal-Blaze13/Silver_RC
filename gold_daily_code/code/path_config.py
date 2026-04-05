"""
path_config.py — Centralised path definitions for the daily pipeline
=====================================================================
All daily scripts import from here instead of hardcoding paths.

Directory layout (relative to this file):
  gold_daily_code/code/            ← this file lives here
  gold_daily_code/data/            ← intermediate .pkl / .npy / .csv files
  gold_daily_code/results/figures/ ← output .png figures
  gold_daily_code/results/tables/  ← output .csv and .json tables
  common_data/                     ← shared raw/preprocessed data (root level)
"""

from pathlib import Path

# Root of the project (two levels up: gold_daily_code/code/ → gold_daily_code/ → project root)
BASE_DIR = Path(__file__).resolve().parents[2]

# Shared input data
DATA_DIR = BASE_DIR / "common_data"

# Daily-specific directories
_DAILY_DIR    = BASE_DIR / "gold_daily_code"
PROCESSED_DIR = _DAILY_DIR / "data"
RESULTS_DIR   = _DAILY_DIR / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
TABLES_DIR    = RESULTS_DIR / "tables"
