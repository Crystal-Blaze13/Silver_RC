"""
Merge manually downloaded Google Trends CSV with existing trends_india.csv
Usage:
  1. Download the 4 missing keywords from trends.google.com as CSV
  2. Place it in this folder as: trends_missing_batch.csv
  3. python merge_manual_trends.py
"""

import pandas as pd
import numpy as np

EXISTING   = "trends_india.csv"
MANUAL_CSV = "trends_missing_batch.csv"
OUT        = "trends_india.csv"

# ── Load existing (8-keyword composite, weekly) ───────────────
existing = pd.read_csv(EXISTING, parse_dates=["date"], index_col="date")
print(f"Existing trends_india.csv: {len(existing)} rows")

# ── Load manual Google Trends export ─────────────────────────
# Google exports with a 2-row header — skip first row, use second as header
raw = pd.read_csv(MANUAL_CSV, skiprows=1)
print(f"Manual CSV columns: {list(raw.columns)}")

# First column is the date
date_col = raw.columns[0]
raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
raw = raw.dropna(subset=[date_col]).set_index(date_col)
raw = raw.apply(pd.to_numeric, errors="coerce").fillna(0)

# Average across the 4 new keywords → one monthly series
new_monthly = raw.mean(axis=1)
new_monthly.name = "new_kw"
print(f"New keywords monthly: {len(new_monthly)} rows, "
      f"{new_monthly.index[0].date()} – {new_monthly.index[-1].date()}")

# ── Resample new keywords to weekly ──────────────────────────
weekly_idx = existing.index
new_weekly = (
    new_monthly
    .reindex(new_monthly.index.union(weekly_idx))
    .interpolate(method="time")
    .reindex(weekly_idx)
    .fillna(0)
    .clip(lower=0)
)

# ── Combine: weighted average (8 existing + 4 new = 12 total) ─
# Weight proportional to keyword count
combined = (existing["trends_raw"] * 8 + new_weekly * 4) / 12

# Rescale to 0-100
max_val = combined.max()
if max_val > 0:
    combined = combined / max_val * 100

combined.index.name = "date"
out_df = combined.reset_index()
out_df.columns = ["date", "trends_raw"]
out_df.to_csv(OUT, index=False)

print(f"\nUpdated {OUT} with all 12 keywords ({len(out_df)} weekly rows)")
print(f"trends_raw stats:\n{out_df['trends_raw'].describe().round(2)}")
print("\nNext: re-run build_master.py → then full pipeline from step1")
