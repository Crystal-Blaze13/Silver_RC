"""
Download VIX (Fear Index) Data
================================
Downloads daily VIX data from Yahoo Finance via yfinance.
VIX = CBOE Volatility Index — measures market fear/uncertainty.
Used as sentiment proxy for silver price modelling.

HOW TO RUN:
-----------
1. pip3 install yfinance pandas
2. python3 fetch_vix.py
3. Uploads vix.csv when done
"""

import yfinance as yf
import pandas as pd

START_DATE = "2016-01-01"
END_DATE   = "2026-02-20"

print("Downloading VIX data...")

df = yf.download("^VIX", start=START_DATE, end=END_DATE, auto_adjust=True)

# Flatten multi-index if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Keep only Close
df = df[["Close"]].rename(columns={"Close": "vix"})
df.index.name = "date"

print(f"  Downloaded {len(df)} rows")
print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
print(f"  VIX range : {df['vix'].min():.1f} → {df['vix'].max():.1f}")
print(f"\n  VIX > 30 (fear periods): {(df['vix'] > 30).sum()} days")
print(f"  VIX < 15 (calm periods) : {(df['vix'] < 15).sum()} days")

df.to_csv("vix.csv")
print("\n✅ Saved: vix.csv")
print("Upload this file to Claude to continue.")