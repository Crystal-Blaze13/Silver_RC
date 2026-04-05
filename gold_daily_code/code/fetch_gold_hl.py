"""
fetch_gold_hl.py  — Run this LOCALLY (needs internet + yfinance)
===================================================================
Downloads daily OHLCV for MCX Gold proxies, merges them into a
single High / Low series that aligns with your master_daily_prices_gold.csv,
and saves:

    financial_data/processed/gold_hl_daily.csv

Columns: date (index), High, Low   (INR/10g, same units as mcx_gold)

HOW TO RUN:
    pip install yfinance pandas
    python fetch_gold_hl.py

TICKERS TRIED (in priority order):
  1. GC=F   — COMEX Silver front-month futures (USD/oz)
     Converted to INR/10g:  × USDINR_rate × 32.1507
  2. XAUUSD=X — Spot silver (USD/oz), same conversion
  3. GC=F   — fallback Gold proxy (not ideal, disabled by default)

The script reads your master_daily_prices_gold.csv to get the USDINR column
for the currency conversion so that High/Low are in INR/10g.
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Install yfinance:  pip install yfinance")

# ── Paths ─────────────────────────────────────────────────────────────────────
MASTER_CSV  = "../data/master_daily_prices_gold.csv"
OUT_CSV     = "../data/gold_hl_daily.csv"

# ── Conversion constants ───────────────────────────────────────────────────────
# 1 troy oz  = 31.1035 g;  1 kg = 1000 g  →  1 oz/kg factor
OZ_PER_10G   = 10.0 / 31.1035          # ≈ 32.1507  oz per 10g

# ── 1. Load master data to get date index and USDINR ──────────────────────────
print(f"Loading {MASTER_CSV} …")
master = pd.read_csv(MASTER_CSV, index_col=0, parse_dates=True)
date_index = master.index

# Try to get USDINR from master; fall back to downloading it
if "usdinr" in master.columns:
    usdinr_s = master["usdinr"].ffill().bfill()
    print(f"  USDINR from master: range {usdinr_s.min():.2f}–{usdinr_s.max():.2f}")
else:
    print("  USDINR not in master — downloading from yfinance…")
    fx = yf.download("USDINR=X", start=str(date_index[0].date()),
                     end=str((date_index[-1] + pd.Timedelta(days=5)).date()),
                     progress=False)["Close"]
    usdinr_s = fx.reindex(date_index).ffill().bfill()
    print(f"  Downloaded USDINR: {len(usdinr_s)} rows")

# ── 2. Download Gold futures OHLCV ──────────────────────────────────────────
start_str = str((date_index[0] - pd.Timedelta(days=10)).date())
end_str   = str((date_index[-1] + pd.Timedelta(days=5)).date())

print(f"\nDownloading gold OHLCV ({start_str} → {end_str}) …")

raw = None
for ticker in ["GC=F", "XAUUSD=X"]:
    try:
        df = yf.download(ticker, start=start_str, end=end_str,
                         auto_adjust=True, progress=False)
        if len(df) > 100:
            print(f"  ✓ {ticker}: {len(df)} rows")
            raw = df[["High", "Low", "Close"]].copy()
            raw.columns = ["High_raw", "Low_raw", "Close_raw"]
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            used_ticker = ticker
            break
        else:
            print(f"  ✗ {ticker}: only {len(df)} rows — skipping")
    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

if raw is None:
    sys.exit("ERROR: Could not download gold data from yfinance. "
             "Check your internet connection or try a different ticker.")

# ── 3. Align to master date index ─────────────────────────────────────────────
raw = raw.reindex(date_index).ffill().bfill()
usdinr_aligned = usdinr_s.reindex(date_index).ffill().bfill()

# ── 4. Convert USD/oz → INR/10g ────────────────────────────────────────────────
# INR/10g = (USD/oz) × (INR/USD) × (oz/kg)
conversion = usdinr_aligned.values * OZ_PER_10G

high_inr = raw["High_raw"].values * conversion
low_inr  = raw["Low_raw"].values  * conversion

# ── 5. Sanity check against mcx_gold close ──────────────────────────────────
if "mcx_gold" in master.columns:
    mcx_close = master["mcx_gold"].values
    ratio_hi  = np.nanmedian(high_inr / np.where(mcx_close > 0, mcx_close, np.nan))
    ratio_lo  = np.nanmedian(low_inr  / np.where(mcx_close > 0, mcx_close, np.nan))
    print(f"\nSanity check (median High/Close = {ratio_hi:.4f}, Low/Close = {ratio_lo:.4f})")
    if ratio_hi < 0.8 or ratio_hi > 1.5:
        print("  WARNING: High/Close ratio looks off — check ticker/conversion.")
    else:
        print("  ✓ Ratio looks reasonable (should be slightly above 1.0 for High).")

# ── 6. Save ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
hl_df = pd.DataFrame(
    {"High": high_inr, "Low": low_inr},
    index=date_index
)
hl_df.index.name = "date"
hl_df.to_csv(OUT_CSV)

print(f"\nSaved: {OUT_CSV}  ({len(hl_df)} rows)")
print(f"  High range : {hl_df['High'].min():,.0f} – {hl_df['High'].max():,.0f} INR/10g")
print(f"  Low  range : {hl_df['Low'].min():,.0f}  – {hl_df['Low'].max():,.0f} INR/10g")
print(f"  Spread (H-L) median: {(hl_df['High'] - hl_df['Low']).median():,.0f} INR/10g")
print(f"\nNow run:  python step6_trading_daily.py")
