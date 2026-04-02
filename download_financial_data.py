import yfinance as yf
import pandas as pd
import os

# Configuration — 25-year window, Indian market focus
START_DATE = "2000-01-01"
END_DATE   = "2026-03-14"
OUTPUT_DIR = "financial_data/pre_processed"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tickers to download
# Indian-market adaptation:
#   sp500  → nifty50  (NSE Nifty 50 index, India's benchmark)
#   dxy    → usdinr   (USD/INR exchange rate, key for Indian precious metals)
tickers = {
    "silver":      "SI=F",       # COMEX silver futures (USD/oz)
    "gold":        "GC=F",       # COMEX gold futures (USD/oz)
    "brent_crude": "BZ=F",       # Brent crude oil futures
    "nifty50":     "^NSEI",      # NSE Nifty 50 (India equity benchmark)
    "usdinr":      "USDINR=X",   # USD/INR spot rate
}

for name, ticker in tickers.items():
    print(f"Downloading {name} ({ticker})...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    
    if df.empty:
        print(f"  ⚠️  No data returned for {ticker}")
        continue

    # Keep only Close price + flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index.name = "date"

    filepath = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(filepath)
    print(f"  ✅ Saved {len(df)} rows → {filepath}")

print("\nAll done! CSVs saved to:", os.path.abspath(OUTPUT_DIR))