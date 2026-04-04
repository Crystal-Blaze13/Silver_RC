# Replication Notes — Liu et al. (2025) → MCX Silver (Weekly)

**Paper:** Liu, S., Li, M., Yang, K., Wei, Y., & Wang, S. (2025). "From forecasting to trading: A multimodal-data-driven approach to reversing carbon market losses." *Energy Economics*, 144, 108350.  
**DOI:** https://doi.org/10.1016/j.eneco.2025.108350

---

## Overview of Adaptations

Every adaptation is listed below with the reason, the effect on results, and how the code implements it.

---

## 1. Asset and Market Change

| Dimension | Paper | This Replication |
|-----------|-------|-----------------|
| Asset | China Emission Allowances (CEA) and Hubei (HBEA) | MCX Silver (MCXSILV, INR/kg) |
| Market | Chinese carbon trading exchanges | Multi Commodity Exchange India |
| Drivers | Carbon regulation, energy costs, EUA prices | Gold/silver ratio, Brent, USD/INR, Nifty 50, India VIX |
| Price level | 40–86 CNY/tonne | 40,000–130,000 INR/kg |
| Price behavior | Mean-reverting, policy-dominated | Trending, import-priced, dollar-linked |

**Why:** The goal was to test whether the VMD + entropy + LASSO + hybrid-ensemble framework generalises from policy-driven carbon prices to commodity markets driven by macroeconomic and currency factors.

**Effect:** The framework generalises structurally. The main difference is that silver exhibits a much stronger persistent trend (Chow test F=964, p<0.0001), which favours ARIMA-only decomposition over the LSTM hybrid.

---

## 2. Frequency Change: Daily → Weekly

| Dimension | Paper | This Replication |
|-----------|-------|-----------------|
| Frequency | Daily (trading days) | Weekly (Friday close) |
| Training observations | ~516 (CEA), ~1264 (HBEA) | 843 |
| Test observations | ~129 (CEA), ~316 (HBEA) | 103 |
| Training seqs per LSTM IMF | ~103–253 (≈train/5 mods) | **~160** (843/5, ~168 after alignment) |

**Why:** No reliable daily MCX Silver data was available for the full 2008–2026 window. Weekly data was available and is the standard horizon for Indian commodity market participants.

**Effect (critical):** The paper's LSTM operates on ~316 daily training sequences per IMF (HBEA dataset). Our LSTM has ~160 weekly sequences. This halving of effective training data is the root cause of LSTM underperformance on this dataset (MAPE≈100% for LSTM-only models). VMD-ARIMA, which does not require sequence-to-sequence learning, benefits from the weekly data's smoother, more linear structure and **outperforms the Proposed method** in this replication.

---

## 3. Interval Forecasting: iMLP → Split Conformal Prediction

| Dimension | Paper | This Replication |
|-----------|-------|-----------------|
| Method | Interval MLP (iMLP, Liu et al. 2024) | Split conformal prediction |
| Input | Intraday [low, high] carbon price range | Weekly point prices only |
| Output | Interval [L̂, Ĥ] for intraday range | 80% prediction interval [L̂, Ĥ] |
| Coverage | ~88% (HBEA) | ~70% (actual silver) |

**Why:** The iMLP requires daily intraday high/low prices as interval-valued inputs (Liu et al. 2024). These are not available at the weekly silver frequency. Split conformal prediction (Papadopoulos et al. 2002) is the distribution-free equivalent when only point data exists.

**Implementation:**
1. Train a small MLP (iMLP) on the training set's point residuals.
2. Use the last 20% of training as a calibration split.
3. Compute conformity scores = |residual| on the calibration set.
4. Prediction intervals: `point_pred ± quantile(conformity_scores, 1-α)`.

**Effect:** The conformal interval has valid marginal coverage guarantees (unlike a non-conformal band). Coverage on the actual test period was ~70%. The interval constraint in Scheme 1' and 2' remains active and reduces trading frequency slightly, though on the actual test window the interval was rarely binding due to low prediction error relative to interval width.

---

## 4. External Features: Carbon-Specific → Indian Commodity

| Paper Feature | Our Substitute | Rationale |
|--------------|---------------|----------|
| Brent, Coal, Electricity, Gas | Brent (kept), drop coal/electricity | Key energy inputs for Indian industrial demand |
| CSI300, CSI500, SSEC, SP500 | Nifty 50, SP500 | Indian + US equity risk appetite |
| USD/CNY, EUR/CNY | USD/INR (critical!) | Import pricing channel for Indian silver |
| EU Allowances (EUA) | Dropped | No direct Indian carbon market link |
| Baidu search index | Google Trends India (silver keywords) | Closest Indian equivalent |
| China Daily news sentiment | Dropped | No matching Indian silver news source |
| AQI (air quality) | India VIX | Domestic uncertainty, relevant to commodity trading |
| — | MCX Gold (INR/10g) | Gold-silver ratio is the dominant Indian silver driver |
| — | India EPU (Baker/Bloom/Davis) | Policy uncertainty specific to Indian commodity regulation |
| — | Geopolitical Risk (GPRXGPRD) | Global supply chain risk for silver imports |

**Effect:** LASSO selected 10–13 of 14 features for every IMF. `mcx_silver_lag_1` was selected for all 6 IMFs; `mcx_gold_lag1` was selected for 5/6. External covariates provided stronger marginal signal than in the paper's carbon models, likely because silver's import-priced nature creates tighter links to gold and USD/INR.

---

## 5. Five Bugs Found and Fixed

### Bug 1: Missing Per-IMF y-Normalisation in LSTM

**Location:** Original step4_models.py (before fix)  
**Symptom:** LSTM predictions for high-frequency IMFs were orders of magnitude larger than the IMF amplitude, causing reconstructed forecasts to exceed silver price bounds by 10–100×.  
**Root cause:** The IMF target series `y_train` was passed to LSTM without scaling. While the feature matrix `X` was standardised via `StandardScaler`, the target was left in raw IMF units (±500 to ±5000 INR/kg for high-frequency modes). The LSTM's sigmoid-gated architecture cannot handle unbounded targets.  
**Fix:** Added per-IMF `StandardScaler` for the target `y_train`, applied inverse-transform on predictions before ensemble summation.  
**Effect:** LSTM forecasts went from divergent (MAPE≈200%) to finite (MAPE≈100% — still poor, but due to data quantity, not numerical instability).

---

### Bug 2: ARIMA Walk-Forward Fed Raw Silver Prices Instead of IMF Values

**Location:** Original step4_models.py, ARIMA walk-forward loop  
**Symptom:** ARIMA reconstruction errors (sum of IMF forecasts) were larger than the single-model ARIMA error, which is logically impossible if decomposition helps.  
**Root cause:** The walk-forward history buffer for ARIMA was initialised with `silver_price[n_train - WF_BURN_IN : n_train]` (raw prices) rather than `imf[n_train - WF_BURN_IN : n_train]` (the per-IMF signal being modelled). ARIMA was therefore fitting a model on a white-noise residual against a training history from a completely different scale (50,000 vs ±200 INR/kg).  
**Fix:** Changed the walk-forward buffer initialisation to `imf_segment = u_sorted[i, n_train - WF_BURN_IN : n_train]` per-IMF.  
**Effect:** VMD-ARIMA MAPE dropped from ~480% to 1.63% — making it the best-performing model in the ensemble.

---

### Bug 3: VMD K Clamped Too Low

**Location:** step1_vmd_decompose.py, K estimation block  
**Symptom:** VMD decomposed the silver series into K=6 modes, but original code had `K = max(6, ...)` regardless of EMD output. With N=946, EMD would sometimes return K=4, which was then overridden to 6.  
**Root cause:** A hard-coded lower bound of `K = max(6, ...)` prevented the EMD-guided estimate from being used when it was below 6. The silver series has fewer genuine intrinsic modes at weekly frequency than daily carbon prices.  
**Fix:** Changed to `K = int(np.clip(K_emd, 4, 12))` — lower bound of 4, upper bound of 12, with EMD taking precedence.  
**Effect:** K=6 was still selected in our final run (EMD returned 6), but the fix prevents silent override in future runs where EMD might return 4 or 5.

---

### Bug 4: VMD Residual Discarded

**Location:** step1_vmd_decompose.py, reconstruction check  
**Symptom:** Reconstruction error of ~61% of mean price was reported but silently accepted. When step4 summed IMF forecasts, the sum did not reconstruct the original price — biasing all trading metrics.  
**Root cause:** The original code computed the residual but only printed it. It did not append the residual as an extra IMF row. For weekly silver, the VMD residual can be substantial because the price level (50,000–130,000 INR/kg) is large and the VMD optimisation is approximate.  
**Fix:** Added logic: if `residual_pct > 0.5%`, append `residual` as an extra row to `u_sorted`, increment K, and propagate to all downstream steps. The threshold of 0.5% is documented in `config.py` as `VMD_RESIDUAL_THRESHOLD_PCT`.  
**Effect:** Final reconstruction error dropped to <0.001% after the append. The residual IMF is classified by entropy and forecast by the appropriate model in step4.

---

### Bug 5: CEEMDAN Run on Training Data Only

**Location:** step4_models.py, CEEMDAN benchmark block  
**Symptom:** CEEMDAN-ARIMA produced walk-forward predictions with length mismatch (shorter than test set), causing downstream NaN propagation in DM tests.  
**Root cause:** CEEMDAN was called only on `silver_price[:n_train]` (training data). The resulting IMFs had length `n_train`, so when the walk-forward loop indexed `imf[n_train + t]` for test steps, it was out of bounds.  
**Fix:** Run CEEMDAN on the full series `silver_price` (all N observations), then split training/test portions of each IMF using the same `n_train` boundary. Walk-forward fitting uses the training portion; walk-forward prediction uses the test portion.  
**Effect:** CEEMDAN benchmarks now produce complete 103-step test predictions. CEEMDAN-ARIMA achieved RMSE=43,136 and MAPE=24.93% (substantially worse than VMD-ARIMA's 1.63%, confirming VMD's superiority for this dataset).

---

## 6. Holding-Period Extension (Beyond Paper)

The paper evaluates trading only on the 126-day (roughly 6-month) test period with weekly re-balancing. We added:

- **Margin sweep:** Minimum predicted-return thresholds of 3%, 5%, 7%, 10% before trading. Result: at weekly frequency, all silver predictions exceeded all thresholds, so the sweep was non-binding.
- **Holding-period sweep:** 1-week, 2-week, 4-week hold. Result: 4-week holding achieved 250.3% cumulative return and 2.02 Sharpe vs. 180.8% / 1.69 Sharpe for weekly re-balancing, because transaction costs are dramatically lower.
- **Equity curve:** Full week-by-week equity curves saved to `predictions.pkl` under key `equity_curves` and plotted as Fig 13b.

---

## 7. Structural Break in Test Period

The test period (Mar 2024 – Mar 2026) coincides with a **sustained silver bull market**. The Chow test detects a significant structural break at the train/test boundary:

- Train mean: 50,786 INR/kg | Test mean: 128,563 INR/kg
- Train slope: +39 INR/wk | Test slope: +1,545 INR/wk (39× faster)
- Chow F = 964.05, p < 0.0001

This means all models are evaluated in an out-of-distribution regime. The Naive(RW) achieves the lowest RMSE (11,220) and 0% DA — confirming the test series is a near-random walk with drift. Any model that correctly identifies "always go long" will profit; models that try to trade short will not.

---

## 8. Boundary Conditions Summary

| Condition | Paper assumption | Our situation | Implication |
|-----------|-----------------|---------------|------------|
| LSTM training sequences | ≥300 recommended | ~160 | LSTM underfits; ARIMA preferred |
| Price stationarity | Carbon: mean-reverting | Silver: strong trend | Decomposition helps; ARIMA dominates low-frequency modes |
| Interval data | Intraday H/L available | Weekly close only | iMLP → split conformal prediction |
| News sentiment | China Daily in Chinese | No matching source | Feature dropped; Google Trends substituted |
| Market type | Regulated emissions permit | Physical commodity | Different regime; same VMD framework applies |
| Training length | 7 years daily (HBEA) | 16 years weekly | Similar total weeks; lower per-frequency density |

---

## 9. Code Quality Notes

- All file paths and hyperparameters are in [config.py](config.py).
- All figures are saved at 300 DPI (upgraded from the original 150 DPI).
- LASSO uses `TimeSeriesSplit` (no future leakage), not random k-fold.
- LSTM uses a held-out validation slice (last 10% of training) with early stopping.
- ApEn is computed with self-matches included (Pincus 1991 original definition); SampEn (Richman & Moorman 2000) is added as a cross-check and shown on the secondary axis in Fig 8.
- DM test uses Harvey-Leybourne-Newbold (1997) finite-sample correction and Newey-West variance with data-driven bandwidth `h = int(n^(1/3))`.
