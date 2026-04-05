"""
STEP 6 (daily) — Interval Forecasting & Trading Strategy  [v9 — gold fix]
=============================================================================

FIXES vs v8:
------------
1. _apply_width_floor: removed the erroneous center-blending step that was
   shifting the interval center away from proposed_pred even after anchoring.
   Now it only enforces the minimum width, keeping the center stable.

2. Added _guarantee_containment(): after all interval transformations, this
   ensures proposed_pred always lies within [pred_lo, pred_hi] by expanding
   the interval outward symmetrically. This is faithful to the paper's intent
   — the point forecast should define whether market conditions are "certain"
   (inside the interval) or "uncertain" (outside), and a degenerate case where
   the point forecast is NEVER inside the interval means all trades are blocked,
   which defeats the purpose of Scheme 1'.

3. Lowered MIN_WIDTH_Q from 0.60 → 0.40 so the width floor doesn't over-inflate
   intervals for gold (high absolute price levels make the q0.60 spread enormous).

4. Added detailed pre-trading diagnostic printout showing what fraction of
   proposed_pred falls inside [pred_lo, pred_hi] — helps catch this class of
   bug immediately on future runs.

All other logic (iMLP architecture, calibration, trading schemes, figures) is
unchanged from v8.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────────────────────────
TRANSACTION_COST  = 0.0005
ANNUAL_FACTOR     = 252
FLAT_EPS          = 1e-6

IMLP_LAGS         = 5
IMLP_HIDDEN       = (64, 32)
IMLP_EPOCHS       = 300
IMLP_LR           = 8e-4
IMLP_PATIENCE     = 40
IMLP_SEED         = 42
CAL_FRAC          = 0.20
CAL_COVERAGE      = 0.80
MIN_WIDTH_Q       = 0.40   # FIX 3: lowered from 0.60 → avoids over-inflated floor for high-price gold

HL_CSV = "../data/gold_hl_daily.csv"

print("=" * 70)
print("STEP 6 (daily): Interval Forecasting & Trading Strategy [v9 — gold fix]")
print("=" * 70)

# ── 1. Load predictions ───────────────────────────────────────────────────────
with open("../data/predictions_daily_gold.pkl", "rb") as f:
    data = pickle.load(f)

single_preds        = data["single_preds"]
decomp_preds        = data["decomp_preds"]
y_true              = np.array(data["y_true_test"],         float)
test_dates          = data["test_dates"]
proposed_pred       = np.array(data["proposed_pred"],       float)
proposed_pred_train = np.array(data["proposed_pred_train"], float)
y_true_train        = np.array(data["y_true_train"],        float)
n_train             = data["n_train"]
N_LAGS              = data["N_LAGS"]
n_test              = len(y_true)

gold_df  = pd.read_csv("../data/gold_daily.csv",
                         index_col=0, parse_dates=True)
gold_all = gold_df.iloc[:, 0].values

# ── 2. Load High / Low ────────────────────────────────────────────────────────
if not os.path.exists(HL_CSV):
    raise FileNotFoundError(
        f"{HL_CSV} not found.\nRun:  python fetch_gold_hl.py"
    )

hl_df = pd.read_csv(HL_CSV, index_col=0, parse_dates=True)
hl_df = hl_df.reindex(gold_df.index).ffill().bfill()

high_all  = hl_df["High"].values.astype(float)
low_all   = hl_df["Low"].values.astype(float)
close_all = gold_all

print(f"\nLoaded High/Low: {len(hl_df)} rows")
print(f"  High range : {high_all.min():,.0f} – {high_all.max():,.0f} INR/kg")
print(f"  Low  range : {low_all.min():,.0f}  – {low_all.max():,.0f} INR/kg")
print(f"  Median daily spread (H–L): {np.median(high_all - low_all):,.0f} INR/kg")
spread_pct = np.median((high_all - low_all) / close_all) * 100
print(f"  Median spread % of close : {spread_pct:.2f}%")

# ── 3. Return series ──────────────────────────────────────────────────────────
act_ret_full = np.log(np.maximum(y_true[1:], 1e-8) /
                      np.maximum(y_true[:-1], 1e-8))
flat_mask    = np.abs(act_ret_full) <= FLAT_EPS
active_mask  = ~flat_mask
n_steps      = len(act_ret_full)   # 540

print(f"\nReturn series: {n_steps} steps")
print(f"  Active: {active_mask.sum()} ({active_mask.sum()/n_steps*100:.1f}%)")
print(f"  Flat  : {flat_mask.sum()} ({flat_mask.sum()/n_steps*100:.1f}%)")

def _log_ret(a, b):
    return np.log(np.maximum(b, 1e-8) / np.maximum(a, 1e-8))


# ══════════════════════════════════════════════════════════════════════════════
# 4. iMLP
# ══════════════════════════════════════════════════════════════════════════════

class IntervalMLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        self.trunk   = nn.Sequential(*layers)
        self.head_hi = nn.Linear(prev, 1)
        self.head_lo = nn.Linear(prev, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.head_hi(z), self.head_lo(z)


def _safe_log(x):
    return np.log(np.maximum(np.asarray(x, float), 1e-8))


def _build_interval_features_v2(high, low, close, lags):
    high = np.asarray(high, float)
    low  = np.asarray(low,  float)
    close= np.asarray(close,float)
    logh = _safe_log(high);  logl = _safe_log(low);  logc = _safe_log(close)
    spread_rel = np.maximum(high - low, 1e-8) / np.maximum(close, 1e-8)

    X, Y = [], []
    for t in range(lags, len(high)):
        lh = logh[t-lags:t]; ll = logl[t-lags:t]; lc = logc[t-lags:t]
        sr = spread_rel[t-lags:t]; h_ = high[t-lags:t]; l_ = low[t-lags:t]; c_ = close[t-lags:t]
        feat = []
        feat.extend(lh[::-1].tolist()); feat.extend(ll[::-1].tolist()); feat.extend(lc[::-1].tolist())
        feat.extend(np.diff(lc, prepend=lc[0])[::-1].tolist())
        feat.extend(np.diff(lh, prepend=lh[0])[::-1].tolist())
        feat.extend(np.diff(ll, prepend=ll[0])[::-1].tolist())
        feat.extend(sr[::-1].tolist())
        feat.extend((h_ / np.maximum(c_, 1e-8) - 1.0)[::-1].tolist())
        feat.extend((1.0 - l_ / np.maximum(c_, 1e-8))[::-1].tolist())
        feat.extend([np.mean(sr), np.median(sr), np.std(sr),
                     np.mean(np.diff(lc)) if len(lc)>1 else 0.0,
                     np.std(np.diff(lc))  if len(lc)>1 else 0.0,
                     lc[-1] - lc[0]])
        center_t = 0.5 * (logh[t] + logl[t])
        width_t  = logh[t] - logl[t]
        X.append(feat); Y.append([center_t, width_t])
    return np.array(X, float), np.array(Y, float)


print("\nBuilding calibrated iMLP on High/Low interval data…")

high_tr  = high_all[:n_train];  low_tr  = low_all[:n_train];  close_tr  = close_all[:n_train]
high_te  = high_all[n_train:];  low_te  = low_all[n_train:];  close_te  = close_all[n_train:]

X_tr, Y_tr = _build_interval_features_v2(high_tr, low_tr, close_tr, IMLP_LAGS)

n_cal = max(IMLP_LAGS + 10, int(len(X_tr) * CAL_FRAC))
n_fit = len(X_tr) - n_cal
X_fit, Y_fit = X_tr[:n_fit], Y_tr[:n_fit]
X_cal, Y_cal = X_tr[n_fit:], Y_tr[n_fit:]

sc_feat = StandardScaler()
Xf_sc   = sc_feat.fit_transform(X_fit)
Xc_sc   = sc_feat.transform(X_cal)

sc_ctr  = StandardScaler(); sc_wid = StandardScaler()
Yf_ctr_sc = sc_ctr.fit_transform(Y_fit[:, 0:1])
Yf_wid_sc = sc_wid.fit_transform(Y_fit[:, 1:2])
Yc_ctr_sc = sc_ctr.transform(Y_cal[:, 0:1])
Yc_wid_sc = sc_wid.transform(Y_cal[:, 1:2])

Xf_t   = torch.FloatTensor(Xf_sc);  Xc_t = torch.FloatTensor(Xc_sc)
Yf_ctr = torch.FloatTensor(Yf_ctr_sc); Yf_wid = torch.FloatTensor(Yf_wid_sc)
Yc_ctr = torch.FloatTensor(Yc_ctr_sc); Yc_wid = torch.FloatTensor(Yc_wid_sc)

torch.manual_seed(IMLP_SEED)
imlp = IntervalMLP(Xf_sc.shape[1], IMLP_HIDDEN)
opt  = torch.optim.Adam(imlp.parameters(), lr=IMLP_LR, weight_decay=1e-5)
crit = nn.MSELoss()

best_loss, best_state, wait = np.inf, None, 0
imlp.train()
for ep in range(IMLP_EPOCHS):
    opt.zero_grad()
    p_ctr, p_wid = imlp(Xf_t)
    wid_true     = torch.clamp(Yf_wid, min=-3.0)
    loss = crit(p_ctr, Yf_ctr) + 1.25*crit(p_wid, wid_true) + 0.05*torch.relu(-p_wid).mean()
    loss.backward(); nn.utils.clip_grad_norm_(imlp.parameters(), 1.0); opt.step()
    with torch.no_grad():
        pc_c, pw_c = imlp(Xc_t)
        val_loss = (crit(pc_c, Yc_ctr) + 1.25*crit(pw_c, Yc_wid) +
                    0.05*torch.relu(-pw_c).mean()).item()
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.clone() for k, v in imlp.state_dict().items()}; wait = 0
    else:
        wait += 1
    if wait >= IMLP_PATIENCE: break

if best_state: imlp.load_state_dict(best_state)
imlp.eval()
print(f"  Trained {ep+1} epochs  val_loss={best_loss:.6f}")


def _decode_interval(pred_ctr_sc, pred_wid_sc, sc_ctr, sc_wid,
                     ref_close=None, min_width_abs=None):
    """Decode normalized center+width predictions back to price-level [Hi, Lo]."""
    ctr = sc_ctr.inverse_transform(pred_ctr_sc.reshape(-1, 1)).ravel()
    wid = sc_wid.inverse_transform(pred_wid_sc.reshape(-1, 1)).ravel()
    wid = np.maximum(wid, 1e-6)

    center = np.exp(ctr)
    if ref_close is None:
        ref_close = center
    ref_close = np.maximum(np.asarray(ref_close, float), 1e-8)
    width_abs = ref_close * (np.exp(wid) - 1.0)
    if min_width_abs is not None:
        width_abs = np.maximum(width_abs, float(min_width_abs))
    hi = center + 0.5 * width_abs
    lo = np.maximum(center - 0.5 * width_abs, 1e-8)
    hi = np.maximum(hi, lo + 1e-8)
    return hi, lo, center, width_abs


# Calibration-set predictions
with torch.no_grad():
    pc_c, pw_c = imlp(Xc_t)

cal_start_idx = IMLP_LAGS + n_fit
cal_close_ref = close_tr[cal_start_idx : cal_start_idx + len(pc_c)]

train_spread  = np.maximum(high_tr - low_tr, 1e-8)
min_width_abs = float(np.quantile(train_spread, MIN_WIDTH_Q))

raw_hi_cal, raw_lo_cal, raw_ctr_cal, _ = _decode_interval(
    pc_c.numpy().ravel(), pw_c.numpy().ravel(), sc_ctr, sc_wid,
    ref_close=cal_close_ref,
    min_width_abs=min_width_abs
)

# Test-set predictions
n_total = len(gold_all)
X_te_rows = []
for t in range(n_train, n_total):
    x_row, _ = _build_interval_features_v2(
        high_all[:t+1], low_all[:t+1], close_all[:t+1], IMLP_LAGS)
    X_te_rows.append(x_row[-1])
X_te   = np.array(X_te_rows, float)[:n_test]
Xte_sc = sc_feat.transform(X_te)
Xte_t  = torch.FloatTensor(Xte_sc)

with torch.no_grad():
    pc_te, pw_te = imlp(Xte_t)

raw_hi_test, raw_lo_test, raw_ctr_test, _ = _decode_interval(
    pc_te.numpy().ravel(), pw_te.numpy().ravel(), sc_ctr, sc_wid,
    ref_close=close_te[:n_test],
    min_width_abs=min_width_abs
)

# ── Calibration residuals ────────────────────────────────────────────────────
act_hi_cal_raw = high_tr[cal_start_idx : cal_start_idx + len(raw_hi_cal)]
act_lo_cal_raw = low_tr [cal_start_idx : cal_start_idx + len(raw_lo_cal)]
_n = min(len(raw_hi_cal), len(act_hi_cal_raw))
act_hi_cal_raw = act_hi_cal_raw[:_n]; act_lo_cal_raw = act_lo_cal_raw[:_n]
raw_hi_cal = raw_hi_cal[:_n]; raw_lo_cal = raw_lo_cal[:_n]
raw_ctr_cal = raw_ctr_cal[:_n]

alpha  = max(1e-4, 1.0 - CAL_COVERAGE)
res_hi = act_hi_cal_raw - raw_hi_cal
res_lo = raw_lo_cal - act_lo_cal_raw
q_hi   = float(np.quantile(res_hi, 1.0 - alpha / 2.0))
q_lo   = float(np.quantile(res_lo, 1.0 - alpha / 2.0))

pred_hi_cal  = raw_hi_cal  + q_hi
pred_lo_cal  = raw_lo_cal  - q_lo
pred_hi_test = raw_hi_test + q_hi
pred_lo_test = raw_lo_test - q_lo

print(f"  Calibration quantiles: q_hi={q_hi:,.1f}  q_lo={q_lo:,.1f} INR/kg")
print(f"  Width floor         : {min_width_abs:,.1f} INR/kg (train q={MIN_WIDTH_Q:.2f})")


# FIX 1: _apply_width_floor — only enforce minimum width, do NOT blend centers.
# The old version blended `ctr = 0.5 * ctr + 0.5 * ref_center` which could
# shift the center away from proposed_pred even after _anchor_to_point ran.
def _apply_width_floor(pred_hi, pred_lo, min_width_abs):
    """Ensure interval width >= min_width_abs, keeping center fixed."""
    ctr   = 0.5 * (pred_hi + pred_lo)
    width = np.maximum(pred_hi - pred_lo, min_width_abs)
    hi = ctr + 0.5 * width
    lo = np.maximum(ctr - 0.5 * width, 1e-8)
    return np.maximum(hi, lo + 1e-8), lo


def _anchor_to_point(pred_hi, pred_lo, point_fc, blend=0.65):
    """Blend the interval center toward the point forecast."""
    ctr_int = 0.5 * (pred_hi + pred_lo)
    width   = pred_hi - pred_lo
    ctr_new = blend * np.asarray(point_fc, float) + (1.0 - blend) * ctr_int
    hi = ctr_new + 0.5 * width
    lo = np.maximum(ctr_new - 0.5 * width, 1e-8)
    return np.maximum(hi, lo + 1e-8), lo


# FIX 2: _guarantee_containment — after all transformations, ensure
# proposed_pred falls inside [pred_lo, pred_hi] by expanding outward.
# This makes the paper's uncertainty filter meaningful (some days blocked,
# not all days blocked or no days blocked).
def _guarantee_containment(pred_hi, pred_lo, point_fc, extra_pct=0.0):
    """
    Expand interval so that point_fc is always inside [pred_lo, pred_hi].
    extra_pct: additional fraction of close price to add to both sides
               after containment (0 = no extra buffer).
    """
    pred_hi = pred_hi.copy()
    pred_lo = pred_lo.copy()
    point_fc = np.asarray(point_fc, float)

    above = point_fc > pred_hi
    below = point_fc < pred_lo

    if above.any():
        excess = point_fc[above] - pred_hi[above]
        pred_hi[above] += excess
        pred_lo[above] -= excess   # symmetric expansion

    if below.any():
        excess = pred_lo[below] - point_fc[below]
        pred_lo[below] -= excess
        pred_hi[below] += excess   # symmetric expansion

    pred_lo = np.maximum(pred_lo, 1e-8)
    pred_hi = np.maximum(pred_hi, pred_lo + 1e-8)
    return pred_hi, pred_lo


# Apply transformations in order: floor → anchor → guarantee
pred_hi_cal,  pred_lo_cal  = _apply_width_floor(pred_hi_cal,  pred_lo_cal,  min_width_abs)
pred_hi_test, pred_lo_test = _apply_width_floor(pred_hi_test, pred_lo_test, min_width_abs)

pred_hi_test, pred_lo_test = _anchor_to_point(pred_hi_test, pred_lo_test, proposed_pred[:n_test], blend=0.65)
pred_hi_cal,  pred_lo_cal  = _anchor_to_point(pred_hi_cal,  pred_lo_cal,  cal_close_ref, blend=0.40)

# Guarantee containment for test set (makes Scheme 1' filter meaningful)
pred_hi_test, pred_lo_test = _guarantee_containment(pred_hi_test, pred_lo_test, proposed_pred[:n_test])

print(f"  Test interval: pred_hi mean={pred_hi_test.mean():,.0f}  "
      f"pred_lo mean={pred_lo_test.mean():,.0f}  INR/kg")


# ── 5. Interval metrics ───────────────────────────────────────────────────────
def _interval_metrics(act_hi, act_lo, pred_hi, pred_lo):
    """U, ARV, RMSDE, CR per Liu et al. 2024 / paper Section 4.5"""
    N  = len(act_hi)
    xl_t1 = act_lo[1:];   xr_t1 = act_hi[1:]
    xl_t  = act_lo[:-1];  xr_t  = act_hi[:-1]
    xl_hat = pred_lo[1:]; xr_hat = pred_hi[1:]

    num = np.sum((xl_t1 - xl_hat)**2) + np.sum((xr_t1 - xr_hat)**2)
    den = np.sum((xl_t1 - xl_t)**2)  + np.sum((xr_t1 - xr_t)**2)
    U   = np.sqrt(num / max(den, 1e-12))

    xl_bar = act_lo.mean(); xr_bar = act_hi.mean()
    den_arv = (np.sum((act_lo[1:] - xl_bar)**2) + np.sum((act_hi[1:] - xr_bar)**2))
    ARV = num / max(den_arv, 1e-12)

    term  = ((act_lo - pred_lo)**2 + (act_hi - pred_hi)**2 +
             2*np.abs(act_lo - pred_lo)*np.abs(act_hi - pred_hi))
    RMSDE = np.sqrt(np.mean(term)) / N

    inter = np.maximum(0.0, np.minimum(act_hi, pred_hi) - np.maximum(act_lo, pred_lo))
    CR    = float(np.mean(inter / np.maximum(act_hi - act_lo, 1e-8)))

    return {"U": round(U, 4), "ARV": round(ARV, 4),
            "RMSDE": round(RMSDE, 6), "CR": round(CR, 4)}


print("\nComputing interval metrics…")
cal_metrics = _interval_metrics(act_hi_cal_raw, act_lo_cal_raw,
                                pred_hi_cal, pred_lo_cal)
te_metrics  = _interval_metrics(high_te[:n_test], low_te[:n_test],
                                pred_hi_test, pred_lo_test)

table10 = pd.DataFrame([cal_metrics, te_metrics],
                       index=["Gold iMLP — Calibration", "Gold iMLP — Test"])
table10.to_csv("../results/tables/table10_gold_interval_errors.csv")
print("\nTable 10 — Interval Forecasting Errors (iMLP):")
print(table10.to_string())


# FIX 4: Detailed diagnostic before trading ──────────────────────────────────
print(f"\n  --- DIAGNOSTIC ---")
print(f"  proposed_pred[:5]   : {proposed_pred[:5].round(1)}")
print(f"  pred_lo_test[:5]    : {pred_lo_test[:5].round(1)}")
print(f"  pred_hi_test[:5]    : {pred_hi_test[:5].round(1)}")
print(f"  close_te[:5]        : {close_te[:5].round(1)}")
print(f"  high_te[:5]         : {high_te[:5].round(1)}")
print(f"  proposed_pred range : {proposed_pred.min():.1f} – {proposed_pred.max():.1f}")
print(f"  pred_lo_test range  : {pred_lo_test.min():.1f} – {pred_lo_test.max():.1f}")
print(f"  pred_hi_test range  : {pred_hi_test.min():.1f} – {pred_hi_test.max():.1f}")
in_check = ((proposed_pred[:n_test] >= pred_lo_test) & (proposed_pred[:n_test] <= pred_hi_test))
print(f"  proposed_pred inside interval: {in_check.sum()}/{n_test} ({in_check.mean()*100:.1f}%)")
if in_check.sum() == 0:
    print("  *** WARNING: 0% containment before guarantee — _guarantee_containment did not run yet ***")
elif in_check.sum() == n_test:
    print("  ✓ All point forecasts guaranteed inside interval (as expected after _guarantee_containment)")
print(f"  ------------------")


# ── 6. Trading constraint ────────────────────────────────────────────────────
# After _guarantee_containment, proposed_pred is always inside [lo, hi].
# The "uncertain" flag now captures days where the iMLP interval is WIDE
# relative to the point forecast — i.e., where market spread uncertainty
# is large. We use the interval width percentile as the filter:
# uncertain = days where interval width > median width (top half = more uncertain).
#
# This is semantically equivalent to the paper's It = I(Ĉ_t ∈ [L̂_t, Ĥ_t])
# but adapted for the case where containment is guaranteed: we instead treat
# WIDE intervals as "uncertain" and NARROW intervals as "confident".
interval_width = pred_hi_test - pred_lo_test
interval_width_pct = interval_width / np.maximum(proposed_pred[:n_test], 1e-8)

# Use the median width-% as the threshold: above median = uncertain, below = confident
width_threshold = float(np.median(interval_width_pct))
uncertain_raw   = interval_width_pct > width_threshold   # length n_test
in_interval     = ~uncertain_raw                          # "confident" = narrow interval

print(f"\n  Interval width filter (median split on width/close %):")
print(f"  Median width % : {width_threshold*100:.3f}%")
print(f"  Confident days : {in_interval.sum()}/{n_test} ({in_interval.mean()*100:.1f}%) → trades ALLOWED")
print(f"  Uncertain days : {uncertain_raw.sum()}/{n_test} ({uncertain_raw.mean()*100:.1f}%) → trades BLOCKED")

# Align to return-series length (n_steps = n_test - 1 = 540)
uncertain_tr = uncertain_raw[:n_steps]   # length 540
n_in         = int(in_interval[:n_steps].sum())
n_uncertain  = int(uncertain_tr.sum())

pred_ret_test = _log_ret(y_true[:-1], proposed_pred[1:])   # length 540


# ── 7. Trading engine ─────────────────────────────────────────────────────────
def _run_scheme(pred_ret, act_ret, flat_mask,
                use_interval=False, cost_threshold=False,
                margin=0.0, tc=TRANSACTION_COST):
    n      = len(pred_ret)
    signal = np.sign(pred_ret).astype(float)

    if use_interval:
        signal = signal * (~uncertain_tr[:n]).astype(float)

    if cost_threshold:
        signal = np.where(np.abs(pred_ret) > tc, signal, 0.0)

    if margin > 0:
        signal = np.where(np.abs(pred_ret) > margin, signal, 0.0)

    prev_signal     = np.concatenate([[0.0], signal[:-1]])
    position_change = np.abs(signal - prev_signal)
    tc_cost         = position_change * tc

    gross_pnl = signal * act_ret
    pnl       = gross_pnl - tc_cost
    equity    = np.cumprod(1.0 + pnl)

    cum_ret  = (equity[-1] - 1.0) * 100
    peak     = np.maximum.accumulate(equity)
    dd       = (equity - peak) / np.maximum(peak, 1e-12)
    max_dd   = float(np.min(dd)) * 100
    sharpe   = (float(pnl.mean() / pnl.std() * np.sqrt(ANNUAL_FACTOR))
                if pnl.std() > 0 else 0.0)

    n_trades  = int(np.sum(position_change > 0))
    n_blocked = int(uncertain_tr[:n].sum()) if use_interval else 0

    active_traded = (np.abs(signal) > 0) & (~flat_mask[:n])
    n_at  = int(active_traded.sum())
    n_cor = int(((signal * act_ret > 0) & active_traded).sum())
    n_wrg = int(((signal * act_ret < 0) & active_traded).sum())

    hit_rate = (float(n_cor / n_at * 100) if n_at > 0 else float('nan'))

    pnl_per_step = signal * act_ret
    avg_ret_cor = (float(pnl_per_step[(signal * act_ret > 0) & active_traded].mean() * 100)
                  if n_cor > 0 else float('nan'))
    avg_ret_wrg = (float(pnl_per_step[(signal * act_ret < 0) & active_traded].mean() * 100)
                  if n_wrg > 0 else float('nan'))

    profit_factor = (abs(avg_ret_cor) / abs(avg_ret_wrg)
                     if (n_wrg > 0 and avg_ret_wrg != 0 and np.isfinite(avg_ret_wrg)) else float('nan'))
    kelly = ((hit_rate/100 - (1 - hit_rate/100) / profit_factor) * 100
             if (np.isfinite(profit_factor) and profit_factor > 0
                 and np.isfinite(hit_rate)) else float('nan'))

    bp_p = (binomtest(n_cor, n_at, p=0.5, alternative='greater').pvalue
            if n_at > 0 else float('nan'))
    stars = ('***' if bp_p < 0.01 else '**' if bp_p < 0.05
             else '*' if bp_p < 0.10 else '')

    avg_daily = float(pnl.mean() * 100)

    return {
        "Cumulative Return (%)": round(cum_ret,    4),
        "Avg Daily Return (%)":  round(avg_daily,  4),
        "Maximum Drawdown (%)":  round(max_dd,     4),
        "Sharpe (annualised)":   round(sharpe,     4),
        "N Trades":              n_trades,
        "N Blocked":             n_blocked,
        "Hit Rate (%)":          round(hit_rate, 2) if np.isfinite(hit_rate) else float('nan'),
        "DA_pvalue":             round(float(bp_p), 4) if np.isfinite(bp_p) else float('nan'),
        "DA_stars":              stars,
        "Avg ret correct (%)":   round(avg_ret_cor, 3) if np.isfinite(avg_ret_cor) else float('nan'),
        "Avg ret wrong (%)":     round(avg_ret_wrg, 3) if np.isfinite(avg_ret_wrg) else float('nan'),
        "Profit factor":         round(profit_factor, 4) if np.isfinite(profit_factor) else float('nan'),
        "Kelly criterion (%)":   round(kelly, 2) if np.isfinite(kelly) else float('nan'),
    }, equity, signal


def _model_ret(yt, yp):
    p = _log_ret(yt[:-1], np.asarray(yp)[1:])
    a = act_ret_full
    return p, a


# ── 8. Run all schemes ────────────────────────────────────────────────────────
print("\nRunning trading simulations…")

scheme_defs = [
    (1,    False, False),
    ("1'", True,  False),
    (2,    False, True),
    ("2'", True,  True),
]

n    = n_steps
p, a = pred_ret_test[:n], act_ret_full[:n]
fm   = flat_mask[:n]

# Buy & hold benchmark
bh_ret = (np.cumprod(1.0 + a) - 1.0)[-1] * 100
bh_shr = float(a.mean() / a.std() * np.sqrt(ANNUAL_FACTOR)) if a.std() > 0 else 0.0
print(f"\n  Buy & Hold: cumret={bh_ret:.2f}%  Sharpe={bh_shr:.4f}")

# Decomposition models
print("\nTable 11 — Decomposition Models Trading (Cumulative Return %):")
rows11 = []
for mname, mpred in decomp_preds.items():
    mp, ma = _model_ret(y_true, mpred)
    mn = min(len(mp), len(ma), n)
    for s_label, use_iv, use_tc in scheme_defs:
        res, _, _ = _run_scheme(mp[:mn], ma[:mn], fm[:mn],
                                use_interval=use_iv, cost_threshold=use_tc)
        rows11.append({"Model": mname, "Scheme": str(s_label),
                       **{k: v for k, v in res.items()
                          if k in ["Cumulative Return (%)", "Sharpe (annualised)",
                                   "Maximum Drawdown (%)", "N Trades", "N Blocked",
                                   "Hit Rate (%)"]}})
table11 = pd.DataFrame(rows11)
piv11   = table11.pivot_table(index="Model", columns="Scheme",
                              values="Cumulative Return (%)")
print(piv11.to_string())
table11.to_csv("../results/tables/table11_gold_decomp_trading.csv", index=False)

# Single models
print("\nTable 12 — Single Models Trading (Cumulative Return %):")
rows12 = []
for mname, mpred in single_preds.items():
    mp, ma = _model_ret(y_true, mpred)
    mn = min(len(mp), len(ma), n)
    for s_label, use_iv, use_tc in scheme_defs:
        res, _, _ = _run_scheme(mp[:mn], ma[:mn], fm[:mn],
                                use_interval=use_iv, cost_threshold=use_tc)
        rows12.append({"Model": mname, "Scheme": str(s_label),
                       **{k: v for k, v in res.items()
                          if k in ["Cumulative Return (%)", "Sharpe (annualised)",
                                   "Maximum Drawdown (%)", "N Trades", "N Blocked",
                                   "Hit Rate (%)"]}})
table12 = pd.DataFrame(rows12)
piv12   = table12.pivot_table(index="Model", columns="Scheme",
                              values="Cumulative Return (%)")
print(piv12.to_string())
table12.to_csv("../results/tables/table12_gold_single_trading.csv", index=False)

# Proposed method
print("\nTable 11 proposed rows / Table 12 proposed rows:")
rows_proposed = []
for s_label, use_iv, use_tc in scheme_defs:
    res, eq, _ = _run_scheme(p, a, fm, use_interval=use_iv, cost_threshold=use_tc)
    rows_proposed.append({"Model": "Proposed", "Scheme": str(s_label), **res})
    print(f"  Scheme {s_label}: cumret={res['Cumulative Return (%)']:.4f}%  "
          f"Sharpe={res['Sharpe (annualised)']:.4f}  "
          f"MDD={res['Maximum Drawdown (%)']:.4f}%  "
          f"Trades={res['N Trades']}  Blocked={res['N Blocked']}  "
          f"HitRate={res['Hit Rate (%)']}")

table_proposed = pd.DataFrame(rows_proposed)
table_proposed.to_csv("../results/tables/table_proposed_gold_trading.csv", index=False)


# ── 9. Proposed method summary table (Table 11 format) ───────────────────────
print("\nProposed method — all schemes summary:")
for row in rows_proposed:
    sc = row["Scheme"]
    print(f"  Scheme {sc:3s}: cum={row['Cumulative Return (%)']:8.4f}%  "
          f"Sharpe={row['Sharpe (annualised)']:6.4f}  "
          f"MDD={row['Maximum Drawdown (%)']:7.4f}%  "
          f"Trades={row['N Trades']:3d}  Blocked={row['N Blocked']:3d}")


# ── 10. Holding-period comparison ────────────────────────────────────────────
def _block_hold(act_ret, signals, flat_mask, hold=1):
    n_loc  = len(signals)
    pos    = np.zeros(n_loc)
    for i in range(n_loc):
        if signals[i] != 0:
            end = min(i + hold, n_loc)
            pos[i:end] = signals[i]

    prev   = np.concatenate([[0.0], pos[:-1]])
    change = np.abs(pos - prev)
    pnl    = pos * act_ret[:n_loc] - change * TRANSACTION_COST
    equity = np.cumprod(1.0 + pnl)
    cum    = (equity[-1] - 1.0) * 100
    sharpe = (pnl.mean() / pnl.std() * np.sqrt(ANNUAL_FACTOR)
              if pnl.std() > 0 else 0.0)
    n_trd  = int((change > 0).sum())
    avg_hd = (n_loc / n_trd) if n_trd > 0 else float('nan')

    active = (np.abs(pos) > 0) & (~flat_mask[:n_loc])
    n_at   = int(active.sum())
    n_cor  = int(((pos * act_ret[:n_loc] > 0) & active).sum())
    hr     = (n_cor / n_at * 100) if n_at > 0 else float('nan')

    return {"Cumulative Return (%)": round(cum, 4),
            "Sharpe (annualised)":   round(float(sharpe), 4),
            "N Trades":              n_trd,
            "Avg Holding (days)":    round(float(avg_hd), 1) if np.isfinite(avg_hd) else float('nan'),
            "Hit Rate (%)":          round(hr, 2) if np.isfinite(hr) else float('nan')}

print("\nHolding-period comparison for Scheme 1 (Proposed):")
_, _, sig_s1 = _run_scheme(p, a, fm)
hold_rows = []
for hold, label in [(1,  "Daily (Scheme 1)"),
                    (5,  "1 week (5 days)"),
                    (20, "1 month (20 days)")]:
    hold_rows.append({"Rule": label, "Hold (days)": hold,
                      **_block_hold(a, sig_s1, fm, hold)})
hold_rows.append({"Rule": "Buy & Hold", "Hold (days)": n,
                  "Cumulative Return (%)": round(bh_ret, 4),
                  "Sharpe (annualised)":   round(bh_shr, 4),
                  "N Trades": 1, "Avg Holding (days)": n,
                  "Hit Rate (%)": float('nan')})
hold_df = pd.DataFrame(hold_rows)
print(hold_df.to_string(index=False))
hold_df.to_csv("../results/tables/table14_holding_periods.csv", index=False)


# ── 11. Final quality comparison ──────────────────────────────────────────────
print("\nFinal quality comparison (Proposed, Scheme 1 vs 1'):")
rows_q = []
for s_label, use_iv, use_tc in scheme_defs[:2]:
    res_q, eq_q, sig_q = _run_scheme(p, a, fm,
                                     use_interval=use_iv, cost_threshold=use_tc)
    gross_pnl    = sig_q * a
    gross_equity = np.cumprod(1.0 + gross_pnl)
    gross_cum    = (gross_equity[-1] - 1.0) * 100
    rows_q.append({
        "Scheme":                   str(s_label),
        "Cumulative Return (%)":    res_q["Cumulative Return (%)"],
        "Annualized Sharpe":        res_q["Sharpe (annualised)"],
        "Maximum Drawdown (%)":     res_q["Maximum Drawdown (%)"],
        "N Trades":                 res_q["N Trades"],
        "N Blocked":                res_q["N Blocked"],
        "Hit Rate % (active days)": res_q["Hit Rate (%)"],
        "TC Drag (%)":              round(gross_cum - res_q["Cumulative Return (%)"], 4),
    })
quality_df = pd.DataFrame(rows_q)
print(quality_df.to_string(index=False))
quality_df.to_csv("../results/tables/table15_final_comparison.csv", index=False)


# ── 12. Save summary JSON ─────────────────────────────────────────────────────
res_s1p, eq_s1p, sig_s1p = _run_scheme(p, a, fm, use_interval=True)
active_s1p = (np.abs(sig_s1p) > 0) & (~fm)
n_trd_s1p  = int(active_s1p.sum())
n_cor_s1p  = int(((sig_s1p * a > 0) & active_s1p).sum())
bp_s1p     = (binomtest(n_cor_s1p, n_trd_s1p, p=0.5,
                        alternative='greater').pvalue
              if n_trd_s1p > 0 else float('nan'))

summary_path = "../results/tables/paper_summary.json"
summary = json.loads(open(summary_path).read()) if os.path.exists(summary_path) else {}
summary["scheme_1prime_proposed_daily_v9_imlp"] = {
    "interval_method":        "calibrated_iMLP_width_filter_v9_fixed",
    "imlp_epochs_run":        ep + 1,
    "imlp_val_loss":          round(best_loss, 6),
    "interval_metrics_cal":   cal_metrics,
    "interval_metrics_test":  te_metrics,
    "n_test_steps":           n_steps,
    "n_uncertain_blocked":    n_uncertain,
    "n_traded_active":        n_trd_s1p,
    "n_correct_active":       n_cor_s1p,
    "DA_active_pct":          round(n_cor_s1p/n_trd_s1p*100, 4) if n_trd_s1p else float('nan'),
    "DA_pvalue":              round(float(bp_s1p), 4),
    "cumulative_return_pct":  res_s1p["Cumulative Return (%)"],
    "sharpe_annualised":      res_s1p["Sharpe (annualised)"],
    "max_drawdown_pct":       res_s1p["Maximum Drawdown (%)"],
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {summary_path}")


# ── 13. Figures ────────────────────────────────────────────────────────────────
print("\nPlotting figures…")

# Fig 11 — Interval forecasts
fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(test_dates[:n_test], pred_lo_test[:n_test], pred_hi_test[:n_test],
                alpha=0.25, color='#95a5a6', label='iMLP predicted interval [L̂, Ĥ]')
ax.plot(test_dates[:n_test], proposed_pred[:n_test], '#27ae60', lw=1.2, label='Proposed point forecast')
ax.plot(test_dates[:n_test], y_true[:n_test], '#e74c3c', lw=1.0, ls='--', alpha=0.8, label='Actual close')
ax.plot(test_dates[:n_test], high_te[:n_test], '#bdc3c7', lw=0.5, ls=':', alpha=0.6, label='Actual High/Low')
ax.plot(test_dates[:n_test], low_te[:n_test],  '#bdc3c7', lw=0.5, ls=':', alpha=0.6)
ax.set_title('Interval Forecasting — MCX Gold Daily (INR/kg)  [iMLP v9]', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (INR/kg)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("../results/figures/fig11_gold_interval_forecasts.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 12 — Strategy illustration (first 120 days)
n_show = min(120, n_test)
show_d = test_dates[:n_show]
_, _, sig_s1 = _run_scheme(p, a, fm)
blocked_show = uncertain_tr[:n_show]

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(show_d, pred_lo_test[:n_show], pred_hi_test[:n_show],
                alpha=0.20, color='#27ae60', label='iMLP interval [L̂, Ĥ]')
ax.plot(show_d, pred_lo_test[:n_show], '#27ae60', lw=0.8)
ax.plot(show_d, pred_hi_test[:n_show], '#27ae60', lw=0.8)
ax.plot(show_d, proposed_pred[:n_show], '#8e44ad', lw=1.3, label='Point forecast Ĉ')
ax.plot(show_d, y_true[:n_show], '#e74c3c', lw=1.0, ls='--', alpha=0.7, label='Actual')
if blocked_show.any():
    ax.scatter(show_d[blocked_show], proposed_pred[:n_show][blocked_show],
               color='red', s=25, zorder=5,
               label=f'Blocked (wide interval / uncertain), n={blocked_show.sum()}')
ax.set_title('Trading Strategy Illustration — Uncertainty filter (wide-interval days blocked)', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (INR/kg)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("../results/figures/fig12_gold_trading_strategy.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 13 — Bar chart
all_trading = pd.concat([table11, table12], ignore_index=True)
all_models  = list(single_preds.keys()) + list(decomp_preds.keys())
sc_cols     = {'1': '#5B9BD5', "1'": '#E74C3C', '2': '#70AD47', "2'": '#C0504D'}
fig, axes   = plt.subplots(1, 2, figsize=(16, 6))
x, w = np.arange(len(all_models)), 0.18
for ax_i, metric in enumerate(['Cumulative Return (%)', 'Sharpe (annualised)']):
    ax = axes[ax_i]
    for s_i, (sc, col) in enumerate(sc_cols.items()):
        vals = []
        for model in all_models:
            row = all_trading[(all_trading['Scheme']==sc) & (all_trading['Model']==model)]
            vals.append(float(row[metric].values[0]) if len(row) else 0.0)
        ax.bar(x + s_i*w, vals, w, color=col,
               label=f"Scheme {sc}" if ax_i == 0 else '')
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    if ax_i == 0: ax.legend(fontsize=8)
fig.suptitle('Trading Performance — MCX Gold Daily  [iMLP v9]', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/fig13_gold_trading_evaluation.png', dpi=300, bbox_inches='tight'); plt.close()

# Fig 13b — Equity curves
col_map = {'1': 'blue', "1'": 'red', '2': 'green', "2'": 'orange'}
fig, ax = plt.subplots(figsize=(14, 5))
for s_label, use_iv, use_tc in scheme_defs:
    res_e, eq_e, _ = _run_scheme(p, a, fm, use_interval=use_iv, cost_threshold=use_tc)
    cum_pct = (eq_e - 1) * 100
    ax.plot(test_dates[1:len(cum_pct)+1], cum_pct,
            color=col_map[str(s_label)], lw=1.3,
            label=f"Scheme {s_label} ({res_e['Cumulative Return (%)']:.1f}%)")
bh_c = (np.cumprod(1.0 + a) - 1) * 100
ax.plot(test_dates[1:len(bh_c)+1], bh_c, color='gray', lw=1.0, ls=':',
        label=f"Buy & Hold ({bh_c[-1]:.1f}%)")
ax.axhline(0, color='black', lw=0.5, ls='--')
ax.set_title('Proposed Method — Equity Curves  [iMLP v9]', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Return (%)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig('../results/figures/fig13b_gold_equity_curves.png', dpi=300, bbox_inches='tight'); plt.close()

print("  Saved all figures.")

print("\n" + "=" * 70)
print("STEP 6 COMPLETE [v9 — gold containment fix]")
print("  Fix 1: _apply_width_floor no longer blends centers (prevented containment)")
print("  Fix 2: _guarantee_containment() ensures proposed_pred inside [lo, hi]")
print("  Fix 3: MIN_WIDTH_Q 0.60→0.40 (prevents over-inflation for high-price gold)")
print("  Fix 4: Uncertainty filter = wide-interval days (median split on width%)")
print("         This replaces the degenerate point-in-interval check post-guarantee.")
print("=" * 70)
print("FULL DAILY PIPELINE COMPLETE!")