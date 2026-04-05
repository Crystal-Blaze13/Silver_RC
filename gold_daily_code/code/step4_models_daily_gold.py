"""
STEP 4 (daily) — Forecasting Models  [v5]
==========================================

ROOT-CAUSE FIX (single-model catastrophe):
  After step3 v2, X_full contains log-return features (stationary, ~0.001
  scale). But y_train_all was still raw silver price LEVELS (~85,000 INR/kg).
  Models faced an impossible regression: stationary near-zero inputs →
  non-stationary 85k-level target. They collapsed to predicting the training
  mean, giving DA=21.7% for every model.

  FIX: single-model benchmarks now predict silver LOG RETURNS, then
  reconstruct price levels via:
    price[t] = price[t-1] * exp(predicted_logret[t])
  
  This is consistent with the feature space (X = log returns → y = log returns)
  and gives the models a stationary, learnable target.

  DECOMPOSITION benchmarks (VMD-ARIMA, VMD-LSTM etc.) are UNCHANGED:
  they predict IMF levels directly, which are already near-stationary
  for High IMFs and handled by ARIMA's d parameter for Low IMFs.
  The LSTM in decomp benchmarks uses X_full (log-return features) to
  predict IMF levels — this is fine because IMF levels are bounded
  and stationary for High IMFs.

  PROPOSED METHOD is UNCHANGED: ARIMA on Low IMF levels, LSTM on High
  IMF levels with LASSO-selected log-return features. Per-IMF LSTM already
  worked (beats naive on IMF5/6/7/8) because IMF levels are stationary.

ARCHITECTURE SUMMARY:
  Low IMF          → ARIMA walk-forward (levels)           [unchanged]
  High IMF         → LSTM on IMF levels, log-return feats   [unchanged]
  Single ES        → direct level forecast                  [unchanged]
  Single ARIMA     → walk-forward on levels                 [unchanged]
  Single SVR/RF/   → predict log return → reconstruct level [FIXED]
    MLP/ELM/LSTM
  VMD-ARIMA        → ARIMA on each IMF level                [unchanged]
  VMD-LSTM         → LSTM on each IMF level                 [unchanged]
  CEEMDAN-*        → same as VMD-*                          [unchanged]
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import itertools
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import binomtest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

warnings.filterwarnings("ignore")


class Config:
    DATA_FILE       = "../data/master_daily_prices_gold.csv"
    IMF_FILE        = "../data/imfs_daily_gold.npy"
    COMPLEXITY_FILE = "../data/imf_complexity_daily_gold.csv"
    LASSO_FILE      = "../data/lasso_selected_features_daily_gold.pkl"
    GOLD_FILE     = "../data/gold_daily.csv"
    N_TRAIN_FILE    = "../data/n_train_daily_gold.npy"

    LSTM_EPOCHS     = 100
    LSTM_HIDDEN     = 64
    LSTM_LR         = 0.001
    LSTM_SEQ_LEN    = 20
    LSTM_BATCH_SIZE = 64
    LSTM_PATIENCE   = 15
    VAL_FRAC        = 0.10
    ALPHA_DIR       = 0.25

    ARIMA_REFIT_EVERY       = 10
    ARIMA_REFIT_TRAIN_EVERY = 20
    WF_BURN_IN              = 120

    ELM_HIDDEN  = 50
    USE_CEEMDAN = True
    N_JOBS      = -1


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    actual_dir = np.sign(np.diff(y_true))
    pred_dir   = np.sign(y_pred[1:] - y_true[:-1])
    correct    = int(np.sum(actual_dir == pred_dir))
    n          = len(actual_dir)
    p_val      = float(binomtest(correct, n, p=0.5, alternative="greater").pvalue)
    return round(correct / n * 100, 2), round(p_val, 4)


def compute_metrics(y_true, y_pred, label=None):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {k: np.nan for k in
                ["RMSE","MAE","MAPE(%)","sMAPE(%)","DA(%)","DA_pval"]}
    yt, yp = y_true[mask], y_pred[mask]
    rmse  = float(np.sqrt(mean_squared_error(yt, yp)))
    mae   = float(mean_absolute_error(yt, yp))
    mape  = float(np.mean(np.abs((yt-yp)/np.maximum(np.abs(yt),1e-8)))*100)
    smape = float(np.mean(np.abs(yt-yp)/((np.abs(yt)+np.abs(yp))/2))*100)
    da, da_p = directional_accuracy(yt, yp)
    if label:
        print(f"    {label}: RMSE={rmse:,.1f}  MAE={mae:,.1f}  "
              f"MAPE={mape:.2f}%  DA={da:.1f}% (p={da_p:.4f})")
    return {"RMSE": round(rmse,4), "MAE": round(mae,4),
            "MAPE(%)": round(mape,4), "sMAPE(%)": round(smape,4),
            "DA(%)": da, "DA_pval": da_p}


def is_stationary(series, alpha=0.05):
    try:
        return adfuller(series, autolag='AIC')[1] < alpha
    except Exception:
        return False


def logret_to_levels(logrets, price_before_first):
    """Reconstruct price levels from log returns."""
    return price_before_first * np.exp(np.cumsum(logrets))


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
_arima_order_cache = {}


def _fit_one_arima(series, p, d, q):
    try:
        return ARIMA(series, order=(p,d,q)).fit().aic, (p,d,q)
    except Exception:
        return np.inf, (p,d,q)


def best_arima_order(series, cache_key=None):
    if cache_key and cache_key in _arima_order_cache:
        return _arima_order_cache[cache_key]
    stationary = is_stationary(series)
    if HAS_PMDARIMA:
        try:
            res = auto_arima(
                series,
                start_p=0, max_p=2 if not stationary else 4,
                start_q=0, max_q=1 if not stationary else 2,
                d=1 if not stationary else None, max_d=1,
                stepwise=True, information_criterion='aic',
                error_action='ignore', suppress_warnings=True)
            order = res.order
        except Exception:
            order = (1,1,0) if not stationary else (1,0,1)
    else:
        cands = (list(itertools.product(range(3),[1],range(2)))
                 if not stationary else
                 list(itertools.product(range(5),range(2),range(3))))
        if HAS_JOBLIB:
            results = Parallel(n_jobs=Config.N_JOBS, prefer='threads')(
                delayed(_fit_one_arima)(series,p,d,q) for p,d,q in cands)
        else:
            results = [_fit_one_arima(series,p,d,q) for p,d,q in cands]
        best_aic, order = np.inf, (1,1,0)
        for aic, o in results:
            if aic < best_aic:
                best_aic, order = aic, o
    if cache_key:
        _arima_order_cache[cache_key] = order
    return order


def safe_arima_fit(history, order):
    p, d, q = order
    for cand in [(p,d,q),(min(2,p),d,min(1,q)),(1,0,1),(1,1,0),(0,1,0)]:
        try:
            return ARIMA(history, order=cand).fit(), cand
        except Exception:
            continue
    return None, None


def arima_walk_forward(seed, observed, order,
                       refit_every=Config.ARIMA_REFIT_EVERY):
    history = list(np.asarray(seed, float))
    observed = np.asarray(observed, float)
    preds, fitted, steps = [], None, 0
    for obs in observed:
        if fitted is None or steps >= refit_every:
            fitted, order = safe_arima_fit(history, order)
            if fitted is None:
                preds.append(float(history[-1]))
                history.append(float(obs)); steps += 1; continue
            steps = 0
        else:
            try:
                fitted = fitted.apply(history)
            except Exception:
                fitted, order = safe_arima_fit(history, order)
                if fitted is None:
                    preds.append(float(history[-1]))
                    history.append(float(obs)); steps += 1; continue
                steps = 0
        try:
            fc = fitted.forecast(steps=1)
            p  = float(fc.iloc[0]) if hasattr(fc,"iloc") else float(fc[0])
        except Exception:
            p = float(history[-1])
        if not np.isfinite(p): p = float(history[-1])
        preds.append(p); history.append(float(obs)); steps += 1
    return np.array(preds, float)


def arima_walk_forward_aligned(series, burn_in=Config.WF_BURN_IN,
                                order=None, refit_every=Config.ARIMA_REFIT_EVERY):
    series = np.asarray(series, float)
    if order is None:
        order = best_arima_order(series[:burn_in])
    preds = np.full(len(series), np.nan, float)
    preds[burn_in:] = arima_walk_forward(
        series[:burn_in], series[burn_in:], order=order, refit_every=refit_every)
    return preds, order


# ─────────────────────────────────────────────────────────────────────────────
# LSTM helpers (unchanged from v4)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:])


class DirMSELoss(nn.Module):
    def __init__(self, alpha=Config.ALPHA_DIR):
        super().__init__()
        self.alpha = alpha
    def forward(self, pred, target):
        mse     = F.mse_loss(pred, target)
        dir_pen = (1.0-(torch.sign(pred)==torch.sign(target)).float()).mean()
        return mse + self.alpha * dir_pen


def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X)-seq_len):
        Xs.append(X[i:i+seq_len]); ys.append(y[i+seq_len])
    return np.array(Xs, np.float32), np.array(ys, np.float32)


def train_lstm(X_train, y_train, device,
               epochs=Config.LSTM_EPOCHS, hidden=Config.LSTM_HIDDEN,
               seq_len=Config.LSTM_SEQ_LEN, lr=Config.LSTM_LR,
               patience=Config.LSTM_PATIENCE,
               batch_size=Config.LSTM_BATCH_SIZE,
               use_dir_loss=True):
    n_val = max(seq_len+1, int(len(X_train)*Config.VAL_FRAC))
    Xs_tr, ys_tr = make_sequences(X_train[:-n_val], y_train[:-n_val], seq_len)
    Xs_va, ys_va = make_sequences(X_train[-n_val-seq_len:],
                                   y_train[-n_val-seq_len:], seq_len)
    if len(Xs_tr)==0:
        Xs_tr, ys_tr = make_sequences(X_train, y_train, seq_len)
        Xs_va, ys_va = Xs_tr, ys_tr
    ds   = TensorDataset(torch.FloatTensor(Xs_tr).to(device),
                         torch.FloatTensor(ys_tr).unsqueeze(1).to(device))
    dl   = DataLoader(ds, batch_size=batch_size, shuffle=False)
    Xv_t = torch.FloatTensor(Xs_va).to(device)
    yv_t = torch.FloatTensor(ys_va).unsqueeze(1).to(device)
    model     = LSTMModel(X_train.shape[1], hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DirMSELoss(Config.ALPHA_DIR) if use_dir_loss else nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr*0.01)
    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for Xb, yb in dl:
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(model(Xv_t), yv_t).item()
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
        if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, seq_len, best_val


def predict_lstm(model, X_train, X_test, seq_len, device):
    model.eval()
    X_all = np.vstack([X_train, X_test]).astype(np.float32)
    n_tr  = len(X_train)
    preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            idx    = n_tr + i
            window = X_all[max(0,idx-seq_len):idx]
            if len(window) < seq_len:
                pad    = np.zeros((seq_len-len(window), window.shape[1]), np.float32)
                window = np.vstack([pad, window])
            preds.append(
                model(torch.FloatTensor(window).unsqueeze(0).to(device)).item())
    return np.array(preds, float)


def lstm_insample_preds(model, X_train, seq_len, device,
                        burn_in=Config.WF_BURN_IN):
    model.eval()
    preds = np.full(len(X_train), np.nan, float)
    with torch.no_grad():
        for i in range(max(seq_len, burn_in), len(X_train)):
            window = X_train[i-seq_len:i].astype(np.float32)
            preds[i] = model(
                torch.FloatTensor(window).unsqueeze(0).to(device)).item()
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# High-IMF forecaster (paper method: LSTM on levels, unchanged from v4)
# ─────────────────────────────────────────────────────────────────────────────
def forecast_high_imf(y_tr, y_te, X_tr_sc, X_te_sc, device, imf_label):
    n_test  = len(y_te)
    burn_in = Config.WF_BURN_IN

    # Train LSTM on scaled IMF levels
    y_sc    = StandardScaler()
    y_tr_sc = y_sc.fit_transform(y_tr.reshape(-1,1)).ravel().astype(np.float32)

    print(f"    [{imf_label}] Training LSTM on levels "
          f"(n_tr={len(y_tr)}, n_feat={X_tr_sc.shape[1]})…")
    model, sl, best_val = train_lstm(
        X_tr_sc, y_tr_sc, device=device, use_dir_loss=True)

    # Naive persistence val loss in normalized space = variance of last val window
    val_size  = max(sl+1, int(len(y_tr)*Config.VAL_FRAC))
    y_val_sc  = y_tr_sc[-val_size:]
    naive_val = float(np.mean((y_val_sc[1:] - y_val_sc[:-1])**2)) \
        if len(y_val_sc) > 1 else np.inf

    if best_val >= naive_val:
        print(f"    [{imf_label}] LSTM did not beat naive persistence "
              f"(val={best_val:.5f} ≥ naive={naive_val:.5f}) → ARIMA fallback")
        ck    = (len(y_tr), "High", imf_label)
        order = best_arima_order(y_tr, cache_key=ck)
        print(f"    [{imf_label}] ARIMA order: {order}")
        preds_test  = arima_walk_forward(y_tr, y_te, order=order)
        preds_train, _ = arima_walk_forward_aligned(
            y_tr, burn_in=burn_in, order=order,
            refit_every=Config.ARIMA_REFIT_TRAIN_EVERY)
        return preds_test, preds_train

    print(f"    [{imf_label}] LSTM beats naive "
          f"(val={best_val:.5f} < naive={naive_val:.5f}) → using LSTM predictions")
    preds_sc    = predict_lstm(model, X_tr_sc, X_te_sc, sl, device)
    preds_test  = y_sc.inverse_transform(preds_sc.reshape(-1,1)).ravel()

    preds_tr_sc = lstm_insample_preds(model, X_tr_sc, sl, device)
    preds_train = np.full(len(y_tr), np.nan, float)
    valid       = np.isfinite(preds_tr_sc)
    if valid.any():
        preds_train[valid] = y_sc.inverse_transform(
            np.nan_to_num(preds_tr_sc[valid]).reshape(-1,1)).ravel()
    return preds_test, preds_train


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition benchmark helper (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def forecast_imf_set(imf_matrix, model_type, label,
                     n_train_adj, n_test, N_LAGS,
                     X_tr_sc_all, X_te_sc_all, device):
    K_    = imf_matrix.shape[0]
    total = np.zeros(n_test, float)
    for i in range(K_):
        imf_  = imf_matrix[i, N_LAGS:]
        y_tr_ = imf_[:n_train_adj]
        y_te_ = imf_[n_train_adj:]
        if len(y_te_) < n_test:
            y_te_ = np.concatenate([y_te_, np.full(n_test-len(y_te_), y_te_[-1])])
        else:
            y_te_ = y_te_[:n_test]
        try:
            if model_type == "ARIMA":
                ck    = (id(imf_matrix), i, "ARIMA")
                order = best_arima_order(y_tr_, cache_key=ck)
                total += arima_walk_forward(y_tr_, y_te_, order=order)
            else:
                # LSTM on IMF levels (paper method)
                y_sc_ = StandardScaler()
                y_tr_n = y_sc_.fit_transform(y_tr_.reshape(-1,1)).ravel()
                m_, sl_, _ = train_lstm(X_tr_sc_all, y_tr_n,
                                        device=device, use_dir_loss=True)
                preds_n = predict_lstm(m_, X_tr_sc_all, X_te_sc_all, sl_, device)
                total  += y_sc_.inverse_transform(preds_n.reshape(-1,1)).ravel()
        except Exception as e:
            print(f"      {label} IMF{i+1}/{K_} FAILED: {e!r}")
            total += float(y_tr_[-1])
        if not np.all(np.isfinite(total)):
            total = np.where(np.isfinite(total), total, float(y_tr_[-1]))
        print(f"      {label} IMF{i+1}/{K_} done")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print("="*65)
    print("STEP 4 (daily): Forecasting Models  [v5]")
    print(f"  Device : {device}")
    print(f"  joblib : {HAS_JOBLIB}   pmdarima: {HAS_PMDARIMA}")
    print("  Single models: predict log returns → reconstruct levels [FIX]")
    print("="*65)
    t0 = time.perf_counter()
    def progress(msg): print(f"[+{time.perf_counter()-t0:7.1f}s] {msg}")

    # ── Load ──────────────────────────────────────────────────────────────
    gold_df    = pd.read_csv(Config.GOLD_FILE, index_col=0, parse_dates=True)
    gold_price = gold_df.iloc[:,0].values
    dates        = gold_df.index

    u_sorted = np.load(Config.IMF_FILE)
    _np, _ni = len(gold_price), u_sorted.shape[1]
    if _ni != _np:
        trim = _ni - _np
        u_sorted = u_sorted[:,trim:] if trim>0 else \
            np.pad(u_sorted, ((0,0),(0,-trim)), mode='edge')

    complexity  = pd.read_csv(Config.COMPLEXITY_FILE)
    n_train     = int(np.load(Config.N_TRAIN_FILE)[0])
    K           = u_sorted.shape[0]

    with open(Config.LASSO_FILE,"rb") as f:
        ld = pickle.load(f)

    selected_features = ld["selected_features"]
    all_feature_names = ld["all_feature_names"]
    N_LAGS            = ld["N_LAGS"]
    X_full            = ld["X_full"]
    n_train_adj       = ld["n_train_adj"]

    # Detect whether features are log-returns (v2) or levels (v1)
    feature_type = ld.get("feature_type", "levels")
    print(f"  Feature type from step3: {feature_type}")
    if feature_type != "log_returns":
        print("  WARNING: step3 v1 features (levels) detected.")
        print("  Re-run step3_lasso_daily.py (v2) before step4 for correct results.")

    n_test       = len(gold_price) - n_train
    y_true_test  = gold_price[n_train:]
    y_true_train = gold_price[N_LAGS:n_train]
    test_dates   = dates[n_train:]

    # Silver log returns (needed for single-model target)
    gold_logret = np.concatenate([[0.0],
        np.log(gold_price[1:] / gold_price[:-1])])
    # logret[t] = log(price[t]/price[t-1]), aligned with gold_price index

    recon   = u_sorted[:,N_LAGS:].sum(axis=0)
    max_err = float(np.max(np.abs(recon - gold_price[N_LAGS:]))) \
        if len(recon)==len(gold_price)-N_LAGS else np.nan
    print(f"  Reconstruction error (max): {max_err:,.4f}")
    print(f"  Train: {n_train} | Test: {n_test} | IMFs: {K}")
    progress("Data loaded")

    # ── Proposed method (unchanged from v4) ───────────────────────────────
    print("\n── Proposed method ──────────────────────────────────────────────")
    progress(f"Starting ({K} IMFs)")

    imf_preds_test  = np.zeros((K,n_test), float)
    imf_preds_train = np.full((K,n_train_adj), np.nan, float)
    imf_da_report   = []

    for i in range(K):
        imf   = u_sorted[i, N_LAGS:]
        comp  = complexity.loc[i,"Complexity"]
        feats = selected_features[i]

        feat_idx = ([all_feature_names.index(f) for f in feats] if feats else [0])
        X_imf    = X_full[:, feat_idx]

        y_tr = imf[:n_train_adj]
        y_te = imf[n_train_adj:]
        y_te = (np.concatenate([y_te, np.full(n_test-len(y_te), y_te[-1])])
                if len(y_te)<n_test else y_te[:n_test])

        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X_imf[:n_train_adj])
        X_te_sc = sc.transform(X_imf[n_train_adj:n_train_adj+n_test])

        imf_std        = float(np.std(y_tr))
        imf_stationary = is_stationary(y_tr)
        print(f"\n  IMF{i+1:02d} ({comp:4s}) std={imf_std:.2f} "
              f"stat={imf_stationary}  n_features={len(feat_idx)}")

        if comp == "Low":
            print(f"    → ARIMA")
            ck    = (len(y_tr),"Low",i)
            order = best_arima_order(y_tr, cache_key=ck)
            print(f"    ARIMA order: {order}")
            preds_test  = arima_walk_forward(y_tr, y_te, order=order)
            preds_train,_ = arima_walk_forward_aligned(
                y_tr, burn_in=Config.WF_BURN_IN, order=order,
                refit_every=Config.ARIMA_REFIT_TRAIN_EVERY)

        elif imf_std > 1000 and not imf_stationary:
            print(f"    → ARIMA override (large non-stationary High IMF)")
            ck    = (len(y_tr),"HighOverride",i)
            order = best_arima_order(y_tr, cache_key=ck)
            print(f"    ARIMA order: {order}")
            preds_test  = arima_walk_forward(y_tr, y_te, order=order)
            preds_train,_ = arima_walk_forward_aligned(
                y_tr, burn_in=Config.WF_BURN_IN, order=order,
                refit_every=Config.ARIMA_REFIT_TRAIN_EVERY)

        else:
            preds_test, preds_train = forecast_high_imf(
                y_tr, y_te, X_tr_sc, X_te_sc, device,
                imf_label=f"IMF{i+1}")

        imf_preds_test[i,:]  = preds_test
        imf_preds_train[i,:] = preds_train[:n_train_adj]

        if len(y_te)>1:
            da_i,_ = directional_accuracy(y_te, preds_test)
            imf_da_report.append((f"IMF{i+1}", comp, da_i))
            print(f"    DA (per-IMF) = {da_i:.1f}%")
        progress(f"IMF {i+1}/{K} done")

    print("\n─── Per-IMF directional accuracy ─────────────────────────────")
    for name,comp_lbl,da_val in imf_da_report:
        print(f"  {name} ({comp_lbl:4s}): {da_val:.1f}%"
              + (" ← LOW" if da_val<40 else ""))
    print("───────────────────────────────────────────────────────────────")

    proposed_pred       = imf_preds_test.sum(axis=0)
    proposed_pred_train = np.nansum(imf_preds_train, axis=0)
    proposed_pred_train[np.all(np.isnan(imf_preds_train), axis=0)] = np.nan

    print("\nProposed method:")
    compute_metrics(y_true_test, proposed_pred, label="  Test")
    progress("Proposed method done")

    # ── Single-model benchmarks ───────────────────────────────────────────
    print("\n── Single-model benchmarks ──────────────────────────────────────")
    print("  Predicting log returns → reconstructing levels")
    progress("Starting single-model benchmarks")

    X_train_all = X_full[:n_train_adj]
    X_test_all  = X_full[n_train_adj:]

    # TARGET: log returns aligned with X_full rows
    # X_full[t] uses features from time N_LAGS+t, predicting silver at N_LAGS+t
    # So y_logret[t] = logret at position N_LAGS+t in the full series
    y_logret_all = gold_logret[N_LAGS:]   # length = len(X_full)
    y_logret_train = y_logret_all[:n_train_adj]
    y_logret_test  = y_logret_all[n_train_adj:n_train_adj+n_test]

    # Last known price before the test period (for level reconstruction)
    price_before_test = float(gold_price[n_train - 1])

    sc_all      = StandardScaler()
    X_tr_sc_all = sc_all.fit_transform(X_train_all)
    X_te_sc_all = sc_all.transform(X_test_all)

    # Scale targets too (log returns, already small but good practice)
    y_sc_lr = StandardScaler()
    y_tr_lr = y_sc_lr.fit_transform(y_logret_train.reshape(-1,1)).ravel()

    def _reconstruct(logret_preds):
        """Convert predicted log returns → price levels."""
        return logret_to_levels(logret_preds, price_before_test)

    def _run_es():
        # ES operates on price levels directly (it ignores features)
        m = ExponentialSmoothing(y_true_train, trend="add").fit(
            smoothing_level=0.2, smoothing_trend=0.8)
        return "ES", m.forecast(n_test)

    def _run_arima():
        # ARIMA on price levels (handles non-stationarity via d)
        ck    = ("single", len(y_true_train))
        order = best_arima_order(y_true_train, cache_key=ck)
        print(f"    Single ARIMA order: {order}")
        return "ARIMA", arima_walk_forward(y_true_train, y_true_test, order=order)

    def _run_svr():
        y_sc = StandardScaler()
        y_n  = y_sc.fit_transform(y_logret_train.reshape(-1,1)).ravel()
        svr  = SVR(kernel="rbf", C=8.1, gamma=0.1)
        svr.fit(X_tr_sc_all, y_n)
        lr_pred = y_sc.inverse_transform(
            svr.predict(X_te_sc_all).reshape(-1,1)).ravel()
        return "SVR", _reconstruct(lr_pred)

    def _run_rf():
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_tr_sc_all, y_logret_train)
        return "RF", _reconstruct(rf.predict(X_te_sc_all))

    def _run_mlp():
        mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=2000,
                           activation="relu", random_state=42,
                           early_stopping=True, validation_fraction=0.1)
        mlp.fit(X_tr_sc_all, y_logret_train)
        return "MLP", _reconstruct(mlp.predict(X_te_sc_all))

    def _run_elm():
        tscv = TimeSeriesSplit(n_splits=5)
        best_r, best_n = np.inf, Config.ELM_HIDDEN
        for n_h in [20, 50, 100]:
            np.random.seed(42)
            W  = np.random.randn(X_tr_sc_all.shape[1], n_h)
            b_ = np.random.randn(n_h)
            H  = np.tanh(X_tr_sc_all @ W + b_)
            rmses = [np.sqrt(mean_squared_error(
                y_logret_train[v],
                H[v] @ np.linalg.pinv(H[tr]) @ y_logret_train[tr]))
                for tr,v in tscv.split(H)]
            if np.mean(rmses) < best_r:
                best_r, best_n = np.mean(rmses), n_h
        np.random.seed(42)
        We = np.random.randn(X_tr_sc_all.shape[1], best_n)
        be = np.random.randn(best_n)
        Htr = np.tanh(X_tr_sc_all @ We + be)
        Hte = np.tanh(X_te_sc_all @ We + be)
        beta = np.linalg.pinv(Htr) @ y_logret_train
        return "ELM", _reconstruct(Hte @ beta)

    def _run_lstm_single():
        y_sc = StandardScaler()
        y_n  = y_sc.fit_transform(y_logret_train.reshape(-1,1)).ravel()
        m, sl, _ = train_lstm(X_tr_sc_all, y_n, device=device, use_dir_loss=True)
        preds_n  = predict_lstm(m, X_tr_sc_all, X_te_sc_all, sl, device)
        lr_pred  = y_sc.inverse_transform(preds_n.reshape(-1,1)).ravel()
        return "LSTM", _reconstruct(lr_pred)

    single_preds, single_metrics = {}, {}
    cpu_tasks = [_run_es, _run_arima, _run_svr, _run_rf, _run_mlp, _run_elm]
    print("  Running ES/ARIMA/SVR/RF/MLP/ELM in parallel…")
    with ThreadPoolExecutor(max_workers=len(cpu_tasks)) as ex:
        futures = {ex.submit(fn): fn.__name__ for fn in cpu_tasks}
        for fut in as_completed(futures):
            try:
                name, pred = fut.result()
                single_preds[name] = pred
                print(f"    {name} done")
            except Exception as e:
                print(f"    {futures[fut]} FAILED: {e}")

    print("  LSTM (single)…")
    name, pred = _run_lstm_single()
    single_preds[name] = pred
    progress("Single benchmarks done")

    for name, pred in single_preds.items():
        single_metrics[name] = compute_metrics(
            y_true_test, pred, label=f"  {name}")

    # ── Decomposition benchmarks ──────────────────────────────────────────
    print("\n── Decomposition benchmarks ─────────────────────────────────────")
    decomp_preds, decomp_metrics = {}, {}

    # Decomp LSTMs predict IMF levels — use full feature matrix, scaled
    sc_full   = StandardScaler()
    X_tr_full = sc_full.fit_transform(X_train_all)
    X_te_full = sc_full.transform(X_test_all)

    kw = dict(n_train_adj=n_train_adj, n_test=n_test, N_LAGS=N_LAGS,
              X_tr_sc_all=X_tr_full, X_te_sc_all=X_te_full, device=device)

    print("  VMD-ARIMA…")
    decomp_preds["VMD-ARIMA"] = forecast_imf_set(
        u_sorted,"ARIMA","VMD-ARIMA",**kw)
    progress("VMD-ARIMA done")

    print("  VMD-LSTM…")
    decomp_preds["VMD-LSTM"] = forecast_imf_set(
        u_sorted,"LSTM","VMD-LSTM",**kw)
    progress("VMD-LSTM done")

    ceemdan_imfs = None
    if Config.USE_CEEMDAN:
        try:
            from PyEMD import CEEMDAN
            print("  CEEMDAN…")
            cem = CEEMDAN(trials=10, parallel=False)
            cem.noise_seed(42)
            c_raw  = cem(gold_price)
            n_full = len(gold_price) - N_LAGS
            c_pad  = np.zeros((c_raw.shape[0], n_full))
            for j in range(c_raw.shape[0]):
                sl = c_raw[j,N_LAGS:]; l = min(len(sl),n_full)
                c_pad[j,:l] = sl[:l]
            ceemdan_imfs = c_pad
            print(f"    {ceemdan_imfs.shape[0]} CEEMDAN IMFs")
            decomp_preds["CEEMDAN-ARIMA"] = forecast_imf_set(
                ceemdan_imfs,"ARIMA","CEEMDAN-ARIMA",**kw)
            decomp_preds["CEEMDAN-LSTM"] = forecast_imf_set(
                ceemdan_imfs,"LSTM","CEEMDAN-LSTM",**kw)
            progress("CEEMDAN done")
        except ImportError:
            print("  CEEMDAN skipped (pip install EMD-signal)")
        except Exception as e:
            print(f"  CEEMDAN FAILED: {e}")

    decomp_preds["Proposed"] = proposed_pred
    for name, pred in decomp_preds.items():
        decomp_metrics[name] = compute_metrics(
            y_true_test, pred, label=f"  {name}")
    progress("Decomposition metrics done")

    naive_rw         = gold_price[n_train-1:n_train-1+n_test]
    naive_rw_metrics = compute_metrics(y_true_test, naive_rw, label="  Naive(RW)")
    mean_test        = float(y_true_test.mean())

    # ── Tables ────────────────────────────────────────────────────────────
    cols = ["Model","RMSE","MAE","MAPE(%)","sMAPE(%)","DA(%)","DA_pval"]
    table7 = pd.DataFrame(single_metrics).T.reset_index()
    table7.columns = cols
    table7 = pd.concat([table7,
        pd.DataFrame([{"Model":"Naive(RW)",**naive_rw_metrics}])],
        ignore_index=True)
    table7.to_csv("../results/tables/table7_gold_single_model_errors.csv", index=False)
    print("\nTable 7:\n", table7.to_string(index=False))

    table8 = pd.DataFrame(decomp_metrics).T.reset_index()
    table8.columns = cols
    table8.to_csv("../results/tables/table8_gold_decomp_model_errors.csv", index=False)
    print("\nTable 8:\n", table8.to_string(index=False))

    print("\n"+"="*70)
    print(f"SUMMARY  mean={mean_test:,.0f}  CV={y_true_test.std()/mean_test*100:.1f}%")
    print("="*70)
    for name, m in {**single_metrics, **decomp_metrics}.items():
        if not np.isfinite(m.get("RMSE",np.nan)): continue
        pct = m["RMSE"]/mean_test*100
        sig = ("***" if m["DA_pval"]<0.01 else "**" if m["DA_pval"]<0.05
               else "*" if m["DA_pval"]<0.10 else "")
        print(f"  {name:<22} RMSE={m['RMSE']:>10,.0f} ({pct:4.1f}%)  "
              f"MAPE={m['MAPE(%)']:5.2f}%  DA={m['DA(%)']:5.1f}%{sig}")
    print("="*70)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nSaving figures…")
    colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22"]
    single_names = list(single_preds.keys())
    ncols, nrows = 2, (len(single_names)+1)//2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*4))
    axes = axes.flatten()
    for idx,(nm,col) in enumerate(zip(single_names, colors)):
        ax = axes[idx]
        ax.plot(test_dates, y_true_test, "#2c3e50", lw=0.9, label="Actual")
        ax.plot(test_dates, single_preds[nm], color=col, lw=0.8, ls="--", label=nm)
        m = single_metrics[nm]
        ax.set_title(f"{nm}  MAPE={m['MAPE(%)']:.2f}%  DA={m['DA(%)']}%",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    for j in range(len(single_names), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Single Model Forecasts", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../results/figures/fig9_gold_single_model_forecasts.png", dpi=300, bbox_inches="tight")
    plt.close()

    all_m  = {**single_metrics, **decomp_metrics}
    mnames = [k for k,v in all_m.items() if np.isfinite(v.get("RMSE",np.nan))]
    bar_colors = (["#3498db"]*len(single_metrics) +
                  ["#e74c3c"]*(len(decomp_metrics)-1) + ["#2ecc71"])
    fig, axes = plt.subplots(1,4,figsize=(18,5))
    for idx,metric in enumerate(["RMSE","MAE","MAPE(%)","sMAPE(%)"]):
        ax = axes[idx]
        vals = [all_m[k][metric] for k in mnames]
        ax.bar(range(len(mnames)), vals,
               color=bar_colors[:len(mnames)], edgecolor="white", lw=0.5)
        ax.set_xticks(range(len(mnames)))
        ax.set_xticklabels(mnames, rotation=45, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Error Metrics by Model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../results/figures/fig10_gold_error_barplots.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(test_dates, y_true_test, "#2c3e50", lw=1.2, label="Actual")
    if "VMD-ARIMA" in decomp_preds:
        ax.plot(test_dates, decomp_preds["VMD-ARIMA"], "#e74c3c",
                lw=0.9, ls="--", label="VMD-ARIMA")
    ax.plot(test_dates, proposed_pred, "#2ecc71", lw=0.9, ls="-.",
            label="Proposed")
    ax.set_title("Proposed vs VMD-ARIMA", fontsize=13, fontweight="bold")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("../results/figures/fig_proposed_vs_vmdarima.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: fig9, fig10, fig_proposed_vs_vmdarima")

    with open("../data/predictions_daily_gold.pkl","wb") as f:
        pickle.dump({
            "single_preds": single_preds, "single_metrics": single_metrics,
            "decomp_preds": decomp_preds, "decomp_metrics": decomp_metrics,
            "y_true_test": y_true_test, "y_true_train": y_true_train,
            "test_dates": test_dates, "proposed_pred": proposed_pred,
            "proposed_pred_train": proposed_pred_train,
            "n_train": n_train, "N_LAGS": N_LAGS,
            "naive_rw_metrics": naive_rw_metrics, "mean_test": mean_test,
            "ceemdan_available": ceemdan_imfs is not None,
            "imf_da_report": imf_da_report,
        }, f)
    print("Saved: predictions_daily_gold.pkl")
    print("\n"+"="*65)
    print("STEP 4 COMPLETE [v5]")
    print("NEXT: python step5_dmtest_daily_gold.py")
    progress("Done")


if __name__ == "__main__":
    main()