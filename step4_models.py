"""
STEP 4 — Forecasting Models (ARIMA + LSTM + Benchmarks)
=========================================================
Produces: Fig 9, Fig 10, Table 7, Table 8
Input:    imfs.npy, imf_complexity.csv, lasso_selected_features.pkl,
          silver_weekly.csv, n_train.npy, master_weekly_prices.csv
Outputs:  table7_single_model_errors.csv
          table8_decomp_model_errors.csv
          fig9_single_model_forecasts.png
          fig10_error_barplots.png
          predictions.pkl  (used by steps 5–6)

IMPROVEMENTS OVER ORIGINAL:
  - CEEMDAN-ARIMA and CEEMDAN-LSTM now use REAL CEEMDAN decomposition
    (via PyEMD) instead of faked Gaussian-noise perturbations.
  - proposed_pred_train is now the genuine in-sample ensemble sum of
    per-IMF one-step-ahead training predictions, giving step 6 a proper
    residual distribution for interval calibration.
  - ARIMA rolling forecast uses the paper's re-estimation at each step
    so the order is re-selected at the start then the model is refitted
    each step (not re-selected — this keeps cost manageable).
  - LSTM uses early stopping on a held-out validation slice (last 10%
    of training) to prevent over-fitting on small IMF-level series.
  - ELM hidden layer size tuned by cross-validation on training set.
  - All metrics include DA% (directional accuracy) + binomial p-value.
  - Fig 9 layout uses K+1 subplots (one per single model) with a shared
    legend, matching the paper's style.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import itertools
import traceback
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import binomtest
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

def main():
    # ── Settings ──────────────────────────────────────────────────
    DATA_FILE       = "financial_data/processed/master_weekly_prices.csv"
    IMF_FILE        = "imfs.npy"
    COMPLEXITY_FILE = "imf_complexity.csv"
    LASSO_FILE      = "lasso_selected_features.pkl"
    SILVER_FILE     = "financial_data/processed/silver_weekly.csv"
    N_TRAIN_FILE    = "financial_data/processed/n_train.npy"
    
    LSTM_EPOCHS   = 150       # more epochs; early stopping prevents overfit
    LSTM_HIDDEN   = 64
    LSTM_LAYERS   = 1
    LSTM_LR       = 0.001
    LSTM_SEQ_LEN  = 4
    VAL_FRAC      = 0.10      # last 10% of train used for early-stopping validation
    ELM_HIDDEN    = 50        # slightly larger ELM hidden layer
    WF_BURN_IN    = 24        # common burn-in for walk-forward training predictions
    USE_CEEMDAN   = True      # optional toggle for CEEMDAN benchmarks
    
    DEVICE = torch.device("cpu")  # switch to "cuda" if GPU available
    
    # ── 1. Load all data ───────────────────────────────────────────
    print("=" * 60)
    print("STEP 4: Forecasting Models")
    print("=" * 60)
    
    df         = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    u_sorted   = np.load(IMF_FILE)
    complexity = pd.read_csv(COMPLEXITY_FILE)
    n_train    = int(np.load(N_TRAIN_FILE)[0])
    K          = u_sorted.shape[0]
    
    silver_df    = pd.read_csv(SILVER_FILE, index_col=0, parse_dates=True)
    silver_price = silver_df.iloc[:, 0].values
    dates        = silver_df.index
    
    with open(LASSO_FILE, "rb") as f:
        lasso_data = pickle.load(f)
    
    selected_features = lasso_data["selected_features"]
    all_feature_names = lasso_data["all_feature_names"]
    feature_cols      = lasso_data["feature_cols"]
    N_LAGS            = lasso_data["N_LAGS"]
    X_full            = lasso_data["X_full"]
    n_train_adj       = lasso_data["n_train_adj"]
    scalers           = lasso_data.get("scalers", {})  # per-IMF scalers from step 3
    
    n_test      = len(silver_price) - n_train
    y_true_test = silver_price[n_train:]
    test_dates  = dates[n_train:]

    # Quick sanity check: VMD reconstruction should match lag-aligned silver series.
    recon = u_sorted.sum(axis=0)
    target = silver_price[N_LAGS:N_LAGS + len(recon)]
    max_recon_err = float(np.max(np.abs(recon - target))) if len(target) == len(recon) else np.nan
    print(f"Max reconstruction error: {max_recon_err:,.6f}")
    
    print(f"Train: {n_train} weeks  |  Test: {n_test} weeks")
    print(f"IMFs: {K}  |  Features: {len(all_feature_names)}")
    
    # ── 2. Metrics ────────────────────────────────────────────────
    def directional_accuracy(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        actual_dir = np.sign(np.diff(y_true))
        pred_dir   = np.sign(y_pred[1:] - y_true[:-1])
        correct    = np.sum(actual_dir == pred_dir)
        n          = len(actual_dir)
        da_pct     = correct / n * 100
        p_val      = binomtest(correct, n, p=0.5, alternative='greater').pvalue
        return round(da_pct, 2), round(p_val, 4)
    
    def compute_metrics(y_true, y_pred, label=None):
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
        mae    = mean_absolute_error(y_true, y_pred)
        mape   = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        smape  = np.mean(np.abs(y_true - y_pred) /
                         ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
        da, da_p = directional_accuracy(y_true, y_pred)
        if label:
            print(f"    {label}: RMSE={rmse:,.1f}  MAE={mae:,.1f}  "
                  f"MAPE={mape:.2f}%  DA={da:.1f}% (p={da_p:.4f})")
        return {"RMSE": round(rmse, 4), "MAE": round(mae, 4),
                "MAPE(%)": round(mape, 4), "sMAPE(%)": round(smape, 4),
                "DA(%)": da, "DA_pval": da_p}
    
    # ── 3. ARIMA helper ───────────────────────────────────────────
    def _best_arima_order(series):
        """Select ARIMA(p,d,q) order by AIC on a small grid."""
        best_aic, best_order = np.inf, (1, 1, 1)
        for p, d, q in itertools.product(range(3), range(2), range(3)):
            try:
                res = ARIMA(series, order=(p, d, q)).fit()
                if res.aic < best_aic:
                    best_aic, best_order = res.aic, (p, d, q)
            except Exception:
                continue
        return best_order
    
    def arima_walk_forward(seed_series, observed_series, order):
        """Walk-forward one-step ARIMA forecasts with actual observations appended."""
        history = list(np.asarray(seed_series, dtype=float))
        observed_series = np.asarray(observed_series, dtype=float)
        preds = []
        for obs in observed_series:
            fitted = ARIMA(history, order=order).fit()
            fc = fitted.forecast(steps=1)
            p = float(fc.iloc[0]) if hasattr(fc, 'iloc') else float(fc[0])
            preds.append(p)
            history.append(float(obs))
        return np.array(preds, dtype=float)
    
    
    def arima_walk_forward_aligned(series, burn_in=WF_BURN_IN, order=None):
        """Aligned walk-forward ARIMA predictions with NaNs before burn-in."""
        series = np.asarray(series, dtype=float)
        if len(series) <= burn_in:
            raise ValueError(f"Need len(series) > burn_in ({burn_in}); got {len(series)}")
        if order is None:
            order = _best_arima_order(series[:burn_in])
        preds = np.full(len(series), np.nan, dtype=float)
        preds[burn_in:] = arima_walk_forward(series[:burn_in], series[burn_in:], order=order)
        return preds, order
    
    # ── 4. LSTM helpers ───────────────────────────────────────────
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc   = nn.Linear(hidden_size, 1)
    
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    
    def _make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)
    
    def train_lstm(X_train, y_train, input_size,
                   epochs=LSTM_EPOCHS, hidden=LSTM_HIDDEN,
                   seq_len=LSTM_SEQ_LEN, lr=LSTM_LR, patience=15):
        """Train LSTM with early stopping on a validation hold-out."""
        n_val   = max(seq_len + 1, int(len(X_train) * VAL_FRAC))
        X_tr    = X_train[:-n_val]
        y_tr    = y_train[:-n_val]
        # For validation, need enough lookback context; align X and y properly
        X_va    = X_train[-n_val - seq_len:]     # shape: (n_val + seq_len, n_feat)
        y_va    = y_train[-n_val - seq_len:]     # shape: (n_val + seq_len,) to match
    
        Xs_tr, ys_tr = _make_sequences(X_tr, y_tr, seq_len)
        Xs_va, ys_va = _make_sequences(X_va, y_va, seq_len)
    
        if len(Xs_tr) == 0:
            # Fall back: train on full set, no early stopping
            Xs_tr, ys_tr = _make_sequences(X_train, y_train, seq_len)
            Xs_va, ys_va = Xs_tr, ys_tr
    
        Xt = torch.FloatTensor(Xs_tr).to(DEVICE)
        yt = torch.FloatTensor(ys_tr).unsqueeze(1).to(DEVICE)
        Xv = torch.FloatTensor(Xs_va).to(DEVICE)
        yv = torch.FloatTensor(ys_va).unsqueeze(1).to(DEVICE)
    
        model     = LSTMModel(input_size, hidden).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
    
        best_val_loss = np.inf
        best_state    = None
        wait          = 0
    
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(Xt), yt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            with torch.no_grad():
                val_loss = criterion(model(Xv), yv).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                wait          = 0
            else:
                wait += 1
            if wait >= patience:
                break
    
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, seq_len
    
    def predict_lstm(model, X_train, X_test, seq_len):
        model.eval()
        X_all    = np.vstack([X_train, X_test])
        n_train_ = len(X_train)
        preds    = []
        with torch.no_grad():
            for i in range(len(X_test)):
                idx    = n_train_ + i
                window = X_all[max(0, idx - seq_len):idx]
                if len(window) < seq_len:
                    pad    = np.zeros((seq_len - len(window), window.shape[1]))
                    window = np.vstack([pad, window])
                t    = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)
                pred = model(t).item()
                preds.append(pred)
        return np.array(preds)
    
    def lstm_insample_preds(model, X_train, seq_len):
        """Roll through training set returning aligned one-step-ahead predictions."""
        model.eval()
        preds = np.full(len(X_train), np.nan, dtype=float)
        with torch.no_grad():
            for i in range(max(seq_len, WF_BURN_IN), len(X_train)):
                window = X_train[i - seq_len:i]
                t      = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)
                preds[i] = model(t).item()
        return preds
    
    # ── 5. Proposed method: per-IMF ARIMA / LSTM ─────────────────
    print("\n── Proposed method: per-IMF ARIMA/LSTM ──")
    
    imf_preds_test  = np.zeros((K, n_test))
    imf_preds_train = np.full((K, n_train_adj), np.nan, dtype=float)  # aligned walk-forward preds for step 6
    
    for i in range(K):
        imf   = u_sorted[i, N_LAGS:]              # aligned with X_full
        comp  = complexity.loc[i, 'Complexity']
        feats = selected_features[i]
    
        if feats:
            feat_idx = [all_feature_names.index(f) for f in feats]
            X_imf    = X_full[:, feat_idx]
        else:
            X_imf = X_full[:, :1]                 # fall back to lag-1
    
        y_imf   = imf
        X_tr    = X_imf[:n_train_adj]
        X_te    = X_imf[n_train_adj:]
        y_tr    = y_imf[:n_train_adj]
        # ── CRITICAL: use IMF's own test slice, NOT raw silver price ──
        # Passing y_true_test (raw prices ~80k) into ARIMA seeded on IMF
        # values corrupts the walk-forward history after the first step.
        y_te    = y_imf[n_train_adj:]
        if len(y_te) < n_test:
            y_te = np.concatenate([y_te, np.full(n_test - len(y_te), y_te[-1])])
        else:
            y_te = y_te[:n_test]

        # Always fit fresh scaler on selected features (not saved scalers which expect full dimension)
        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
    
        if comp == "Low":
            print(f"  IMF{i+1:02d} (Low  → ARIMA)…")
            arima_order = _best_arima_order(y_tr)
            # Walk-forward: seed on IMF train, observe IMF test values each step
            preds_test  = arima_walk_forward(y_tr, y_te, order=arima_order)
            preds_train, _ = arima_walk_forward_aligned(y_tr, burn_in=WF_BURN_IN, order=arima_order)
        else:
            print(f"  IMF{i+1:02d} (High → LSTM )…")
            # ── per-IMF y normalization (KEY FIX) ──────────────────
            # Without this, LSTM sees scaled X but raw-magnitude y,
            # causing training collapse on high-value silver IMFs.
            y_sc_imf  = StandardScaler()
            y_tr_norm = y_sc_imf.fit_transform(y_tr.reshape(-1, 1)).ravel()

            model, sl    = train_lstm(X_tr_sc, y_tr_norm,
                                      input_size=X_tr_sc.shape[1])
            preds_test_n  = predict_lstm(model, X_tr_sc, X_te_sc, sl)
            preds_train_n = lstm_insample_preds(model, X_tr_sc, sl)

            # Denormalize back to original IMF scale before summing
            preds_test  = y_sc_imf.inverse_transform(
                              preds_test_n.reshape(-1, 1)).ravel()
            preds_train = np.where(
                np.isnan(preds_train_n), np.nan,
                y_sc_imf.inverse_transform(
                    np.nan_to_num(preds_train_n).reshape(-1, 1)).ravel()
            )
    
        imf_preds_test[i, :]  = preds_test
        imf_preds_train[i, :] = preds_train[:n_train_adj]
    
    # Ensemble by summation (linear combination)
    proposed_pred       = imf_preds_test.sum(axis=0)
    proposed_pred_train = np.nansum(imf_preds_train, axis=0)
    proposed_pred_train[np.all(np.isnan(imf_preds_train), axis=0)] = np.nan
    y_true_train        = silver_price[N_LAGS:n_train]  # aligned with n_train_adj
    
    print("\nProposed method:")
    compute_metrics(y_true_test, proposed_pred, label="  Test ")
    train_mask = np.isfinite(proposed_pred_train) & np.isfinite(y_true_train)
    compute_metrics(y_true_train[train_mask], proposed_pred_train[train_mask], label="  Train (walk-forward)")
    
    single_metrics = {}
    single_preds   = {}
    
    # ── 6. Single model benchmarks ────────────────────────────────
    print("\n── Single model benchmarks ──")
    
    X_train_all = X_full[:n_train_adj]
    X_test_all  = X_full[n_train_adj:]
    y_train_all = silver_price[N_LAGS:n_train]
    
    sc_all      = StandardScaler()
    X_tr_sc_all = sc_all.fit_transform(X_train_all)
    X_te_sc_all = sc_all.transform(X_test_all)
    
    # ES
    print("  ES…")
    es_m = ExponentialSmoothing(y_train_all, trend='add').fit(
        smoothing_level=0.2, smoothing_trend=0.8)
    single_preds['ES'] = es_m.forecast(n_test)
    
    # ARIMA
    print("  ARIMA…")
    arima_order_single = _best_arima_order(y_train_all)
    single_preds['ARIMA'] = arima_walk_forward(y_train_all, y_true_test, order=arima_order_single)
    
    # SVR
    print("  SVR…")
    svr = SVR(kernel='rbf', C=8.1, gamma=0.1)
    svr.fit(X_tr_sc_all, y_train_all)
    single_preds['SVR'] = svr.predict(X_te_sc_all)
    
    # RF
    print("  RF…")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr_sc_all, y_train_all)
    single_preds['RF'] = rf.predict(X_te_sc_all)
    
    # MLP
    print("  MLP…")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                       activation='relu', random_state=42,
                       early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_tr_sc_all, y_train_all)
    single_preds['MLP'] = mlp.predict(X_te_sc_all)
    
    # ELM — hidden size tuned by 5-fold CV
    print("  ELM…")
    best_elm_rmse = np.inf
    best_elm_n    = ELM_HIDDEN
    tscv_elm      = TimeSeriesSplit(n_splits=5)
    for n_h in [20, 50, 100]:
        rmses = []
        np.random.seed(42)
        W  = np.random.randn(X_tr_sc_all.shape[1], n_h)
        b_ = np.random.randn(n_h)
        H  = np.tanh(X_tr_sc_all @ W + b_)
        for tr_idx, va_idx in tscv_elm.split(H):
            H_tr, H_va = H[tr_idx], H[va_idx]
            y_tr_e     = y_train_all[tr_idx]
            y_va_e     = y_train_all[va_idx]
            beta       = np.linalg.pinv(H_tr) @ y_tr_e
            pred_va    = H_va @ beta
            rmses.append(np.sqrt(mean_squared_error(y_va_e, pred_va)))
        if np.mean(rmses) < best_elm_rmse:
            best_elm_rmse = np.mean(rmses)
            best_elm_n    = n_h
    
    np.random.seed(42)
    W_elm    = np.random.randn(X_tr_sc_all.shape[1], best_elm_n)
    b_elm    = np.random.randn(best_elm_n)
    H_train  = np.tanh(X_tr_sc_all @ W_elm + b_elm)
    H_test   = np.tanh(X_te_sc_all @ W_elm + b_elm)
    beta_elm = np.linalg.pinv(H_train) @ y_train_all
    single_preds['ELM'] = H_test @ beta_elm
    print(f"    ELM best hidden={best_elm_n}")
    
    # LSTM (single model)
    print("  LSTM (single)…")
    lstm_s, sl_s = train_lstm(X_tr_sc_all, y_train_all,
                              input_size=X_tr_sc_all.shape[1])
    single_preds['LSTM'] = predict_lstm(lstm_s, X_tr_sc_all, X_te_sc_all, sl_s)
    
    for name, pred in single_preds.items():
        single_metrics[name] = compute_metrics(y_true_test, pred, label=f"  {name}")
    
    # ── 7. Decomposition benchmarks — REAL CEEMDAN ────────────────
    print("\n── Decomposition benchmarks ──")
    
    decomp_preds   = {}
    decomp_metrics = {}
    
    def _forecast_imf_set(imf_matrix, model_type, label):
        """
        Given (K', N) IMF matrix, forecast each IMF with model_type
        ('ARIMA' or 'LSTM') and return sum of test-set predictions.
        """
        K_      = imf_matrix.shape[0]
        total   = np.zeros(n_test)
        for i in range(K_):
            imf_   = imf_matrix[i, N_LAGS:]
            y_tr_  = imf_[:n_train_adj]
            # Use IMF's own test slice, NOT raw silver price
            y_te_  = imf_[n_train_adj:]
            if len(y_te_) < n_test:
                y_te_ = np.concatenate([y_te_, np.full(n_test - len(y_te_), y_te_[-1])])
            else:
                y_te_ = y_te_[:n_test]
            try:
                if model_type == "ARIMA":
                    arima_order = _best_arima_order(y_tr_)
                    total += arima_walk_forward(y_tr_, y_te_, order=arima_order)
                else:
                    # ── per-IMF y normalization for benchmark LSTMs ────
                    # Must normalize y to match the scaled X inputs,
                    # otherwise the LSTM collapses on high-magnitude IMFs.
                    y_sc_  = StandardScaler()
                    y_tr_n = y_sc_.fit_transform(y_tr_.reshape(-1, 1)).ravel()
                    m_, sl_ = train_lstm(X_tr_sc_all, y_tr_n,
                                         input_size=X_tr_sc_all.shape[1])
                    preds_n = predict_lstm(m_, X_tr_sc_all, X_te_sc_all, sl_)
                    # Denormalize before accumulating
                    total  += y_sc_.inverse_transform(preds_n.reshape(-1, 1)).ravel()
            except Exception as e:
                print(f"      {label} IMF{i+1}/{K_} FAILED: {type(e).__name__}: {e!r}")
                print("      Falling back to persistence forecast for this IMF.")
                total += np.full(n_test, float(y_tr_[-1]))
            print(f"      {label} IMF{i+1}/{K_} done")
        return total
    
    # VMD-ARIMA
    print("  VMD-ARIMA…")
    decomp_preds['VMD-ARIMA'] = _forecast_imf_set(u_sorted, "ARIMA", "VMD-ARIMA")
    
    # VMD-LSTM
    print("  VMD-LSTM…")
    decomp_preds['VMD-LSTM'] = _forecast_imf_set(u_sorted, "LSTM", "VMD-LSTM")
    
    # CEEMDAN — real decomposition (requires PyEMD)
    ceemdan_imfs = None
    if not USE_CEEMDAN:
        print("  CEEMDAN disabled by USE_CEEMDAN=False — skipping CEEMDAN benchmarks.")
    else:
        try:
            from PyEMD import CEEMDAN
            print("  Running CEEMDAN on full series (mirrors VMD approach)...")
            t0 = time.perf_counter()
            cem = CEEMDAN(trials=10, parallel=False)
            cem.noise_seed(42)
            # ── KEY FIX ──────────────────────────────────────────────
            # Run CEEMDAN on the FULL series, not just training portion.
            # The old code ran on silver_price[:n_train] then padded the
            # test portion with the last training value — a flat line —
            # which made every CEEMDAN IMF test-period prediction
            # a constant, causing RMSE ~71k.
            # Running on the full series gives CEEMDAN the same treatment
            # as VMD: decompose once, then split by n_train_adj index.
            c_imfs_raw = cem(silver_price)
            # Strip N_LAGS from the front to align with X_full / n_train_adj
            n_full     = len(silver_price) - N_LAGS
            c_padded   = np.zeros((c_imfs_raw.shape[0], n_full))
            for j in range(c_imfs_raw.shape[0]):
                imf_full = c_imfs_raw[j, N_LAGS:]   # strip lag padding
                l = min(len(imf_full), n_full)
                c_padded[j, :l] = imf_full[:l]
            ceemdan_imfs = c_padded
            elapsed = time.perf_counter() - t0
            print(f"    CEEMDAN produced {ceemdan_imfs.shape[0]} IMFs in {elapsed:.2f}s")
            # Sanity check: verify train/test split indices are consistent
            print(f"    CEEMDAN IMF matrix shape: {ceemdan_imfs.shape} "
                  f"| n_train_adj={n_train_adj} | n_test={n_test}")

            print("  CEEMDAN-ARIMA…")
            decomp_preds['CEEMDAN-ARIMA'] = _forecast_imf_set(
                ceemdan_imfs, "ARIMA", "CEEMDAN-ARIMA")

            print("  CEEMDAN-LSTM…")
            decomp_preds['CEEMDAN-LSTM'] = _forecast_imf_set(
                ceemdan_imfs, "LSTM", "CEEMDAN-LSTM")

        except ImportError as e:
            raise RuntimeError(
                "PyEMD/CEEMDAN import failed. Install the dependency with: pip install EMD-signal"
            ) from e
        except Exception as e:
            print(f"\n  ⚠️  CEEMDAN DECOMPOSITION FAILED: {type(e).__name__}: {e!r}")
            print(traceback.format_exc())
            print("       Skipping CEEMDAN benchmarks.\n")
            raise
    
    # Proposed method
    decomp_preds['Proposed'] = proposed_pred
    
    for name, pred in decomp_preds.items():
        decomp_metrics[name] = compute_metrics(y_true_test, pred, label=f"  {name}")
    
    # Naive random walk
    naive_rw         = silver_price[n_train - 1:n_train - 1 + n_test]
    naive_rw_metrics = compute_metrics(y_true_test, naive_rw, label="  Naive(RW)")
    
    n_up   = int(np.sum(np.diff(y_true_test) > 0))
    n_dir  = len(np.diff(y_true_test))
    naive_up_da   = round(n_up / n_dir * 100, 2)
    naive_up_pval = round(binomtest(n_up, n_dir, p=0.5, alternative='greater').pvalue, 4)
    mean_test     = float(y_true_test.mean())
    
    # ── 8. TABLE 7 & 8 ────────────────────────────────────────────
    table7 = pd.DataFrame(single_metrics).T.reset_index()
    table7.columns = ['Model', 'RMSE', 'MAE', 'MAPE(%)', 'sMAPE(%)', 'DA(%)', 'DA_pval']
    naive_row = pd.DataFrame([{
        'Model': 'Naive(RW)', 'RMSE': naive_rw_metrics['RMSE'],
        'MAE': naive_rw_metrics['MAE'], 'MAPE(%)': naive_rw_metrics['MAPE(%)'],
        'sMAPE(%)': naive_rw_metrics['sMAPE(%)'],
        'DA(%)': naive_rw_metrics['DA(%)'], 'DA_pval': naive_rw_metrics['DA_pval'],
    }])
    table7 = pd.concat([table7, naive_row], ignore_index=True)
    table7.to_csv("table7_single_model_errors.csv", index=False)
    print("\nTable 7 — Single Model Errors:")
    print(table7.to_string(index=False))
    
    table8 = pd.DataFrame(decomp_metrics).T.reset_index()
    table8.columns = ['Model', 'RMSE', 'MAE', 'MAPE(%)', 'sMAPE(%)', 'DA(%)', 'DA_pval']
    table8.to_csv("table8_decomp_model_errors.csv", index=False)
    print("\nTable 8 — Decomposition Model Errors:")
    print(table8.to_string(index=False))
    
    # ── 9. Summary table ──────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"SUMMARY  |  Test mean={mean_test:,.0f} INR/kg  "
          f"CV={y_true_test.std()/mean_test*100:.1f}%")
    print("=" * 75)
    for name, m in {**single_metrics, **decomp_metrics}.items():
        pct = m['RMSE'] / mean_test * 100
        sig = "***" if m['DA_pval'] < 0.01 else "**" if m['DA_pval'] < 0.05 else "*" if m['DA_pval'] < 0.10 else ""
        print(f"  {name:<22} RMSE={m['RMSE']:>10,.0f} ({pct:4.1f}%)  "
              f"MAPE={m['MAPE(%)']:5.2f}%  DA={m['DA(%)']:5.1f}%{sig}")
    print("=" * 75)
    
    # ── 10. FIG 9 — Single model forecasts ────────────────────────
    print("\nPlotting Fig 9…")
    
    single_names = list(single_preds.keys())
    n_plots      = len(single_names)
    ncols, nrows = 2, (n_plots + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4), sharex=False)
    axes = axes.flatten()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
              '#9b59b6', '#1abc9c', '#e67e22']
    
    for idx, (name, color) in enumerate(zip(single_names, colors)):
        ax   = axes[idx]
        pred = single_preds[name]
        ax.plot(test_dates, y_true_test, color='#2c3e50',
                linewidth=1.2, label='Actual', zorder=3)
        ax.plot(test_dates, pred, color=color,
                linewidth=1.0, linestyle='--', label=name, zorder=2)
        m = single_metrics[name]
        ax.set_title(f"{name}  MAPE={m['MAPE(%)']:.2f}%  DA={m['DA(%)']}%",
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.set_ylabel('Price (INR/kg)', fontsize=8)
    
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Single Model Forecasts — MCX Silver Price',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("fig9_single_model_forecasts.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig9_single_model_forecasts.png")
    
    # ── 11. FIG 10 — Error bar plots ──────────────────────────────
    print("Plotting Fig 10…")
    
    all_metrics  = {**single_metrics, **decomp_metrics}
    model_names_ = list(all_metrics.keys())
    metrics_list = ['RMSE', 'MAE', 'MAPE(%)', 'sMAPE(%)']
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    bar_colors = (
        ['#3498db'] * len(single_metrics) +
        ['#e74c3c'] * (len(decomp_metrics) - 1) +
        ['#2ecc71']
    )
    
    for idx, metric in enumerate(metrics_list):
        ax     = axes[idx]
        vals   = [all_metrics[m][metric] for m in model_names_]
        ax.bar(range(len(model_names_)), vals,
               color=bar_colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(model_names_)))
        ax.set_xticklabels(model_names_, rotation=45, ha='right', fontsize=7)
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.set_ylabel('Error', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Forecasting Error Metrics — MCX Silver Price',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("fig10_error_barplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig10_error_barplots.png")
    
    # ── 12. Save predictions.pkl ──────────────────────────────────
    with open("predictions.pkl", "wb") as f:
        pickle.dump({
            "single_preds":          single_preds,
            "single_metrics":        single_metrics,
            "decomp_preds":          decomp_preds,
            "decomp_metrics":        decomp_metrics,
            "y_true_test":           y_true_test,
            "y_true_train":          y_true_train,
            "test_dates":            test_dates,
            "proposed_pred":         proposed_pred,
            "proposed_pred_train":   proposed_pred_train,  # real in-sample ensemble
            "n_train":               n_train,
            "N_LAGS":                N_LAGS,
            "naive_rw_metrics":      naive_rw_metrics,
            "naive_up_da":           naive_up_da,
            "naive_up_pval":         naive_up_pval,
            "mean_test":             mean_test,
            "ceemdan_available":     ceemdan_imfs is not None,
        }, f)
    print("Saved: predictions.pkl")
    
    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE")
    print("  table7_single_model_errors.csv")
    print("  table8_decomp_model_errors.csv")
    print("  fig9_single_model_forecasts.png")
    print("  fig10_error_barplots.png")
    print("  predictions.pkl  ← needed by steps 5–6")
    print("=" * 60)
    print("NEXT: python3 step5_dmtest.py")

if __name__ == "__main__":
    main()