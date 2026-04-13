"""
STEP 7 (daily) — Sliding-Window Trading Robustness (Gold)

Uses saved Step 6 per-day proposed Scheme 1' series (no retraining).
Window: 90 days, stride: 30 days.
"""

import os
import numpy as np
import pandas as pd

WINDOW_DAYS = 90
STEP_DAYS = 30
INPUT_CSV = "../results/tables/proposed_daily_series.csv"
OUTPUT_CSV = "../results/tables/sliding_window_trading.csv"
ANNUAL_FACTOR = 252


def _window_metrics(df_w: pd.DataFrame) -> dict:
    pnl = df_w["pnl"].to_numpy(dtype=float)
    sig = df_w["signal_scheme_1prime"].to_numpy(dtype=float)
    blocked = df_w["blocked"].to_numpy(dtype=int)

    equity = np.cumprod(1.0 + pnl)
    cum_ret = float((equity[-1] - 1.0) * 100.0)

    pnl_std = float(np.std(pnl))
    sharpe = float(np.mean(pnl) / pnl_std * np.sqrt(ANNUAL_FACTOR)) if pnl_std > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    max_dd = float(np.min(dd) * 100.0)

    prev = np.concatenate([[0.0], sig[:-1]])
    n_trades = int(np.sum(np.abs(sig - prev) > 0))

    blocked_days = int(blocked.sum())
    total_days = int(len(df_w))

    active = np.abs(sig) > 0
    n_active = int(active.sum())
    n_correct = int(((sig * df_w["actual_ret"].to_numpy(dtype=float)) > 0)[active].sum()) if n_active > 0 else 0
    hit_rate = float(n_correct / n_active * 100.0) if n_active > 0 else float("nan")

    return {
        "cumulative_return_pct": round(cum_ret, 4),
        "annualized_sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "n_trades": n_trades,
        "blocked_days": blocked_days,
        "total_days": total_days,
        "blocked_ratio_pct": round(blocked_days / total_days * 100.0, 2),
        "hit_rate_active_pct": round(hit_rate, 2) if np.isfinite(hit_rate) else float("nan"),
    }


def main() -> None:
    print("=" * 72)
    print("STEP 7 (daily): Sliding-Window Trading Robustness — Gold")
    print("=" * 72)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Missing input: {INPUT_CSV}. Run Step 6 first to generate per-day series."
        )

    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < WINDOW_DAYS:
        raise ValueError(f"Not enough rows for window={WINDOW_DAYS}. Found n={n}.")

    rows = []
    window_id = 1
    for start in range(0, n - WINDOW_DAYS + 1, STEP_DAYS):
        end = start + WINDOW_DAYS
        w = df.iloc[start:end].copy()
        m = _window_metrics(w)
        m.update({
            "asset": "gold",
            "window_id": window_id,
            "start_date": w["date"].iloc[0].date().isoformat(),
            "end_date": w["date"].iloc[-1].date().isoformat(),
        })
        rows.append(m)
        window_id += 1

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Windows: {len(out)}  |  Window={WINDOW_DAYS}  Step={STEP_DAYS}")

    print("\nWindow-level results:")
    print(out.to_string(index=False))

    print("\nRobustness summary (mean ± std):")
    for col in [
        "cumulative_return_pct",
        "annualized_sharpe",
        "max_drawdown_pct",
        "blocked_ratio_pct",
        "hit_rate_active_pct",
    ]:
        s = out[col].astype(float)
        print(f"  {col}: {s.mean():.4f} ± {s.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
