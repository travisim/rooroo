#!/usr/bin/env python3
"""
Risk management comparison for vec_rsi_regime 4h.

Tests four sizing schemes on the same strategy + params:
  A. Fixed 25% per trade
  B. Fixed 30% per trade  (backtest baseline)
  C. Fixed 40% per trade
  D. Dynamic inverse-ATR  (what the live bot uses)

Dynamic sizing formula (per bar):
    size_pct = min(TARGET_RISK_PCT / (ATR_SL_MULT × ATR%), MAX_POSITION_PCT)

Usage:
    /opt/anaconda3/envs/vectorbtpro/bin/python backtest/risk_comparison.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbtpro as vbt

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.sweep import compute_metrics, _strategy_params
from bot.indicators import compute_signals, atr as compute_atr

DATA_DIR   = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

FEES       = 0.001
INIT_CASH  = 1_000_000

# Dynamic sizing params (mirror config.py)
TARGET_RISK_PCT  = 0.02
ATR_SL_MULT      = 2.0
ATR_PERIOD       = 14
MAX_POSITION_PCT = 0.40


def dynamic_size_array(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """
    Per-bar position fraction for each symbol.
        size_pct = min(TARGET_RISK_PCT / (ATR_SL_MULT × ATR%), MAX_POSITION_PCT)
    Returns a DataFrame with the same shape as close.
    """
    atr_vals = compute_atr(high, low, close, ATR_PERIOD)
    atr_pct  = atr_vals / close.replace(0, np.nan)

    raw = TARGET_RISK_PCT / (ATR_SL_MULT * atr_pct)
    clipped = raw.clip(upper=MAX_POSITION_PCT)
    # Fill NaN (first ATR_PERIOD bars) with the cap
    return clipped.fillna(MAX_POSITION_PCT)


def run_variant(label: str, cfg: dict, data: vbt.Data, size, size_type="percent") -> dict:
    close  = data.get("Close")
    high   = data.get("High")
    low    = data.get("Low")

    params   = _strategy_params(cfg)
    entries, exits = compute_signals(cfg["strategy"], close, high, low, **params)

    pf_kwargs = dict(
        close=close, entries=entries, exits=exits,
        fees=FEES, init_cash=INIT_CASH,
        size=size, size_type=size_type,
        group_by=True, cash_sharing=True, call_seq="default",
    )
    if cfg.get("sl_stop"):
        pf_kwargs["sl_stop"] = cfg["sl_stop"]

    pf = vbt.Portfolio.from_signals(**pf_kwargs)
    m  = compute_metrics(pf, cfg["timeframe"])
    m["label"] = label
    return m


def main():
    best_path = RESULTS_DIR / "best_params.json"
    cfg = json.loads(best_path.read_text())
    tf  = cfg["timeframe"]   # "4h"

    print(f"\nRisk management comparison: {cfg['strategy']} / {tf}")
    print(f"sl_stop={cfg.get('sl_stop')} | params: length={cfg['length']} mult={cfg['mult']} "
          f"rsi={cfg['rsi_lower']}/{cfg['rsi_upper']} thr={cfg['atr_ratio_threshold']}\n")

    data_path = DATA_DIR / f"binance_{tf}.h5"
    data = vbt.HDFData.pull(str(data_path))

    close = data.get("Close")
    high  = data.get("High")
    low   = data.get("Low")

    # Precompute dynamic sizes
    dyn_size = dynamic_size_array(close, high, low)

    variants = [
        ("A. Fixed 25%",        0.25,     "percent"),
        ("B. Fixed 30% (baseline)", 0.30, "percent"),
        ("C. Fixed 40%",        0.40,     "percent"),
        ("D. Dynamic ATR",      dyn_size, "percent"),
    ]

    results = []
    for label, size, size_type in variants:
        print(f"Running {label}...", end="  ", flush=True)
        m = run_variant(label, cfg, data, size, size_type)
        results.append(m)
        print(f"C={m['composite']:.3f}  So={m['sortino']:.3f}  Sh={m['sharpe']:.3f}  "
              f"Ca={m['calmar']:.3f}  t/d={m['trades_per_day']:.2f}  "
              f"dd={m['max_drawdown']*100:.1f}%  ret={m['total_return']*100:.1f}%")

    # Summary table
    cols = ["label", "composite", "sortino", "sharpe", "calmar",
            "trades_per_day", "max_drawdown", "total_return", "n_trades"]
    df = pd.DataFrame(results)[cols].set_index("label")
    df["max_drawdown"] = df["max_drawdown"].map("{:.1%}".format)
    df["total_return"] = df["total_return"].map("{:.1%}".format)
    df["trades_per_day"] = df["trades_per_day"].map("{:.3f}".format)
    for c in ["composite", "sortino", "sharpe", "calmar"]:
        df[c] = df[c].map("{:.4f}".format)

    print("\n" + "=" * 90)
    print("RISK MANAGEMENT COMPARISON — vec_rsi_regime / 4h")
    print("=" * 90)
    print(df.to_string())

    # Recommendation
    raw = pd.DataFrame(results)
    best = raw.loc[raw["composite"].idxmax()]
    print(f"\nBest composite: {best['label']}  →  C={best['composite']:.4f}")

    # Save
    out = RESULTS_DIR / "risk_comparison.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
