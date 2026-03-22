#!/usr/bin/env python3
"""
Walk-forward out-of-sample validation.

After the parameter sweep finds the best configuration, this script
validates it on held-out data (the last 2 months) to confirm the
strategy generalises beyond the optimisation period.

Usage:
    conda run -n vbtpro python backtest/walk_forward.py
    conda run -n vbtpro python backtest/walk_forward.py --oos-months 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbtpro as vbt

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.sweep import compute_metrics
from bot.indicators import compute_signals

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

FEES = 0.001
INIT_CASH = 1_000_000
POSITION_SIZE_PCT = 0.30


def _strategy_params_from_cfg(cfg: dict) -> dict:
    s = cfg["strategy"]
    if s == "ema_crossover":
        return {"fast_window": cfg["fast_window"], "slow_window": cfg["slow_window"]}
    elif s == "rsi":
        return {"window": cfg["window"], "lower": cfg["lower"], "upper": cfg["upper"]}
    elif s == "atr_trend":
        return {"ma_len": cfg["ma_len"], "atr_len": cfg["atr_len"], "atr_mult": cfg["atr_mult"]}
    elif s == "rsi_ema_filter":
        return {"ema_len": cfg["ema_len"], "rsi_period": cfg["rsi_period"],
                "oversold": cfg["oversold"], "overbought": cfg["overbought"]}
    raise ValueError(f"Unknown strategy: {s}")


def run_period(cfg: dict, data: vbt.Data, start=None, end=None, label="") -> dict:
    """Run a single backtest on a (optionally sliced) data period."""
    close = data.get("Close")
    high = data.get("High")
    low = data.get("Low")

    if start is not None or end is not None:
        close = close.loc[start:end]
        high = high.loc[start:end]
        low = low.loc[start:end]

    params = _strategy_params_from_cfg(cfg)
    entries, exits = compute_signals(cfg["strategy"], close, high, low, **params)

    pf_kwargs = dict(
        close=close,
        entries=entries,
        exits=exits,
        fees=FEES,
        init_cash=INIT_CASH,
        size=POSITION_SIZE_PCT,
        size_type="percent",
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
    )
    if cfg.get("sl_stop"):
        pf_kwargs["sl_stop"] = cfg["sl_stop"]
    if cfg.get("tp_stop"):
        pf_kwargs["tp_stop"] = cfg["tp_stop"]

    pf = vbt.Portfolio.from_signals(**pf_kwargs)
    m = compute_metrics(pf, cfg["timeframe"])
    m["period"] = label
    m["start"] = str(close.index[0].date())
    m["end"] = str(close.index[-1].date())
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos-months", type=int, default=2,
                        help="Number of months to reserve for out-of-sample (default: 2)")
    args = parser.parse_args()

    best_path = RESULTS_DIR / "best_params.json"
    if not best_path.exists():
        print("ERROR: results/best_params.json not found.")
        print("Run backtest/sweep.py first to generate the best config.")
        sys.exit(1)

    cfg = json.loads(best_path.read_text())
    tf = cfg["timeframe"]
    print(f"\nWalk-forward validation for: {cfg['strategy']} / {tf}")
    print(f"OOS period: last {args.oos_months} month(s)\n")

    data_path = DATA_DIR / f"binance_{tf}.h5"
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}. Run fetch_data.py first.")
        sys.exit(1)

    data = vbt.HDFData.pull(str(data_path))
    close = data.get("Close")

    # Split date
    split_date = close.index[-1] - pd.DateOffset(months=args.oos_months)

    is_start = close.index[0]
    is_end = split_date
    oos_start = split_date
    oos_end = close.index[-1]

    print(f"In-sample  : {is_start.date()} → {is_end.date()}")
    print(f"Out-of-sample: {oos_start.date()} → {oos_end.date()}\n")

    is_metrics = run_period(cfg, data, end=is_end, label="in-sample")
    oos_metrics = run_period(cfg, data, start=oos_start, label="out-of-sample")

    # Display comparison table
    cols = ["composite", "sortino", "sharpe", "calmar",
            "total_return", "max_drawdown", "trades_per_day", "n_trades"]

    rows = []
    for m in [is_metrics, oos_metrics]:
        rows.append({
            "period": f"{m['period']} ({m['start']} to {m['end']})",
            **{c: m[c] for c in cols},
        })

    df = pd.DataFrame(rows).set_index("period")
    df["total_return"] = df["total_return"].map("{:.1%}".format)
    df["max_drawdown"] = df["max_drawdown"].map("{:.1%}".format)
    df["trades_per_day"] = df["trades_per_day"].map("{:.2f}".format)
    for col in ["composite", "sortino", "sharpe", "calmar"]:
        df[col] = df[col].map("{:.3f}".format)

    print("=" * 80)
    print("WALK-FORWARD RESULTS")
    print("=" * 80)
    print(df.to_string())
    print()

    # Verdict
    is_c = float(is_metrics["composite"])
    oos_c = float(oos_metrics["composite"])
    ratio = oos_c / is_c if is_c > 0 else 0

    print(f"OOS/IS composite ratio: {ratio:.2f}")
    if ratio >= 0.7:
        print("VERDICT: Strategy generalises well (OOS >= 70% of IS performance).")
    elif ratio >= 0.5:
        print("VERDICT: Acceptable. Some overfitting — monitor live performance closely.")
    else:
        print("VERDICT: Poor generalisation. Consider relaxing sweep parameters.")

    # Save results
    out_path = RESULTS_DIR / "walk_forward.json"
    out_path.write_text(json.dumps({
        "config": cfg,
        "in_sample": is_metrics,
        "out_of_sample": oos_metrics,
        "oos_is_ratio": ratio,
    }, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
