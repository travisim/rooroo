#!/usr/bin/env python3
"""
Parameter sweep — the "ralph loop".

Iterates all strategy × timeframe × parameter combinations and evaluates
each using the competition composite score:
    composite = 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar

Stops early ("ralph loop break") when:
    composite > COMPOSITE_TARGET
    AND trades_per_day >= MIN_TRADES_PER_DAY
    AND max_drawdown < MAX_DRAWDOWN_PCT

Usage (in vbtpro conda env):
    conda run -n vbtpro python backtest/sweep.py [--timeframes 1h 4h] [--no-stop]

Results saved to results/sweep_results.csv and results/best_params.json.
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbtpro as vbt

sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.indicators import compute_signals

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Ralph loop stop conditions ────────────────────────────────────────────────
COMPOSITE_TARGET = 1.5
MIN_TRADES_PER_DAY = 0.9        # Slightly under 1 to allow for low-signal days
MAX_DRAWDOWN_PCT = 0.25         # 25% max drawdown

FEES = 0.001                    # 0.1% taker (conservative)
INIT_CASH = 1_000_000
POSITION_SIZE_PCT = 0.30        # 30% of portfolio per asset per signal

# ── Parameter grid ────────────────────────────────────────────────────────────
# Each entry: (strategy_name, timeframe, param_dict, sl_stop, tp_stop)
def build_sweep_configs(timeframes):
    configs = []

    for tf in timeframes:
        # ── EMA crossover ──────────────────────────────────────────────────
        for fw, sw, sl, tp in product(
            [8, 9, 12, 20],
            [21, 26, 50],
            [0.03, 0.05, 0.08],
            [0.06, 0.10, 0.15, None],
        ):
            if sw <= fw:
                continue
            configs.append({
                "strategy": "ema_crossover",
                "timeframe": tf,
                "fast_window": fw, "slow_window": sw,
                "sl_stop": sl, "tp_stop": tp,
            })

        # ── RSI ────────────────────────────────────────────────────────────
        for window, lower, upper, sl in product(
            [14, 21],
            [30, 35, 40],
            [60, 65, 70],
            [0.03, 0.05, 0.08],
        ):
            configs.append({
                "strategy": "rsi",
                "timeframe": tf,
                "window": window, "lower": lower, "upper": upper,
                "sl_stop": sl, "tp_stop": None,
            })

        # ── ATR Trend (mirrors Pine Script) ───────────────────────────────
        for ma_len, atr_mult, atr_len in product(
            [50, 100, 200],
            [2.0, 3.0, 4.0],
            [7, 14],
        ):
            configs.append({
                "strategy": "atr_trend",
                "timeframe": tf,
                "ma_len": ma_len, "atr_len": atr_len, "atr_mult": atr_mult,
                "sl_stop": 0.05, "tp_stop": None,
            })

        # ── RSI + EMA filter ───────────────────────────────────────────────
        for ema_len, rsi_p, lower, upper in product(
            [50, 100, 200],
            [14, 21],
            [30, 35, 40],
            [60, 65, 70],
        ):
            configs.append({
                "strategy": "rsi_ema_filter",
                "timeframe": tf,
                "ema_len": ema_len, "rsi_period": rsi_p,
                "oversold": lower, "overbought": upper,
                "sl_stop": 0.05, "tp_stop": None,
            })

    return configs


def _strategy_params(cfg: dict) -> dict:
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


ANN_FACTORS = {     # Periods per year for annualization
    "1m":  525_600,
    "5m":  105_120,
    "15m":  35_040,
    "30m":  17_520,
    "1h":    8_760,
    "4h":    2_190,
    "1d":      365,
}


def safe_float(val) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def compute_metrics(pf: vbt.Portfolio, timeframe: str) -> dict:
    """
    Manually compute Sharpe, Sortino, Calmar to avoid vbt/pandas 3.0 freq issues.
    Uses pf.returns (per-period portfolio returns).
    """
    ann = ANN_FACTORS.get(timeframe, 365)

    try:
        rets = pf.returns
        if hasattr(rets, "columns"):
            rets = rets.iloc[:, 0]  # take first group column if grouped
    except Exception:
        return {"sortino": 0, "sharpe": 0, "calmar": 0, "composite": 0,
                "max_drawdown": 1, "total_return": 0, "trades_per_day": 0, "n_trades": 0}

    mean_ret = rets.mean()
    std_ret = rets.std()
    neg_rets = rets[rets < 0]
    downside_std = neg_rets.std() if len(neg_rets) > 0 else np.nan

    sharpe = (mean_ret / std_ret * np.sqrt(ann)) if std_ret > 0 else 0
    sortino = (mean_ret / downside_std * np.sqrt(ann)) if (downside_std and downside_std > 0) else 0

    total_return = safe_float(pf.total_return)
    n_periods = len(rets)
    years = n_periods / ann
    cagr = ((1 + total_return) ** (1 / max(years, 1e-9)) - 1) if total_return > -1 else -1

    try:
        cum = (1 + rets).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = abs(float(drawdown.min()))
    except Exception:
        max_dd = 1.0

    calmar = (cagr / max_dd) if max_dd > 1e-6 else 0
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    try:
        n_trades = int(pf.trades.count())
    except Exception:
        n_trades = 0

    close = pf.close
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    n_days = max((close.index[-1] - close.index[0]).days, 1)
    trades_per_day = n_trades / n_days

    return {
        "sortino": safe_float(sortino),
        "sharpe": safe_float(sharpe),
        "calmar": safe_float(calmar),
        "composite": safe_float(composite),
        "max_drawdown": safe_float(max_dd),
        "total_return": total_return,
        "trades_per_day": trades_per_day,
        "n_trades": n_trades,
    }


def run_single(cfg: dict, data: vbt.Data) -> dict:
    """Run one backtest configuration. Returns metrics dict."""
    close = data.get("Close")
    high = data.get("High")
    low = data.get("Low")
    params = _strategy_params(cfg)

    # Compute signals (multi-column DataFrame: BTCUSDT / ETHUSDT / SOLUSDT)
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
    metrics = compute_metrics(pf, cfg["timeframe"])
    return {**cfg, **metrics}


def run_sweep(timeframes: list[str], stop_on_target: bool = True):
    configs = build_sweep_configs(timeframes)
    print(f"\n{'='*65}")
    print(f"RALPH LOOP — sweeping {len(configs)} configurations")
    print(f"Target : composite > {COMPOSITE_TARGET}")
    print(f"         trades/day >= {MIN_TRADES_PER_DAY}")
    print(f"         max_drawdown < {MAX_DRAWDOWN_PCT*100:.0f}%")
    print(f"{'='*65}\n")

    results = []
    best = {"composite": -np.inf, "config": None}
    loaded: dict[str, vbt.Data] = {}

    csv_path = RESULTS_DIR / "sweep_results.csv"
    # Write header immediately
    pd.DataFrame(columns=[
        "strategy", "timeframe", "composite", "sortino", "sharpe", "calmar",
        "trades_per_day", "max_drawdown", "total_return", "n_trades",
    ]).to_csv(csv_path, index=False)

    for i, cfg in enumerate(configs):
        tf = cfg["timeframe"]

        # Lazy-load data per timeframe
        if tf not in loaded:
            path = DATA_DIR / f"binance_{tf}.h5"
            if not path.exists():
                if i == 0:
                    print(f"[WARN] Data not found for {tf}. Run fetch_data.py first.")
                continue
            print(f"Loading {tf} data...", end=" ", flush=True)
            loaded[tf] = vbt.HDFData.pull(str(path))
            print("OK")

        t0 = time.time()
        try:
            metrics = run_single(cfg, loaded[tf])
        except Exception as e:
            print(f"[{i+1:>5}/{len(configs)}] SKIP {cfg['strategy']}/{tf}: {e}")
            continue
        elapsed = time.time() - t0

        results.append(metrics)

        # Progress line
        c = metrics["composite"]
        flag = " *** NEW BEST ***" if c > best["composite"] and metrics["trades_per_day"] >= MIN_TRADES_PER_DAY else ""
        print(
            f"[{i+1:>5}/{len(configs)}] {cfg['strategy']:<15} {tf:<4} "
            f"C={c:>6.3f} (So={metrics['sortino']:.2f} Sh={metrics['sharpe']:.2f} Ca={metrics['calmar']:.2f}) "
            f"dd={metrics['max_drawdown']*100:.1f}% t/d={metrics['trades_per_day']:.2f} "
            f"ret={metrics['total_return']*100:.1f}% ({elapsed:.1f}s){flag}"
        )

        # Track best (must meet trade frequency AND drawdown criteria)
        meets_criteria = (
            metrics["trades_per_day"] >= MIN_TRADES_PER_DAY
            and metrics["max_drawdown"] < MAX_DRAWDOWN_PCT
        )
        if meets_criteria and c > best["composite"]:
            best = {"composite": c, "config": metrics}

        # Append to CSV incrementally
        pd.DataFrame([metrics]).to_csv(csv_path, mode="a", header=False, index=False)

        # Ralph loop: stop early if target met
        if stop_on_target and meets_criteria and c > COMPOSITE_TARGET:
            print(f"\n{'='*65}")
            print(f"RALPH LOOP TARGET REACHED at config {i+1}/{len(configs)}")
            print(f"Composite = {c:.4f} > {COMPOSITE_TARGET}")
            print(f"{'='*65}")
            break

    # ── Final report ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path).sort_values("composite", ascending=False)
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*65}")
    print("TOP 10 CONFIGURATIONS")
    print("=" * 65)
    cols = ["strategy", "timeframe", "composite", "sortino", "sharpe",
            "calmar", "trades_per_day", "max_drawdown", "total_return"]
    print(df.head(10)[cols].to_string(index=False))

    if best["config"]:
        best_path = RESULTS_DIR / "best_params.json"
        best_path.write_text(json.dumps(best["config"], indent=2, default=str))
        print(f"\nBest params saved to {best_path}")
        print(f"\n>>> Update config.py with these settings and redeploy the bot <<<\n")
        print(json.dumps(best["config"], indent=2, default=str))
    else:
        print("\nNo configuration met all criteria. Consider relaxing thresholds.")

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframes", nargs="+", default=["1h", "4h"],
                        help="Timeframes to sweep (default: 1h 4h)")
    parser.add_argument("--all-timeframes", action="store_true",
                        help="Sweep all timeframes: 15m 30m 1h 4h 1d")
    parser.add_argument("--no-stop", action="store_true",
                        help="Run full sweep without early stopping")
    args = parser.parse_args()

    tfs = ["15m", "30m", "1h", "4h", "1d"] if args.all_timeframes else args.timeframes
    run_sweep(tfs, stop_on_target=not args.no_stop)


if __name__ == "__main__":
    main()
