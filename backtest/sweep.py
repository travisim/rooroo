#!/usr/bin/env python3
"""
Parameter sweep — the "ralph loop".

Iterates strategy × timeframe × parameter combinations and evaluates each
using the competition composite score:
    composite = 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar

Stops early ("ralph loop break") when:
    composite > COMPOSITE_TARGET
    AND trades_per_day >= MIN_TRADES_PER_DAY
    AND max_drawdown < MAX_DRAWDOWN_PCT

Usage (vectorbtpro conda env):
    /opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py
    /opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py --timeframes 1h 4h --no-stop
    /opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py --strategies volty_expan_close rsi
    /opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py --strategies xgboost --timeframes 1h 4h

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
MIN_TRADES_PER_DAY = 0.9
MAX_DRAWDOWN_PCT = 0.25

FEES = 0.001            # 0.1% taker
INIT_CASH = 1_000_000
POSITION_SIZE_PCT = 0.30

# Bars per month per timeframe — used to compute XGBoost rolling windows
TF_BARS_PER_MONTH = {
    "1m": 43_200, "5m": 8_640, "15m": 2_880, "30m": 1_440,
    "1h": 720, "4h": 180, "1d": 30,
}


# ── Parameter grid ────────────────────────────────────────────────────────────

def build_sweep_configs(timeframes: list[str]) -> list[dict]:
    configs = []

    for tf in timeframes:

        # ── 1. EMA Crossover ───────────────────────────────────────────────
        for fw, sw, sl, tp in product(
            [8, 9, 12, 20], [21, 26, 50],
            [0.03, 0.05, 0.08], [0.06, 0.10, 0.15, None],
        ):
            if sw <= fw:
                continue
            configs.append({
                "strategy": "ema_crossover", "timeframe": tf,
                "fast_window": fw, "slow_window": sw,
                "sl_stop": sl, "tp_stop": tp,
            })

        # ── 2. RSI ─────────────────────────────────────────────────────────
        for window, lower, upper, sl in product(
            [14, 21], [30, 35, 40], [60, 65, 70], [0.03, 0.05, 0.08],
        ):
            configs.append({
                "strategy": "rsi", "timeframe": tf,
                "window": window, "lower": lower, "upper": upper,
                "sl_stop": sl, "tp_stop": None,
            })

        # ── 3. ATR Trend ───────────────────────────────────────────────────
        for ma_len, atr_mult, atr_len in product(
            [50, 100, 200], [2.0, 3.0, 4.0], [7, 14],
        ):
            configs.append({
                "strategy": "atr_trend", "timeframe": tf,
                "ma_len": ma_len, "atr_len": atr_len, "atr_mult": atr_mult,
                "sl_stop": 0.05, "tp_stop": None,
            })

        # ── 4. RSI + EMA Filter ────────────────────────────────────────────
        for ema_len, rsi_p, lower, upper in product(
            [50, 100, 200], [14, 21], [30, 35, 40], [60, 65, 70],
        ):
            configs.append({
                "strategy": "rsi_ema_filter", "timeframe": tf,
                "ema_len": ema_len, "rsi_period": rsi_p,
                "oversold": lower, "overbought": upper,
                "sl_stop": 0.05, "tp_stop": None,
            })

        # ── 5. Volty Expan Close ───────────────────────────────────────────
        # Extended grid: includes 1m/5m + larger length/mult values
        for length, mult, sl in product(
            [3, 5, 7, 10, 20, 30],
            [0.5, 0.75, 1.0, 1.5, 2.0, 2.5],
            [0.03, 0.05, 0.08],
        ):
            configs.append({
                "strategy": "volty_expan_close", "timeframe": tf,
                "length": length, "mult": mult,
                "sl_stop": sl, "tp_stop": None,
            })

        # ── 6. Volty Expan + ATR Ratio (expanding vol regime gate) ─────────
        # Only on 15m+ (ATR ratio needs some warmup)
        if tf not in ("1m", "5m"):
            for length, mult, sl, threshold in product(
                [5, 7, 10], [0.75, 1.0, 1.5, 2.0],
                [0.03, 0.05], [1.0, 1.1, 1.2],
            ):
                configs.append({
                    "strategy": "volty_expan_close+atr_ratio", "timeframe": tf,
                    "length": length, "mult": mult,
                    "sl_stop": sl, "tp_stop": None,
                    "atr_ratio_threshold": threshold,
                })

        # ── 7. RSI + ATR Ratio Low (contracting vol regime gate) ───────────
        if tf not in ("1m", "5m"):
            for window, lower, upper, sl, threshold in product(
                [14, 21], [30, 35], [60, 65],
                [0.03, 0.05], [0.8, 0.9],
            ):
                configs.append({
                    "strategy": "rsi+atr_ratio_low", "timeframe": tf,
                    "window": window, "lower": lower, "upper": upper,
                    "sl_stop": sl, "tp_stop": None,
                    "atr_ratio_threshold": threshold,
                })

        # ── 8. RSI + EMA Filter + ATR Ratio Low ───────────────────────────
        if tf not in ("1m", "5m"):
            for ema_len, rsi_p, lower, upper, sl, threshold in product(
                [100, 200], [14], [30, 35], [60, 65],
                [0.03, 0.05], [0.8, 0.9],
            ):
                configs.append({
                    "strategy": "rsi_ema_filter+atr_ratio_low", "timeframe": tf,
                    "ema_len": ema_len, "rsi_period": rsi_p,
                    "oversold": lower, "overbought": upper,
                    "sl_stop": sl, "tp_stop": None,
                    "atr_ratio_threshold": threshold,
                })

        # ── 9. GARCH-filtered combos (1h/4h only — model fitting is slow) ──
        if tf in ("1h", "4h"):
            for window, lower, upper, sl, gw in product(
                [14, 21], [30, 35], [60, 65], [0.03, 0.05], [126, 252],
            ):
                configs.append({
                    "strategy": "rsi+garch", "timeframe": tf,
                    "window": window, "lower": lower, "upper": upper,
                    "sl_stop": sl, "tp_stop": None, "garch_window": gw,
                })
            for fw, sw, sl, gw in product(
                [9, 12], [21, 26], [0.03, 0.05], [126, 252],
            ):
                if sw <= fw:
                    continue
                configs.append({
                    "strategy": "ema_crossover+garch", "timeframe": tf,
                    "fast_window": fw, "slow_window": sw,
                    "sl_stop": sl, "tp_stop": None, "garch_window": gw,
                })
            for length, mult, sl, gw in product(
                [5, 7, 10], [0.75, 1.0, 1.5, 2.0], [0.03, 0.05], [126, 252],
            ):
                configs.append({
                    "strategy": "volty_expan_close+garch", "timeframe": tf,
                    "length": length, "mult": mult,
                    "sl_stop": sl, "tp_stop": None, "garch_window": gw,
                })
            # Full combo: VEC + ATR ratio + GARCH
            for length, mult, sl, threshold, gw in product(
                [5, 7, 10], [1.0, 1.5, 2.0], [0.03, 0.05], [1.0, 1.1], [126, 252],
            ):
                configs.append({
                    "strategy": "volty_expan_close+atr_ratio+garch", "timeframe": tf,
                    "length": length, "mult": mult,
                    "sl_stop": sl, "tp_stop": None,
                    "atr_ratio_threshold": threshold, "garch_window": gw,
                })

        # ── 10. Lorentzian KNN (1h/4h only — bar-by-bar loop) ─────────────
        if tf in ("1h", "4h"):
            for k_val, features, mbk, use_kernel in product(
                [6, 8, 10],
                ["rsi_wt_cci", "rsi_wt_cci_adx"],
                [500, 1000, 2000],
                [False, True],
            ):
                configs.append({
                    "strategy": "lorentzian_knn", "timeframe": tf,
                    "k": k_val, "features": features,
                    "max_bars_back": mbk, "use_kernel": use_kernel,
                    "sl_stop": 0.05, "tp_stop": None,
                })
            # Lorentzian KNN + GARCH
            for k_val, features, mbk in product(
                [8], ["rsi_wt_cci", "rsi_wt_cci_adx"], [500, 1000],
            ):
                for gw in [126, 252]:
                    configs.append({
                        "strategy": "lorentzian_knn+garch", "timeframe": tf,
                        "k": k_val, "features": features,
                        "max_bars_back": mbk, "use_kernel": False,
                        "sl_stop": 0.05, "tp_stop": None, "garch_window": gw,
                    })

        # ── 11b. VEC+RSI Regime meta-strategy (Round 2 narrow grid) ──────────
        # Focused on 1h and 4h where VEC shines; target: composite>1.5 + t/d≥0.9
        if tf in ("1h", "4h"):
            for length, mult, rsi_lower, rsi_upper, sl, thr in product(
                [5, 7, 10, 15],          # VEC length
                [1.0, 1.5, 2.0, 2.5],   # VEC mult
                [30, 35],                # RSI oversold
                [65, 70],                # RSI overbought
                [0.03, 0.05],            # SL
                [0.8, 1.0, 1.1],         # ATR ratio threshold
            ):
                configs.append({
                    "strategy": "vec_rsi_regime", "timeframe": tf,
                    "length": length, "mult": mult,
                    "rsi_window": 14, "rsi_lower": rsi_lower, "rsi_upper": rsi_upper,
                    "atr_ratio_threshold": thr,
                    "sl_stop": sl, "tp_stop": None,
                })

        # ── 11. XGBoost (1h/4h only — rolling train/predict) ──────────────
        if tf in ("1h", "4h"):
            bars_month = TF_BARS_PER_MONTH[tf]
            for n_est, max_d, mcw, thr, sl in product(
                [100, 200], [3, 5], [1, 3], [0.50, 0.55, 0.60], [0.03, 0.05],
            ):
                configs.append({
                    "strategy": "xgboost", "timeframe": tf,
                    "train_window_bars": bars_month * 6,
                    "predict_window_bars": bars_month,
                    "n_estimators": n_est, "max_depth": max_d,
                    "min_child_weight": mcw, "threshold": thr,
                    "sl_stop": sl, "tp_stop": None,
                })

    return configs


def _strategy_params(cfg: dict) -> dict:
    """Extract signal params from a config dict. Handles modifier chains."""
    s = cfg["strategy"]
    base = s.split("+")[0]

    if base == "ema_crossover":
        params = {"fast_window": cfg["fast_window"], "slow_window": cfg["slow_window"]}
    elif base == "rsi":
        params = {"window": cfg["window"], "lower": cfg["lower"], "upper": cfg["upper"]}
    elif base == "atr_trend":
        params = {"ma_len": cfg["ma_len"], "atr_len": cfg["atr_len"], "atr_mult": cfg["atr_mult"]}
    elif base == "rsi_ema_filter":
        params = {
            "ema_len": cfg["ema_len"], "rsi_period": cfg["rsi_period"],
            "oversold": cfg["oversold"], "overbought": cfg["overbought"],
        }
    elif base == "volty_expan_close":
        params = {"length": cfg["length"], "mult": cfg["mult"]}
    elif base == "vec_rsi_regime":
        params = {
            "length": cfg["length"], "mult": cfg["mult"],
            "rsi_window": cfg.get("rsi_window", 14),
            "rsi_lower": cfg.get("rsi_lower", 30),
            "rsi_upper": cfg.get("rsi_upper", 70),
            "atr_ratio_threshold": cfg.get("atr_ratio_threshold", 1.0),
        }
    elif base == "lorentzian_knn":
        params = {
            "k": cfg["k"], "features": cfg["features"],
            "max_bars_back": cfg["max_bars_back"],
            "use_kernel": cfg.get("use_kernel", False),
        }
    elif base == "xgboost":
        params = {
            "train_window_bars": cfg["train_window_bars"],
            "predict_window_bars": cfg["predict_window_bars"],
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "min_child_weight": cfg["min_child_weight"],
            "threshold": cfg["threshold"],
        }
    else:
        raise ValueError(f"Unknown base strategy: {base}")

    # Modifier params
    if "atr_ratio" in s:
        params["atr_ratio_threshold"] = cfg.get("atr_ratio_threshold", 1.1)
    if "garch" in s:
        params["garch_window"] = cfg.get("garch_window", 252)

    return params


# ── Metrics ───────────────────────────────────────────────────────────────────

ANN_FACTORS = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "4h": 2_190, "1d": 365,
}


def safe_float(val) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def compute_metrics(pf: vbt.Portfolio, timeframe: str) -> dict:
    """
    Compute Sharpe, Sortino, Calmar and composite score from portfolio returns.
    Manually computed to avoid vbt/pandas frequency inference issues.
    """
    ann = ANN_FACTORS.get(timeframe, 365)
    try:
        rets = pf.returns
        if hasattr(rets, "columns"):
            rets = rets.iloc[:, 0]
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
        "sortino": safe_float(sortino), "sharpe": safe_float(sharpe),
        "calmar": safe_float(calmar), "composite": safe_float(composite),
        "max_drawdown": safe_float(max_dd), "total_return": total_return,
        "trades_per_day": trades_per_day, "n_trades": n_trades,
    }


def run_single(cfg: dict, data: vbt.Data) -> dict:
    """Run one backtest configuration. Returns metrics dict."""
    close = data.get("Close")
    high = data.get("High")
    low = data.get("Low")
    params = _strategy_params(cfg)

    entries, exits = compute_signals(cfg["strategy"], close, high, low, **params)

    pf_kwargs = dict(
        close=close, entries=entries, exits=exits,
        fees=FEES, init_cash=INIT_CASH,
        size=POSITION_SIZE_PCT, size_type="percent",
        group_by=True, cash_sharing=True, call_seq="default",
    )
    if cfg.get("sl_stop"):
        pf_kwargs["sl_stop"] = cfg["sl_stop"]
    if cfg.get("tp_stop"):
        pf_kwargs["tp_stop"] = cfg["tp_stop"]

    pf = vbt.Portfolio.from_signals(**pf_kwargs)
    metrics = compute_metrics(pf, cfg["timeframe"])
    return {**cfg, **metrics}


# ── Main sweep loop ───────────────────────────────────────────────────────────

def run_sweep(
    timeframes: list[str],
    strategies: list[str] | None = None,
    stop_on_target: bool = True,
    csv_path: Path | None = None,
):
    all_configs = build_sweep_configs(timeframes)

    # Filter by strategy if --strategies flag provided
    if strategies:
        configs = [c for c in all_configs if c["strategy"] in strategies]
    else:
        configs = all_configs

    if not configs:
        print("No configs matched the given --strategies filter.")
        return {}

    if csv_path is None:
        csv_path = RESULTS_DIR / "sweep_results.csv"

    strat_filter_str = f" (strategies: {', '.join(strategies)})" if strategies else ""
    print(f"\n{'='*65}")
    print(f"RALPH LOOP — {len(configs)} configs{strat_filter_str}")
    print(f"Target : composite > {COMPOSITE_TARGET}, trades/day >= {MIN_TRADES_PER_DAY}, dd < {MAX_DRAWDOWN_PCT*100:.0f}%")
    print(f"{'='*65}\n")

    best = {"composite": -np.inf, "config": None}
    loaded: dict[str, vbt.Data] = {}

    CSV_COLS = [
        "strategy", "timeframe", "composite", "sortino", "sharpe", "calmar",
        "trades_per_day", "max_drawdown", "total_return", "n_trades",
    ]

    # Write or append CSV header
    write_header = not csv_path.exists()
    if write_header:
        pd.DataFrame(columns=CSV_COLS).to_csv(csv_path, index=False)

    for i, cfg in enumerate(configs):
        tf = cfg["timeframe"]

        if tf not in loaded:
            path = DATA_DIR / f"binance_{tf}.h5"
            if not path.exists():
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

        c = metrics["composite"]
        meets = (
            metrics["trades_per_day"] >= MIN_TRADES_PER_DAY
            and metrics["max_drawdown"] < MAX_DRAWDOWN_PCT
        )
        flag = " *** NEW BEST ***" if meets and c > best["composite"] else ""
        print(
            f"[{i+1:>5}/{len(configs)}] {cfg['strategy']:<32} {tf:<4} "
            f"C={c:>6.3f} (So={metrics['sortino']:.2f} Sh={metrics['sharpe']:.2f} Ca={metrics['calmar']:.2f}) "
            f"dd={metrics['max_drawdown']*100:.1f}% t/d={metrics['trades_per_day']:.2f} "
            f"ret={metrics['total_return']*100:.1f}% ({elapsed:.1f}s){flag}"
        )

        if meets and c > best["composite"]:
            best = {"composite": c, "config": metrics}

        pd.DataFrame([{k: metrics[k] for k in CSV_COLS}]).to_csv(csv_path, mode="a", header=False, index=False)

        if stop_on_target and meets and c > COMPOSITE_TARGET:
            print(f"\n{'='*65}")
            print(f"RALPH LOOP TARGET REACHED at config {i+1}/{len(configs)}: composite={c:.4f}")
            print(f"{'='*65}")
            break

    # Sort final CSV by composite
    if csv_path.exists():
        df = pd.read_csv(csv_path).sort_values("composite", ascending=False)
        df.to_csv(csv_path, index=False)

        print(f"\n{'='*65}\nTOP 10 CONFIGURATIONS\n{'='*65}")
        cols = ["strategy", "timeframe", "composite", "sortino", "sharpe",
                "calmar", "trades_per_day", "max_drawdown", "total_return"]
        print(df.head(10)[cols].to_string(index=False))

    if best["config"]:
        best_path = RESULTS_DIR / "best_params.json"
        best_path.write_text(json.dumps(best["config"], indent=2, default=str))
        print(f"\nBest params → {best_path}")
        print(json.dumps(best["config"], indent=2, default=str))
    else:
        print("\nNo configuration met all criteria. Consider relaxing thresholds or running --no-stop.")

    return best


def main():
    parser = argparse.ArgumentParser(description="Ralph Loop — parameter sweep for the Roostoo bot.")
    parser.add_argument("--timeframes", nargs="+", default=["1h", "4h"],
                        help="Timeframes to sweep (default: 1h 4h)")
    parser.add_argument("--all-timeframes", action="store_true",
                        help="Sweep all timeframes: 1m 5m 15m 30m 1h 4h 1d")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Only sweep these strategies (e.g. --strategies rsi volty_expan_close)")
    parser.add_argument("--no-stop", action="store_true",
                        help="Run full sweep without early stopping")
    args = parser.parse_args()

    tfs = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"] if args.all_timeframes else args.timeframes
    run_sweep(tfs, strategies=args.strategies, stop_on_target=not args.no_stop)


if __name__ == "__main__":
    main()
