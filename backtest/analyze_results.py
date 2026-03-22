#!/usr/bin/env python3
"""
Claude Analysis script — reads sweep_results.csv and produces:
  1. Top 10 overall by composite score
  2. Best per strategy family
  3. Best per timeframe
  4. Regime routing suggestion (which strategy wins in each regime)
  5. Round 2 narrow-grid recommendations

Usage:
    /opt/anaconda3/bin/conda run -n vectorbtpro python backtest/analyze_results.py
    /opt/anaconda3/bin/conda run -n vectorbtpro python backtest/analyze_results.py --min-trades 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Minimum quality filters
MIN_TRADES_PER_DAY = 0.5   # lower bar for analysis (strict bar is 0.9 for deploy)
MAX_DRAWDOWN = 0.40         # allow up to 40% for analysis
DEPLOY_MIN_TRADES = 0.9
DEPLOY_MAX_DD = 0.25
DEPLOY_TARGET = 1.5


def load_results(min_trades: float = MIN_TRADES_PER_DAY) -> pd.DataFrame:
    path = RESULTS_DIR / "sweep_results.csv"
    if not path.exists():
        print("ERROR: results/sweep_results.csv not found. Run sweep.py first.")
        raise SystemExit(1)

    df = pd.read_csv(path)
    df = df[df["n_trades"] > 0]
    df = df[df["composite"].notna()]
    df = df[df["max_drawdown"] < MAX_DRAWDOWN]
    df = df[df["trades_per_day"] >= min_trades]
    return df.sort_values("composite", ascending=False)


def strategy_family(s: str) -> str:
    base = s.split("+")[0]
    return base


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze(df: pd.DataFrame):
    print_section("ROUND 1 SWEEP ANALYSIS")
    print(f"Total configs with trades: {len(df)}")
    print(f"Configs meeting deploy criteria "
          f"(C>{DEPLOY_TARGET}, t/d>={DEPLOY_MIN_TRADES}, dd<{DEPLOY_MAX_DD*100:.0f}%): "
          f"{len(df[(df['composite']>DEPLOY_TARGET) & (df['trades_per_day']>=DEPLOY_MIN_TRADES) & (df['max_drawdown']<DEPLOY_MAX_DD)])}")

    # ── Top 10 overall ────────────────────────────────────────────────────────
    print_section("TOP 10 OVERALL (composite score)")
    cols = ["strategy", "timeframe", "composite", "sortino", "sharpe", "calmar",
            "trades_per_day", "max_drawdown", "total_return", "n_trades"]
    top10 = df.head(10)
    for col in ["composite", "sortino", "sharpe", "calmar"]:
        top10 = top10.copy()
        top10[col] = top10[col].map("{:.3f}".format)
    top10["max_drawdown"] = top10["max_drawdown"].map("{:.1%}".format)
    top10["total_return"] = top10["total_return"].map("{:.1%}".format)
    top10["trades_per_day"] = top10["trades_per_day"].map("{:.2f}".format)
    print(top10[cols].to_string(index=False))

    # ── Best per strategy family ──────────────────────────────────────────────
    print_section("BEST CONFIG PER STRATEGY FAMILY")
    df2 = df.copy()
    df2["family"] = df2["strategy"].apply(strategy_family)
    best_per_family = (
        df2.groupby("family")
        .apply(lambda g: g.nlargest(1, "composite").iloc[0])
        .reset_index(drop=True)
        .sort_values("composite", ascending=False)
    )
    for _, row in best_per_family.iterrows():
        deploy_ok = (
            row["composite"] > DEPLOY_TARGET
            and row["trades_per_day"] >= DEPLOY_MIN_TRADES
            and row["max_drawdown"] < DEPLOY_MAX_DD
        )
        flag = " ← DEPLOY READY" if deploy_ok else ""
        print(f"  {row['family']:<30} {row['timeframe']:<5}  "
              f"C={row['composite']:>6.3f}  t/d={row['trades_per_day']:.2f}  "
              f"dd={row['max_drawdown']*100:.1f}%{flag}")

    # ── Best per timeframe ────────────────────────────────────────────────────
    print_section("BEST CONFIG PER TIMEFRAME")
    best_per_tf = (
        df.groupby("timeframe")
        .apply(lambda g: g.nlargest(1, "composite").iloc[0])
        .reset_index(drop=True)
        .sort_values("composite", ascending=False)
    )
    for _, row in best_per_tf.iterrows():
        print(f"  {row['timeframe']:<5}  {row['strategy']:<32}  "
              f"C={row['composite']:>6.3f}  t/d={row['trades_per_day']:.2f}  "
              f"dd={row['max_drawdown']*100:.1f}%")

    # ── Regime routing suggestion ─────────────────────────────────────────────
    print_section("REGIME ROUTING SUGGESTION")
    df2["family"] = df2["strategy"].apply(strategy_family)

    breakout_families = ["volty_expan_close"]
    meanrev_families = ["rsi", "rsi_ema_filter"]
    trend_families = ["ema_crossover", "atr_trend"]
    filtered_families = ["volty_expan_close", "rsi", "rsi_ema_filter", "ema_crossover", "atr_trend"]

    regime_map = {
        "VOL_EXPANDING (atr_ratio > 1.2)": breakout_families,
        "VOL_CONTRACTING (atr_ratio < 0.8)": [f + "+atr_ratio_low" if f in meanrev_families else f
                                               for f in meanrev_families] + ["rsi+atr_ratio_low", "rsi_ema_filter+atr_ratio_low"],
        "TRENDING (strong)": trend_families,
        "REGIME_UNCERTAIN": ["lorentzian_knn", "xgboost"],
    }

    # Just show which family scored best in each regime bucket
    for regime, families in regime_map.items():
        subset = df2[df2["family"].isin(families) | df2["strategy"].isin(families)]
        if subset.empty:
            print(f"  {regime}: No results yet")
            continue
        best = subset.nlargest(1, "composite").iloc[0]
        print(f"  {regime}:")
        print(f"    Best: {best['strategy']} / {best['timeframe']}  "
              f"C={best['composite']:.3f}  t/d={best['trades_per_day']:.2f}  dd={best['max_drawdown']*100:.1f}%")

    # ── Round 2 recommendations ───────────────────────────────────────────────
    print_section("ROUND 2 NARROW GRID RECOMMENDATIONS")
    print("Focus Round 2 on the top-3 families per regime region.\n")

    # Top 3 families overall
    top3_families = best_per_family.head(5)
    for _, row in top3_families.iterrows():
        print(f"  • {row['family']:<30} C={row['composite']:.3f}")
        # Show the winning timeframe
        family_df = df2[df2["family"] == row["family"]]
        best_tfs = family_df.groupby("timeframe")["composite"].max().sort_values(ascending=False)
        tf_str = ", ".join([f"{tf}={v:.2f}" for tf, v in best_tfs.head(3).items()])
        print(f"    Best TFs: {tf_str}")
        print()

    print("Suggested Round 2 action:")
    print("  1. Narrow param ranges around top-3 configs per winning family")
    print("  2. Add GARCH / ATR ratio modifiers to winning base strategies")
    print("  3. Run Lorentzian KNN + XGBoost on top-2 timeframes")

    # ── Save summary ─────────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / "analysis_r1.json"
    summary = {
        "total_configs_with_trades": len(df),
        "deploy_ready_count": int(len(df[
            (df["composite"] > DEPLOY_TARGET) &
            (df["trades_per_day"] >= DEPLOY_MIN_TRADES) &
            (df["max_drawdown"] < DEPLOY_MAX_DD)
        ])),
        "top10": df.head(10)[cols].to_dict(orient="records"),
        "best_per_family": best_per_family[
            ["family", "strategy", "timeframe", "composite", "sortino",
             "sharpe", "calmar", "trades_per_day", "max_drawdown", "total_return"]
        ].to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-trades", type=float, default=MIN_TRADES_PER_DAY)
    args = parser.parse_args()

    df = load_results(min_trades=args.min_trades)
    analyze(df)


if __name__ == "__main__":
    main()
