# Roostoo Trading Bot

Autonomous cryptocurrency trading bot for the Roostoo hackathon (Mar 22–31 2026). Trades BTC/USD, ETH/USD, and SOL/USD on spot markets. Strategy selected via a full parameter sweep on 4 years of Binance OHLCV data with walk-forward out-of-sample validation.

## Deployed Strategy

**`vec_rsi_regime` on 4h** — a regime-switching meta-strategy that routes to different signals based on the current volatility environment, measured by ATR(5)/ATR(50).

| Regime | Condition | Signal |
|--------|-----------|--------|
| High-vol | ATR(5)/ATR(50) > 0.8 | Volatility Expansion Close (VEC) breakout |
| Low-vol  | ATR(5)/ATR(50) ≤ 0.8 | RSI(14) mean-reversion |

**Parameters** (`config.py` / `results/best_params.json`):

| Parameter | Value |
|-----------|-------|
| VEC length | 15 |
| VEC multiplier | 1.0 |
| RSI period | 14 |
| RSI oversold | 35 |
| RSI overbought | 65 |
| ATR ratio threshold | 0.8 |
| Stop loss | 3% hard stop |
| Position size | 30% of portfolio per asset |
| Fees | 0.1% taker |

**Backtest + walk-forward results:**

| Period | Composite | Sortino | Sharpe | Calmar | Return | Max DD | t/day |
|--------|-----------|---------|--------|--------|--------|--------|-------|
| Full 1-year | 1.530 | 1.490 | 1.285 | 1.830 | +28% | 15.3% | 0.91 |
| IS (10 months) | 1.140 | — | — | — | — | — | 0.94 |
| OOS Jan–Mar 2026 | 11.49 | — | — | — | — | 3.3% | 0.61 |
| **OOS/IS ratio** | **10.08** | — | — | — | — | — | — |

Competition scoring: `composite = 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar`
Deploy criteria: composite > 1.5, trades/day ≥ 0.9, max_drawdown < 25%.

---

## Sweep History

All sweeps used 4-year Binance data (BTC/ETH/SOL, 2022–2026), fees=0.1%. Run sequentially (`--workers 0`) — peak RAM 4–6 GB, safe on 16 GB machines.

### Round 1 — Full Grid (4-year, ~4,100 configs)

**File**: `results/sweep_results.csv`

Swept 11 strategies across 1h and 4h: `ema_crossover`, `rsi`, `atr_trend`, `rsi_ema_filter`, `volty_expan_close`, `volty_expan_close+atr_ratio`, `rsi+atr_ratio_low`, `rsi+garch`, `ema_crossover+garch`, `volty_expan_close+garch`, `vec_rsi_regime`, `lorentzian_knn`, `vec_garch_regime`.

**Key finding**: `vec_rsi_regime` on 4h was the **only** configuration meeting all three deployment criteria simultaneously. High composite and high trade frequency are inherently in tension — the regime-switching approach is what bridges the gap.

**Note**: `lorentzian_knn` appeared to score C=1.979 in Round 1 — this was invalidated by a lookahead bug (see Bugs section below). After the fix, KNN dropped to C=1.129.

**Round 1 winner (only config meeting all criteria):**

| Strategy | TF | Composite | Sortino | Sharpe | Calmar | t/day | Max DD | Return |
|---|---|---|---|---|---|---|---|---|
| vec_rsi_regime | 4h | 1.530 | 1.490 | 1.285 | 1.830 | 0.91 | 15.3% | +28% |

**vec_rsi_regime on full 4-year data**: Composite=1.01, Sharpe=1.01, Sortino=1.16, Calmar=0.81, Max DD=27.1%, Return=121.5%, Win Rate=37.8%, Profit Factor=1.23, 1291 trades.

---

### Round 2 — Narrow KNN Grid (288 configs)

**File**: `results/sweep_results_r2.csv`

Goal: find a KNN configuration with C>1.5 AND t/d≥0.9 after fixing the lookahead bug. Swept `sample_step`, `k`, `horizon`, `kernel_lookback`, `kernel_alpha` on 4h and 1h.

**Outcome**: no KNN config achieves both C>1.5 AND t/d≥0.9 on 4h. The trade-off is structural — smaller `sample_step` increases trade frequency but dilutes signal quality.

**Best (relaxed to t/d≥0.7):**

| Strategy | TF | Composite | t/day | Max DD | Params |
|---|---|---|---|---|---|
| lorentzian_knn | 4h | 1.668 | 0.825 | 20.9% | k=2, h=8, ss=2, kl=8, ka=2.0, sl=0.03 |

**Walk-forward on KNN**: 0 trades generated in Jan–Mar 2026 OOS window. KNN trained on 2022–2025 market structure does not generalise to early 2026. Not deployed.

---

### Round 3 — XGBoost + GARCH Features (64 configs)

**File**: `results/sweep_results_r3.csv`

Two approaches tested on 4h only:

**A. Standalone XGBoost with 21 features (including GARCH conditional vol)**

All 24 configs produced t/d ~2.1–2.2 (chronic over-trading), best composite=0.715 with dd=27%. The model generates too many low-conviction signals regardless of threshold. Not viable.

**B. `vec_rsi_xgb_regime` — XGBoost as an AND-gate precision filter on `vec_rsi_regime`**

XGBoost was trained rolling (6-month windows, 1-month prediction steps) to predict whether a vec_rsi_regime entry would be profitable. The AND-gate improved signal quality on 4-year data (best C=1.396 vs bare vec_rsi_regime C=1.01 on same period; max DD halved to 14%). But double-gating two already-selective signals collapsed trade frequency to t/d=0.34–0.51 — below the 0.9 threshold.

**Round 3 best:**

| Strategy | TF | Composite | Sortino | Calmar | t/day | Max DD | Return (4yr) |
|---|---|---|---|---|---|---|---|
| vec_rsi_xgb_regime | 4h | 1.396 | 1.299 | 1.547 | 0.41 | 14.1% | +121% |

**Conclusion**: XGBoost as AND-gate doesn't meet hackathon criteria. Post-hackathon directions: use as exit timer, OR-gate, or position-size multiplier rather than entry filter.

---

## Bugs Found and Fixed

### 1. KNN Lookahead (invalidated all pre-fix KNN results)

**Symptom**: `lorentzian_knn` returned C=8.484 and +4483% return — clearly fake.

**Root cause** (`bot/indicators.py`): the candidate validity check `valid = (cand_idx >= horizon) & ((cand_idx + horizon) < n)` only verified dataset bounds, not that the candidate's label bar was strictly *before the current query bar*. With small `sample_step` (e.g. 1), offsets like 1/2/3 with `horizon=4` produced labels using bars q+3, q+2, q+1 — future data at query time. Even `sample_step=4, horizon=8` was affected: offset=4 gave a label bar at q+4 > q.

**Fix**:
```python
# Before — only bounds check, allows future labels:
valid = (cand_idx >= horizon) & ((cand_idx + horizon) < n)

# After — label bar must be ≤ current bar:
valid = (cand_idx >= 0) & (cand_idx + horizon <= q_bars[:, None])
```

**Impact**: R1 KNN champion dropped from C=1.979 → C=1.129. All KNN results prior to this fix are invalid.

---

### 2. GARCH DataFrame Bug (17-min hangs)

**Symptom**: `vec_garch_regime` strategy caused 17-minute hangs and 3 GB RAM per config.

**Root cause** (`bot/indicators.py`): `vec_garch_regime_signals` called `garch_vol_mask(close, ...)` where `close` was a multi-symbol DataFrame. `arch_model` doesn't handle DataFrames — it silently ran on the wrong data or looped internally.

**Fix**: iterate per column explicitly:
```python
if isinstance(close, pd.DataFrame):
    low_vol_mask = pd.DataFrame(
        {col: garch_vol_mask(close[col], ...) for col in close.columns},
        index=close.index,
    ).reindex(vec_e.index, fill_value=False)
```

**Impact**: runtime dropped from 17 min to 6.7s per config (150× speedup).

---

## Feature Engineering (XGBoost, 21 features)

`bot/indicators.py: build_xgb_features()` — all strictly past-data, no lookahead.

| Group | Features |
|-------|---------|
| Momentum | RSI(14), RSI(7) |
| Trend | EMA diff(9/21), dist\_ema50, dist\_ema200 |
| Volatility | ATR ratio(5/50), realized vol ratio(5/50), BB %B(20), BB width(20) |
| Oscillators | CCI(20), ADX(14), MACD histogram(12/26/9) |
| Returns | 1-bar, 3-bar, 5-bar, 10-bar, 20-bar |
| Calendar | day\_of\_week, hour |
| GARCH (optional, `use_garch=True`) | GARCH(1,1) conditional vol, rolling percentile rank |

---

## Project Structure

```
├── config.py                  # Strategy parameters — single source of truth
├── roostoo_client.py          # Roostoo API client (HMAC auth, orders, balance)
├── bot/
│   ├── runner.py              # Main bot loop — runs every 4h
│   ├── indicators.py          # All indicators + signal generators (12 strategies)
│   ├── ml_models.py           # GARCH(1,1) volatility model + XGBoost rolling signals
│   └── market_data.py         # Binance public REST for live OHLCV
├── backtest/
│   ├── fetch_data.py          # Downloads Binance OHLCV (7 timeframes, up to 4 years)
│   ├── sweep.py               # Ralph loop — 1,600+ configs, 3 rounds
│   └── walk_forward.py        # IS/OOS split validation on best_params.json
├── data/                      # Cached OHLCV (HDF5, git-ignored)
└── results/
    ├── best_params.json        # Deployed config (vec_rsi_regime 4h, C=1.530)
    ├── sweep_results.csv       # Round 1: full grid results
    ├── sweep_results_r2.csv    # Round 2: narrow KNN results
    ├── sweep_results_r3.csv    # Round 3: XGBoost+GARCH results
    └── walk_forward.json       # Walk-forward OOS output
```

---

## Available Strategies

| Strategy | Description | Status |
|---|---|---|
| `vec_rsi_regime` | **[DEPLOYED]** VEC in high-vol, RSI in low-vol, ATR gated | ✅ Live |
| `lorentzian_knn` | KNN with Lorentzian distance + RQK kernel smoother | Research — t/d too low |
| `vec_garch_regime` | VEC gated by GARCH(1,1) high-vol regime | Research |
| `vec_rsi_xgb_regime` | `vec_rsi_regime` filtered by XGBoost quality gate | Research — t/d too low |
| `volty_expan_close` | Pure VEC breakout | Baseline |
| `rsi` | RSI mean-reversion | Baseline |
| `ema_crossover` | EMA fast/slow crossover | Baseline |
| `xgboost` | Rolling train/predict XGBoost classifier (21 features) | Research |

Modifier chains composable via `+`: `+atr_ratio`, `+atr_ratio_low`, `+garch`.

---

## Running the Sweep

```bash
CONDA="conda run -n vectorbtpro"

# One-time: download 4 years of OHLCV (~20 min)
$CONDA python backtest/fetch_data.py

# Round 1: full grid on 4h (sequential, ~45 min)
$CONDA python backtest/sweep.py --timeframes 4h --no-stop

# Round 2: narrow KNN grid targeting t/d ≥ 0.9
$CONDA python backtest/sweep.py --round 2 --timeframes 4h --no-stop

# Round 3: XGBoost+GARCH features + vec_rsi_xgb_regime (~40 min)
$CONDA python backtest/sweep.py --round 3 --no-stop

# Walk-forward validation on current best_params.json
$CONDA python backtest/walk_forward.py

# Filter to specific strategies
$CONDA python backtest/sweep.py --strategies vec_rsi_regime --timeframes 4h --no-stop
```

All sweeps run sequentially by default — safe on 16 GB RAM. GARCH strategies add ~7s per config per symbol.

---

## Setup

```bash
cp .env.example .env
# Fill in ROOSTOO_API_KEY and ROOSTOO_API_SECRET

pip install -r requirements.txt

python bot/runner.py   # run locally
./deploy.sh            # deploy to EC2
```
