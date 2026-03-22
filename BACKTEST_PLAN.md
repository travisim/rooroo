# Backtest & Parameter Tuning Plan

**Goal: composite score > 1.5 AND trades/day ≥ 1 AND max_drawdown < 25%**

```
composite = 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar
```

---

## Backtesting vs. Walk-Forward Validation

These are two distinct steps — both are required.

**Backtesting** is running a strategy on historical data to see how it *would have* performed. The parameter sweep (`sweep.py`) does this — it tests thousands of configs on 1 year of past data and ranks them by composite score. This tells you which params *fit the past best*.

**The problem with backtesting alone**: the winning config is optimized to that specific historical period. It may have simply gotten lucky with those exact market conditions rather than having a genuine edge. This is called **overfitting** or **curve-fitting** — the strategy looks amazing on paper but falls apart the moment it trades live.

**Walk-forward validation** is the test for whether a backtest result is real or overfitted. The 1-year dataset is split:

```
|←────────── 10 months (in-sample / IS) ──────────→|←── 2 months (out-of-sample / OOS) ──→|
  sweep.py finds best params here (backtest)           walk_forward.py tests here — blind,
                                                        params locked, never re-optimized
```

- **In-sample (IS)**: the backtest period. Params are tuned here.
- **Out-of-sample (OOS)**: hidden data the optimizer never saw. Run the *exact same params* here with no changes.
- **OOS/IS ratio**: if a strategy scored composite=2.0 IS and 1.6 OOS, ratio = 0.80 — the edge is real. If OOS = 0.4, ratio = 0.20 — it was overfit noise.

**The workflow is always**: backtest (find params) → walk-forward (verify they generalize) → deploy.
Never deploy based on backtest results alone.

---

## Current State

| Item | Status |
|---|---|
| Data | ✅ All 7 TFs downloaded (1m–1d, BTC/ETH/SOL, 1 year) |
| Pass A sweep (8 strategies, 6 TFs) | ✅ Done — 2812 configs, ~7 min |
| Pass B sweep (GARCH+KNN+XGB+meta, 1h/4h) | ⏳ Running now |
| Round 1 analysis | ✅ Done — see below |
| Bot deployed with | RSI(14)/15m → to be replaced with best found config |

### Round 1 Findings (Pass A: ema/rsi/atr_trend/rsi_ema/vec and combos)

**Winner: `volty_expan_close+atr_ratio` on 4h**

Top configs found:
- `VEC+ATR_RATIO 4h` (len=7, mult=2.0, sl=0.05, thr=1.0): composite=**2.44**, t/d=0.13 (too few trades)
- `VEC+ATR_RATIO 4h` (len=5, mult=0.75, sl=0.03, thr=1.0): composite=1.45, t/d=**0.89** (almost!)
- `VEC 1h` (len=7, mult=2.5, sl=0.03): composite=**1.55**, t/d=0.41 (too few trades)
- `VEC 4h` plain: composite=1.15, t/d=0.99 (meets trade freq, needs better composite)

**Key insight:** High composite → fewer trades. Meeting composite>1.5 AND t/d≥0.9 simultaneously is the core challenge. GARCH + Lorentzian KNN (Pass B) may provide independent signal sources to bridge this gap.

**Regimes:**
- 1m/5m: all terrible (too noisy, massive drawdowns)
- 30m: decent with VEC (composite ~1.0)
- 1h: best standalone composite (VEC 1.55)
- 4h: best balanced combo (VEC+ATR_RATIO 1.45 at 0.89 t/d)

---

## End Goal: Regime-Switching Meta-Strategy

The funnel does not find one single best strategy. It finds the **best strategy per regime**. The final deployed bot is a meta-strategy that detects the current market regime on every check cycle and routes to whichever strategy has demonstrated edge in that condition.

```
Each check cycle (per asset):
    │
    ├─ Compute ATR5/ATR50 ratio + GARCH predicted vol
    │
    ├─ ATR ratio > 1.2 (vol expanding)
    │       └─ → Volty Expan Close
    │
    ├─ ATR ratio < 0.8 + price > EMA200 (quiet uptrend)
    │       └─ → RSI+EMA filter
    │
    ├─ ATR ratio < 0.8 + price < EMA200 (quiet, no trend)
    │       └─ → RSI mean reversion
    │
    ├─ Strong trend (ATR ratio neutral, EMA200 clearly rising)
    │       └─ → EMA crossover or ATR trend
    │
    └─ Unclear / high vol + no direction
            └─ → Lorentzian KNN or XGBoost
```

Each asset runs this independently — BTC can be in a Volty Expan trade while ETH is in an RSI trade at the same time. Dynamic position sizing (inverse ATR) then scales each position to risk the same dollar amount regardless of which strategy fired.

**The funnel's real job:** find the best params for each strategy *in its target regime*, validate each one out-of-sample, then assemble them into the meta-strategy. The composite score of the meta-strategy should exceed any individual strategy because each component is only active when conditions favour it.

---

## Master Strategy List

All strategies that will be backtested and funnelled. Grouped by the regime each targets.

### Trend / Crossover

| # | Strategy | Description | Regimes it works in |
|---|---|---|---|
| 1 | `ema_crossover` | Fast EMA crosses above/below slow EMA | Trending, low noise |
| 2 | `atr_trend` | Price breaks above/below EMA ± ATR band | Strong trending |

### Mean Reversion

| # | Strategy | Description | Regimes it works in |
|---|---|---|---|
| 3 | `rsi` | Buy RSI oversold, sell overbought | Ranging, low vol |
| 4 | `rsi_ema_filter` | RSI signal gated: only enter when price > EMA200 | Ranging within uptrend |

### Volatility Breakout

| # | Strategy | Description | Regimes it works in |
|---|---|---|---|
| 5 | `volty_expan_close` | Entry when price breaks close ± SMA(ATR) band | Expanding vol, breakouts |

### Regime-Filtered Combos
These are existing strategies above with a volatility regime filter layered on. The filter uses either the **ATR5/ATR50 ratio** (short-term ATR ÷ long-term ATR — detects if vol is expanding *right now*) or **GARCH** (predicts whether vol will be high on the next bar). They complement each other: ATR ratio is reactive, GARCH is predictive.

| # | Strategy | Base | Filter | Why combine |
|---|---|---|---|---|
| 6 | `volty_expan_close+atr_ratio` | Volty Expan | ATR5/ATR50 > threshold | Only enter breakouts when vol is confirmed expanding |
| 7 | `volty_expan_close+garch` | Volty Expan | GARCH predicts high vol | GARCH anticipates the vol spike before ATR confirms it |
| 8 | `volty_expan_close+atr_ratio+garch` | Volty Expan | Both ATR ratio AND GARCH | Highest precision: both axes confirm expansion |
| 9 | `rsi+atr_ratio_low` | RSI | ATR5/ATR50 < threshold | RSI mean reversion is most reliable when vol is contracting |
| 10 | `rsi+garch` | RSI | GARCH predicts low vol | Suppress RSI entries during noisy/chaotic vol regimes |
| 11 | `ema_crossover+garch` | EMA crossover | GARCH predicts low vol | EMA crossovers are cleaner in low-noise environments |
| 12 | `rsi_ema_filter+atr_ratio_low` | RSI+EMA filter | ATR5/ATR50 < threshold | Quiet uptrend bounces — highest quality RSI setups |

### Machine Learning

| # | Strategy | Description | Regimes it works in |
|---|---|---|---|
| 13 | `lorentzian_knn` | KNN classifier using Lorentzian distance on RSI, WT, CCI, ADX features | Any — learns from historical patterns |
| 14 | `lorentzian_knn+garch` | Lorentzian KNN gated by GARCH | Filters out high-noise regimes where pattern matching breaks down |
| 15 | `xgboost` | Gradient boosted classifier on 11 technical features, rolling train/predict | Any — data-driven |

---

## Iteration Loop

```
Implement strategy → Sweep params → Walk-forward validate → Deploy if OOS ≥ 70% IS
        ↑                                                              |
        └──────────────── Relax params / pivot strategy ←─────────────┘
                          if OOS < 70% or composite < 1.5
```

---

## Phase 1 — Implement New Strategies

Work through these in order. Each phase is a prerequisite for the next sweep run.

### 1a. Volty Expan Close (VEC) + ATR Regime Indicator
**Files:** `bot/indicators.py`, `backtest/sweep.py`

Add to `bot/indicators.py`:
- `sma(series, period)` — simple moving average
- `true_range(high, low, close)` — TR for each bar
- `atr_expansion_ratio(high, low, close, short=5, long=50)` — `atr(short) / atr(long)`:
  - > 1.2: vol expanding (breakout regime) → Volty Expan entry allowed
  - < 0.8: vol contracting (mean reversion regime) → RSI entry allowed
  - Used as a filter modifier via the `+atr_ratio` / `+atr_ratio_low` strategy name pattern
- `volty_expan_close_signals(close, high, low, length, mult)`:
  - `band = sma(true_range, length) * mult`
  - Entry: `high[t] > close[t-1] + band[t-1]`
  - Exit: `low[t] < close[t-1] - band[t-1]`

Sweep grid (6 lengths × 6 mults × 3 SLs × 6 TFs = 648 configs):

| Param | Values |
|---|---|
| `length` | 3, 5, 7, 10, 20, 30 |
| `mult` | 0.5, 0.75, 1.0, 1.5, 2.0, 2.5 |
| `sl_stop` | 3%, 5%, 8% |
| Timeframes | 1m, 5m, 15m, 30m, 1h, 4h |

### 1b. Lorentzian KNN
**Files:** `bot/indicators.py`, `backtest/sweep.py`

Add to `bot/indicators.py`:
- `wave_trend(close, high, low, n1=10, n2=21)` — WT oscillator
- `cci(close, high, low, period=20)` — Commodity Channel Index
- `adx(high, low, close, period=14)` — Average Directional Index (Wilder)
- `normalize_series(series, lookback)` — rolling min-max → [0, 1]
- `lorentzian_knn_signals(close, high, low, k, max_bars_back, use_adx, use_kernel)`:
  - Features: normalized RSI(14), WT(10,21), CCI(20), optionally ADX(14)
  - Distance: `Σ log(1 + |f_current − f_historical[j]|)` for each feature
  - Sample every 4 bars back; keep k nearest; label = `sign(close[j] − close[j−4])`
  - Signal: long if `sum(labels of k neighbors) > 0`
  - **Lookahead rule**: label at bar j uses only `close[j]` and `close[j-4]` — never bar i (current)

> **Restrict to 1h and 4h only** — the bar-by-bar loop is too slow on 15m/30m.

Sweep grid (36 configs × timeframe):

| Param | Values |
|---|---|
| `k` | 6, 8, 10 |
| `features` | `rsi_wt_cci`, `rsi_wt_cci_adx` |
| `max_bars_back` | 500, 1000, 2000 |
| `use_kernel` | False, True |
| Timeframes | 1h, 4h |

### 1c. GARCH Volatility Filter
**Files:** `bot/ml_models.py` (new), `bot/indicators.py`, `backtest/sweep.py`

Create `bot/ml_models.py`. Add `garch_vol_mask(close, window, vol_percentile)`:
- Fit GARCH(1,1) on rolling `window` bars of log-returns
- **Refit every 24 bars** (not every bar) for speed
- Gate: return `True` (entry allowed) when predicted vol < rolling 75th percentile
- **Lookahead rule**: fit window is strictly `[t-window, t-1]`, never includes bar t

Integrated as `"rsi+garch"`, `"ema_crossover+garch"` — `compute_signals()` splits on `+`.

Sweep grid (per base strategy, 1h/4h only):

| Param | Values |
|---|---|
| Base strategy | rsi (14/35/65), ema_crossover (9/21) |
| `garch_window` | 126, 252 |
| `sl_stop` | 3%, 5% |

### 1d. XGBoost Rolling Classifier
**Files:** `bot/ml_models.py`, `bot/indicators.py`, `backtest/sweep.py`

Add to `bot/indicators.py`:
- `build_xgb_features(close, high, low)` → DataFrame:
  - RSI(14), RSI(7), EMA diff normalized `(ema9-ema21)/close`, ATR/close, CCI(20), ADX(14), returns(1,5,20 bars), day_of_week, hour

Add to `bot/ml_models.py`:
- `build_xgb_labels(close, horizon=4, threshold=0.002)` — **only called on training windows**:
  - Label = 1 if `close[t+4] / close[t] − 1 > 0.002`
- `xgboost_rolling_signals(close, high, low, train_window_bars, predict_window_bars, ...)`:
  - Roll: train on `[t-train_window, t-horizon]`, predict `[t, t+predict_window]`
  - `train_end = start - horizon` — **hard lookahead guard**
  - Entry: predicted probability > threshold
  - Exit: probability drops back below threshold

> **Restrict to 1h and 4h only.** Run as a separate sweep pass with `--strategies xgboost`.

Sweep grid (48 configs × timeframe):

| Param | Values |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 3, 5 |
| `min_child_weight` | 1, 3 |
| `threshold` | 0.50, 0.55, 0.60 |
| `sl_stop` | 3%, 5% |
| Train window | 6 months in bars |
| Predict window | 1 month in bars |

---

## The Funnel: 3 Rounds of Backtest → Claude Analysis → Refine

Each round narrows the search space. The goal is not to find one winner — it's to find the **best validated config for each regime**. By Round 3, each regime slot in the meta-strategy has a confirmed, walk-forward-validated strategy and param set.

```
Round 1: Wide sweep (all strategies, broad grid, ~1500 configs)
              ↓
         Claude Analysis #1
         – Which strategy wins in expanding vol regime?
         – Which wins in quiet/ranging regime?
         – Which wins in trending regime?
         – Which ML strategy (KNN or XGBoost) adds value?
              ↓
Round 2: Narrow sweep (best candidate per regime, refined grid, ~300 configs)
              ↓
         Claude Analysis #2
         – Did each regime's candidate hold up?
         – Are there regime overlaps (two strategies fighting for same condition)?
         – Pick top 2 configs per regime for Round 3
              ↓
Round 3: Fine-tune (top 2 per regime, tight ±1 step grid, ~100 configs)
              ↓
         Claude Analysis #3
         – Walk-forward results per regime candidate
         – Confirm each regime slot: which strategy + params to use
         – Flag any regime that has no validated candidate (leave empty = no trade)
              ↓
         Assemble meta-strategy → sizing sweep → walk-forward full portfolio → deploy
```

---

## Phase 2 — Round 1: Wide Sweep (Broad Grid)

Cast a wide net. Goal: eliminate losing strategy families, not find the perfect config.

### Pass A — Fast strategies + ATR ratio combos (all TFs, run first)
```bash
cd "/Users/k/Downloads/cody/hypex capital/rooroo"
/opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py \
  --strategies ema_crossover rsi atr_trend rsi_ema_filter \
              volty_expan_close \
              volty_expan_close+atr_ratio \
              rsi+atr_ratio_low \
              rsi_ema_filter+atr_ratio_low \
  --timeframes 15m 30m 1h 4h \
  --no-stop
```
Expected: ~45–90 min. The ATR ratio combos add minimal overhead (no model fitting).

### Pass B — Lorentzian KNN (1h/4h only)
```bash
/opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py \
  --strategies lorentzian_knn lorentzian_knn+garch \
  --timeframes 1h 4h \
  --no-stop
```
Expected: ~1–2 hours.

### Pass C — GARCH combos + XGBoost (run overnight)
```bash
/opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py \
  --strategies rsi+garch ema_crossover+garch \
              volty_expan_close+garch \
              volty_expan_close+atr_ratio+garch \
              xgboost \
  --timeframes 1h 4h \
  --no-stop
```
Expected: ~4–7 hours. `volty_expan+atr_ratio+garch` is the key combo to watch — highest precision Volty signal.

---

## Claude Analysis #1 — After Round 1

Paste `results/sweep_results.csv` into Claude and ask:

> "Here are the Round 1 backtest results. Analyze the top 20 configs.
> 1. Which strategy families (ema_crossover / rsi / atr_trend / lorentzian_knn / xgboost / etc.) dominate?
> 2. Which metric is the weakest bottleneck — Sortino, Sharpe, or Calmar?
> 3. Are there param values (specific SL %, timeframe, window size) that consistently appear in the top 20?
> 4. Any configs with suspiciously high IS composite that look overfit (very high return, very few trades)?
> 5. Which 2–3 strategy families should Round 2 focus on, and what param ranges should the narrow grid use?"

### What Claude will identify

| Signal | Interpretation | Action for Round 2 |
|---|---|---|
| One strategy family occupies top 10 | Strong structural edge | Focus Round 2 entirely on this family |
| Top configs cluster around one SL % | SL level is critical | Narrow SL grid to ±1 step around that value |
| Sortino high but Calmar low across all | Wins are small, losses are large | Add TP variants; test tighter SL |
| Calmar high but Sortino low | Few big wins, many small losses | Widen exit threshold; test longer TF |
| No config above composite 0.8 | Strategy families aren't working | Consider regime filter or pivot to 1d TF |
| Top configs have <0.5 trades/day | Too few signals | Loosen entry threshold; try lower TF |

---

## Phase 3 — Round 2: Narrow Sweep (Refined Grid)

Based on Claude Analysis #1, pick the top 2–3 strategy families and build a refined grid that:
- Centres param values around the Round 1 best
- Adds intermediate steps the broad grid skipped
- Tests TP variants if Calmar was weak
- Tests GARCH filter on top strategy if Sortino was weak

Example — if Round 1 top was `rsi / 1h / window=14 / lower=35 / upper=65 / sl=0.05`:

```python
# Round 2 narrow grid for RSI
rsi_windows  = [12, 13, 14, 15, 16]
lower_thresh = [32, 33, 35, 37, 38]
upper_thresh = [62, 63, 65, 67, 68]
sl_values    = [0.03, 0.04, 0.05]
tp_values    = [None, 0.08, 0.10, 0.12]   # add TP variants
timeframes   = ["1h"]                      # focus on winner TF only
```

```bash
/opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py \
  --strategies rsi rsi+garch \
  --timeframes 1h \
  --no-stop
```

Expected: ~15–30 min (much smaller grid).

---

## Claude Analysis #2 — After Round 2

Paste `results/sweep_results.csv` (Round 2 only) into Claude and ask:

> "Here are the Round 2 backtest results from a narrow grid around the Round 1 best configs.
> 1. Did the Round 1 leaders hold up or did different params win?
> 2. Did adding TP improve Calmar without hurting trade frequency?
> 3. Did the GARCH filter improve or hurt composite?
> 4. Pick the top 5 candidates with the most stable composite across nearby param values (not just the single best).
> 5. Which of these 5 looks least likely to be overfit? (consistent across param neighbours = more robust)
> 6. Recommend the final 3 configs to take into Round 3 fine-tuning."

### Robustness check Claude should perform

A config that scores composite=2.0 but all its neighbours score 0.5 is likely a fluke. Claude should look for configs where:
- Composite stays > 1.2 across at least 3 adjacent param combinations
- Metrics are balanced (Sortino, Sharpe, Calmar all > 0.5) — not one metric carrying the others
- Trade frequency is stable (±0.3 trades/day) across param variants

---

## Phase 4 — Round 3: Fine-Tune Top Configs

Take the 3 configs recommended by Claude Analysis #2. For each, build an ultra-tight grid ±1 step:

```python
# Example: winning config was RSI window=14, lower=35, upper=65, sl=0.04, tp=0.10
rsi_windows  = [13, 14, 15]
lower_thresh = [34, 35, 36]
upper_thresh = [64, 65, 66]
sl_values    = [0.035, 0.04, 0.045]
tp_values    = [0.09, 0.10, 0.11]
```

Goal: not to find a new winner, but to confirm the config is at a true local optimum (scores don't improve meaningfully with micro-adjustments).

```bash
/opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py \
  --strategies rsi \
  --timeframes 1h \
  --no-stop
```

Expected: ~5–15 min per config.

---

## Claude Analysis #3 — After Round 3

Run walk-forward on all 3 final candidates first:

```bash
/opt/anaconda3/bin/conda run -n vectorbtpro python backtest/walk_forward.py --oos-months 2
```

Then paste both `results/sweep_results.csv` and `results/walk_forward.json` into Claude and ask:

> "Here are the Round 3 fine-tuned backtest results and walk-forward validation.
> We are building a regime-switching meta-strategy with these regime slots:
>   - Expanding vol (ATR5/ATR50 > 1.2) → candidate: volty_expan or similar
>   - Quiet uptrend (ATR < 0.8, price > EMA200) → candidate: rsi_ema_filter or similar
>   - Quiet ranging (ATR < 0.8, no trend) → candidate: rsi or similar
>   - Strong trend → candidate: ema_crossover or atr_trend
>   - ML fallback → candidate: lorentzian_knn or xgboost
>
> 1. For each regime slot, which config has the best OOS/IS ratio?
> 2. Are there any regime slots where no candidate passes walk-forward (OOS/IS < 0.50)? If so, leave that slot empty — no trade is better than a bad trade.
> 3. Do any two strategies target the same regime? If so, pick the one with better Sortino.
> 4. What is the expected composite score of the assembled meta-strategy vs. the best single strategy?
> 5. Flag any regime candidate that looks overfit (IS composite >> OOS composite)."

### Walk-forward pass/fail rule

| OOS/IS ratio | Decision |
|---|---|
| ≥ 0.70 | ✅ Deploy |
| 0.50–0.70 | ⚠️ Deploy with reduced position size (50%) |
| < 0.50 | ❌ Reject — go back to Round 2 with a different family |

---

## Phase 5 — Dynamic Position Sizing (Risk Management Layer)

**Applied after the funnel finds the best strategy. Not before.**

The strategy funnel finds *which signal has edge*. This phase maximises the risk-adjusted return of that signal by sizing positions dynamically. The bot currently has no intelligent sizing — this is where Sharpe, Sortino, and Calmar get their final lift.

### Why sizing comes after strategy selection

Position sizing doesn't change entry/exit signals. It changes how much you make or lose on each trade. Optimising sizing before knowing which strategy has edge wastes time — you'd be tuning the volume knob before knowing if the song is any good.

### Three sizing methods to sweep on the winning strategy

**Method 1 — Inverse ATR Sizing** (recommended starting point)

```
position_size = (portfolio_value × target_risk_pct) / ATR_pct
```

Where `ATR_pct = ATR(14) / price`. If BTC ATR is 3% → position = `$1M × 1% / 0.03 = $333K`. If ATR spikes to 6% → position = `$167K`.

Every trade risks approximately the same dollar amount regardless of current volatility. Return std decreases → Sharpe increases directly.

Sweep: `target_risk_pct` ∈ [0.5%, 1%, 1.5%, 2%]

**Method 2 — Half Kelly**

```
f* = (p × b − q) / b
position_size = portfolio_value × (f* / 2)
```

Where `p` = win rate from IS backtest, `b` = avg_win / avg_loss, `q` = 1 − p.

Uses IS backtest stats to compute the theoretically optimal fraction, then halves it to reduce variance. Sizes proportionally to the measured edge.

Requirement: IS backtest must have ≥ 50 trades for a reliable estimate. Below that, skip Kelly and use inverse ATR.

**Method 3 — Regime-Scaled Sizing**

Multiply the base size (from Method 1 or 2) by a regime multiplier:

| Regime | Multiplier |
|---|---|
| GARCH high vol + ATR5/ATR50 expanding | 0.5× — uncertain, reduce exposure |
| Neutral | 1.0× |
| GARCH low vol + price > EMA200 | 1.25× — high conviction, size up |

Best combined with Method 1 as the base: `inverse_ATR_size × regime_multiplier`.

### How to sweep sizing methods

After Round 3 produces the winning strategy config, run a dedicated sizing sweep on that config only:

```bash
/opt/anaconda3/bin/conda run -n vectorbtpro python backtest/sweep.py \
  --strategies <winning_strategy> \
  --timeframes <winning_tf> \
  --sizing-methods inverse_atr half_kelly regime_scaled \
  --no-stop
```

Compare composite scores across sizing methods. The winning sizing method + winning strategy config goes to walk-forward together.

### What good sizing looks like in results

| Metric | Effect of good sizing |
|---|---|
| Sharpe | Increases — consistent $ return per trade |
| Sortino | Increases — downside trades are smaller in high-vol |
| Calmar | Increases — max drawdown shrinks as positions scale down in risky regimes |
| Total return | May decrease slightly — that's acceptable if composite improves |
| Trades/day | Unchanged — sizing never blocks a signal |

### ATR-Based Dynamic Stop Loss (replaces fixed 5%)

While here, also replace the fixed 5% SL with an ATR-based SL:

```
stop_loss = entry_price − N × ATR(14)
```

Sweep `N` ∈ [1.5, 2.0, 2.5, 3.0] alongside sizing method. In quiet markets ATR is small → tight SL. In volatile markets ATR is larger → wider SL that respects normal price noise, avoiding premature stop-outs.

---

## Phase 6 — Deploy

When the winning strategy + sizing method passes walk-forward:

1. Update `config.py`:
   - Remove `POSITION_SIZE_USD`
   - Add `TARGET_RISK_PCT`, `ATR_SL_MULTIPLIER`, sizing method params
2. Update `bot/runner.py` to compute position size dynamically using ATR at entry time
3. Commit with full context:
   ```bash
   git add config.py bot/runner.py
   git commit -m "strategy: <strategy> <tf> composite=X.XX oos_ratio=0.XX + dynamic sizing"
   ```
4. Redeploy:
   ```bash
   ./deploy.sh
   ```
5. Monitor first 24h — if live drawdown > 10%, pause and reassess

---

## Iteration Schedule

| Day | Action |
|---|---|
| **Mar 22 (today)** | Implement Phase 1 strategies (VEC → Lorentzian → GARCH → XGBoost) |
| **Mar 23 morning** | Run Round 1 Pass A+B. Claude Analysis #1. |
| **Mar 23 evening** | Run Round 1 Pass C overnight. |
| **Mar 24 morning** | Complete Claude Analysis #1. Run Round 2. Claude Analysis #2. |
| **Mar 24 evening** | Run Round 3. Walk-forward. Claude Analysis #3. |
| **Mar 25 morning** | Run sizing sweep on winning strategy. Walk-forward with sizing. Deploy. |
| **Mar 25–27** | Monitor live. Re-run narrow sweep if live drawdown > 10%. |
| **Mar 28** | Repo submission deadline — freeze code. |
| **Mar 29–31** | Final trading days. Emergency fix only. |

---

## Decision Tree — What to Do When Stuck

```
Round 1 sweep: no config above composite 0.8?
  → Run --no-stop to see full distribution
  → Check vbt index mismatch warning in output
  → Lower target to 0.6 temporarily to find best available
  → Try 1d timeframe — most robust, fewest params

Claude Analysis says all strategies look overfit?
  → Increase OOS window to 3 months (--oos-months 3)
  → Simplify strategy (EMA crossover only, no filters)
  → Check for data quality issues (NaN rows, gap in data)

Walk-forward always fails OOS/IS < 0.50 across all 3 candidates?
  → The market regime in the OOS window differs from IS
  → Try splitting IS/OOS differently (--oos-months 1 for a recent slice)
  → Pivot to a regime-agnostic strategy (e.g. ATR trend with wide SL)

Bot not trading (0 trades/day live)?
  → Check bot log: ssh to EC2, tail bot.log
  → Widen entry thresholds by 10%
  → Confirm config.py was updated and bot restarted

Portfolio drawdown > 15% live?
  → Manually close all positions via Roostoo dashboard immediately
  → Redeploy with tighter SL (3%) or flat (no position)
  → Do not wait for next signal — act within 1 check cycle
```

---

## Files Reference

| File | Purpose |
|---|---|
| `backtest/sweep.py` | Main parameter sweep — extend with new strategies |
| `backtest/walk_forward.py` | IS/OOS validation — run after sweep |
| `backtest/fetch_data.py` | Data download — already done ✅ |
| `bot/indicators.py` | All signal logic — add new strategy functions here |
| `bot/ml_models.py` | Heavy ML (GARCH, XGBoost) — new file to create |
| `config.py` | Live bot params — update after each deploy |
| `results/sweep_results.csv` | Full sweep output — sorted by composite |
| `results/best_params.json` | Winning config — input to walk_forward.py |
| `results/walk_forward.json` | IS/OOS validation output |
