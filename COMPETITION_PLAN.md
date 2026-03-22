# Roostoo Hackathon — Competition Plan
**Goal: Win the composite score leaderboard and advance to finals.**

---

## Current Status (Mar 22)

| Item | Status |
|---|---|
| Bot deployed to EC2 | ✅ Running (`rooroo-bot.service`) |
| Trades today | ✅ 6 trades (3 BUY + 3 SELL, BTC/ETH/SOL) |
| Active strategy | RSI(14) on 15m, threshold 35/65 |
| Data fetch | ⏳ Running (1 year OHLCV for 7 timeframes) |
| Backtest sweep | ⏳ Pending (run after data fetch) |

---

## Competition Timeline

| Date | Action |
|---|---|
| **Mar 22 (TODAY)** | ✅ Bot deployed, 6 trades placed, strategy running |
| **Mar 23** | Run full backtest sweep overnight → update config → redeploy |
| **Mar 24–27** | Let optimized bot trade. Monitor daily, fix any issues. |
| **Mar 28** | Submit open-source repo link by this date |
| **Mar 29–31** | Final trading days. Freeze strategy unless critical fix needed. |

---

## Evaluation Breakdown — How to Win Each Screen

### Screen 1: Rule Compliance (mandatory)
- **Requirement**: Autonomous execution, no manual API calls, traceable commit history
- **Our approach**:
  - Bot runs as `systemd` service on EC2 — 100% autonomous
  - Every strategy change = new commit before redeploying
  - `OrderSource: "PUBLIC_API"` in every order proves bot execution (not manual)
  - Commit message convention: `strategy: <description>` for every param change

### Screen 2: Portfolio Returns (top 20 qualify)
- **Requirement**: Maximize (Final Value − Initial Value) / Initial Value
- **Our approach**:
  - Current: RSI oversold dips → buy, recover → sell (mean reversion)
  - After sweep: use best-composite-score strategy (likely trend following on 1h/4h)
  - Target: >10% return over 10 days (aggressive but achievable in crypto)
  - Position size: $280K × 3 assets = up to $840K deployed when all in signal

### Screen 3: Composite Score (40% of judging)
```
Score = 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar
```
- **Sortino** (highest weight): minimize downside volatility → use hard stop loss (5%)
- **Sharpe**: consistent positive returns → avoid whipsaw strategies
- **Calmar**: CAGR / max drawdown → cut losses fast, let winners run
- **Our approach**: Backtest sweep optimizes directly for this composite formula

### Screen 4: Code Quality (60% of judging — most important!)
- **Strategy clarity (30%)**: `bot/indicators.py` has clean, commented signal logic
- **Clean code (20%)**: modular structure (bot/, backtest/, config.py), typed functions
- **Roostoo compatibility (10%)**: `roostoo_client.py` → proven working API client

---

## Strategy Architecture

```
Binance OHLCV (data-api.binance.vision)
         ↓
   bot/indicators.py   ←── config.py (all params)
         ↓
   Signal: entry/exit boolean
         ↓
   roostoo_client.py  →  Roostoo Mock API
         ↓
   bot/bot.log  (every decision logged)
```

### Current Strategy (RSI/15m)
- **Entry**: RSI(14) crosses BELOW 35 → BUY $280K
- **Exit**: RSI(14) crosses ABOVE 65 → SELL all
- **Stop**: 5% below entry price (hard stop, checked each cycle)
- **Frequency**: Checks every 15 min, expects 2-4 trades/day per asset

### After Sweep (to be deployed Mar 23-24)
Best strategy from `backtest/sweep.py` — likely EMA crossover or RSI+EMA filter on 1h.

---

## Backtest Sweep Plan

### Step 1: Fetch data (run once)
```bash
conda run -n vbtpro python backtest/fetch_data.py
```
Downloads 1 year of 1m/5m/15m/30m/1h/4h/1d OHLCV for BTC/ETH/SOL.

### Step 2: Run ralph loop
```bash
# Fast (1h + 4h only, ~15 min):
conda run -n vbtpro python backtest/sweep.py --timeframes 1h 4h

# Full sweep (all timeframes, ~2 hours):
conda run -n vbtpro python backtest/sweep.py --all-timeframes
```
Sweeps ~1,500 configurations, stops when `composite > 1.5` AND `trades/day ≥ 1`.

### Step 3: Review results
```
results/sweep_results.csv   — full results sorted by composite score
results/best_params.json    — winning configuration
```

### Step 4: Update + redeploy
1. Update `config.py` with winning params
2. `git add config.py && git commit -m "strategy: update to <strategy> <tf> (composite=X.XX)"`
3. `./deploy.sh`

---

## Strategies in the Sweep

| Strategy | Timeframes | Key Params | Expected Trade Freq |
|---|---|---|---|
| **EMA Crossover** | 1h, 4h | fast=8-20, slow=21-100 | 1-3/day |
| **RSI** | 15m, 30m, 1h, 4h | window=14/21, thresh=30-40/60-70 | 2-5/day |
| **ATR Trend** | 1h, 4h | ma=50-200, mult=2-4, atr=7-14 | 1-2/day |
| **RSI+EMA Filter** | 1h, 4h | ema=50-200, rsi same | 1-3/day |

Stop loss: 3-8% tested across all. Take profit: 6-15% tested.

---

## Risk Management Rules (Non-negotiable)

1. **Hard stop loss**: 5% below entry, checked every 15 min cycle
2. **Position limit**: $280K per asset, max 3 open = $840K (84% deployed)
3. **No pyramiding**: one position per asset at a time
4. **Long only**: no shorting (spot constraints + competition rules)
5. **Fees**: 0.1% taker per trade — strategy must have >0.2% edge per round trip

---

## Daily Workflow (Mar 23–31)

**Each morning:**
1. Check EC2 bot log: `ssh ... 'tail -50 bot/bot.log'`
2. Check Roostoo portfolio P&L on the Roostoo dashboard
3. If large drawdown (>10% portfolio): review and optionally redeploy different config
4. If bot crashed: `ssh ... 'sudo systemctl restart rooroo-bot'`

**Evening:**
1. Review day's trades in bot log
2. Check if sweep found better params
3. Commit any config changes with clear messages

---

## Monitoring Commands

```bash
# EC2 bot log (live)
ssh -i ~/Downloads/Sydney.pem ubuntu@ec2-3-104-220-58.ap-southeast-2.compute.amazonaws.com \
  'tail -f /home/ubuntu/rooroo/bot/bot.log'

# Bot service status
ssh -i ~/Downloads/Sydney.pem ubuntu@ec2-3-104-220-58.ap-southeast-2.compute.amazonaws.com \
  'sudo systemctl status rooroo-bot'

# Roostoo balance check (local)
conda run -n vbtpro python -c "
import sys; sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv()
import roostoo_client as rc, json
print(json.dumps(rc.get_balance(), indent=2))
"

# Run sweep (local, vbtpro env)
conda run -n vbtpro python backtest/sweep.py --timeframes 1h 4h

# Redeploy after config change
./deploy.sh
```

---

## Winning Formula

```
Rank 1 (Returns) + Rank 1 (Composite) + Clean Code = WIN
```

**Returns**: RSI mean-reversion catches oversold bounces. Trend following on 1h/4h catches sustained moves.

**Composite**: Stop losses control Calmar. Consistent edge controls Sharpe. Asymmetric wins control Sortino.

**Code**: Clean Python, modular, single `config.py` for all params, clear git history.

---

## Repo Submission Checklist (due Mar 28)

- [ ] `README.md` — strategy description, setup instructions, backtest results table
- [ ] `config.py` — current winning params clearly documented
- [ ] `bot/` — clean, well-commented autonomous bot
- [ ] `backtest/` — sweep methodology + walk-forward validation
- [ ] `results/sweep_results.csv` — show the research work
- [ ] Git history — clean commits with strategy rationale in messages
- [ ] `.env.example` — no real keys in repo
- [ ] `requirements.txt` — reproducible environment
