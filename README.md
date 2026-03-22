# Roostoo Trading Bot

Autonomous cryptocurrency trading bot for the Roostoo hackathon. Trades BTC/USD, ETH/USD, and SOL/USD on spot markets using systematic momentum and trend-following strategies.

## Strategy

The active strategy is configured in `config.py`. The current default is an **EMA crossover (9/21)** on the 1-hour timeframe with a 5% stop loss:

- **Entry**: Fast EMA (9) crosses above Slow EMA (21)
- **Exit**: Fast EMA crosses below Slow EMA, or stop loss/take profit triggered
- **Assets**: BTC/USD, ETH/USD, SOL/USD (independent signals per asset)
- **Position size**: ~$280,000 per asset (≤ 84% portfolio exposure at any time)
- **Fees**: 0.1% taker (MARKET orders)

Available strategies: `ema_crossover`, `rsi`, `atr_trend`, `rsi_ema_filter`.

## Project Structure

```
├── config.py              # Strategy parameters (edit after running sweep)
├── roostoo_client.py      # Roostoo API client (auth, orders, balance)
├── bot/
│   ├── runner.py          # Main bot loop — runs autonomously every hour
│   ├── indicators.py      # Pure-pandas indicators (EMA, RSI, ATR, signals)
│   └── market_data.py     # Binance public REST API for live OHLCV
├── backtest/
│   ├── fetch_data.py      # Downloads 1-year OHLCV from Binance (all timeframes)
│   └── sweep.py           # Parameter sweep ("ralph loop") — finds best strategy
├── data/                  # Cached Binance OHLCV (HDF5, git-ignored)
└── results/               # Sweep results CSV and best_params.json
```

## Setup

### Live Bot (EC2 deployment)

```bash
cp .env.example .env
# Fill in ROOSTOO_API_KEY and ROOSTOO_API_SECRET

pip install -r requirements.txt

# Run locally:
python bot/runner.py

# Deploy to EC2:
./deploy.sh
```

### Backtest & Strategy Optimization (local, requires vbtpro conda env)

```bash
# Install extra dependency in vbtpro env
conda run -n vbtpro pip install python-dotenv

# Step 1: Download 1 year of OHLCV data for all timeframes
conda run -n vbtpro python backtest/fetch_data.py

# Step 2: Run parameter sweep (ralph loop) across all strategies
conda run -n vbtpro python backtest/sweep.py --timeframes 1h 4h

# Full sweep including short timeframes (takes longer):
conda run -n vbtpro python backtest/sweep.py --all-timeframes

# Step 3: Update config.py with best_params from results/best_params.json
# Step 4: Redeploy: ./deploy.sh
```

## Optimization Metric

The sweep optimizes for the **competition composite score**:

```
Composite = 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar
```

Subject to: `trades/day ≥ 1`, `max_drawdown < 25%`.

The sweep stops early (ralph loop) when a configuration exceeds the composite threshold.

## Risk Management

- **Stop loss**: Hard stop at 5% below entry price (configurable via `STOP_LOSS_PCT`)
- **Take profit**: Optional, disabled by default (strategy exit signal used instead)
- **Max exposure**: 3 × $280K = $840K of $1M portfolio
- **No pyramiding**: One position per asset at a time
- **Spot only**: No leverage, no short selling

## Backtested Timeframes

| Timeframe | Use case |
|-----------|----------|
| 1m, 5m    | Short-term signal research |
| 15m, 30m  | Intraday momentum |
| 1h        | Primary: balances signal frequency and reliability |
| 4h        | Trend confirmation |
| 1d        | Long-term trend filter |
