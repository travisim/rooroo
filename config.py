"""
Trading bot configuration.
Run `python backtest/sweep.py` to find optimal parameters, then update this file.
"""

# ── Active Strategy ────────────────────────────────────────────────────────────
STRATEGY = "vec_rsi_regime"  # best from full sweep — composite=1.530, t/d=0.906, dd=15.3%
TIMEFRAME = "4h"

# ── EMA Crossover Parameters ───────────────────────────────────────────────────
EMA_FAST = 9
EMA_SLOW = 21

# ── RSI Parameters ────────────────────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERSOLD = 35               # Buy when RSI crosses below this (sweep optimised)
RSI_OVERBOUGHT = 65             # Sell when RSI crosses above this (sweep optimised)

# ── RSI + EMA Filter Parameters ───────────────────────────────────────────────
RSI_EMA_FILTER_LEN = 200        # Only buy when price > this EMA

# ── ATR Trend Parameters (mirrors Pine Script) ─────────────────────────────────
ATR_MA_LEN = 100
ATR_LEN = 7
ATR_MULT = 3.0

# ── Volty Expan Close Parameters ─────────────────────────────────────────────
VEC_LENGTH = 15                 # SMA period for true-range band (sweep optimised)
VEC_MULT = 1.0                  # Band multiplier (sweep optimised)
VEC_ATR_RATIO_THRESHOLD = 0.8  # ATR5/ATR50 threshold for regime gate

# ── vec_rsi_regime Parameters ────────────────────────────────────────────────
VEC_RSI_ATR_THRESHOLD = 0.8    # above → VEC; below → RSI (sweep optimised)

# ── Risk Management ───────────────────────────────────────────────────────────
# ATR-based dynamic position sizing:  size = portfolio × TARGET_RISK_PCT / (ATR_SL_MULT × ATR_pct)
TARGET_RISK_PCT = 0.02          # Risk 2% of portfolio per trade
ATR_SL_MULT = 2.0               # Stop loss = N × ATR(14) below entry
ATR_PERIOD = 14                 # ATR lookback for sizing and stop-loss
MAX_POSITION_PCT = 0.40         # Hard cap: max 40% of portfolio per position
TAKE_PROFIT_PCT = None          # None = rely on exit signal
FIXED_SL_PCT = 0.03             # Fixed 3% stop loss (matches backtest sl_stop=0.03)

# ── Portfolio ─────────────────────────────────────────────────────────────────
ASSETS = ["BTC/USD", "ETH/USD", "SOL/USD"]
CANDLE_HISTORY = 300            # Historical candles to fetch for indicator warmup

# ── Execution ─────────────────────────────────────────────────────────────────
ORDER_TYPE = "MARKET"           # "MARKET" (taker 0.1%) | "LIMIT" (maker 0.05%)
CHECK_INTERVAL_SEC = 900        # Signal check frequency in seconds (900 = 15m)

# ── Asset Mapping ─────────────────────────────────────────────────────────────
ROOSTOO_TO_BINANCE = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
}
BINANCE_INTERVAL = {            # Binance kline interval codes
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}
