"""
Trading bot configuration.
Run `python backtest/sweep.py` to find optimal parameters, then update this file.
"""

# ── Active Strategy ────────────────────────────────────────────────────────────
STRATEGY = "rsi"                # "ema_crossover" | "rsi" | "atr_trend" | "rsi_ema_filter"
TIMEFRAME = "15m"               # Binance kline interval: "15m", "30m", "1h", "4h"

# ── EMA Crossover Parameters ───────────────────────────────────────────────────
EMA_FAST = 9
EMA_SLOW = 21

# ── RSI Parameters ────────────────────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERSOLD = 35               # Buy when RSI crosses below this
RSI_OVERBOUGHT = 65             # Sell when RSI crosses above this

# ── RSI + EMA Filter Parameters ───────────────────────────────────────────────
RSI_EMA_FILTER_LEN = 200        # Only buy when price > this EMA

# ── ATR Trend Parameters (mirrors Pine Script) ─────────────────────────────────
ATR_MA_LEN = 100
ATR_LEN = 7
ATR_MULT = 3.0

# ── Risk Management ───────────────────────────────────────────────────────────
STOP_LOSS_PCT = 0.05            # 5% hard stop loss from entry
TAKE_PROFIT_PCT = None          # None = rely on exit signal; e.g. 0.10 = 10% TP

# ── Portfolio ─────────────────────────────────────────────────────────────────
ASSETS = ["BTC/USD", "ETH/USD", "SOL/USD"]
POSITION_SIZE_USD = 280_000     # USD per position (max 3 open = $840K / $1M)
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
