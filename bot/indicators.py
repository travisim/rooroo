"""
Pure-pandas technical indicators — no external dependencies.
Used by both the live bot and the backtest sweep for consistency.
"""
import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) — matches Pine Script ta.rma()."""
    return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, period)


def crossed_above(a, b):
    """True on the bar where a crosses above b. Works for Series, DataFrames, and scalars."""
    b_prev = b if isinstance(b, (pd.Series, pd.DataFrame)) else b
    return (a > b) & (a.shift(1) <= b_prev)


def crossed_below(a, b):
    """True on the bar where a crosses below b. Works for Series, DataFrames, and scalars."""
    b_prev = b if isinstance(b, (pd.Series, pd.DataFrame)) else b
    return (a < b) & (a.shift(1) >= b_prev)


# ── Signal generators ──────────────────────────────────────────────────────────

def ema_crossover_signals(close: pd.Series, fast: int, slow: int):
    """EMA crossover: buy fast>slow crossover, sell fast<slow crossover."""
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    entries = crossed_above(fast_ema, slow_ema)
    exits = crossed_below(fast_ema, slow_ema)
    return entries, exits


def rsi_signals(close: pd.Series, period: int, oversold: float, overbought: float):
    """RSI: buy on oversold cross, sell on overbought cross."""
    rsi_vals = rsi(close, period)
    entries = crossed_below(rsi_vals, oversold)
    exits = crossed_above(rsi_vals, overbought)
    return entries, exits


def atr_trend_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    ma_len: int = 100,
    atr_len: int = 7,
    atr_mult: float = 3.0,
):
    """ATR trend breakout — mirrors the existing Pine Script strategy."""
    ma_val = ema(close, ma_len)
    atr_val = atr(high, low, close, atr_len)
    upper = ma_val + atr_val * atr_mult
    lower = ma_val - atr_val * atr_mult
    entries = crossed_above(close, upper)
    exits = crossed_below(close, lower)
    return entries, exits


def rsi_ema_filter_signals(
    close: pd.Series,
    ema_len: int = 200,
    rsi_period: int = 14,
    oversold: float = 35,
    overbought: float = 65,
):
    """RSI entry gated by EMA trend filter — only longs when above EMA."""
    trend_up = close > ema(close, ema_len)
    rsi_vals = rsi(close, rsi_period)
    entries = crossed_below(rsi_vals, oversold) & trend_up
    exits = crossed_above(rsi_vals, overbought)
    return entries, exits


def compute_signals(strategy: str, close, high=None, low=None, **params):
    """Dispatch to the correct signal generator."""
    if strategy == "ema_crossover":
        return ema_crossover_signals(close, params["fast_window"], params["slow_window"])
    elif strategy == "rsi":
        return rsi_signals(close, params["window"], params["lower"], params["upper"])
    elif strategy == "atr_trend":
        return atr_trend_signals(close, high, low, params["ma_len"], params["atr_len"], params["atr_mult"])
    elif strategy == "rsi_ema_filter":
        return rsi_ema_filter_signals(close, params["ema_len"], params["rsi_period"], params["oversold"], params["overbought"])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def latest_signal(strategy: str, df: pd.DataFrame, **params) -> tuple[bool, bool]:
    """
    Compute entry/exit signal for the latest COMPLETE candle.
    df must have columns: open, high, low, close (lowercase).
    Returns (entry, exit) booleans.
    """
    close = df["close"]
    high = df.get("high", df["close"])
    low = df.get("low", df["close"])

    entries, exits = compute_signals(strategy, close, high, low, **params)

    # Use second-to-last row: last row may be an open (incomplete) candle
    idx = -2
    return bool(entries.iloc[idx]), bool(exits.iloc[idx])
