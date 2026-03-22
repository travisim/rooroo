"""
Pure-pandas technical indicators — no external dependencies.
Used by both the live bot and the backtest sweep for consistency.
"""
import pandas as pd
import numpy as np


# ── Core smoothing helpers ─────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) — matches Pine Script ta.rma()."""
    return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ── Core indicators ────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(high, low, close):
    """True Range — works for both pd.Series (single symbol) and pd.DataFrame (multi-symbol)."""
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    # numpy fmax is NaN-safe element-wise maximum; preserves original type
    tr_vals = np.fmax(np.fmax(hl.values, hc.values), lc.values)
    if isinstance(high, pd.Series):
        return pd.Series(tr_vals, index=high.index)
    return pd.DataFrame(tr_vals, index=high.index, columns=high.columns)


def atr(high, low, close, period: int = 14):
    tr = true_range(high, low, close)
    return rma(tr, period)


def atr_expansion_ratio(
    high: pd.Series, low: pd.Series, close: pd.Series,
    short: int = 5, long: int = 50,
) -> pd.Series:
    """
    ATR(short) / ATR(long).
    > 1.2 = volatility expanding (breakout regime → Volty Expan).
    < 0.8 = volatility contracting (mean-reversion regime → RSI).
    """
    return atr(high, low, close, short) / atr(high, low, close, long).replace(0, np.nan)


def wave_trend(
    close: pd.Series, high: pd.Series, low: pd.Series,
    n1: int = 10, n2: int = 21,
) -> pd.Series:
    """Wave Trend oscillator (WT1). Used as a feature in Lorentzian KNN."""
    hlc3 = (high + low + close) / 3
    esa = ema(hlc3, n1)
    d = ema((hlc3 - esa).abs(), n1)
    ci = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
    return ema(ci, n2)


def cci(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = sma(tp, period)
    mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder's)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_val = atr(high, low, close, period)
    safe_atr = atr_val.replace(0, np.nan)
    plus_di = 100 * rma(plus_dm, period) / safe_atr
    minus_di = 100 * rma(minus_dm, period) / safe_atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx.fillna(0), period)


def normalize_series(series: pd.Series, lookback: int) -> pd.Series:
    """Rolling min-max normalization → [0, 1]. Used to normalise KNN features."""
    roll_min = series.rolling(lookback).min()
    roll_max = series.rolling(lookback).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    return (series - roll_min) / denom


# ── Crossover helpers ──────────────────────────────────────────────────────────

def crossed_above(a, b):
    """True on the bar where a crosses above b."""
    return (a > b) & (a.shift(1) <= b)


def crossed_below(a, b):
    """True on the bar where a crosses below b."""
    return (a < b) & (a.shift(1) >= b)


# ── XGBoost feature matrix ─────────────────────────────────────────────────────

def build_xgb_features(
    close: pd.Series, high: pd.Series, low: pd.Series,
) -> pd.DataFrame:
    """
    Feature matrix for XGBoost. All features use only close[t] or earlier — no lookahead.
    """
    feats = pd.DataFrame(index=close.index)
    feats["rsi_14"] = rsi(close, 14)
    feats["rsi_7"] = rsi(close, 7)
    feats["ema_diff"] = (ema(close, 9) - ema(close, 21)) / close
    feats["atr_ratio"] = atr_expansion_ratio(high, low, close, 5, 50)
    feats["cci_20"] = cci(close, high, low, 20)
    feats["adx_14"] = adx(high, low, close, 14)
    feats["return_1"] = close.pct_change(1)
    feats["return_5"] = close.pct_change(5)
    feats["return_20"] = close.pct_change(20)
    feats["day_of_week"] = close.index.dayofweek.astype(float)
    feats["hour"] = close.index.hour.astype(float)
    return feats


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
    close: pd.Series, high: pd.Series, low: pd.Series,
    ma_len: int = 100, atr_len: int = 7, atr_mult: float = 3.0,
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
    close: pd.Series, ema_len: int = 200, rsi_period: int = 14,
    oversold: float = 35, overbought: float = 65,
):
    """RSI entry gated by EMA trend filter — only longs when price is above EMA."""
    trend_up = close > ema(close, ema_len)
    rsi_vals = rsi(close, rsi_period)
    entries = crossed_below(rsi_vals, oversold) & trend_up
    exits = crossed_above(rsi_vals, overbought)
    return entries, exits


def volty_expan_close_signals(
    close: pd.Series, high: pd.Series, low: pd.Series,
    length: int = 5, mult: float = 1.0,
):
    """
    Volatility Expansion Close (VEC).
    Entry : high[t] > close[t-1] + SMA(TR, length)[t-1] * mult  (upside breakout)
    Exit  : low[t]  < close[t-1] - SMA(TR, length)[t-1] * mult  (downside break)
    Works best when ATR5/ATR50 > 1.0 (vol already expanding).
    """
    band = sma(true_range(high, low, close), length) * mult
    prev_close = close.shift(1)
    prev_band = band.shift(1)
    entries = high > prev_close + prev_band
    exits = low < prev_close - prev_band
    return entries, exits


def vec_rsi_regime_signals(
    close, high, low,
    length: int = 7, mult: float = 2.0,
    rsi_window: int = 14, rsi_lower: float = 30, rsi_upper: float = 70,
    atr_ratio_threshold: float = 1.0,
):
    """
    Blended meta-strategy: VEC + RSI with soft regime weighting.
      - VEC entries gated to high-vol regime (ATR5/ATR50 > threshold)
      - RSI entries gated to low-vol  regime (ATR5/ATR50 <= threshold)
      - Exits: each strategy closes its own regime independently,
        but either strategy can close an open position (safe behavior)

    This boosts trade frequency vs pure VEC while retaining high-composite
    VEC signals — measured against RSI as the frequency-filling component.
    """
    ratio = atr_expansion_ratio(high, low, close)
    high_vol = ratio > atr_ratio_threshold
    low_vol = ~high_vol

    vec_e, vec_x = volty_expan_close_signals(close, high, low, length, mult)
    rsi_e, rsi_x = rsi_signals(close, rsi_window, rsi_lower, rsi_upper)

    # Gate entries by regime; exits are unioned so any open position can close
    entries = (vec_e & high_vol) | (rsi_e & low_vol)
    exits = vec_x | rsi_x
    return entries, exits


def lorentzian_knn_signals(
    close: pd.Series, high: pd.Series, low: pd.Series,
    k: int = 8, max_bars_back: int = 1000,
    use_adx: bool = False, use_kernel: bool = False,
    kernel_lookback: int = 8,
):
    """
    Lorentzian KNN Classification.

    Features (normalized to [0,1] over max_bars_back rolling window):
        RSI(14), WaveTrend(10,21), CCI(20), optionally ADX(14)

    Distance metric: sum of log(1 + |f_current - f_historical[j]|) per feature.
    Samples every 4 bars back to ensure chronological spacing.

    Label at bar j = sign(close[j] - close[j-4])  — purely historical, no lookahead.
    Signal = long if sum(k-nearest labels) > 0.

    Optional kernel filter: suppress entries when EMA(close, kernel_lookback) is falling.

    Restricted to 1h+ timeframes in the sweep (bar-by-bar loop is too slow on 1m/5m).
    """
    # Multi-symbol DataFrame: dispatch per column
    if isinstance(close, pd.DataFrame):
        entries_d, exits_d = {}, {}
        for col in close.columns:
            e, x = lorentzian_knn_signals(
                close[col], high[col], low[col],
                k=k, max_bars_back=max_bars_back,
                use_adx=use_adx, use_kernel=use_kernel,
                kernel_lookback=kernel_lookback,
            )
            entries_d[col] = e
            exits_d[col] = x
        return pd.DataFrame(entries_d), pd.DataFrame(exits_d)

    n = len(close)
    close_arr = close.values.astype(float)

    # Pre-compute normalized features
    f1 = normalize_series(rsi(close, 14), max_bars_back).values.astype(float)
    f2 = normalize_series(wave_trend(close, high, low), max_bars_back).values.astype(float)
    f3 = normalize_series(cci(close, high, low, 20), max_bars_back).values.astype(float)
    feat_list = [f1, f2, f3]
    if use_adx:
        f4 = normalize_series(adx(high, low, close, 14), max_bars_back).values.astype(float)
        feat_list.append(f4)
    feat_arr = np.stack(feat_list, axis=1)  # (n, n_features)

    # Kernel estimate for optional trend filter
    kernel_arr = ema(close, kernel_lookback).values if use_kernel else None

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    for i in range(max_bars_back, n):
        if np.any(np.isnan(feat_arr[i])):
            continue

        # Candidate bars: every 4 bars, within max_bars_back, j >= 4 (need j-4 index)
        start = max(i - max_bars_back, 4)
        cand_idx = np.arange(i - 4, start - 1, -4)
        if len(cand_idx) == 0:
            continue

        # Filter candidates with valid features
        hist_feats = feat_arr[cand_idx]
        valid = ~np.any(np.isnan(hist_feats), axis=1)
        if not np.any(valid):
            continue
        cand_valid = cand_idx[valid]
        feats_valid = hist_feats[valid]

        # Lorentzian distances (vectorized)
        diffs = np.abs(feat_arr[i] - feats_valid)     # (n_valid, n_features)
        distances = np.sum(np.log1p(diffs), axis=1)   # (n_valid,)

        # Labels: sign(close[j] - close[j-4]) — strictly historical
        labels = np.sign(close_arr[cand_valid] - close_arr[cand_valid - 4])

        # k-nearest vote
        if len(distances) > k:
            top_k = np.argpartition(distances, k)[:k]
            vote = labels[top_k].sum()
        else:
            vote = labels.sum()

        # Apply kernel trend filter
        if use_kernel and i > 0:
            kernel_bullish = kernel_arr[i] > kernel_arr[i - 1]
            if vote > 0 and kernel_bullish:
                entries[i] = True
            elif vote < 0:
                exits[i] = True
        else:
            if vote > 0:
                entries[i] = True
            elif vote < 0:
                exits[i] = True

    return pd.Series(entries, index=close.index), pd.Series(exits, index=close.index)


# ── Signal dispatcher ──────────────────────────────────────────────────────────

def _apply_atr_ratio_mask(entries, exits, high, low, close, modifier: str, params: dict):
    """Apply ATR expansion ratio regime filter to entry signals."""
    is_low = modifier == "atr_ratio_low"
    threshold = params.get("atr_ratio_threshold", 0.9 if is_low else 1.1)
    ratio = atr_expansion_ratio(high, low, close)
    mask = (ratio < threshold) if is_low else (ratio > threshold)
    if isinstance(entries, pd.DataFrame):
        mask = mask.reindex(entries.index, fill_value=False)
    else:
        mask = mask.reindex(entries.index, fill_value=False)
    return entries & mask, exits


def compute_signals(strategy: str, close, high=None, low=None, **params):
    """
    Dispatch to the correct signal generator.

    Supports modifier chains via '+':
        "rsi+atr_ratio_low"            → RSI gated to contracting-vol regime
        "volty_expan_close+atr_ratio"  → VEC gated to expanding-vol regime
        "rsi+garch"                    → RSI gated by GARCH predicted vol
        "volty_expan_close+atr_ratio+garch" → VEC with both regime filters
    """
    # ── Modifier chain (e.g. "rsi+garch", "volty_expan_close+atr_ratio+garch") ──
    if "+" in strategy:
        parts = strategy.split("+")
        base_strat = parts[0]
        modifiers = parts[1:]
        entries, exits = compute_signals(base_strat, close, high, low, **params)

        for mod in modifiers:
            if mod in ("atr_ratio", "atr_ratio_low"):
                entries, exits = _apply_atr_ratio_mask(entries, exits, high, low, close, mod, params)
            elif mod == "garch":
                from bot.ml_models import garch_vol_mask
                garch_window = params.get("garch_window", 252)
                if isinstance(close, pd.DataFrame):
                    masks = {
                        col: garch_vol_mask(close[col], window=garch_window).reindex(
                            close.index, fill_value=False
                        )
                        for col in close.columns
                    }
                    mask = pd.DataFrame(masks)
                else:
                    mask = garch_vol_mask(close, window=garch_window).reindex(
                        entries.index, fill_value=False
                    )
                entries = entries & mask
        return entries, exits

    # ── Base strategies ────────────────────────────────────────────────────────
    if strategy == "ema_crossover":
        return ema_crossover_signals(close, params["fast_window"], params["slow_window"])
    elif strategy == "rsi":
        return rsi_signals(close, params["window"], params["lower"], params["upper"])
    elif strategy == "atr_trend":
        return atr_trend_signals(close, high, low, params["ma_len"], params["atr_len"], params["atr_mult"])
    elif strategy == "rsi_ema_filter":
        return rsi_ema_filter_signals(
            close, params["ema_len"], params["rsi_period"], params["oversold"], params["overbought"]
        )
    elif strategy == "volty_expan_close":
        return volty_expan_close_signals(close, high, low, params["length"], params["mult"])
    elif strategy == "vec_rsi_regime":
        return vec_rsi_regime_signals(
            close, high, low,
            length=params["length"], mult=params["mult"],
            rsi_window=params.get("rsi_window", 14),
            rsi_lower=params.get("rsi_lower", 30),
            rsi_upper=params.get("rsi_upper", 70),
            atr_ratio_threshold=params.get("atr_ratio_threshold", 1.0),
        )
    elif strategy == "lorentzian_knn":
        return lorentzian_knn_signals(
            close, high, low,
            k=params.get("k", 8),
            max_bars_back=params.get("max_bars_back", 1000),
            use_adx=(params.get("features", "") == "rsi_wt_cci_adx"),
            use_kernel=params.get("use_kernel", False),
        )
    elif strategy == "xgboost":
        from bot.ml_models import xgboost_rolling_signals
        if isinstance(close, pd.DataFrame):
            entries_d, exits_d = {}, {}
            for col in close.columns:
                e, x = xgboost_rolling_signals(close[col], high[col], low[col], **params)
                entries_d[col] = e
                exits_d[col] = x
            return pd.DataFrame(entries_d), pd.DataFrame(exits_d)
        return xgboost_rolling_signals(close, high, low, **params)
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
