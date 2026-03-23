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
    use_garch: bool = False,
) -> pd.DataFrame:
    """
    Feature matrix for XGBoost. All features use only close[t] or earlier — no lookahead.

    Base features (always computed, ~0.1s):
        RSI(14), RSI(7), EMA diff(9/21), ATR ratio(5/50), CCI(20), ADX(14),
        Bollinger %B(20), Bollinger width(20), MACD histogram(12/26/9),
        dist_ema50, dist_ema200, realized vol ratio(5/50),
        returns at 1/3/5/10/20 bars, day_of_week, hour.

    Optional GARCH features (use_garch=True, adds ~6s per symbol):
        garch_vol: GARCH(1,1) conditional volatility
        garch_vol_pct: rolling percentile rank of garch_vol (0-1)
    """
    feats = pd.DataFrame(index=close.index)

    # Momentum / mean-reversion
    feats["rsi_14"] = rsi(close, 14)
    feats["rsi_7"] = rsi(close, 7)

    # Trend
    feats["ema_diff_9_21"] = (ema(close, 9) - ema(close, 21)) / close
    feats["dist_ema50"] = (close - ema(close, 50)) / close
    feats["dist_ema200"] = (close - ema(close, 200)) / close

    # Volatility regime
    feats["atr_ratio"] = atr_expansion_ratio(high, low, close, 5, 50)
    short_vol = close.pct_change().rolling(5).std()
    long_vol = close.pct_change().rolling(50).std()
    feats["realized_vol_ratio"] = short_vol / long_vol.replace(0, np.nan)

    # Oscillators
    feats["cci_20"] = cci(close, high, low, 20)
    feats["adx_14"] = adx(high, low, close, 14)

    # Bollinger Bands
    bb_mid = sma(close, 20)
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feats["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    feats["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    # MACD histogram (normalised by price)
    macd_line = ema(close, 12) - ema(close, 26)
    feats["macd_hist"] = (macd_line - ema(macd_line, 9)) / close

    # Returns at multiple horizons
    feats["return_1"] = close.pct_change(1)
    feats["return_3"] = close.pct_change(3)
    feats["return_5"] = close.pct_change(5)
    feats["return_10"] = close.pct_change(10)
    feats["return_20"] = close.pct_change(20)

    # Calendar
    feats["day_of_week"] = close.index.dayofweek.astype(float)
    feats["hour"] = close.index.hour.astype(float)

    # Optional GARCH features
    if use_garch:
        from bot.ml_models import garch_vol_series
        gvol = garch_vol_series(close, window=252)
        feats["garch_vol"] = gvol
        # Rolling percentile rank: fraction of past 252 bars where vol was lower
        feats["garch_vol_pct"] = (
            gvol.rolling(252, min_periods=126)
            .apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan, raw=True)
        )

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


def vec_garch_regime_signals(
    close, high, low,
    length: int = 5, mult: float = 1.0,
    garch_window: int = 252, vol_percentile: float = 60.0,
):
    """
    VEC gated by GARCH(1,1) volatility regime.
    Only takes VEC entries when predicted conditional vol is ABOVE the rolling
    percentile threshold (i.e., we're in a high-vol / volatile regime).
    Exits use VEC exit signal.
    """
    from bot.ml_models import garch_vol_mask
    vec_e, vec_x = volty_expan_close_signals(close, high, low, length, mult)
    # GARCH mask: True where vol is HIGH (above percentile) — invert the low-vol mask
    # Must iterate per column when close is a DataFrame (garch_vol_mask expects Series)
    if isinstance(close, pd.DataFrame):
        low_vol_mask = pd.DataFrame(
            {col: garch_vol_mask(close[col], window=garch_window, vol_percentile=vol_percentile)
             for col in close.columns},
            index=close.index,
        ).reindex(vec_e.index, fill_value=False)
    else:
        low_vol_mask = garch_vol_mask(close, window=garch_window, vol_percentile=vol_percentile)
    high_vol_mask = ~low_vol_mask
    entries = vec_e & high_vol_mask
    return entries, vec_x


def vec_rsi_xgb_regime_signals(
    close, high, low,
    length: int = 15, mult: float = 1.0,
    rsi_window: int = 14, rsi_lower: float = 35, rsi_upper: float = 65,
    atr_ratio_threshold: float = 0.8,
    train_window_bars: int = 1080,   # 6 months on 4h
    predict_window_bars: int = 180,  # 1 month on 4h
    n_estimators: int = 100,
    max_depth: int = 3,
    xgb_threshold: float = 0.50,
):
    """
    vec_rsi_regime entries filtered by a rolling XGBoost quality gate.

    An entry fires only when BOTH conditions hold:
      1. vec_rsi_regime regime signal is active (VEC in high-vol OR RSI in low-vol)
      2. XGBoost predicted probability of an up-move exceeds xgb_threshold

    Exits use vec_rsi_regime's exit signals only — XGBoost does not affect exits.
    This preserves the base strategy's risk control while adding a precision filter.
    """
    if isinstance(close, pd.DataFrame):
        entries_d, exits_d = {}, {}
        for col in close.columns:
            e, x = vec_rsi_xgb_regime_signals(
                close[col], high[col], low[col],
                length=length, mult=mult,
                rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper,
                atr_ratio_threshold=atr_ratio_threshold,
                train_window_bars=train_window_bars,
                predict_window_bars=predict_window_bars,
                n_estimators=n_estimators, max_depth=max_depth,
                xgb_threshold=xgb_threshold,
            )
            entries_d[col] = e
            exits_d[col] = x
        return pd.DataFrame(entries_d), pd.DataFrame(exits_d)

    from bot.ml_models import xgboost_rolling_signals

    base_entries, base_exits = vec_rsi_regime_signals(
        close, high, low,
        length=length, mult=mult,
        rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper,
        atr_ratio_threshold=atr_ratio_threshold,
    )
    xgb_entries, _ = xgboost_rolling_signals(
        close, high, low,
        train_window_bars=train_window_bars,
        predict_window_bars=predict_window_bars,
        n_estimators=n_estimators,
        max_depth=max_depth,
        threshold=xgb_threshold,
    )
    entries = base_entries & xgb_entries
    return entries, base_exits


def rational_quadratic_kernel(src: np.ndarray, lookback: int, alpha: float = 1.0) -> np.ndarray:
    """
    Nadaraya-Watson Rational Quadratic Kernel smoother — vectorised (no Python loops).
    Matches the Pine Script implementation used in jdehorty's Lorentzian Classification.

    Weight for lag j: (1 + j² / (2α·lookback²))^(−α)
    Uses stride_tricks to build a sliding window matrix [n, lookback], then a
    single matrix-vector dot product replaces the O(n·lookback) Python loop.
    Memory: n × lookback × 4 bytes ≈ 35k × 16 × 4 = ~2 MB — safe on 16 GB.
    """
    n = len(src)
    # Precompute normalised weights once — shape [lookback]
    lags = np.arange(1, lookback + 1, dtype=np.float32)
    weights = (1.0 + lags ** 2 / (2.0 * alpha * lookback ** 2)) ** (-alpha)
    weights /= weights.sum()

    # Sliding window matrix via stride tricks: shape [n, lookback]
    # Row i contains [src[i], src[i-1], ..., src[i-lookback+1]]
    src_f = src.astype(np.float32)
    padded = np.pad(src_f, (lookback - 1, 0), mode="edge")
    s = padded.strides[0]
    window = np.lib.stride_tricks.as_strided(
        padded[lookback - 1:], shape=(n, lookback), strides=(s, -s)
    )

    out = window @ weights          # shape [n], fully vectorised
    out = out.astype(np.float64)
    out[:lookback] = np.nan         # warmup period
    return out


def lorentzian_knn_signals(
    close: pd.Series, high: pd.Series, low: pd.Series,
    k: int = 8, max_bars_back: int = 2000,
    use_adx: bool = True, use_kernel: bool = True,
    kernel_lookback: int = 8, kernel_alpha: float = 1.0,
    horizon: int = 4, sample_step: int = 4,
):
    """
    Lorentzian KNN Classification — fixed to match jdehorty's original Pine Script.

    KEY FIX: Labels are FORWARD-looking for historical bars (no lookahead because
    those future bars are already in the past when predicting the current bar):
        label[j] = 1 if close[j + horizon] > close[j] else -1
    Valid candidates: j + horizon < i  (strictly no lookahead)

    Distance metric: Lorentzian = sum(log(1 + |f_i - f_j|)) per feature.
    Samples every `sample_step` bars to avoid autocorrelation.

    Features (rolling min-max normalised over max_bars_back):
        RSI(14), WaveTrend(10,21), CCI(20), RSI(9), optionally ADX(14)

    Optional Rational Quadratic Kernel: smooths the raw vote series and derives
    crossover-based entry/exit, matching the original script's kernel regression.
    """
    if isinstance(close, pd.DataFrame):
        entries_d, exits_d = {}, {}
        for col in close.columns:
            e, x = lorentzian_knn_signals(
                close[col], high[col], low[col],
                k=k, max_bars_back=max_bars_back,
                use_adx=use_adx, use_kernel=use_kernel,
                kernel_lookback=kernel_lookback, kernel_alpha=kernel_alpha,
                horizon=horizon, sample_step=sample_step,
            )
            entries_d[col] = e
            exits_d[col] = x
        return pd.DataFrame(entries_d), pd.DataFrame(exits_d)

    n = len(close)
    close_arr = close.values.astype(float)

    # Features — 5 features matching jdehorty's default set
    f1 = normalize_series(rsi(close, 14), max_bars_back).values.astype(float)
    f2 = normalize_series(wave_trend(close, high, low), max_bars_back).values.astype(float)
    f3 = normalize_series(cci(close, high, low, 20), max_bars_back).values.astype(float)
    f4 = normalize_series(rsi(close, 9), max_bars_back).values.astype(float)
    feat_list = [f1, f2, f3, f4]
    if use_adx:
        f5 = normalize_series(adx(high, low, close, 14), max_bars_back).values.astype(float)
        feat_list.append(f5)
    feat_arr = np.stack(feat_list, axis=1)

    # ── Vectorised KNN vote (chunked to cap RAM at ~120 MB; MLX GPU if available) ──
    # Replaces the O(n) Python bar loop with batched numpy/MLX ops.
    # chunk_size=4000 → peak memory per chunk ≈ 4000×500×5×4 bytes ≈ 40 MB.
    raw_vote = np.full(n, np.nan)
    warmup = max(max_bars_back, horizon + sample_step)
    n_cand_max = max_bars_back // sample_step          # e.g. 2000/4 = 500
    offsets = np.arange(1, n_cand_max + 1, dtype=np.int32) * sample_step
    feat_f32 = feat_arr.astype(np.float32)

    try:
        import mlx.core as _mx
        _USE_MLX = True
    except Exception:
        _USE_MLX = False

    CHUNK = 4000
    for chunk_start in range(warmup, n, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n)
        q_bars = np.arange(chunk_start, chunk_end, dtype=np.int32)   # [q]
        q = len(q_bars)

        # Candidate indices: [q, n_cand_max] — fixed sliding window
        cand_idx = q_bars[:, None] - offsets[None, :]                  # [q, C]
        # Lookahead guard: label bar (cand + horizon) must be <= current bar (q)
        # i.e. offset >= horizon; prevents future-data leakage when sample_step < horizon
        valid = (cand_idx >= 0) & (cand_idx + horizon <= q_bars[:, None])  # [q, C]
        cand_safe = np.clip(cand_idx, 0, n - 1)

        # Feature validity for queries
        has_nan = np.any(np.isnan(feat_f32[q_bars]), axis=1)           # [q]

        # Gather candidate features: [q, C, d]
        c_feats = feat_f32[cand_safe]                                  # [q, C, d]

        # Lorentzian distances: sum(log1p(|qi - cj|)) — GPU if MLX available
        if _USE_MLX:
            q_mlx = _mx.array(feat_f32[q_bars])[:, None, :]           # [q, 1, d]
            c_mlx = _mx.array(c_feats)                                 # [q, C, d]
            dists_mlx = _mx.sum(_mx.log1p(_mx.abs(q_mlx - c_mlx)), axis=-1)
            _mx.eval(dists_mlx)
            dists = np.array(dists_mlx)                                # [q, C]
        else:
            diffs = np.abs(feat_f32[q_bars][:, None, :] - c_feats)    # [q, C, d]
            dists = np.sum(np.log1p(diffs), axis=2)                    # [q, C]

        # Labels: +1/−1; 0 for invalid candidates
        future_idx = np.clip(cand_safe + horizon, 0, n - 1)
        labels = np.where(close_arr[future_idx] > close_arr[cand_safe], 1.0, -1.0)
        labels[~valid] = 0.0
        dists[~valid] = 1e10
        dists[has_nan] = 1e10   # exclude NaN-feature query bars

        # Vectorised top-k vote for all q bars at once
        n_valid = (dists < 1e9).sum(axis=1)                           # [q]
        k_safe = min(k, n_cand_max - 1)
        top_idx = np.argpartition(dists, k_safe, axis=1)[:, :k_safe]  # [q, k]
        votes_topk = np.take_along_axis(labels, top_idx, axis=1).sum(axis=1)
        votes_all = labels.sum(axis=1)                                 # [q] (invalid=0)

        votes = np.where(n_valid >= k, votes_topk, votes_all)
        votes[n_valid == 0] = np.nan
        votes[has_nan]       = np.nan
        raw_vote[chunk_start:chunk_end] = votes

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    if use_kernel:
        # Rational Quadratic Kernel smoother on the vote series — matches original script
        vote_filled = np.where(np.isnan(raw_vote), 0.0, raw_vote)
        smoothed = rational_quadratic_kernel(vote_filled, kernel_lookback, kernel_alpha)
        # Entry: smoothed crosses above 0; Exit: smoothed crosses below 0
        for i in range(1, n):
            if np.isnan(smoothed[i]) or np.isnan(smoothed[i - 1]):
                continue
            if smoothed[i] > 0 and smoothed[i - 1] <= 0:
                entries[i] = True
            elif smoothed[i] < 0 and smoothed[i - 1] >= 0:
                exits[i] = True
    else:
        for i in range(n):
            if np.isnan(raw_vote[i]):
                continue
            if raw_vote[i] > 0:
                entries[i] = True
            elif raw_vote[i] < 0:
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
        features = params.get("features", "rsi_wt_cci_rsi9_adx")
        return lorentzian_knn_signals(
            close, high, low,
            k=params.get("k", 8),
            max_bars_back=params.get("max_bars_back", 2000),
            use_adx=("adx" in features),
            use_kernel=params.get("use_kernel", True),
            kernel_lookback=params.get("kernel_lookback", 8),
            kernel_alpha=params.get("kernel_alpha", 1.0),
            horizon=params.get("horizon", 4),
            sample_step=params.get("sample_step", 4),
        )
    elif strategy == "vec_garch_regime":
        return vec_garch_regime_signals(
            close, high, low,
            length=params.get("length", 5),
            mult=params.get("mult", 1.0),
            garch_window=params.get("garch_window", 252),
            vol_percentile=params.get("vol_percentile", 60.0),
        )
    elif strategy == "vec_rsi_xgb_regime":
        return vec_rsi_xgb_regime_signals(
            close, high, low,
            length=params.get("length", 15),
            mult=params.get("mult", 1.0),
            rsi_window=params.get("rsi_window", 14),
            rsi_lower=params.get("rsi_lower", 35),
            rsi_upper=params.get("rsi_upper", 65),
            atr_ratio_threshold=params.get("atr_ratio_threshold", 0.8),
            train_window_bars=params.get("train_window_bars", 1080),
            predict_window_bars=params.get("predict_window_bars", 180),
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            xgb_threshold=params.get("xgb_threshold", 0.50),
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
