"""
Heavy ML models — GARCH and XGBoost.
Kept separate from indicators.py so the live bot never imports these
unless explicitly running a GARCH or XGBoost strategy.
"""
import numpy as np
import pandas as pd


# ── GARCH Volatility Filter ────────────────────────────────────────────────────

def garch_vol_series(
    close: pd.Series,
    window: int = 252,
    refit_every: int = 96,
) -> pd.Series:
    """
    Returns GARCH(1,1) predicted conditional volatility as a float Series.
    Aligned to close.index; NaN for warmup bars.

    Use as a raw feature for ML models. garch_vol_mask wraps this
    with a rolling-percentile threshold to produce a boolean entry gate.

    Lookahead rule: fit window is strictly [t-window, t-1].
    """
    from arch import arch_model

    log_rets = (np.log(close / close.shift(1)).dropna() * 100)
    n = len(log_rets)

    predicted_vols = []
    last_params = None
    last_var = None

    for i in range(window, n):
        if i % refit_every == 0 or last_params is None:
            train = log_rets.iloc[i - window:i]
            try:
                am = arch_model(train, vol="Garch", p=1, q=1, dist="normal", rescale=False)
                res = am.fit(disp="off", show_warning=False)
                last_params = res.params
                forecast = res.forecast(horizon=1)
                pred_var = float(forecast.variance.iloc[-1, 0])
                last_var = pred_var
            except Exception:
                predicted_vols.append(np.nan)
                continue
        else:
            # Fast GARCH(1,1) one-step variance update:
            #   σ²_t = omega + alpha * ε²_{t-1} + beta * σ²_{t-1}
            try:
                p = last_params
                omega = float(p.get("omega", p.iloc[0]))
                alpha = float(p.get("alpha[1]", p.iloc[1]))
                beta  = float(p.get("beta[1]",  p.iloc[2]))
                eps_sq = float(log_rets.iloc[i - 1]) ** 2
                pred_var = omega + alpha * eps_sq + beta * (last_var if last_var else omega)
                last_var = pred_var
            except Exception:
                predicted_vols.append(predicted_vols[-1] if predicted_vols else np.nan)
                continue

        predicted_vols.append(np.sqrt(max(pred_var, 0)))

    result = pd.Series(np.nan, index=close.index)
    if predicted_vols:
        result.loc[log_rets.index[window:]] = predicted_vols
    return result


def garch_vol_mask(
    close: pd.Series,
    window: int = 252,
    vol_percentile: float = 75.0,
    refit_every: int = 96,   # 24→96: 4× faster; GARCH params stable over weeks
) -> pd.Series:
    """
    Returns a boolean Series: True where predicted conditional vol is BELOW
    the rolling vol_percentile threshold (i.e., entry is permitted).

    Fits GARCH(1,1) on a rolling window of log-returns using only past data.
    Refits every `refit_every` bars (not every bar) for speed.

    Lookahead rule: fit window is strictly [t-window, t-1]. Forecast is for
    bar t+1 and is used as the gate for bar t+1's entry signal.
    """
    vol_s = garch_vol_series(close, window=window, refit_every=refit_every)
    vol_series = vol_s.dropna()
    rolling_pct = vol_series.rolling(window, min_periods=window // 2).quantile(vol_percentile / 100)
    allowed = vol_series < rolling_pct
    mask = pd.Series(False, index=close.index)
    mask.loc[allowed.index] = allowed.values
    return mask


# ── XGBoost Rolling Classifier ─────────────────────────────────────────────────

def build_xgb_labels(
    close: pd.Series,
    horizon: int = 4,
    threshold: float = 0.002,
) -> pd.Series:
    """
    Label = 1 if close[t+horizon] / close[t] - 1 > threshold, else 0.

    CRITICAL: This function uses future data. Call ONLY on training windows.
    Never call on the prediction (current) window.
    """
    future_ret = close.shift(-horizon) / close - 1
    return (future_ret > threshold).astype(int)


def xgboost_rolling_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    train_window_bars: int = 4320,   # ~6 months on 1h
    predict_window_bars: int = 720,  # ~1 month on 1h
    n_estimators: int = 100,
    max_depth: int = 3,
    min_child_weight: int = 1,
    threshold: float = 0.5,
    horizon: int = 4,
    label_threshold: float = 0.002,
    use_garch: bool = False,
    **kwargs,
) -> tuple[pd.Series, pd.Series]:
    """
    Rolling train/predict XGBoost classifier.

    Structure:
        Train on [start - train_window, start - horizon]  (strictly past)
        Predict on [start, start + predict_window)        (out-of-sample)
        Roll forward by predict_window_bars.

    Lookahead guard: train_end = start - horizon ensures the last `horizon`
    bars of the training window have no valid forward label, so build_xgb_labels
    produces NaN there — dropna() excludes them automatically.

    Entry: predicted probability > threshold.
    Exit:  predicted probability <= threshold.

    use_garch: if True, adds GARCH conditional volatility features (~6s extra per symbol).
    """
    from xgboost import XGBClassifier
    from bot.indicators import build_xgb_features

    features = build_xgb_features(close, high, low, use_garch=use_garch)
    labels_full = build_xgb_labels(close, horizon=horizon, threshold=label_threshold)

    n = len(close)
    pred_proba = pd.Series(np.nan, index=close.index)

    start = train_window_bars
    while start < n:
        end = min(start + predict_window_bars, n)

        # Training window: exclude last `horizon` bars (no valid labels there)
        train_start = start - train_window_bars
        train_end = start - horizon

        if train_end <= train_start + 50:
            start = end
            continue

        X_train = features.iloc[train_start:train_end].dropna()
        y_train = labels_full.iloc[train_start:train_end].loc[X_train.index].dropna()
        X_train = X_train.loc[y_train.index]

        if len(X_train) < 50 or y_train.nunique() < 2:
            start = end
            continue

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
        model.fit(X_train.values, y_train.values)

        # Predict on out-of-sample window — strictly no training data
        X_pred = features.iloc[start:end].dropna()
        if len(X_pred) > 0:
            proba = model.predict_proba(X_pred.values)[:, 1]
            pred_proba.loc[X_pred.index] = proba

        start = end

    entries = (pred_proba > threshold).fillna(False)
    exits = (pred_proba <= threshold).fillna(False)
    return entries, exits
