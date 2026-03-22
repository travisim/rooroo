"""
Binance public REST API — no authentication required.
Fetches OHLCV klines for live signal computation.
"""
import requests
import pandas as pd
import logging

log = logging.getLogger(__name__)

BINANCE_BASE = "https://data-api.binance.vision"  # Public data API — no geo-restrictions


def fetch_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance public API.

    Args:
        symbol:   e.g. "BTCUSDT"
        interval: e.g. "1h", "4h", "15m"
        limit:    number of candles (max 1000)

    Returns:
        DataFrame with columns: open_time, open, high, low, close, volume
        Index is DatetimeIndex (UTC).
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Binance klines fetch failed for {symbol}/{interval}: {e}")
        raise

    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    log.debug(f"Fetched {len(df)} candles for {symbol}/{interval}")
    return df


def fetch_ticker_price(symbol: str) -> float:
    """Fetch the latest spot price for a symbol."""
    url = f"{BINANCE_BASE}/api/v3/ticker/price"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])
