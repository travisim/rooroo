#!/usr/bin/env python3
"""
Pull and cache Binance OHLCV data, stored as vectorbtpro-compatible HDF5.

Uses data-api.binance.vision (no geo-restrictions) instead of api.binance.com.

Usage (in vbtpro conda env):
    conda run -n vbtpro python backtest/fetch_data.py
    conda run -n vbtpro python backtest/fetch_data.py --force   # re-download
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests
import vectorbtpro as vbt

sys.path.insert(0, str(Path(__file__).parent.parent))

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
# 5s/10s are NOT available via Binance historical REST API (min is 1m).
# 3m is the smallest practically useful step between 1m and 5m.
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
START_DATE = "2022-03-22"    # 4 years back for rigorous validation
DATA_DIR = Path(__file__).parent.parent / "data"

BINANCE_BASE = "https://data-api.binance.vision"  # geo-unrestricted


def _fetch_klines_raw(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetch all klines for a symbol/interval between start and end (ms timestamps)."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    all_rows = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        all_rows.extend(rows)
        # Advance past the last fetched candle
        current = rows[-1][0] + 1
        if len(rows) < 1000:
            break
        time.sleep(0.1)  # be polite

    return all_rows


def _rows_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    df.index.name = "Date"
    return df


def fetch_symbol(symbol: str, interval: str, start_str: str) -> pd.DataFrame:
    """Fetch full OHLCV history for one symbol."""
    start_ms = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows = _fetch_klines_raw(symbol, interval, start_ms, end_ms)
    return _rows_to_df(rows)


def fetch_all(force: bool = False):
    DATA_DIR.mkdir(exist_ok=True)

    for tf in TIMEFRAMES:
        path = DATA_DIR / f"binance_{tf}.h5"
        if path.exists() and not force:
            print(f"[{tf:>4}] cached → skipping (--force to re-download)")
            continue

        print(f"[{tf:>4}] downloading {SYMBOLS}...", flush=True)
        symbol_dfs = {}

        for sym in SYMBOLS:
            print(f"       {sym}", end=" ", flush=True)
            try:
                df = fetch_symbol(sym, tf, START_DATE)
                symbol_dfs[sym] = df
                print(f"({len(df)} rows)", flush=True)
            except Exception as e:
                print(f"FAILED: {e}")

        if not symbol_dfs:
            print(f"[{tf}] No data fetched, skipping.")
            continue

        # Build vbt-compatible multi-symbol Data object
        # vbt.Data.from_data expects dict of {symbol: DataFrame}
        data = vbt.Data.from_data(symbol_dfs)
        data.to_hdf(str(path))
        print(f"[{tf:>4}] saved to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeframes", nargs="+", default=None,
                        help="Only fetch these timeframes e.g. --timeframes 1h 4h 1d")
    args = parser.parse_args()
    if args.timeframes:
        TIMEFRAMES = args.timeframes
    fetch_all(force=args.force)
    print("\nDone. Run `conda run -n vbtpro python backtest/sweep.py` next.")
