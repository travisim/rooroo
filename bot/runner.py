#!/usr/bin/env python3
"""
Roostoo Autonomous Trading Bot
================================
Polls Binance for OHLCV data, computes strategy signals, and places
orders on the Roostoo mock exchange. Runs indefinitely on a schedule.

Usage:
    python bot/runner.py

Environment variables (from .env):
    ROOSTOO_API_KEY, ROOSTOO_API_SECRET, ROOSTOO_BASE_URL
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

import config as cfg
import roostoo_client as rc
from bot.indicators import latest_signal
from bot.market_data import fetch_klines, fetch_ticker_price

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot/bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── State file ────────────────────────────────────────────────────────────────
STATE_FILE = Path("bot/state.json")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {asset: {"in_position": False, "entry_price": 0.0, "quantity": 0.0}
            for asset in cfg.ASSETS}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Strategy params from config ───────────────────────────────────────────────
def _strategy_params() -> dict:
    s = cfg.STRATEGY
    if s == "ema_crossover":
        return {"fast_window": cfg.EMA_FAST, "slow_window": cfg.EMA_SLOW}
    elif s == "rsi":
        return {"window": cfg.RSI_PERIOD, "lower": cfg.RSI_OVERSOLD, "upper": cfg.RSI_OVERBOUGHT}
    elif s == "atr_trend":
        return {"ma_len": cfg.ATR_MA_LEN, "atr_len": cfg.ATR_LEN, "atr_mult": cfg.ATR_MULT}
    elif s == "rsi_ema_filter":
        return {"ema_len": cfg.RSI_EMA_FILTER_LEN, "rsi_period": cfg.RSI_PERIOD,
                "oversold": cfg.RSI_OVERSOLD, "overbought": cfg.RSI_OVERBOUGHT}
    else:
        raise ValueError(f"Unknown strategy: {s}")


# ── Order helpers ─────────────────────────────────────────────────────────────
def _usd_to_qty(pair: str, usd: float) -> str:
    binance_sym = cfg.ROOSTOO_TO_BINANCE[pair]
    price = fetch_ticker_price(binance_sym)
    info = rc.get_exchange_info().get("TradePairs", {})
    precision = info.get(pair, {}).get("AmountPrecision", 5)
    qty = round(usd / price, precision)
    log.info(f"  ${usd:,.0f} @ ${price:,.2f} = {qty} {pair.split('/')[0]}")
    return str(qty)


def _get_coin_qty(pair: str) -> float:
    """Return actual coin balance for the pair's base asset."""
    coin = pair.split("/")[0]
    bal = rc.get_balance()
    return float(bal.get("SpotWallet", {}).get(coin, {}).get("Free", 0))


def _stop_triggered(state: dict, pair: str, current_price: float) -> bool:
    pos = state[pair]
    if not pos["in_position"] or not cfg.STOP_LOSS_PCT:
        return False
    entry = pos["entry_price"]
    if entry <= 0:
        return False
    loss_pct = (entry - current_price) / entry
    if loss_pct >= cfg.STOP_LOSS_PCT:
        log.warning(f"  STOP LOSS triggered for {pair}: entry={entry:.2f} current={current_price:.2f} loss={loss_pct*100:.1f}%")
        return True
    if cfg.TAKE_PROFIT_PCT:
        gain_pct = (current_price - entry) / entry
        if gain_pct >= cfg.TAKE_PROFIT_PCT:
            log.info(f"  TAKE PROFIT triggered for {pair}: gain={gain_pct*100:.1f}%")
            return True
    return False


# ── Core loop ─────────────────────────────────────────────────────────────────
def run_once(state: dict) -> dict:
    """
    Check signals for all assets and place orders if warranted.
    Returns updated state.
    """
    params = _strategy_params()
    binance_interval = cfg.BINANCE_INTERVAL[cfg.TIMEFRAME]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log.info(f"=== Signal check {ts} | strategy={cfg.STRATEGY} tf={cfg.TIMEFRAME} ===")

    for pair in cfg.ASSETS:
        binance_sym = cfg.ROOSTOO_TO_BINANCE[pair]
        log.info(f"  {pair} ({binance_sym})")

        try:
            df = fetch_klines(binance_sym, binance_interval, limit=cfg.CANDLE_HISTORY)
        except Exception as e:
            log.error(f"  Failed to fetch data for {pair}: {e}")
            continue

        try:
            entry_signal, exit_signal = latest_signal(cfg.STRATEGY, df, **params)
        except Exception as e:
            log.error(f"  Signal computation failed for {pair}: {e}")
            continue

        pos = state[pair]
        current_price = df["close"].iloc[-1]

        log.info(f"  price={current_price:.4f} | entry={entry_signal} exit={exit_signal} | in_pos={pos['in_position']}")

        # ── Check stop loss / take profit ──────────────────────────────────
        if _stop_triggered(state, pair, current_price):
            exit_signal = True

        # ── Exit ──────────────────────────────────────────────────────────
        if exit_signal and pos["in_position"]:
            actual_qty = _get_coin_qty(pair)
            if actual_qty <= 0:
                log.warning(f"  EXIT skipped — no coins held for {pair}")
                state[pair]["in_position"] = False
            else:
                log.info(f"  SELL {actual_qty} {pair} (MARKET)")
                result = rc.place_order(pair, "SELL", "MARKET", str(actual_qty))
                if result.get("Success"):
                    od = result.get("OrderDetail", {})
                    log.info(f"  SOLD @ ${od.get('FilledAverPrice', 0):.4f} | order={od.get('OrderID')}")
                    state[pair] = {"in_position": False, "entry_price": 0.0, "quantity": 0.0}
                else:
                    log.error(f"  SELL FAILED: {result.get('ErrMsg', result)}")

        # ── Entry ──────────────────────────────────────────────────────────
        elif entry_signal and not pos["in_position"]:
            try:
                qty_str = _usd_to_qty(pair, cfg.POSITION_SIZE_USD)
            except Exception as e:
                log.error(f"  USD→qty conversion failed: {e}")
                continue

            log.info(f"  BUY {qty_str} {pair} (MARKET, ~${cfg.POSITION_SIZE_USD:,.0f})")
            result = rc.place_order(pair, "BUY", "MARKET", qty_str)
            if result.get("Success"):
                od = result.get("OrderDetail", {})
                fill_price = float(od.get("FilledAverPrice", current_price))
                log.info(f"  BOUGHT @ ${fill_price:.4f} | order={od.get('OrderID')}")
                state[pair] = {
                    "in_position": True,
                    "entry_price": fill_price,
                    "quantity": float(qty_str),
                }
            else:
                log.error(f"  BUY FAILED: {result.get('ErrMsg', result)}")
        else:
            log.info(f"  No action.")

    save_state(state)
    return state


def main():
    log.info("=" * 60)
    log.info("Roostoo Trading Bot starting")
    log.info(f"Strategy : {cfg.STRATEGY}")
    log.info(f"Timeframe: {cfg.TIMEFRAME}")
    log.info(f"Assets   : {cfg.ASSETS}")
    log.info(f"Pos size : ${cfg.POSITION_SIZE_USD:,.0f}")
    log.info(f"Interval : {cfg.CHECK_INTERVAL_SEC}s")
    log.info("=" * 60)

    # Verify connectivity
    try:
        bal = rc.get_balance()
        wallet = bal.get("SpotWallet", {})
        usd_free = float(wallet.get("USD", {}).get("Free", 0))
        log.info(f"Roostoo connected — USD balance: ${usd_free:,.2f}")
    except Exception as e:
        log.error(f"Roostoo connection failed: {e}")
        sys.exit(1)

    state = load_state()
    log.info(f"Loaded state: {json.dumps(state, indent=2)}")

    while True:
        try:
            state = run_once(state)
        except Exception as e:
            log.exception(f"Unexpected error in run_once: {e}")

        log.info(f"Sleeping {cfg.CHECK_INTERVAL_SEC}s until next check...")
        time.sleep(cfg.CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
