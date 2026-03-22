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
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
    base = cfg.STRATEGY.split("+")[0]
    if base == "ema_crossover":
        params = {"fast_window": cfg.EMA_FAST, "slow_window": cfg.EMA_SLOW}
    elif base == "rsi":
        params = {"window": cfg.RSI_PERIOD, "lower": cfg.RSI_OVERSOLD, "upper": cfg.RSI_OVERBOUGHT}
    elif base == "atr_trend":
        params = {"ma_len": cfg.ATR_MA_LEN, "atr_len": cfg.ATR_LEN, "atr_mult": cfg.ATR_MULT}
    elif base == "rsi_ema_filter":
        params = {"ema_len": cfg.RSI_EMA_FILTER_LEN, "rsi_period": cfg.RSI_PERIOD,
                  "oversold": cfg.RSI_OVERSOLD, "overbought": cfg.RSI_OVERBOUGHT}
    elif base == "volty_expan_close":
        params = {"length": cfg.VEC_LENGTH, "mult": cfg.VEC_MULT}
    elif base == "vec_rsi_regime":
        params = {
            "length": cfg.VEC_LENGTH, "mult": cfg.VEC_MULT,
            "rsi_window": cfg.RSI_PERIOD, "rsi_lower": cfg.RSI_OVERSOLD,
            "rsi_upper": cfg.RSI_OVERBOUGHT, "atr_ratio_threshold": cfg.VEC_RSI_ATR_THRESHOLD,
        }
    else:
        raise ValueError(f"Unknown strategy: {cfg.STRATEGY}")
    # Modifier params
    if "atr_ratio" in cfg.STRATEGY:
        params["atr_ratio_threshold"] = cfg.VEC_ATR_RATIO_THRESHOLD
    if "garch" in cfg.STRATEGY:
        params["garch_window"] = getattr(cfg, "GARCH_WINDOW", 252)
    return params


# ── Order helpers ─────────────────────────────────────────────────────────────
def _portfolio_usd() -> float:
    """Total portfolio value in USD (SpotWallet USD + coin values at market price)."""
    try:
        bal = rc.get_balance()
        wallet = bal.get("SpotWallet", {})
        total = float(wallet.get("USD", {}).get("Free", 0))
        for pair in cfg.ASSETS:
            coin = pair.split("/")[0]
            coin_qty = float(wallet.get(coin, {}).get("Free", 0))
            if coin_qty > 0:
                price = fetch_ticker_price(cfg.ROOSTOO_TO_BINANCE[pair])
                total += coin_qty * price
        return total
    except Exception:
        return 1_000_000.0  # fallback


def _dynamic_position_usd(df, portfolio_usd: float) -> float:
    """
    Inverse ATR position sizing:
        size = portfolio × TARGET_RISK_PCT / (ATR_SL_MULT × ATR%)
    Caps at MAX_POSITION_PCT of portfolio.
    """
    from bot.indicators import atr as compute_atr
    try:
        atr_val = compute_atr(df["high"], df["low"], df["close"], cfg.ATR_PERIOD).iloc[-1]
        current_price = df["close"].iloc[-1]
        if current_price > 0 and np.isfinite(atr_val) and atr_val > 0:
            atr_pct = atr_val / current_price
            size = portfolio_usd * cfg.TARGET_RISK_PCT / (cfg.ATR_SL_MULT * atr_pct)
        else:
            size = portfolio_usd * cfg.MAX_POSITION_PCT
    except Exception:
        size = portfolio_usd * cfg.MAX_POSITION_PCT
    return min(size, portfolio_usd * cfg.MAX_POSITION_PCT)


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
    if not pos["in_position"]:
        return False
    entry = pos["entry_price"]
    if entry <= 0:
        return False
    # Fixed % stop loss (matches backtest sl_stop param)
    fixed_sl = getattr(cfg, "FIXED_SL_PCT", None)
    if fixed_sl:
        loss_pct = (entry - current_price) / entry
        if loss_pct >= fixed_sl:
            log.warning(f"  STOP LOSS ({fixed_sl*100:.0f}%) triggered for {pair}: entry={entry:.4f} current={current_price:.4f}")
            return True
    else:
        # ATR-based stop loss: stop at entry - ATR_SL_MULT × ATR at entry
        atr_at_entry = pos.get("atr_at_entry", 0)
        if atr_at_entry > 0:
            sl_price = entry - cfg.ATR_SL_MULT * atr_at_entry
            if current_price <= sl_price:
                log.warning(f"  ATR STOP triggered for {pair}: sl={sl_price:.4f} current={current_price:.4f}")
                return True
        else:
            loss_pct = (entry - current_price) / entry
            if loss_pct >= 0.05:
                log.warning(f"  STOP LOSS (fixed 5%) triggered for {pair}")
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

    portfolio_usd = _portfolio_usd()
    log.info(f"  Portfolio value: ${portfolio_usd:,.0f}")

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
                    state[pair] = {"in_position": False, "entry_price": 0.0, "quantity": 0.0, "atr_at_entry": 0.0}
                else:
                    log.error(f"  SELL FAILED: {result.get('ErrMsg', result)}")

        # ── Entry ──────────────────────────────────────────────────────────
        elif entry_signal and not pos["in_position"]:
            position_usd = _dynamic_position_usd(df, portfolio_usd)
            try:
                qty_str = _usd_to_qty(pair, position_usd)
            except Exception as e:
                log.error(f"  USD→qty conversion failed: {e}")
                continue

            from bot.indicators import atr as compute_atr
            try:
                atr_val = float(compute_atr(df["high"], df["low"], df["close"], cfg.ATR_PERIOD).iloc[-1])
            except Exception:
                atr_val = 0.0

            log.info(f"  BUY {qty_str} {pair} (MARKET, ~${position_usd:,.0f}, ATR={atr_val:.4f})")
            result = rc.place_order(pair, "BUY", "MARKET", qty_str)
            if result.get("Success"):
                od = result.get("OrderDetail", {})
                fill_price = float(od.get("FilledAverPrice", current_price))
                log.info(f"  BOUGHT @ ${fill_price:.4f} | order={od.get('OrderID')}")
                state[pair] = {
                    "in_position": True,
                    "entry_price": fill_price,
                    "quantity": float(qty_str),
                    "atr_at_entry": atr_val,
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
    log.info(f"Sizing   : inverse-ATR, risk={cfg.TARGET_RISK_PCT*100:.1f}%/trade, SL={cfg.ATR_SL_MULT}×ATR, max={cfg.MAX_POSITION_PCT*100:.0f}%")
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
