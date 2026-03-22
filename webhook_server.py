"""
TradingView Webhook -> Roostoo Trading Server

Receives TradingView alert webhooks and places orders on Roostoo.
Automatically converts USD quantity to coin quantity using live price.
Logs all trades to strategy_trades.csv for comparison with TradingView.

Endpoints:
  GET  /health   - Health check
  GET  /balance  - Account balance
  GET  /trades   - View trade log
  POST /webhook  - Simple BUY/SELL orders
  POST /strategy - ATR strategy signals (LONG_ENTRY, LONG_EXIT)
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify

import roostoo_client

load_dotenv()

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trades.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "rooroo-tv-webhook-2024")
DEFAULT_PAIR = os.getenv("TRADE_PAIR", "BTC/USD")
DEFAULT_ORDER_TYPE = os.getenv("ORDER_TYPE", "MARKET")
DEFAULT_QTY = os.getenv("DEFAULT_QTY", "0.01")

# CSV trade log for easy comparison with TradingView
TRADE_LOG_CSV = "strategy_trades.csv"
CSV_HEADERS = ["timestamp", "signal", "side", "pair", "usd_amount", "coin_qty", "fill_price", "order_id", "status", "balance_after_usd"]


def _init_csv():
    if not Path(TRADE_LOG_CSV).exists():
        with open(TRADE_LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def _log_trade(signal, side, pair, usd_amount, coin_qty, fill_price, order_id, status, balance_after):
    """Append a trade row to the CSV log."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    row = [ts, signal, side, pair, f"{usd_amount:.2f}", coin_qty, f"{fill_price:.2f}", order_id, status, f"{balance_after:.2f}"]
    with open(TRADE_LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    log.info(f"TRADE LOG | {ts} | {signal} | {side} | {pair} | ${usd_amount:.2f} | {coin_qty} | @${fill_price:.2f} | #{order_id} | {status} | bal=${balance_after:.2f}")


_init_csv()

# Cache exchange info for precision lookups
_exchange_info = {}


def _load_exchange_info():
    global _exchange_info
    if not _exchange_info:
        try:
            info = roostoo_client.get_exchange_info()
            _exchange_info = info.get("TradePairs", {})
            log.info(f"Loaded exchange info for {len(_exchange_info)} pairs")
        except Exception as e:
            log.error(f"Failed to load exchange info: {e}")


def _get_precision(pair: str) -> int:
    _load_exchange_info()
    pair_info = _exchange_info.get(pair, {})
    return pair_info.get("AmountPrecision", 5)


def _usd_to_coin_qty(pair: str, usd_amount: float) -> str:
    """Convert a USD amount to coin quantity using live ticker price."""
    try:
        ticker = roostoo_client.get_ticker(pair)
        if isinstance(ticker, dict):
            price = None
            data = ticker.get("Data", {})
            if pair in data:
                price = float(data[pair].get("LastPrice", 0))
            elif pair in ticker:
                price = float(ticker[pair].get("LastPrice", 0))

            if price and price > 0:
                precision = _get_precision(pair)
                coin_qty = round(usd_amount / price, precision)
                log.info(f"Converted ${usd_amount} -> {coin_qty} {pair} @ ${price}")
                return str(coin_qty)

        log.warning(f"Could not get price for {pair}, ticker response: {ticker}")
        return str(usd_amount)
    except Exception as e:
        log.error(f"Price lookup failed for {pair}: {e}")
        return str(usd_amount)


def _get_usd_balance() -> float:
    """Get current USD free balance."""
    try:
        bal = roostoo_client.get_balance()
        return float(bal.get("SpotWallet", {}).get("USD", {}).get("Free", 0))
    except Exception:
        return 0.0


# Track positions for exit sizing
positions = {}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


@app.route("/balance", methods=["GET"])
def balance():
    try:
        result = roostoo_client.get_balance()
        return jsonify(result)
    except Exception as e:
        log.error(f"Balance check failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/trades", methods=["GET"])
def trades():
    """Return the trade log CSV as JSON for easy viewing."""
    try:
        trades_list = []
        with open(TRADE_LOG_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades_list.append(row)
        return jsonify({"trades": trades_list, "count": len(trades_list)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
    except Exception:
        log.warning("Bad request body")
        return jsonify({"error": "invalid json"}), 400

    log.info(f"Webhook received: {json.dumps(data)}")

    if data.get("secret") != WEBHOOK_SECRET:
        log.warning("Invalid webhook secret")
        return jsonify({"error": "unauthorized"}), 401

    action = data.get("action", "").upper()
    if action not in ("BUY", "SELL"):
        log.warning(f"Invalid action: {action}")
        return jsonify({"error": "invalid action"}), 400

    pair = data.get("pair", DEFAULT_PAIR)
    raw_qty = float(data.get("quantity", DEFAULT_QTY))
    order_type = data.get("order_type", DEFAULT_ORDER_TYPE).upper()
    price = data.get("price")

    quantity = _usd_to_coin_qty(pair, raw_qty)

    log.info(f"Placing {action} {order_type} order: {quantity} {pair} (from ${raw_qty})")

    try:
        result = roostoo_client.place_order(
            pair=pair,
            side=action,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        log.info(f"Order result: {json.dumps(result)}")

        if result.get("Success"):
            od = result.get("OrderDetail", {})
            _log_trade(action, action, pair, raw_qty, quantity,
                       od.get("FilledAverPrice", 0), od.get("OrderID", ""), od.get("Status", ""),
                       _get_usd_balance())

        return jsonify({"status": "ok", "result": result})

    except Exception as e:
        log.error(f"Order failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/strategy", methods=["POST"])
def strategy_webhook():
    """
    Handles ATR strategy signals with automatic USD-to-coin conversion.
    Logs every trade to strategy_trades.csv for comparison with TradingView.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400

    log.info(f"Strategy webhook received: {json.dumps(data)}")

    if data.get("secret") != WEBHOOK_SECRET:
        return jsonify({"error": "unauthorized"}), 401

    signal = data.get("signal", "").upper()
    pair = data.get("pair", DEFAULT_PAIR)
    raw_qty = float(data.get("quantity", DEFAULT_QTY))
    order_type = data.get("order_type", DEFAULT_ORDER_TYPE).upper()

    signal_map = {
        "LONG_ENTRY": "BUY",
        "SHORT_ENTRY": "SELL",
        "LONG_EXIT": "SELL",
        "SHORT_EXIT": "BUY",
    }

    action = signal_map.get(signal)
    if not action:
        log.warning(f"Unknown signal: {signal}")
        return jsonify({"error": f"unknown signal: {signal}"}), 400

    # For exits, check actual wallet balance to avoid insufficient-balance errors
    if signal in ("LONG_EXIT", "SHORT_EXIT"):
        actual_balance = roostoo_client.get_coin_balance(pair)
        if actual_balance > 0 and signal == "LONG_EXIT":
            quantity = str(actual_balance)
            log.info(f"Selling actual balance: {quantity}")
        else:
            quantity = _usd_to_coin_qty(pair, raw_qty)
    else:
        quantity = _usd_to_coin_qty(pair, raw_qty)

    log.info(f"Signal {signal} -> {action} {quantity} {pair} (from ${raw_qty})")

    try:
        result = roostoo_client.place_order(
            pair=pair,
            side=action,
            order_type=order_type,
            quantity=quantity,
        )
        log.info(f"Order result: {json.dumps(result)}")

        # Log to CSV and track positions
        if result.get("Success"):
            od = result.get("OrderDetail", {})
            usd_bal = _get_usd_balance()
            _log_trade(signal, action, pair, raw_qty, quantity,
                       od.get("FilledAverPrice", 0), od.get("OrderID", ""), od.get("Status", ""),
                       usd_bal)

            if signal in ("LONG_ENTRY", "SHORT_ENTRY"):
                positions[pair] = {
                    "side": action,
                    "quantity": quantity,
                    "entry_price": od.get("FilledAverPrice", 0),
                    "time": datetime.now(timezone.utc).isoformat(),
                }
            elif signal in ("LONG_EXIT", "SHORT_EXIT"):
                positions.pop(pair, None)
        else:
            # Log failed trades too
            _log_trade(signal, action, pair, raw_qty, quantity, 0, "", f"FAILED: {result.get('ErrMsg', '')}", _get_usd_balance())

        return jsonify({"status": "ok", "signal": signal, "action": action, "quantity": quantity, "result": result})

    except Exception as e:
        log.error(f"Order failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    log.info("Starting Roostoo Webhook Server...")
    log.info(f"Default pair: {DEFAULT_PAIR}, Default qty: {DEFAULT_QTY}")

    try:
        st = roostoo_client.get_server_time()
        log.info(f"Roostoo server time: {st}")
        bal = roostoo_client.get_balance()
        log.info(f"Account balance: {json.dumps(bal)}")
        _load_exchange_info()
    except Exception as e:
        log.warning(f"Could not connect to Roostoo on startup: {e}")

    app.run(host="0.0.0.0", port=80, debug=False)
