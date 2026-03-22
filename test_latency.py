"""
Signal-to-execution latency test: places a $1 BUY then $1 SELL on BTC/USD
and reports timing for each leg.
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import roostoo_client as rc

PAIR = "BTC/USD"
TARGET_USD = 1.0


def get_btc_price() -> float:
    """Get current BTC price from Roostoo ticker."""
    ticker = rc.get_ticker(PAIR)
    info = ticker.get("Data", {}).get(PAIR, {})
    return float(info.get("LastPrice", 0))


def get_qty_precision() -> int:
    """Get amount precision for BTC/USD from exchange info."""
    info = rc.get_exchange_info()
    pairs = info.get("TradePairs", {})
    return int(pairs.get(PAIR, {}).get("AmountPrecision", 6))


def usd_to_qty(usd: float, price: float, precision: int) -> str:
    import math
    # Round up so the order value always meets the minimum USD threshold
    raw = usd / price
    step = 10 ** -precision
    qty = math.ceil(raw / step) * step
    return f"{qty:.{precision}f}"


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def run():
    print(f"=== Roostoo Signal-to-Execution Latency Test ===")
    print(f"Pair: {PAIR}  Target: ${TARGET_USD}\n")

    # --- Setup ---
    t_setup_start = time.perf_counter()
    price = get_btc_price()
    precision = get_qty_precision()
    t_setup_end = time.perf_counter()

    if price == 0:
        print("ERROR: Could not fetch BTC price from Roostoo ticker")
        print("Raw ticker response:", rc.get_ticker(PAIR))
        sys.exit(1)

    qty_str = usd_to_qty(TARGET_USD, price, precision)
    print(f"BTC price:  ${price:,.2f}")
    print(f"Precision:  {precision} decimals")
    print(f"Order qty:  {qty_str} BTC  (~${float(qty_str) * price:.4f})")
    print(f"Setup time: {ms(t_setup_end - t_setup_start)}\n")

    # --- BUY ---
    print("--- BUY ---")
    t_signal = time.perf_counter()          # "signal fired"
    t_buy_sent = time.perf_counter()
    buy_result = rc.place_order(PAIR, "BUY", "MARKET", qty_str)
    t_buy_done = time.perf_counter()

    buy_latency = t_buy_done - t_signal
    print(f"Signal → buy response: {ms(buy_latency)}")
    print(f"Success: {buy_result.get('Success')}")
    if buy_result.get("OrderDetail"):
        od = buy_result["OrderDetail"]
        print(f"Order ID: {od.get('OrderID', 'n/a')}")
        print(f"Fill price: {od.get('FilledAverPrice', 'n/a')}")
        print(f"Status: {od.get('Status', 'n/a')}")
    elif not buy_result.get("Success"):
        print(f"Error: {buy_result.get('ErrMsg', buy_result)}")
    print()

    # Brief pause so sell doesn't collide with unsettled buy
    time.sleep(0.5)

    # --- SELL ---
    print("--- SELL ---")
    # Re-fetch actual BTC balance in case fill was partial
    btc_balance = rc.get_coin_balance(PAIR)
    sell_qty = f"{min(float(qty_str), btc_balance):.{precision}f}" if btc_balance > 0 else qty_str
    print(f"BTC balance: {btc_balance}  using qty: {sell_qty}")

    t_sell_signal = time.perf_counter()
    sell_result = rc.place_order(PAIR, "SELL", "MARKET", sell_qty)
    t_sell_done = time.perf_counter()

    sell_latency = t_sell_done - t_sell_signal
    print(f"Signal → sell response: {ms(sell_latency)}")
    print(f"Success: {sell_result.get('Success')}")
    if sell_result.get("OrderDetail"):
        od = sell_result["OrderDetail"]
        print(f"Order ID: {od.get('OrderID', 'n/a')}")
        print(f"Fill price: {od.get('FilledAverPrice', 'n/a')}")
        print(f"Status: {od.get('Status', 'n/a')}")
    elif not sell_result.get("Success"):
        print(f"Error: {sell_result.get('ErrMsg', sell_result)}")
    print()

    # --- Summary ---
    total = t_sell_done - t_signal
    print("=== Summary ===")
    print(f"Buy  latency: {ms(buy_latency)}")
    print(f"Sell latency: {ms(sell_latency)}")
    print(f"Total (signal → sell done, excl. 500ms pause): {ms(total - 0.5)}")


if __name__ == "__main__":
    run()
