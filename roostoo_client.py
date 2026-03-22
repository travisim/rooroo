import hashlib
import hmac
import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ROOSTOO_API_KEY")
API_SECRET = os.getenv("ROOSTOO_API_SECRET")
BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")


def _timestamp():
    return str(int(time.time() * 1000))


def _sign(params: dict) -> str:
    sorted_params = sorted(params.items())
    query_string = "&".join(f"{k}={v}" for k, v in sorted_params)
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return signature


def _headers(signature: str) -> dict:
    return {
        "RST-API-KEY": API_KEY,
        "MSG-SIGNATURE": signature,
        "Content-Type": "application/x-www-form-urlencoded",
    }


def get_server_time():
    r = requests.get(f"{BASE_URL}/v3/serverTime")
    return r.json()


def get_exchange_info():
    r = requests.get(f"{BASE_URL}/v3/exchangeInfo")
    return r.json()


def get_ticker(pair=None):
    params = {"timestamp": _timestamp()}
    if pair:
        params["pair"] = pair
    sig = _sign(params)
    r = requests.get(
        f"{BASE_URL}/v3/ticker",
        params=params,
        headers={"RST-API-KEY": API_KEY, "MSG-SIGNATURE": sig},
    )
    return r.json()


def get_balance():
    params = {"timestamp": _timestamp()}
    sig = _sign(params)
    r = requests.get(
        f"{BASE_URL}/v3/balance",
        params=params,
        headers={"RST-API-KEY": API_KEY, "MSG-SIGNATURE": sig},
    )
    return r.json()


def place_order(pair: str, side: str, order_type: str, quantity: str, price: str = None):
    params = {
        "pair": pair,
        "side": side,
        "type": order_type,
        "quantity": str(quantity),
        "timestamp": _timestamp(),
    }
    if order_type == "LIMIT" and price:
        params["price"] = str(price)

    sig = _sign(params)
    r = requests.post(
        f"{BASE_URL}/v3/place_order",
        data=params,
        headers=_headers(sig),
    )
    return r.json()


def query_orders(pair: str = None, pending_only: bool = False):
    params = {"timestamp": _timestamp()}
    if pair:
        params["pair"] = pair
    if pending_only:
        params["pending_only"] = "true"
    sig = _sign(params)
    r = requests.post(
        f"{BASE_URL}/v3/query_order",
        data=params,
        headers=_headers(sig),
    )
    return r.json()


def get_coin_balance(pair: str) -> float:
    """Get the free balance of the coin in a pair (e.g., 'BTC/USD' -> BTC free balance)."""
    coin = pair.split("/")[0]
    bal = get_balance()
    wallet = bal.get("SpotWallet", {})
    coin_data = wallet.get(coin, {})
    return float(coin_data.get("Free", 0))


def cancel_order(order_id: str = None, pair: str = None):
    params = {"timestamp": _timestamp()}
    if order_id:
        params["order_id"] = order_id
    if pair:
        params["pair"] = pair
    sig = _sign(params)
    r = requests.post(
        f"{BASE_URL}/v3/cancel_order",
        data=params,
        headers=_headers(sig),
    )
    return r.json()


if __name__ == "__main__":
    print("=== Server Time ===")
    print(get_server_time())
    print("\n=== Balance ===")
    print(get_balance())
    print("\n=== Exchange Info ===")
    print(get_exchange_info())
