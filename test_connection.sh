#!/bin/bash
# Test the webhook server locally or on EC2
# Usage: ./test_connection.sh [host]

HOST="${1:-http://ec2-3-104-220-58.ap-southeast-2.compute.amazonaws.com:5000}"

echo "=== Testing Webhook Server at $HOST ==="

echo ""
echo "1. Health Check:"
curl -s "$HOST/health" | python3 -m json.tool 2>/dev/null || echo "FAILED"

echo ""
echo "2. Balance Check:"
curl -s "$HOST/balance" | python3 -m json.tool 2>/dev/null || echo "FAILED"

echo ""
echo "3. Test BUY Order (small amount):"
curl -s -X POST "$HOST/strategy" \
  -H "Content-Type: application/json" \
  -d '{
    "secret": "rooroo-tv-webhook-2024",
    "signal": "LONG_ENTRY",
    "pair": "BTC/USD",
    "quantity": "10",
    "order_type": "MARKET"
  }' | python3 -m json.tool 2>/dev/null || echo "FAILED"

echo ""
echo "4. Test SELL Order (close position):"
curl -s -X POST "$HOST/strategy" \
  -H "Content-Type: application/json" \
  -d '{
    "secret": "rooroo-tv-webhook-2024",
    "signal": "LONG_EXIT",
    "pair": "BTC/USD",
    "quantity": "10",
    "order_type": "MARKET"
  }' | python3 -m json.tool 2>/dev/null || echo "FAILED"

echo ""
echo "=== Done ==="
