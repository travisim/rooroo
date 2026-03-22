#!/bin/bash
# Deploy Roostoo Trading Bot to EC2
# Usage: ./deploy.sh

set -e

EC2_HOST="ubuntu@ec2-3-104-220-58.ap-southeast-2.compute.amazonaws.com"
PEM_KEY="${SYDNEY_PEM:-$HOME/Downloads/Sydney.pem}"
REMOTE_DIR="/home/ubuntu/rooroo"

echo "=== Deploying Roostoo Trading Bot ==="

chmod 400 "$PEM_KEY"

ssh -i "$PEM_KEY" "$EC2_HOST" "mkdir -p $REMOTE_DIR/bot $REMOTE_DIR/data $REMOTE_DIR/results"

echo "Syncing files..."
scp -i "$PEM_KEY" \
    config.py \
    roostoo_client.py \
    requirements.txt \
    .env \
    "$EC2_HOST:$REMOTE_DIR/"

scp -i "$PEM_KEY" \
    bot/__init__.py \
    bot/indicators.py \
    bot/market_data.py \
    bot/runner.py \
    "$EC2_HOST:$REMOTE_DIR/bot/"

echo "Setting up environment and service..."
ssh -i "$PEM_KEY" "$EC2_HOST" << 'REMOTE_SCRIPT'
cd /home/ubuntu/rooroo

sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv

python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

echo "=== Testing Roostoo connectivity ==="
python3 -c "import roostoo_client as rc; bal=rc.get_balance(); print('USD balance:', bal.get('SpotWallet',{}).get('USD',{}).get('Free',0))"

sudo tee /etc/systemd/system/rooroo-bot.service > /dev/null << 'SERVICE'
[Unit]
Description=Roostoo Autonomous Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/rooroo
ExecStart=/home/ubuntu/rooroo/venv/bin/python3 bot/runner.py
Restart=always
RestartSec=10
Environment=PATH=/home/ubuntu/rooroo/venv/bin:/usr/bin

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable rooroo-bot
sudo systemctl restart rooroo-bot

sleep 3
echo "=== Bot Service Status ==="
sudo systemctl status rooroo-bot --no-pager -l
REMOTE_SCRIPT

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Monitor logs:"
echo "  ssh -i $PEM_KEY $EC2_HOST 'tail -f /home/ubuntu/rooroo/bot/bot.log'"
echo ""
echo "Check status:"
echo "  ssh -i $PEM_KEY $EC2_HOST 'sudo systemctl status rooroo-bot'"
