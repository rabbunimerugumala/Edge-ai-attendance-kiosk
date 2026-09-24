#!/usr/bin/env bash
# ==============================================================================
# TOUCHLESS SMART ATTENDANCE KIOSK - RASPBERRY PI ONE-CLICK INSTALLER
# ==============================================================================
set -e

echo "===================================================================="
echo "    TOUCHLESS SMART ATTENDANCE KIOSK - RASPBERRY PI 3B+ INSTALLER   "
echo "===================================================================="

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] Please run with sudo: sudo bash deploy/setup_pi.sh"
  exit 1
fi

ACTUAL_USER="${SUDO_USER:-pi}"
INSTALL_DIR=$(pwd)

echo "[1/5] Updating APT repositories..."
apt-get update -y

echo "[2/5] Installing pre-compiled edge AI & web packages..."
apt-get install -y \
  python3-opencv \
  python3-flask \
  python3-requests \
  python3-numpy \
  python3-psutil \
  python3-pil \
  curl

echo "[3/5] Configuring systemd auto-start service..."
SERVICE_FILE="/etc/systemd/system/touchless-kiosk.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=Touchless Smart Attendance Kiosk Web & Auto-Scan Service
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${ACTUAL_USER}
WorkingDirectory=${INSTALL_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable touchless-kiosk.service

echo "[4/5] Configuring video device permissions..."
usermod -aG video "${ACTUAL_USER}"

echo "[5/5] Starting service..."
systemctl restart touchless-kiosk.service

# Retrieve local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "===================================================================="
echo " [SUCCESS] Touchless Smart Attendance Kiosk is Installed & Running! "
echo "===================================================================="
echo "  • Web Dashboard URL : http://${LOCAL_IP}:5000"
echo "  • Video Stream URL  : http://${LOCAL_IP}:5000/video_feed"
echo "  • Service Status    : sudo systemctl status touchless-kiosk"
echo "  • View Live Logs    : sudo journalctl -u touchless-kiosk -f"
echo "===================================================================="
echo "Open http://${LOCAL_IP}:5000 in your browser or phone to manage the kiosk!"
echo ""
