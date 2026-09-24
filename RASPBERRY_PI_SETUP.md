# 🍓 Raspberry Pi 3B+ Deployment & 24/7 Kiosk Master Guide

This guide details the complete, production-ready setup for running the **Touchless Smart Attendance Kiosk** on a **Raspberry Pi 3 Model B+** (or Pi 4 / 5). 

Following these instructions ensures your kiosk **automatically boots, starts the camera engine, listens on the local network, and syncs to Google Sheets 24/7 without manual intervention.**

---

## 📋 Hardware Requirements

* **Single Board Computer:** Raspberry Pi 3 Model B+ (1GB RAM) or higher.
* **Operating System:** Raspberry Pi OS (Bullseye or Bookworm, 32-bit or 64-bit with Desktop).
* **Storage:** 16GB+ MicroSD Card (Class 10 / A1 or A2 recommended).
* **Power Supply:** Official 5.1V / 2.5A power adapter (essential to prevent USB camera brownouts).
* **Camera:** 
  * USB Webcam: **Logitech C270** (Recommended budget) or **Logitech C920**.
  * OR CSI Camera: **Raspberry Pi Camera Module 3** (via 15-pin ribbon cable).
* **Display (Optional):** Any HDMI monitor or official 7" touchscreen.

---

## 🚀 Step 1: Clone the Codebase to Your Pi

Open the terminal on your Raspberry Pi (or connect via SSH):

```bash
# 1. Navigate to home directory
cd /home/pi

# 2. Clone the repository
git clone https://github.com/rabbunimerugumala/Edge-ai-attendance-kiosk.git Touchless-Smart-Attendance-Kiosk

# 3. Enter the project directory
cd Touchless-Smart-Attendance-Kiosk
```

---

## 🔐 Step 2: Configure Environment Secrets (`.env`)

Because your Google Sheets webhook URL is private and ignored by Git, create the `.env` file on your Pi:

```bash
nano .env
```

Paste your configuration inside the editor:

```ini
# Google Sheets Apps Script Webhook URL
GOOGLE_SHEETS_WEBHOOK_URL="https://script.google.com/macros/s/AKfycbyzGYKZaN5f0I8ZnRmlueP-VPO4zeLLCOSb_Jah7gwVEBRqGuwS7BrslUUh6ntqNDBc_Q/exec"

# Flask Security Secret
FLASK_SECRET_KEY="touchless-edge-kiosk-secret-key"
```

*Press <kbd>Ctrl</kbd> + <kbd>O</kbd>, then <kbd>Enter</kbd> to save. Press <kbd>Ctrl</kbd> + <kbd>X</kbd> to exit.*

---

## 🧠 Step 3: Transfer Your Trained Biometric Face Model

If you already enrolled student faces on your laptop and want to use that model immediately on the Pi without re-enrolling:

Open **PowerShell** on your laptop and run:

```powershell
# Copy the trained LBPH model and student registry to the Pi
scp "c:\Users\rabbu\Downloads\Touchless Smart Attendance Kiosk\face-trainer.yml" pi@<PI_IP_ADDRESS>:/home/pi/Touchless-Smart-Attendance-Kiosk/
scp "c:\Users\rabbu\Downloads\Touchless Smart Attendance Kiosk\students.json" pi@<PI_IP_ADDRESS>:/home/pi/Touchless-Smart-Attendance-Kiosk/
```
*(Replace `<PI_IP_ADDRESS>` with your Raspberry Pi's actual local IP address, e.g. `192.168.1.50`).*

---

## ⚡ Step 4: Run the 1-Click Automated Installer

On your Raspberry Pi terminal, run:

```bash
cd /home/pi/Touchless-Smart-Attendance-Kiosk
sudo bash deploy/setup_pi.sh
```

### What `setup_pi.sh` does automatically:
1. **Installs System-Optimized ARM Packages:** Uses `apt-get` to install pre-compiled `python3-opencv`, `python3-flask`, `python3-numpy`, and `python3-pil`.  
   *(⚠️ **Note:** Compiling OpenCV via `pip install opencv-python` on a 1GB RAM Pi will freeze the board. The APT package installs in ~45 seconds with zero RAM stress!)*
2. **Grants Video Permissions:** Adds the user to the `video` hardware group (`usermod -aG video`).
3. **Installs Systemd Auto-Start Service:** Registers `/etc/systemd/system/touchless-kiosk.service`.
4. **Enables Auto-Boot & Watchdog:** Configures the kiosk daemon to launch on boot and automatically recover within 5 seconds if a crash or power drop occurs.

---

## 🔄 Step 5: Industrial 24/7 Resilience (`systemd`)

The kiosk is managed by Linux `systemd`. It runs as a background daemon with continuous watchdog monitoring:

```ini
[Unit]
Description=Touchless Smart Attendance Kiosk Web & Auto-Scan Service
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Touchless-Smart-Attendance-Kiosk
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Key Reliability Features:
* **`Restart=always`:** If someone unplugs the camera or power flickers, the service restarts automatically.
* **`RestartSec=5`:** Waits 5 seconds before reviving to give USB drivers time to re-initialize.
* **`After=network-online.target`:** Ensures Google Sheets synchronization only starts once Wi-Fi or Ethernet is connected.

---

## 🖥️ Step 6: Fullscreen Kiosk Mode on HDMI Monitor (Optional)

If your kiosk is connected to an HDMI display and you want the screen to **automatically open full-screen into the attendance dashboard** when the Pi turns on:

### 1. Create the Autostart Entry
```bash
mkdir -p ~/.config/autostart
nano ~/.config/autostart/kiosk.desktop
```

### 2. Paste the Kiosk Profile
```ini
[Desktop Entry]
Type=Application
Name=Attendance Kiosk Display
Exec=chromium-browser --kiosk --noerrdialogs --disable-infobars --check-for-update-interval=31536000 http://localhost:5000
```
*Save with <kbd>Ctrl</kbd> + <kbd>O</kbd>, <kbd>Enter</kbd>, <kbd>Ctrl</kbd> + <kbd>X</kbd>.*

### 3. Disable Screen Blanking / Sleep
To keep the display on 24/7 without turning black:
1. Open Raspberry Pi Configuration:
   ```bash
   sudo raspi-config
   ```
2. Navigate to: **Display Options** $\rightarrow$ **Screen Blanking** $\rightarrow$ Select **No (Disable)**.
3. Select **Finish** and reboot.

---

## 🛠️ Step 7: Kiosk Management & Diagnostics

### Checking Service Status
```bash
sudo systemctl status touchless-kiosk
```

### Viewing Real-Time Logs
Watch facial recognitions and Google Sheets cloud syncs happen live:
```bash
sudo journalctl -u touchless-kiosk -f
```

### Restarting or Stopping the Kiosk
```bash
# Restart
sudo systemctl restart touchless-kiosk

# Stop
sudo systemctl stop touchless-kiosk

# Start
sudo systemctl start touchless-kiosk
```

### Verifying Hardware Camera Detection
To verify that your USB webcam is recognized by the Linux kernel:
```bash
# Check USB device enumeration
lsusb

# Check video capture nodes
v4l2-ctl --list-devices
```
*(Your webcam will appear as `/dev/video0`).*

### Monitoring Hardware Thermals & CPU
```bash
# View SoC Temperature (Ideal: < 65°C)
vcgencmd measure_temp

# View CPU & Memory Usage
htop
```

---

## 🌐 Remote Dashboard Access

Once your Raspberry Pi is running on your network, any phone, tablet, or PC on the same Wi-Fi can view the dashboard:

```text
http://<RASPBERRY_PI_IP>:5000
```
*(Example: `http://192.168.1.50:5000`)*

From the web portal you can:
* Monitor live attendance stats.
* View the real-time MJPEG camera stream.
* Enroll new students directly from any browser.
* 1-Click retrain the LBPH biometric model.
* Force instant cloud sync to Google Sheets.
