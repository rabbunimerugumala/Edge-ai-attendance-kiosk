# Edge-ai-attendance-kiosk

# 👁️ Touchless Smart Attendance Kiosk (Edge AI)

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%203B+%20%7C%20Laptop-blue.svg)](https://www.raspberrypi.com/)
[![Computer Vision](https://img.shields.io/badge/CV-OpenCV%20Haar%20%2B%20LBPH-green.svg)](https://opencv.org/)
[![RAM Footprint](https://img.shields.io/badge/RAM-<%20120MB%20Peak-brightgreen.svg)]()
[![Database](https://img.shields.io/badge/Database-SQLite3%20(WAL%20Mode)-orange.svg)](https://www.sqlite.org/)
[![Cloud Sync](https://img.shields.io/badge/Cloud-Google%20Sheets%20API-red.svg)](https://developers.google.com/apps-script)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

> A complete, production-grade, 100% offline Edge AI Face Attendance Kiosk designed for laptop development and engineered for direct migration to a **Raspberry Pi 3B+ (1GB RAM)** with **Pi Camera Module 3**, **16mm metal desk button**, and **I2C OLED screen**. Includes a modern **Flask Web Portal**, **REST API**, and **resilient Google Sheets cloud synchronization**.

---

## 📑 Table of Contents
1. [System Architecture & Design](#-system-architecture--design)
2. [Repository Directory & File Breakdown](#-repository-directory--file-breakdown)
3. [Quick Start: Laptop / Local Testing](#-quick-start-laptop--local-testing)
4. [How to Upload to Raspberry Pi](#-how-to-upload-to-raspberry-pi)
5. [Raspberry Pi Installation & Deployment](#-raspberry-pi-installation--deployment)
6. [Modes of Operation](#-modes-of-operation)
7. [Google Sheets Cloud Webhook Setup](#-google-sheets-cloud-webhook-setup)
8. [Raspberry Pi Hardware Wiring Guide](#-raspberry-pi-hardware-wiring-guide)
9. [Complete REST API Reference](#-complete-rest-api-reference)
10. [Background Service & Auto-Start (systemd)](#-background-service--auto-start-systemd)
11. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🏗️ System Architecture & Design

The system is designed with strict **Edge AI constraints** to guarantee real-time performance on resource-limited hardware (Raspberry Pi 3B+ with 1GB RAM) without memory swapping or thermal throttling.

```
                           +----------------------------------------+
                           |  Camera Input (640x480 Standardized)   |
                           |  Webcam / Raspberry Pi Camera Module 3 |
                           +-------------------+--------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |  Haar Cascade Face Detection           |
                           |  haarcascade_frontalface_default.xml   |
                           +-------------------+--------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |  Preprocessing & Normalization         |
                           |  - Grayscale Conversion                |
                           |  - Resize to 200x200                   |
                           |  - Histogram Equalization (equalizeHist)
                           +-------------------+--------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |  LBPH Face Recognizer (cv2.face)       |
                           |  Trained mathematical model (YML)      |
                           +-------------------+--------------------+
                                               |
                           +-------------------+--------------------+
                           |                                        |
           [Match: Distance < 75.0]                   [Distance >= 75.0]
                           |                                        |
                           v                                        v
             +---------------------------+              +-----------------------+
             | 10-Second Anti-Spam Check |              | Display UNRECOGNIZED  |
             +-------------+-------------+              +-----------------------+
                           |
            +--------------+--------------+
            |                             |
      [Not Cooled]                   [Cooling Down]
            |                             |
            v                             v
+-----------------------+     +-------------------------+
| Fast SQLite Logging   |     | Skip DB Write           |
| attendance.db (< 5ms) |     | Show Cooldown Alert     |
+-----------+-----------+     +-------------------------+
            |
            v
+-------------------------------------------------------+
| Background Cloud Sync Daemon (Every 15s)              |
| - Queries unsynced rows (synced = 0)                  |
| - JSON POST to Google Apps Script Webhook             |
| - On HTTP 200: Sets synced = 1                        |
| - Offline Resilience: Silent retry on internet drop   |
+-------------------------------------------------------+
```

### Key Architectural Decisions:
- **No Heavy Deep Learning**: Eliminates MediaPipe, TensorFlow, PyTorch, and ONNX Runtime. The complete pipeline uses OpenCV's lightweight Haar + LBPH algorithms, keeping RAM usage **under 120MB** at all times.
- **Microsecond Database Writes**: SQLite configured with `PRAGMA journal_mode = WAL;` (Write-Ahead Logging) and `PRAGMA synchronous = NORMAL;`. Local database transactions complete in **1 to 4 milliseconds**, preventing UI freeze.
- **Offline Resilience & Crash Immunity**: If Wi-Fi disconnects, all logs accumulate safely in local SQLite. Once internet is restored, the daemon thread automatically pushes backlogged rows in batches of 50.
- **Anti-Spam Cooldown Engine**: In-memory hash map tracks `{student_id: last_logged_timestamp}` with a 10-second lockout window, preventing duplicate logs when a student stands in front of the camera.

---

## 📁 Repository Directory & File Breakdown

Here is what every file in this repository does and how it fits into the system:

| File / Folder | Type | Detailed Description & Role |
| :--- | :--- | :--- |
| **`app.py`** | Python | **All-in-One Web Kiosk & REST API Server.** Runs a lightweight Flask app with an integrated camera thread, MJPEG live video stream (`/video_feed`), web enrollment, attendance dashboard, and background Google Sheets sync daemon. |
| **`kiosk_laptop.py`** | Python | **Desktop Kiosk with Spacebar Trigger.** Uses `cv2.imshow` for desktop preview. Pressing `[SPACEBAR]` triggers recognition, draws on-screen visual banners, logs to SQLite in < 5ms, and syncs via background thread. Includes the Raspberry Pi migration blueprint header. |
| **`kiosk_autoscan_test.py`** | Python | **Hands-Free Auto-Scan Kiosk.** Continuous face-scanning script designed for bare Raspberry Pi or laptop testing without any physical buttons. Prints large, color-coded ANSI terminal cards with match quality %, latency, and sync results. |
| **`enroll.py`** | Python | **CLI Face Enrollment Tool.** Prompts for Student ID and Name, displays live alignment reticle on webcam, captures 20 normalized grayscale crops (200x200, histogram-equalized) into `dataset/<id>/`, and updates `students.json`. |
| **`train.py`** | Python | **LBPH Model Trainer.** Scans `dataset/`, trains OpenCV's `LBPHFaceRecognizer`, outputs `face-trainer.yml` (< 2 seconds), and prints training metrics. |
| **`google_sheets_webhook.js`** | JavaScript | **Google Apps Script Webhook.** Script pasted into Google Sheets (`Extensions -> Apps Script`). Handles JSON POST requests from the kiosk, locks sheet to prevent write collisions, and appends `[Date, Time, Student ID, Student Name, Status]`. |
| **`haarcascade_frontalface_default.xml`** | XML | Official OpenCV Haar Cascade frontal face detection weights bundled directly in the repo, ensuring zero missing dependency errors across operating systems. |
| **`requirements.txt`** | Text | Core Python pip dependencies (`opencv-contrib-python`, `requests`, `numpy`, `Flask`, `psutil`) with platform instructions for Raspberry Pi OS. |
| **`templates/`** | HTML5 | Jinja2 templates for the web portal: `base.html` (cyber nav & telemetry), `index.html` (dashboard & live stream), `enroll.html` (web enrollment), `students.html` (roster), and `attendance.html` (ledger). |
| **`static/style.css`** | CSS3 | Custom modern dark cyber design system with glassmorphic cards, responsive grid, status badges, and progress animations. |
| **`deploy/setup_pi.sh`** | Bash | One-click automated setup script for Raspberry Pi OS (installs packages, configures video permissions, enables systemd service). |
| **`deploy/touchless-kiosk.service`**| Systemd | Linux systemd unit file to automatically launch and restart `app.py` on Raspberry Pi boot. |
| **`students.json`** | JSON | Auto-generated mapping registry storing `{student_id: {name, enrolled_at}}`. |
| **`face-trainer.yml`** | YAML | Auto-generated mathematical model containing trained LBPH histogram weights. |
| **`attendance.db`** | SQLite | Auto-generated local database storing attendance entries with `synced` flags. |
| **`dataset/`** | Directory | Auto-generated directory containing enrolled face image samples organized by student ID (`dataset/<student_id>/sample_XX.jpg`). |

---

## 💻 Quick Start: Laptop / Local Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note**: `opencv-contrib-python` is strictly required (instead of plain `opencv-python`) because LBPH Face Recognizer resides in the `cv2.face` module.

### 2. Configure Your Google Sheets Webhook
Open `app.py`, `kiosk_laptop.py`, and `kiosk_autoscan_test.py`, and ensure `GOOGLE_SHEETS_WEBHOOK_URL` is set to your deployed Google Apps Script URL:
```python
GOOGLE_SHEETS_WEBHOOK_URL = "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"
```

### 3. Launch the System (Choose Your Preferred Interface)

#### Option A: Modern Web Portal (Recommended)
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser:
- Navigate to **Enroll Student** (`/enroll`) -> Enter Student ID & Name -> Click **Start 20-Sample Capture** -> Click **Retrain LBPH Model Now**.
- Navigate back to **Dashboard** (`/`) -> Look at camera -> Face is recognized and logged automatically!

#### Option B: Hands-Free Auto-Scan in Terminal
```bash
python kiosk_autoscan_test.py
```
Continuously detects faces, displays real-time color-coded ANSI terminal banners, logs to SQLite, and syncs to Google Sheets.

#### Option C: Desktop Window with Spacebar Trigger
```bash
python kiosk_laptop.py
```
Press **`[SPACEBAR]`** to trigger recognition, display green confirmation banners, and log attendance.

---

## 📤 How to Upload to Raspberry Pi

You can transfer the entire project from your laptop to the Raspberry Pi using any of the following 3 methods:

### Method 1: Via SCP (Secure Copy over Local Wi-Fi) — Fastest
Open PowerShell or Terminal on your laptop and run:
```bash
# Syntax: scp -r <local-folder> <pi-username>@<pi-ip-address>:<destination>
scp -r "c:\Users\rabbu\Downloads\Touchless Smart Attendance Kiosk" pi@192.168.1.50:/home/pi/Touchless-Smart-Attendance-Kiosk
```
*(Replace `192.168.1.50` with your Raspberry Pi's actual IP address, and `pi` with your username).*

### Method 2: Via Git (GitHub / GitLab)
On your laptop, push this repository to GitHub:
```bash
git init
git add .
git commit -m "Initial commit of Touchless Attendance Kiosk"
git remote add origin https://github.com/your-username/touchless-attendance-kiosk.git
git push -u origin main
```
Then, on your Raspberry Pi:
```bash
git clone https://github.com/your-username/touchless-attendance-kiosk.git /home/pi/Touchless-Smart-Attendance-Kiosk
```

### Method 3: Via USB Flash Drive
1. Copy the `Touchless Smart Attendance Kiosk` folder onto a USB flash drive.
2. Insert the USB drive into any USB port on the Raspberry Pi.
3. Open terminal on the Pi and copy the folder:
   ```bash
   cp -r /media/pi/*/"Touchless Smart Attendance Kiosk" /home/pi/Touchless-Smart-Attendance-Kiosk
   ```

---

## 🍓 Raspberry Pi Installation & Deployment

### Method A: One-Click Automated Deployment (Recommended)
Once the files are on your Raspberry Pi:
```bash
cd /home/pi/Touchless-Smart-Attendance-Kiosk
sudo bash deploy/setup_pi.sh
```
The script will automatically update APT, install pre-built system OpenCV and Python packages, configure video permissions, install the `systemd` auto-start service, and launch the kiosk.

### Method B: Manual Step-by-Step Installation
If you prefer configuring manually on Raspberry Pi OS:

```bash
# 1. Update system repositories
sudo apt update && sudo apt upgrade -y

# 2. Install pre-compiled Edge AI packages (Prevents compilation freezes on 1GB RAM)
sudo apt install -y python3-opencv python3-flask python3-requests python3-numpy python3-psutil python3-pil python3-gpiozero

# 3. Add user to video group
sudo usermod -aG video $USER

# 4. Enable Camera and I2C interfaces
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
# Navigate to: Interface Options -> I2C -> Enable
sudo reboot
```

After rebooting, launch the application:
```bash
cd /home/pi/Touchless-Smart-Attendance-Kiosk
python3 app.py
```
Open `http://<raspberry-pi-ip>:5000` from any phone or PC on your network.

---

## 🕹️ Modes of Operation

| Mode | Command | Best Used For | Features |
| :--- | :--- | :--- | :--- |
| **Flask Web Kiosk & Portal** | `python app.py` | Production deployment on Pi & remote management | Remote access via phone/PC, live MJPEG stream, web face enrollment, 1-click model retraining, REST API, SQLite WAL logging. |
| **Auto-Scan CLI Kiosk** | `python kiosk_autoscan_test.py` | Bare Pi testing with USB webcam | Hands-free continuous scanning, 10s anti-spam lock, color-coded ANSI terminal cards, headless SSH resilience. |
| **Manual Desktop Kiosk** | `python kiosk_laptop.py` | Desktop/laptop testing with physical button simulation | Spacebar trigger, on-screen OpenCV banners, 10s cooldown, SQLite logging, background cloud sync. |
| **Standalone CLI Enrollment** | `python enroll.py` | Terminal-based student registration | Guide reticle, captures 20 normalized crops into `dataset/<id>/`, updates `students.json`. |
| **Standalone Model Trainer** | `python train.py` | Offline batch training | Trains LBPH on `dataset/`, outputs `face-trainer.yml`. |

---

## ☁️ Google Sheets Cloud Webhook Setup

Follow these exact steps to connect your kiosk to Google Sheets:

1. Open [Google Sheets](https://sheets.new) in your web browser.
2. Create a new sheet and rename it to **`Attendance`**.
3. In the top menu, go to **Extensions** -> **Apps Script**.
4. Delete any code inside `Code.gs` and paste the entire contents of [`google_sheets_webhook.js`](google_sheets_webhook.js).
5. Click **Save** (disk icon).
6. Click **Deploy** (top right) -> **New deployment**.
7. Click the gear icon next to "Select type" -> choose **Web app**.
8. Configure the deployment settings:
   - **Description**: `Attendance Kiosk Webhook`
   - **Execute as**: `Me (your_email@gmail.com)`
   - **Who has access**: **`Anyone`** *(CRITICAL: Allows unauthenticated HTTP POST requests from the kiosk)*
9. Click **Deploy** and grant permissions when prompted.
10. Copy the **Web app URL** (looks like: `https://script.google.com/macros/s/AKfycb.../exec`).
11. Paste this URL into `app.py`, `kiosk_laptop.py`, and `kiosk_autoscan_test.py`:
    ```python
    GOOGLE_SHEETS_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycb.../exec"
    ```

### Columns Automatically Appended to Google Sheets:
| Column A | Column B | Column C | Column D | Column E | Column F |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Date** | **Time** | **Student ID** | **Student Name** | **Status** | **Synced At (Cloud)** |
| 2026-09-23 | 19:45:10 | 101 | Alex Rivera | PRESENT | 2026-09-23T14:15:10.000Z |

---

## 🔌 Raspberry Pi Hardware Wiring Guide

When you are ready to connect the physical hardware to your Raspberry Pi 3B+, use the pinout diagram below:

### Pinout Mapping

```
     Raspberry Pi 3B+ GPIO Header
          3.3V Power [01] [02] 5V Power
     GPIO 2 (SDA1) --[03] [04] 5V Power
     GPIO 3 (SCL1) --[05] [06] Ground ------------- Button LED (-)
                     [07] [08] GPIO 14 (TX)
            Ground --[09] [10] GPIO 15 (RX)
    Button Input ----[11] [12] GPIO 18
                     [13] [14] Ground ------------- OLED Ground
                     [15] [16] GPIO 23
       OLED 3.3V ----[17] [18] GPIO 24
                     [19] [20] Ground
```

| Peripheral | Component Pin | Raspberry Pi Header | Function |
| :--- | :--- | :--- | :--- |
| **16mm Metal Button** | Contact Terminal 1 | Physical Pin 11 | **GPIO 17** (Pull-Up Input) |
| | Contact Terminal 2 | Physical Pin 9 | **GND** |
| | LED Terminal (+) | Physical Pin 1 | **3.3V Power** |
| | LED Terminal (-) | Physical Pin 6 | **GND** |
| **0.96" I2C OLED** | VCC | Physical Pin 17 | **3.3V Power** |
| | GND | Physical Pin 14 | **GND** |
| | SDA | Physical Pin 3 | **GPIO 2 (I2C1 SDA)** |
| | SCL | Physical Pin 5 | **GPIO 3 (I2C1 SCL)** |
| **Pi Camera Module 3** | Ribbon Cable | CSI Camera Port | Blue tape faces Ethernet jack |

---

## 🛠️ Complete REST API Reference

The Flask application (`app.py`) exposes a full REST API for remote integration:

### 1. System Telemetry
- **Endpoint**: `GET /api/system/status`
- **Response**:
  ```json
  {
    "status": "online",
    "camera_active": true,
    "cpu_percent": 14.2,
    "memory_mb": 98,
    "memory_percent": 9.8,
    "pending_sync_count": 0,
    "total_attendance": 42,
    "total_students": 5,
    "cloud_sync_status": "Idle (Synced)"
  }
  ```

### 2. Query Attendance Records
- **Endpoint**: `GET /api/attendance?limit=50&date=2026-09-23`
- **Response**:
  ```json
  [
    {
      "id": 1,
      "student_id": 101,
      "student_name": "Alex Rivera",
      "timestamp": "2026-09-23 19:45:10",
      "synced": 1
    }
  ]
  ```

### 3. Force Cloud Sync
- **Endpoint**: `POST /api/sync/trigger`
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Successfully pushed 3 rows to Google Sheets",
    "last_sync_status": "Synced 3 rows (19:45:25)"
  }
  ```

### 4. Capture Face Crop for Enrollment
- **Endpoint**: `POST /api/students/capture_crop`
- **Body**:
  ```json
  {
    "student_id": 101,
    "student_name": "Alex Rivera",
    "sample_index": 1
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "student_id": 101,
    "sample_index": 1,
    "file": "dataset/101/sample_01.jpg"
  }
  ```

### 5. Retrain Model
- **Endpoint**: `POST /api/model/train`
- **Response**:
  ```json
  {
    "status": "success",
    "duration": 0.85,
    "students_count": 5,
    "samples_count": 100
  }
  ```

### 6. Delete Student
- **Endpoint**: `DELETE /api/students/<student_id>`
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Student 101 removed"
  }
  ```

---

## ⚙️ Background Service & Auto-Start (systemd)

To ensure the kiosk runs 24/7 without manual intervention, configure it as a Linux system service:

### Install Service:
```bash
sudo cp deploy/touchless-kiosk.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable touchless-kiosk.service
sudo systemctl start touchless-kiosk.service
```

### Essential Commands:
```bash
# Check if service is active and running:
sudo systemctl status touchless-kiosk

# View real-time output & logs:
sudo journalctl -u touchless-kiosk -f

# Restart service after editing code:
sudo systemctl restart touchless-kiosk

# Stop service:
sudo systemctl stop touchless-kiosk
```

---

## ❓ Troubleshooting & FAQs

### Q1: `ModuleNotFoundError: No module named 'cv2.face'`
**Cause**: The standard `opencv-python` package was installed instead of `opencv-contrib-python`.  
**Fix**:
```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python>=4.8.0.76
```
*(On Raspberry Pi, simply run: `sudo apt install -y python3-opencv`)*

### Q2: Camera fails to open (`Unable to open camera`)
- Ensure no other application (Zoom, Teams, another Python script) is currently using the camera.
- On Raspberry Pi, check that the user has video permissions:
  ```bash
  sudo usermod -aG video $USER
  ```
- If using Raspberry Pi Camera Module 3, make sure ribbon cable is firmly seated with blue side facing the Ethernet jack.

### Q3: How do I adjust the recognition sensitivity?
In `app.py`, `kiosk_laptop.py`, or `kiosk_autoscan_test.py`, modify:
```python
LBPH_CONFIDENCE_THRESHOLD = 75.0
```
- **Lower value (e.g., 60.0 - 68.0)**: Stricter recognition, prevents false positives.
- **Higher value (e.g., 75.0 - 82.0)**: More forgiving match, useful in dim or uneven lighting.

### Q4: Google Sheets sync says `HTTP 401` or `HTTP 302`
- Ensure that when deploying the Apps Script Web App, **Who has access** is set to **`Anyone`**, not *Only myself*. If set to *Only myself*, Google will block requests that do not provide OAuth credentials.

---

## 📄 License
This project is open-source and released under the **MIT License**. Free for educational, commercial, and personal use.
