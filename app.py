"""
====================================================================================================
TOUCHLESS SMART ATTENDANCE KIOSK - FLASK ADMIN PORTAL & REST API (app.py)
====================================================================================================
Modern Executive Light Theme & Comprehensive Student Details Architecture:
  1. Captures Full Name, Alphanumeric Roll Number (e.g. 21B91A0501), Year of Study, and Branch.
  2. Mapped 1-to-1 to an internal integer label for OpenCV LBPH compliance.
  3. Records arrival Date & discrete In-Time in SQLite WAL (< 5ms) and Google Sheets.
  4. Real-time MJPEG live video stream (/video_feed) with HUD overlays.
  5. 100% Hands-free Auto-Scan Kiosk engine with 10s anti-spam lock.
====================================================================================================
"""

import os
import sys
import json
import time
import shutil
import sqlite3
import threading
from datetime import datetime
import cv2
import numpy as np
import requests
import psutil
from flask import Flask, render_template, Response, request, jsonify

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FACE_CROP_SIZE = (200, 200)

DATASET_DIR = "dataset"
DB_PATH = "attendance.db"
TRAINER_FILE = "face-trainer.yml"
STUDENTS_FILE = "students.json"

# LBPH Biometric Distance Thresholds:
#   OpenCV LBPH measures Euclidean distance between Local Binary Pattern Histograms.
#   Lower distance = closer biometric match.
#     • < 50.0: Strict match (Same person under lighting variations)
#     • 50.0 - 65.0: Attendance recognition match (Suitable for daily check-in)
#     • > 65.0: Different individual (Siblings, family members, classmates)
LBPH_RECOGNITION_THRESHOLD = 65.0  # Used for daily attendance scanning
LBPH_DUPLICATE_THRESHOLD = 50.0    # Strict threshold for duplicate identity detection (<= 50.0 & >= 55% similarity)
COOLDOWN_SECONDS = 10.0
SYNC_INTERVAL_SECONDS = 15.0

# -----------------------------------------------------------------------------
# Environment & Secrets Configuration (Loaded from .env if present)
# -----------------------------------------------------------------------------
def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            print(f"[WARN] Failed to parse .env file: {e}")

_load_env_file()

# Google Sheets Apps Script Webhook URL (Loaded from .env)
GOOGLE_SHEETS_WEBHOOK_URL = os.getenv("GOOGLE_SHEETS_WEBHOOK_URL", "")

# UI Theme Color Tokens (BGR format for OpenCV overlays)
UI_BG_HEADER = (245, 247, 250)
UI_BG_BANNER = (255, 255, 255)
UI_ACCENT = (229, 70, 79)         # Royal Indigo / Blue
UI_TEXT_MAIN = (15, 23, 42)       # Dark Slate
UI_TEXT_MUTED = (100, 116, 139)   # Muted Slate
UI_SUCCESS = (5, 150, 105)        # Emerald Green
UI_DANGER = (220, 38, 38)         # Red
UI_WARNING = (217, 119, 6)        # Amber
UI_INFO = (2, 132, 199)           # Cyan/Blue

# Initialize Flask App
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "touchless-edge-kiosk-secret-key")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


# -----------------------------------------------------------------------------
# Haar Cascade XML Path Resolver
# -----------------------------------------------------------------------------
def get_cascade_path() -> str:
    """Resolves Haar Cascade XML from local directory first, then OpenCV data."""
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        return local_path
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        opencv_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(opencv_path):
            return opencv_path
    return "haarcascade_frontalface_default.xml"


# -----------------------------------------------------------------------------
# Local SQLite Database Engine (WAL Mode, < 5ms latency & Auto-Migration)
# -----------------------------------------------------------------------------
def init_database() -> None:
    """Initializes SQLite database with Write-Ahead Logging (WAL) and auto-migrates columns."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        
        # Create core table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                roll_number TEXT DEFAULT '',
                student_name TEXT NOT NULL,
                year TEXT DEFAULT '',
                branch TEXT DEFAULT '',
                date TEXT DEFAULT '',
                in_time TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                synced INTEGER DEFAULT 0
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_synced ON attendance (synced);")
        
        # Check and perform safe schema migration for existing databases
        cursor.execute("PRAGMA table_info(attendance);")
        existing_cols = [row[1] for row in cursor.fetchall()]
        
        migrations = [
            ("roll_number", "TEXT DEFAULT ''"),
            ("year", "TEXT DEFAULT ''"),
            ("branch", "TEXT DEFAULT ''"),
            ("date", "TEXT DEFAULT ''"),
            ("in_time", "TEXT DEFAULT ''")
        ]
        for col_name, col_type in migrations:
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE attendance ADD COLUMN {col_name} {col_type};")
                print(f"[DB MIGRATION] Added missing column '{col_name}' to attendance table.")

        conn.commit()
    finally:
        conn.close()


def log_attendance_local(student_id: int, roll_number: str, student_name: str, 
                         year: str, branch: str, date_str: str, in_time_str: str, timestamp_str: str) -> bool:
    """Records full student attendance entry in SQLite. Executes in ~1-3ms."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO attendance 
               (student_id, roll_number, student_name, year, branch, date, in_time, timestamp, synced) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0);""",
            (student_id, roll_number, student_name, year, branch, date_str, in_time_str, timestamp_str)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[DB ERROR] {e}")
        return False
    finally:
        conn.close()


def get_sync_stats() -> tuple[int, int]:
    """Returns (pending_count, total_count) from SQLite."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE synced = 0;")
        pending = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM attendance;")
        total = cursor.fetchone()[0]
        return pending, total
    except Exception:
        return 0, 0
    finally:
        conn.close()


def get_today_count() -> int:
    """Returns total attendance entries logged today."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE timestamp LIKE ? OR date = ?;", (f"{today_str}%", today_str))
        return cursor.fetchone()[0]
    except Exception:
        return 0
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Students Registry Helper
# -----------------------------------------------------------------------------
def load_students_registry() -> dict:
    if os.path.exists(STUDENTS_FILE):
        try:
            with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_students_registry(data: dict) -> None:
    with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_or_create_numeric_id(roll_number: str) -> int:
    """
    OpenCV LBPH requires integer labels. Maps an alphanumeric roll number
    (e.g. '21B91A0501') to a consistent internal integer ID.
    """
    registry = load_students_registry()
    clean_roll = roll_number.strip().upper()
    
    # Check if roll number already exists
    for sid, info in registry.items():
        if info.get("roll_number", "").upper() == clean_roll:
            return int(sid)
            
    # Allocate next available integer ID
    existing_ids = [int(k) for k in registry.keys() if k.isdigit()]
    return (max(existing_ids) + 1) if existing_ids else 101


# -----------------------------------------------------------------------------
# Anti-Spam Cooldown Manager
# -----------------------------------------------------------------------------
class AntiSpamManager:
    def __init__(self, cooldown_seconds: float = 10.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_logged: dict[int, float] = {}

    def is_cooling_down(self, student_id: int) -> tuple[bool, float]:
        now = time.time()
        last_time = self.last_logged.get(student_id, 0.0)
        elapsed = now - last_time
        if elapsed < self.cooldown_seconds:
            return True, (self.cooldown_seconds - elapsed)
        return False, 0.0

    def register_log(self, student_id: int) -> None:
        self.last_logged[student_id] = time.time()


# -----------------------------------------------------------------------------
# Background Cloud Synchronization Worker
# -----------------------------------------------------------------------------
class CloudSyncWorker:
    def __init__(self, webhook_url: str, interval_sec: float = 15.0):
        self.webhook_url = webhook_url
        self.interval_sec = interval_sec
        self.running = True
        self.last_sync_status = "Initialized"
        self.last_sync_time = None
        self.thread = threading.Thread(target=self._run_loop, name="CloudSyncWorker", daemon=True)

    def start(self):
        self.thread.start()

    def _run_loop(self):
        while self.running:
            try:
                self.sync_now()
            except Exception as e:
                self.last_sync_status = f"Err: {str(e)[:20]}"
            for _ in range(int(self.interval_sec)):
                if not self.running:
                    break
                time.sleep(1.0)

    def sync_now(self) -> tuple[bool, str]:
        if not self.webhook_url or "YOUR-WEBHOOK-TOKEN-HERE" in self.webhook_url:
            self.last_sync_status = "Webhook not set"
            return False, "Webhook URL not configured"

        conn = sqlite3.connect(DB_PATH, timeout=5.0)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, student_id, roll_number, student_name, year, branch, date, in_time, timestamp 
                FROM attendance WHERE synced = 0 LIMIT 50;
            """)
            rows = cursor.fetchall()
            if not rows:
                self.last_sync_status = "Idle (Synced)"
                return True, "No pending records to sync"

            records = []
            row_ids = []
            for r in rows:
                row_ids.append(r[0])
                records.append({
                    "id": r[0],
                    "student_id": r[1],
                    "roll_number": r[2] or str(r[1]),
                    "student_name": r[3],
                    "year": r[4] or "N/A",
                    "branch": r[5] or "N/A",
                    "date": r[6] or (r[8].split(" ")[0] if " " in r[8] else ""),
                    "in_time": r[7] or (r[8].split(" ")[1] if " " in r[8] else ""),
                    "timestamp": r[8],
                    "status": "PRESENT"
                })

            payload = {"records": records}
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=25.0
            )

            if response.status_code == 200:
                placeholders = ",".join(["?"] * len(row_ids))
                cursor.execute(f"UPDATE attendance SET synced = 1 WHERE id IN ({placeholders});", row_ids)
                conn.commit()
                self.last_sync_time = datetime.now().strftime("%H:%M:%S")
                self.last_sync_status = f"Synced {len(records)} rows ({self.last_sync_time})"
                return True, f"Successfully pushed {len(records)} rows to Google Sheets"
            else:
                self.last_sync_status = f"HTTP {response.status_code}"
                return False, f"Google Webhook returned HTTP {response.status_code}"

        except requests.exceptions.RequestException as e:
            self.last_sync_status = "Offline"
            return False, f"Network drop: {e}"
        finally:
            conn.close()


# -----------------------------------------------------------------------------
# Global Video Stream & Auto-Scan Camera Manager
# -----------------------------------------------------------------------------
class CameraManager:
    """
    Centralized Camera Controller.
    Captures 640x480 frames, performs continuous auto-scan recognition when enabled,
    provides normalized face crops for web enrollment, and yields MJPEG frames.
    Supports complete hardware release (LED off, zero CPU) via physical power toggle.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.cap = None
        self.latest_raw_frame = None
        self.latest_display_frame = self._generate_camera_off_frame()
        self.latest_face_crop = None
        self.camera_active = False
        
        # Auto-scan toggle state (allows pausing scanner from web or during enrollment)
        self.autoscan_enabled = True
        self.enrollment_active = False

        cascade_file = get_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

        self.recognizer = None
        self.reload_recognizer()

        self.banner_title = ""
        self.banner_subtitle = ""
        self.banner_color = UI_BG_BANNER
        self.banner_expiry = 0.0

        self.thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)

    def _generate_camera_off_frame(self) -> np.ndarray:
        """Generates a crisp Executive Light Theme visual frame when camera is turned off."""
        canvas = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (250, 250, 248), dtype=np.uint8)
        
        # Outer subtle card border
        cv2.rectangle(canvas, (35, 45), (FRAME_WIDTH - 35, FRAME_HEIGHT - 45), (226, 232, 240), 2)
        cv2.rectangle(canvas, (37, 47), (FRAME_WIDTH - 37, FRAME_HEIGHT - 47), (255, 255, 255), -1)
        
        # Camera Disabled Badge Circle
        cv2.circle(canvas, (FRAME_WIDTH // 2, 150), 36, (241, 245, 249), -1)
        cv2.circle(canvas, (FRAME_WIDTH // 2, 150), 36, (203, 213, 225), 1)
        cv2.putText(canvas, "OFF", (FRAME_WIDTH // 2 - 25, 158), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 38, 38), 2, cv2.LINE_AA)
        
        # Main Title
        cv2.putText(canvas, "LAPTOP CAMERA IS POWERED OFF", (FRAME_WIDTH // 2 - 215, 230),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, (15, 23, 42), 1, cv2.LINE_AA)
        
        # Explanation
        cv2.putText(canvas, "Physical webcam hardware is safely released to protect privacy.",
                    (FRAME_WIDTH // 2 - 225, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1, cv2.LINE_AA)
        
        cv2.putText(canvas, "Click 'Camera: OFF' in Navbar or 'Turn Camera ON' on Dashboard.",
                    (FRAME_WIDTH // 2 - 235, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (79, 70, 229), 1, cv2.LINE_AA)
        
        # Standby Footer Bar
        cv2.rectangle(canvas, (37, FRAME_HEIGHT - 85), (FRAME_WIDTH - 37, FRAME_HEIGHT - 47), (248, 250, 252), -1)
        cv2.putText(canvas, "HARDWARE STATUS: STANDBY / IDLE", (FRAME_WIDTH // 2 - 135, FRAME_HEIGHT - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)
        
        return canvas

    def _open_camera(self) -> bool:
        """Connects to the physical webcam with graceful backend fallback."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            print(f"[CAM MANAGER] Initializing camera device at {FRAME_WIDTH}x{FRAME_HEIGHT}...")
            cap = None
            if sys.platform.startswith("win"):
                try:
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                except Exception:
                    cap = None
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)

            if not cap or not cap.isOpened():
                print("[CAM MANAGER ERROR] Could not open camera device 0.")
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap = cap
            return True
        except Exception as e:
            print(f"[CAM MANAGER ERROR] Failed to open camera: {e}")
            return False

    def _release_camera(self):
        """Explicitly releases physical capture hardware, turning off LED."""
        try:
            if self.cap is not None:
                print("[CAM MANAGER] Releasing physical camera device (turning off LED)...")
                self.cap.release()
                self.cap = None
        except Exception as e:
            print(f"[CAM MANAGER WARN] Error releasing camera: {e}")

    def set_camera_power(self, enable: bool) -> tuple[bool, str]:
        """Toggles camera power on or off dynamically."""
        with self.lock:
            if enable == self.camera_active:
                state_str = "ON" if self.camera_active else "OFF"
                return True, f"Camera is already {state_str}"

            if enable:
                success = self._open_camera()
                if success:
                    self.camera_active = True
                    return True, "Camera powered ON successfully"
                else:
                    self.camera_active = False
                    self.latest_display_frame = self._generate_camera_off_frame()
                    return False, "Could not open camera device"
            else:
                self._release_camera()
                self.camera_active = False
                self.latest_face_crop = None
                self.latest_display_frame = self._generate_camera_off_frame()
                return True, "Camera powered OFF and hardware released"

    def toggle_camera_power(self) -> tuple[bool, str]:
        """Toggles between ON and OFF."""
        return self.set_camera_power(not self.camera_active)

    def reload_recognizer(self):
        if os.path.exists(TRAINER_FILE) and hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            try:
                rec = cv2.face.LBPHFaceRecognizer_create()
                rec.read(TRAINER_FILE)
                self.recognizer = rec
                print(f"[CAM MANAGER] LBPH model reloaded from '{TRAINER_FILE}'.")
            except Exception as e:
                print(f"[CAM MANAGER WARN] Could not load LBPH model: {e}")
                self.recognizer = None
        else:
            self.recognizer = None

    def start(self):
        self.thread.start()

    def set_banner(self, title: str, subtitle: str, color: tuple, duration: float = 3.0):
        self.banner_title = title
        self.banner_subtitle = subtitle
        self.banner_color = color
        self.banner_expiry = time.time() + duration

    def _capture_loop(self):
        # Attempt initial hardware camera connection
        if self._open_camera():
            self.camera_active = True
        else:
            self.camera_active = False
            with self.lock:
                self.latest_display_frame = self._generate_camera_off_frame()

        frame_skip = 0

        while self.running:
            # If camera is powered off, idle without reading from released hardware
            if not self.camera_active or self.cap is None:
                with self.lock:
                    self.latest_display_frame = self._generate_camera_off_frame()
                    self.latest_face_crop = None
                time.sleep(0.12)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face Detection (every 2nd frame for low CPU on Pi)
            frame_skip += 1
            face_box = None
            if frame_skip % 2 == 0:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(110, 110),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                    face_box = faces[0]

            now = time.time()

            if face_box is not None:
                x, y, fw, fh = face_box
                box_color = (229, 70, 79) if self.autoscan_enabled else (100, 116, 139)
                cv2.rectangle(display, (x, y), (x + fw, y + fh), box_color, 2)

                # Extract normalized face crop
                crop = gray[y:y + fh, x:x + fw]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, FACE_CROP_SIZE, interpolation=cv2.INTER_AREA)
                    crop_eq = cv2.equalizeHist(crop_resized)
                    with self.lock:
                        self.latest_face_crop = crop_eq

                    # Run Auto-Scan Recognition ONLY if scanner is active and not in enrollment mode
                    if self.enrollment_active:
                        cv2.putText(display, "READY FOR ENROLLMENT", (x, max(y - 8, 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (79, 70, 229), 1, cv2.LINE_AA)
                    elif not self.autoscan_enabled:
                        cv2.putText(display, "SCANNER PAUSED", (x, max(y - 8, 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1, cv2.LINE_AA)
                    elif self.recognizer is not None:
                        try:
                            student_id, distance = self.recognizer.predict(crop_eq)
                            if distance <= LBPH_RECOGNITION_THRESHOLD:
                                is_cooling, remaining = anti_spam.is_cooling_down(student_id)
                                registry = load_students_registry()
                                student_info = registry.get(str(student_id), {})
                                student_name = student_info.get("student_name", f"Student {student_id}")
                                roll_number = student_info.get("roll_number", str(student_id))
                                year = student_info.get("year", "")
                                branch = student_info.get("branch", "")

                                if not is_cooling:
                                    # Log attendance instantly with In-Time
                                    now_dt = datetime.now()
                                    date_str = now_dt.strftime("%Y-%m-%d")
                                    in_time_str = now_dt.strftime("%H:%M:%S")
                                    timestamp_str = f"{date_str} {in_time_str}"

                                    log_attendance_local(student_id, roll_number, student_name, 
                                                         year, branch, date_str, in_time_str, timestamp_str)
                                    anti_spam.register_log(student_id)

                                    self.set_banner(
                                        title=f"VERIFIED: {student_name}",
                                        subtitle=f"Roll: {roll_number} | In-Time: {in_time_str} | {branch}",
                                        color=(5, 150, 105), # Emerald Green
                                        duration=3.5
                                    )
                                else:
                                    cv2.putText(display, f"COOLDOWN ({int(remaining)}s)", (x, max(y - 8, 30)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (2, 132, 199), 1, cv2.LINE_AA)
                            else:
                                cv2.putText(display, "UNREGISTERED", (x, max(y - 8, 30)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 38, 38), 1, cv2.LINE_AA)
                        except Exception:
                            pass

            # Draw HUD Overlay Banner on MJPEG Frame
            h, w = display.shape[:2]
            if now < self.banner_expiry:
                banner_h = 68
                banner_y = h - 25 - banner_h
                overlay = display.copy()
                cv2.rectangle(overlay, (12, banner_y), (w - 12, banner_y + banner_h), (255, 255, 255), -1)
                cv2.addWeighted(overlay, 0.92, display, 0.08, 0, display)
                cv2.rectangle(display, (12, banner_y), (20, banner_y + banner_h), self.banner_color, -1)
                cv2.rectangle(display, (12, banner_y), (w - 12, banner_y + banner_h), (226, 232, 240), 1)

                cv2.putText(display, self.banner_title, (30, banner_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.60, (15, 23, 42), 1, cv2.LINE_AA)
                cv2.putText(display, self.banner_subtitle, (30, banner_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (71, 85, 105), 1, cv2.LINE_AA)

            with self.lock:
                self.latest_raw_frame = frame
                self.latest_display_frame = display

            time.sleep(0.025)

        self._release_camera()

    def get_jpeg_bytes(self) -> bytes:
        with self.lock:
            if self.latest_display_frame is None:
                frame = self._generate_camera_off_frame()
            else:
                frame = self.latest_display_frame

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes() if ret else b""

    def capture_normalized_face(self) -> np.ndarray:
        with self.lock:
            return self.latest_face_crop


# -----------------------------------------------------------------------------
# Global Singletons
# -----------------------------------------------------------------------------
init_database()
anti_spam = AntiSpamManager(cooldown_seconds=COOLDOWN_SECONDS)
sync_worker = CloudSyncWorker(webhook_url=GOOGLE_SHEETS_WEBHOOK_URL, interval_sec=SYNC_INTERVAL_SECONDS)
sync_worker.start()

camera_mgr = CameraManager()
camera_mgr.start()


# -----------------------------------------------------------------------------
# Web Routes (HTML Views)
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    pending_count, _ = get_sync_stats()
    today_count = get_today_count()
    registry = load_students_registry()
    student_count = len(registry)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, student_id, roll_number, student_name, year, branch, in_time, timestamp, synced 
        FROM attendance ORDER BY id DESC LIMIT 8;
    """)
    rows = cursor.fetchall()
    conn.close()

    recent_logs = [
        {
            "id": r[0], 
            "student_id": r[1], 
            "roll_number": r[2] or str(r[1]), 
            "student_name": r[3], 
            "year": r[4], 
            "branch": r[5], 
            "in_time": r[6] or (r[7].split(" ")[1] if " " in r[7] else ""), 
            "timestamp": r[7], 
            "synced": r[8]
        }
        for r in rows
    ]

    return render_template(
        "index.html",
        pending_count=pending_count,
        today_count=today_count,
        student_count=student_count,
        recent_logs=recent_logs
    )


@app.route("/enroll")
def enroll_page():
    return render_template("enroll.html")


@app.route("/students")
def students_page():
    registry = load_students_registry()
    
    for sid in registry:
        folder = os.path.join(DATASET_DIR, str(sid))
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            count = 0
        registry[sid]["sample_count"] = count

    return render_template("students.html", students=registry)


@app.route("/attendance")
def attendance_page():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, student_id, roll_number, student_name, year, branch, date, in_time, timestamp, synced 
        FROM attendance ORDER BY id DESC LIMIT 500;
    """)
    rows = cursor.fetchall()
    conn.close()

    logs = [
        {
            "id": r[0], 
            "student_id": r[1], 
            "roll_number": r[2] or str(r[1]), 
            "student_name": r[3], 
            "year": r[4], 
            "branch": r[5], 
            "date": r[6] or (r[8].split(" ")[0] if " " in r[8] else ""), 
            "in_time": r[7] or (r[8].split(" ")[1] if " " in r[8] else ""), 
            "timestamp": r[8], 
            "synced": r[9]
        }
        for r in rows
    ]
    return render_template("attendance.html", logs=logs)


# -----------------------------------------------------------------------------
# Video Streaming Route (MJPEG)
# -----------------------------------------------------------------------------
def generate_mjpeg_stream():
    while True:
        frame_bytes = camera_mgr.get_jpeg_bytes()
        if frame_bytes:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.04)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/system/status")
def api_system_status():
    """Returns hardware resource telemetry and kiosk status."""
    pending, total = get_sync_stats()
    registry = load_students_registry()

    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mem_used_mb = int(mem.used / (1024 * 1024))

    return jsonify({
        "status": "online",
        "camera_active": camera_mgr.camera_active,
        "autoscan_enabled": camera_mgr.autoscan_enabled,
        "enrollment_active": camera_mgr.enrollment_active,
        "cpu_percent": cpu_pct,
        "memory_mb": mem_used_mb,
        "memory_percent": mem.percent,
        "pending_sync_count": pending,
        "total_attendance": total,
        "total_students": len(registry),
        "cloud_sync_status": sync_worker.last_sync_status
    })


@app.route("/api/camera/toggle", methods=["POST"])
def api_camera_toggle():
    """Toggles laptop camera hardware power on/off (releases device on OFF)."""
    success, msg = camera_mgr.toggle_camera_power()
    return jsonify({
        "status": "success" if success else "error",
        "camera_active": camera_mgr.camera_active,
        "message": msg
    })


@app.route("/api/camera/set", methods=["POST"])
def api_camera_set():
    """Explicitly turns camera power on or off."""
    data = request.get_json() or {}
    enable = data.get("active", True)
    success, msg = camera_mgr.set_camera_power(bool(enable))
    return jsonify({
        "status": "success" if success else "error",
        "camera_active": camera_mgr.camera_active,
        "message": msg
    })


@app.route("/api/kiosk/toggle_scan", methods=["POST"])
def api_toggle_scan():
    """Toggles continuous face scanning on/off."""
    camera_mgr.autoscan_enabled = not camera_mgr.autoscan_enabled
    state_str = "ACTIVE" if camera_mgr.autoscan_enabled else "PAUSED"
    return jsonify({
        "status": "success",
        "autoscan_enabled": camera_mgr.autoscan_enabled,
        "message": f"Auto-Scan Recognition is now {state_str}"
    })


@app.route("/api/kiosk/set_scan", methods=["POST"])
def api_set_scan():
    """Explicitly enables or disables auto-scan recognition."""
    data = request.get_json() or {}
    enabled = data.get("enabled", True)
    camera_mgr.autoscan_enabled = bool(enabled)
    return jsonify({
        "status": "success",
        "autoscan_enabled": camera_mgr.autoscan_enabled
    })


@app.route("/api/kiosk/set_enrollment_mode", methods=["POST"])
def api_set_enrollment_mode():
    """Enables enrollment mode to pause auto-scan while capturing new students."""
    data = request.get_json() or {}
    active = data.get("active", False)
    camera_mgr.enrollment_active = bool(active)
    return jsonify({
        "status": "success",
        "enrollment_active": camera_mgr.enrollment_active
    })


@app.route("/api/students/check_duplicate", methods=["POST"])
def api_check_duplicate():
    """
    Multi-layer duplicate verification:
      1. Checks if roll number already exists.
      2. Biometrically checks live face against LBPH model to detect if the same
         person is trying to register under a different roll number/identity.
    """
    data = request.get_json() or {}
    roll_number = data.get("roll_number", "").strip().upper()
    registry = load_students_registry()

    roll_exists = False
    existing_student = None
    for sid, info in registry.items():
        if info.get("roll_number", "").upper() == roll_number:
            roll_exists = True
            existing_student = info
            break

    # If recognizer is not currently loaded in memory, attempt reload
    if camera_mgr.recognizer is None and os.path.exists(TRAINER_FILE):
        camera_mgr.reload_recognizer()

    # Check if face in camera matches any existing student
    face_match = {"matched": False}
    crop = camera_mgr.capture_normalized_face()
    if crop is not None and camera_mgr.recognizer is not None:
        try:
            pred_id, dist = camera_mgr.recognizer.predict(crop)
            # Only consider a biometric duplicate if the match is strictly high confidence (dist <= 50.0)
            if dist <= LBPH_DUPLICATE_THRESHOLD:
                matched_info = registry.get(str(pred_id), {})
                matched_roll = matched_info.get("roll_number", str(pred_id)).upper()
                is_same_roll = (matched_roll == roll_number) if roll_number else False
                is_duplicate_identity = not is_same_roll
                confidence = max(10, min(99, int((1.0 - (dist / 80.0)) * 100)))

                # Only block if similarity is actually high (>= 55%)
                if confidence >= 55 and is_duplicate_identity:
                    face_match = {
                        "matched": True,
                        "student_id": pred_id,
                        "roll_number": matched_roll,
                        "student_name": matched_info.get("student_name", "Unknown"),
                        "year": matched_info.get("year", ""),
                        "branch": matched_info.get("branch", ""),
                        "distance": round(float(dist), 1),
                        "confidence": confidence,
                        "is_same_roll": is_same_roll,
                        "is_duplicate_identity": True
                    }
        except Exception as e:
            print(f"[CHECK DUPLICATE WARN] LBPH prediction error: {e}")

    return jsonify({
        "status": "success",
        "roll_exists": roll_exists,
        "existing_student": existing_student,
        "face_match": face_match
    })


@app.route("/api/attendance")
def api_attendance():
    """Returns attendance records in JSON with optional limit and date filter."""
    limit = request.args.get("limit", 100, type=int)
    date_filter = request.args.get("date", "", type=str)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if date_filter:
        cursor.execute(
            """SELECT id, student_id, roll_number, student_name, year, branch, date, in_time, timestamp, synced 
               FROM attendance WHERE date = ? OR timestamp LIKE ? ORDER BY id DESC LIMIT ?;""",
            (date_filter, f"{date_filter}%", limit)
        )
    else:
        cursor.execute(
            """SELECT id, student_id, roll_number, student_name, year, branch, date, in_time, timestamp, synced 
               FROM attendance ORDER BY id DESC LIMIT ?;""",
            (limit,)
        )
    rows = cursor.fetchall()
    conn.close()

    logs = [
        {
            "id": r[0], 
            "student_id": r[1], 
            "roll_number": r[2] or str(r[1]), 
            "student_name": r[3], 
            "year": r[4], 
            "branch": r[5], 
            "date": r[6] or (r[8].split(" ")[0] if " " in r[8] else ""), 
            "in_time": r[7] or (r[8].split(" ")[1] if " " in r[8] else ""), 
            "timestamp": r[8], 
            "synced": r[9]
        }
        for r in rows
    ]
    return jsonify(logs)


@app.route("/api/sync/trigger", methods=["POST"])
def api_sync_trigger():
    """Manually forces the cloud synchronization worker to push pending rows."""
    success, msg = sync_worker.sync_now()
    return jsonify({
        "status": "success" if success else "error",
        "message": msg,
        "last_sync_status": sync_worker.last_sync_status
    })


@app.route("/api/students/capture_crop", methods=["POST"])
def api_capture_crop():
    """
    Captures a normalized 200x200 face crop from the live video feed.
    Enforces server-side biometric uniqueness and automatically retrains LBPH on sample 20.
    """
    data = request.get_json() or {}
    student_name = data.get("student_name", "").strip()
    roll_number = data.get("roll_number", "").strip().upper()
    year = data.get("year", "3rd Year")
    branch = data.get("branch", "Computer Science & Engineering")
    sample_index = data.get("sample_index", 1)
    overwrite = data.get("overwrite", False)

    if not student_name or not roll_number:
        return jsonify({"status": "error", "message": "Missing student_name or roll_number"}), 400

    crop = camera_mgr.capture_normalized_face()
    if crop is None:
        return jsonify({"status": "error", "message": "No face currently detected in viewfinder"}), 422

    registry = load_students_registry()

    # SERVER-SIDE BIOMETRIC DUPLICATE PROTECTION (on 1st sample if not an intentional overwrite)
    if sample_index == 1 and not overwrite and camera_mgr.recognizer is not None:
        try:
            pred_id, dist = camera_mgr.recognizer.predict(crop)
            if dist <= LBPH_DUPLICATE_THRESHOLD:
                confidence = max(10, min(99, int((1.0 - (dist / 80.0)) * 100)))
                if confidence >= 55:
                    matched_info = registry.get(str(pred_id), {})
                    matched_roll = matched_info.get("roll_number", str(pred_id)).upper()
                    if matched_roll and matched_roll != roll_number:
                        return jsonify({
                            "status": "duplicate_biometric",
                            "message": f"Biometric duplicate: Face strongly matches enrolled student '{matched_info.get('student_name')}' (Roll No: {matched_roll}) with {confidence}% similarity.",
                            "matched_student": matched_info,
                            "distance": round(float(dist), 1),
                            "confidence": confidence
                        }), 409
        except Exception:
            pass

    # Map alphanumeric roll number to internal integer ID for LBPH
    student_id = get_or_create_numeric_id(roll_number)

    # Save crop to dataset/<student_id>/
    folder = os.path.join(DATASET_DIR, str(student_id))
    if overwrite and sample_index == 1 and os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"sample_{sample_index:02d}.jpg")
    cv2.imwrite(filename, crop)

    # Save full details in students.json
    registry[str(student_id)] = {
        "student_id": student_id,
        "roll_number": roll_number,
        "student_name": student_name,
        "year": year,
        "branch": branch,
        "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_students_registry(registry)

    # REAL-TIME AUTOMATIC TRAINING: If 20th sample was captured, train immediately!
    auto_trained = False
    if sample_index == 20:
        try:
            from train import prepare_training_data
            faces, labels, stats = prepare_training_data(DATASET_DIR)
            if faces and labels:
                rec = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
                rec.train(faces, np.array(labels, dtype=np.int32))
                rec.write(TRAINER_FILE)
                camera_mgr.reload_recognizer()
                auto_trained = True
                print(f"[AUTO-TRAIN] LBPH model automatically trained on {len(stats)} students ({len(faces)} samples).")
        except Exception as e:
            print(f"[AUTO-TRAIN ERROR] Failed to auto-train: {e}")

    return jsonify({
        "status": "success",
        "student_id": student_id,
        "roll_number": roll_number,
        "sample_index": sample_index,
        "auto_trained": auto_trained,
        "file": filename
    })


@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def api_delete_student(student_id):
    """Deletes student directory and removes entry from students.json."""
    folder = os.path.join(DATASET_DIR, str(student_id))
    if os.path.exists(folder):
        shutil.rmtree(folder)

    registry = load_students_registry()
    if str(student_id) in registry:
        del registry[str(student_id)]
        save_students_registry(registry)

        # Retrain remaining dataset so deleted face is no longer recognized
        try:
            from train import prepare_training_data
            faces, labels, _ = prepare_training_data(DATASET_DIR)
            if faces and labels:
                rec = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
                rec.train(faces, np.array(labels, dtype=np.int32))
                rec.write(TRAINER_FILE)
                camera_mgr.reload_recognizer()
            else:
                if os.path.exists(TRAINER_FILE):
                    os.remove(TRAINER_FILE)
                camera_mgr.recognizer = None
        except Exception:
            pass

        return jsonify({"status": "success", "message": f"Student {student_id} removed"})
    return jsonify({"status": "error", "message": "Student not found"}), 404


@app.route("/api/system/reset_all", methods=["POST"])
def api_system_reset_all():
    """Wipes all dataset images, LBPH model, registered students, and attendance logs."""
    try:
        # 1. Clean dataset folder
        if os.path.exists(DATASET_DIR):
            for item in os.listdir(DATASET_DIR):
                p = os.path.join(DATASET_DIR, item)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
        else:
            os.makedirs(DATASET_DIR, exist_ok=True)

        # 2. Reset students.json
        save_students_registry({})

        # 3. Delete face-trainer.yml
        if os.path.exists(TRAINER_FILE):
            os.remove(TRAINER_FILE)

        # 4. Clear model in camera manager
        camera_mgr.recognizer = None

        # 5. Clear attendance records in SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance;")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='attendance';")
        conn.commit()
        conn.close()

        print("[RESET ALL] All kiosk face data and attendance logs have been cleanly wiped.")
        return jsonify({
            "status": "success",
            "message": "All face datasets, LBPH model, registered students, and attendance logs have been completely reset."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model/train", methods=["POST"])
def api_model_train():
    """Retrains the LBPH model and reloads recognizer into camera manager."""
    try:
        from train import prepare_training_data
        t0 = time.time()
        faces, labels, stats = prepare_training_data(DATASET_DIR)
        
        if not faces or not labels:
            return jsonify({"status": "error", "message": "No training samples found in dataset/"}), 400

        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        recognizer.train(faces, np.array(labels, dtype=np.int32))
        recognizer.write(TRAINER_FILE)

        camera_mgr.reload_recognizer()

        duration = round(time.time() - t0, 2)
        return jsonify({
            "status": "success",
            "duration": duration,
            "students_count": len(stats),
            "samples_count": len(faces)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------------------------------------------------------------
# Server Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  TOUCHLESS SMART ATTENDANCE KIOSK - FLASK SERVER (WHITE THEME) ")
    print("=" * 70)
    print("  • Web Dashboard & Video Stream : http://0.0.0.0:5000")
    print("  • Background Auto-Scan Engine  : ACTIVE (cv2.VideoCapture(0))")
    print("  • Cloud Synchronization        : ACTIVE (Every 15s)")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
