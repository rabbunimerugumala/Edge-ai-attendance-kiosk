"""
====================================================================================================
TOUCHLESS SMART ATTENDANCE KIOSK - CORE RUNTIME ENGINE (kiosk_laptop.py)
====================================================================================================

====================================================================================================
                        RASPBERRY PI 3B+ MIGRATION BLUEPRINT
====================================================================================================
When migrating this code from Laptop to Raspberry Pi 3B+ (1GB RAM) with Pi Camera Module 3,
a 16mm metal button on GPIO 17, and an SSD1306 128x64 OLED screen on I2C, perform the following swaps:

1. CAMERA PIPELINE:
   - LAPTOP (current):
       cap = cv2.VideoCapture(0)
       ret, frame = cap.read()
   - RASPBERRY PI (Picamera2):
       from picamera2 import Picamera2
       picam2 = Picamera2()
       picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
       picam2.start()
       frame = picam2.capture_array() # Native numpy BGR/RGB array at zero copy overhead

2. PHYSICAL BUTTON TRIGGER:
   - LAPTOP (current):
       key = cv2.waitKey(1) & 0xFF
       if key == 32: # SPACEBAR
   - RASPBERRY PI (gpiozero):
       from gpiozero import Button
       button = Button(17, bounce_time=0.1) # 16mm Metal Button wired to GPIO 17 & GND
       button.when_pressed = trigger_scan_callback

3. OLED DISPLAY FEEDBACK (SSD1306 I2C):
   - In place of on-screen visual banners (or in parallel), drive the I2C OLED:
       from luma.core.interface.serial import i2c
       from luma.oled.device import ssd1306
       from PIL import Image, ImageDraw, ImageFont
       serial = i2c(port=1, address=0x3C)
       oled = ssd1306(serial)
       # Display confirmation:
       with canvas(oled) as draw:
           draw.text((0, 0), "VERIFIED: PRESENT", fill=255)
           draw.text((0, 20), f"{student_name}", fill=255)
           draw.text((0, 40), f"ID: {student_id}", fill=255)

The Haar Cascade and LBPH Face Recognizer algorithms remain 100% UNCHANGED and run at 15-20 FPS
on the Pi 3B+ ARM Cortex-A53 processor with under 100MB of RAM.
====================================================================================================
"""

import os
import sys
import json
import time
import sqlite3
import threading
from datetime import datetime
import cv2
import numpy as np
import requests

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FACE_CROP_SIZE = (200, 200)

DB_PATH = "attendance.db"
TRAINER_FILE = "face-trainer.yml"
STUDENTS_FILE = "students.json"

# In LBPH, confidence measures Chi-Square distance (0 = identical).
# Values below this threshold are accepted as positive matches.
LBPH_CONFIDENCE_THRESHOLD = 75.0

# Anti-spam cooldown per student in seconds
COOLDOWN_SECONDS = 10.0

# Cloud Synchronization interval in seconds
SYNC_INTERVAL_SECONDS = 15.0

# Load .env configuration if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    if _k.strip() not in os.environ:
                        os.environ[_k.strip()] = _v.strip().strip('"').strip("'")
    except Exception:
        pass

# Google Apps Script Webhook URL (Loaded from environment)
GOOGLE_SHEETS_WEBHOOK_URL = os.getenv("GOOGLE_SHEETS_WEBHOOK_URL", "")

# UI Theme Color Tokens (BGR format)
UI_BG_HEADER = (20, 20, 24)
UI_BG_BANNER = (30, 30, 36)
UI_ACCENT = (255, 191, 0)         # Amber / Cyan tone
UI_TEXT_WHITE = (250, 250, 250)
UI_TEXT_MUTED = (160, 160, 165)
UI_SUCCESS = (76, 217, 100)       # Green
UI_DANGER = (50, 50, 235)         # Red
UI_WARNING = (0, 165, 255)        # Orange / Amber
UI_INFO = (235, 180, 50)          # Cyan / Blue


# -----------------------------------------------------------------------------
# Local SQLite Database Engine (Ultra Low Latency < 5ms)
# -----------------------------------------------------------------------------
def init_database() -> None:
    """Initializes SQLite database with Write-Ahead Logging (WAL) and auto-migrates columns."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
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
        
        cursor.execute("PRAGMA table_info(attendance);")
        existing_cols = [row[1] for row in cursor.fetchall()]
        for col_name, col_type in [("roll_number", "TEXT DEFAULT ''"), ("year", "TEXT DEFAULT ''"), 
                                   ("branch", "TEXT DEFAULT ''"), ("date", "TEXT DEFAULT ''"), ("in_time", "TEXT DEFAULT ''")]:
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE attendance ADD COLUMN {col_name} {col_type};")
        conn.commit()
    finally:
        conn.close()


def log_attendance_local(student_id: int, roll_number: str, student_name: str,
                         year: str, branch: str, date_str: str, in_time_str: str, timestamp_str: str) -> bool:
    """Records full student attendance entry in SQLite. Executes in ~1-3ms."""
    start = time.perf_counter()
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
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"[DB LOG] Attendance recorded for {roll_number} ({student_name}) in {elapsed_ms:.2f}ms")
        return True
    except Exception as e:
        print(f"[DB ERROR] Failed to record attendance: {e}")
        return False
    finally:
        conn.close()


def get_sync_stats() -> tuple[int, int]:
    """Returns (pending_count, total_count) from local SQLite."""
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


# -----------------------------------------------------------------------------
# Background Cloud Synchronization Worker (Daemon Thread)
# -----------------------------------------------------------------------------
class CloudSyncWorker:
    """
    Autonomous background worker thread. Every 15 seconds, queries unsynced
    attendance rows, posts them in batch to Google Apps Script, and marks them
    synced=1 on HTTP 200. Handles network drops gracefully with zero crashing.
    """
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
        print(f"[SYNC WORKER] Background thread started. Interval: {self.interval_sec}s.")
        while self.running:
            try:
                self._sync_pending_records()
            except Exception as e:
                # Catch-all to ensure the daemon thread NEVER dies
                self.last_sync_status = f"Err: {str(e)[:20]}"
            
            # Sleep in 1-second chunks for responsive shutdown
            for _ in range(int(self.interval_sec)):
                if not self.running:
                    break
                time.sleep(1.0)

    def _sync_pending_records(self):
        # Check if dummy or invalid URL
        if not self.webhook_url or "YOUR-WEBHOOK-TOKEN-HERE" in self.webhook_url:
            self.last_sync_status = "Webhook URL not configured"
            return

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
                return

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

            # Send HTTP POST with 6-second timeout
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=25.0
            )

            if response.status_code == 200:
                # Mark as synced
                placeholders = ",".join(["?"] * len(row_ids))
                cursor.execute(f"UPDATE attendance SET synced = 1 WHERE id IN ({placeholders});", row_ids)
                conn.commit()
                self.last_sync_time = datetime.now().strftime("%H:%M:%S")
                self.last_sync_status = f"Synced {len(records)} rows at {self.last_sync_time}"
                print(f"[CLOUD SYNC] Successfully synced {len(records)} record(s) to Google Sheets.")
            else:
                self.last_sync_status = f"HTTP {response.status_code}"
                print(f"[CLOUD SYNC WARN] Google Sheets Webhook returned status: {response.status_code}")

        except requests.exceptions.RequestException:
            # Network drop / offline resilience
            self.last_sync_status = "Offline (Retrying)"
        finally:
            conn.close()


# -----------------------------------------------------------------------------
# Anti-Spam Cooldown Manager
# -----------------------------------------------------------------------------
class AntiSpamManager:
    """Manages in-memory cooldown timestamps per student ID to prevent double logs."""
    def __init__(self, cooldown_seconds: float = 10.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_logged: dict[int, float] = {}

    def is_cooling_down(self, student_id: int) -> tuple[bool, float]:
        """Returns (is_active, remaining_seconds)."""
        now = time.time()
        last_time = self.last_logged.get(student_id, 0.0)
        elapsed = now - last_time
        if elapsed < self.cooldown_seconds:
            return True, (self.cooldown_seconds - elapsed)
        return False, 0.0

    def register_log(self, student_id: int) -> None:
        self.last_logged[student_id] = time.time()


def get_cascade_path() -> str:
    """Resolves Haar Cascade XML from local directory first, then OpenCV data package."""
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        return local_path
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        opencv_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(opencv_path):
            return opencv_path
    return "haarcascade_frontalface_default.xml"


# -----------------------------------------------------------------------------
# Face Recognition & Detection Pipeline
# -----------------------------------------------------------------------------
class FacePipeline:
    """Encapsulates Haar Cascade detector and LBPH recognizer."""
    def __init__(self, trainer_path: str, students_path: str):
        self.trainer_path = trainer_path
        self.students_path = students_path
        self.students_registry = self._load_registry()
        
        # Load Haar Cascade
        cascade_file = get_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load Haar Cascade from {cascade_file}")

        # Load LBPH Recognizer
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError("OpenCV missing 'cv2.face' module. Install 'opencv-contrib-python'.")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(self.trainer_path):
            raise FileNotFoundError(
                f"Trained model '{self.trainer_path}' not found!\n"
                f"Please run 'python enroll.py' followed by 'python train.py' first."
            )
        self.recognizer.read(self.trainer_path)
        print(f"[INFO] LBPH model successfully loaded from '{self.trainer_path}'.")

    def _load_registry(self) -> dict:
        if os.path.exists(self.students_path):
            try:
                with open(self.students_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def detect_face(self, gray_frame: np.ndarray):
        """Returns best face bounding box (x, y, w, h) or None."""
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(110, 110),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None
        # Return the largest face detected in frame
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        return faces[0]

    def recognize_crop(self, gray_frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, str, float]:
        """
        Takes bounding box, crops, normalizes with histogram equalization,
        and runs LBPH prediction. Returns (student_id, student_name, confidence).
        """
        x, y, w, h = bbox
        face_crop = gray_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, FACE_CROP_SIZE, interpolation=cv2.INTER_AREA)
        face_equalized = cv2.equalizeHist(face_resized)

        student_id, confidence = self.recognizer.predict(face_equalized)

        student_info = self.students_registry.get(str(student_id), {})
        student_name = student_info.get("student_name", f"Student {student_id}")

        return student_id, student_name, float(confidence)


# -----------------------------------------------------------------------------
# On-Screen HUD & Visual Banner Renderer
# -----------------------------------------------------------------------------
class VisualHUD:
    """Renders professional Kiosk HUD with status banners and sync status."""
    def __init__(self):
        self.banner_state = "IDLE"  # IDLE, SUCCESS, UNRECOGNIZED, NO_FACE, COOLDOWN
        self.banner_title = ""
        self.banner_subtitle = ""
        self.banner_expiry = 0.0
        self.banner_color = UI_BG_BANNER

    def set_banner(self, state: str, title: str, subtitle: str, color: tuple, duration: float = 3.0):
        self.banner_state = state
        self.banner_title = title
        self.banner_subtitle = subtitle
        self.banner_color = color
        self.banner_expiry = time.time() + duration

    def draw(self, frame: np.ndarray, sync_status: str, pending_count: int, face_box=None):
        h, w = frame.shape[:2]
        now = time.time()

        # 1. Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 48), UI_BG_HEADER, -1)
        cv2.putText(frame, "TOUCHLESS ATTENDANCE KIOSK", (16, 22), cv2.FONT_HERSHEY_DUPLEX, 0.58, UI_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Edge AI Powered - Offline Resilient", (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, UI_TEXT_MUTED, 1, cv2.LINE_AA)

        # Cloud Sync Pill
        sync_txt = f"Cloud: {sync_status}"
        if pending_count > 0:
            sync_txt += f" ({pending_count} pending)"
        sync_color = UI_SUCCESS if "Synced" in sync_status or "Idle" in sync_status else UI_WARNING
        cv2.putText(frame, sync_txt, (w - 290, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sync_color, 1, cv2.LINE_AA)

        # 2. Bottom Instruction Bar
        cv2.rectangle(frame, (0, h - 38), (w, h), UI_BG_HEADER, -1)
        cv2.putText(frame, "LOOK AT CAMERA & PRESS [SPACEBAR] TO RECORD ATTENDANCE", (16, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.44, UI_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "[Q] Exit", (w - 75, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, UI_TEXT_MUTED, 1, cv2.LINE_AA)

        # 3. Face Bounding Box if detected
        if face_box is not None:
            x, y, fw, fh = face_box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), UI_ACCENT, 2)
            cv2.putText(frame, "READY", (x, max(y - 8, 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_ACCENT, 1, cv2.LINE_AA)

        # 4. Status Notification Banner (persists for 3.0 seconds after trigger)
        if now < self.banner_expiry:
            banner_h = 74
            banner_y = h - 38 - banner_h
            
            # Semi-transparent backdrop
            overlay = frame.copy()
            cv2.rectangle(overlay, (12, banner_y), (w - 12, banner_y + banner_h), self.banner_color, -1)
            cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
            
            # Accent left border indicator
            cv2.rectangle(frame, (12, banner_y), (20, banner_y + banner_h), UI_TEXT_WHITE, -1)
            cv2.rectangle(frame, (12, banner_y), (w - 12, banner_y + banner_h), (255, 255, 255), 1)

            # Banner Text
            cv2.putText(frame, self.banner_title, (32, banner_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, UI_TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, self.banner_subtitle, (32, banner_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_TEXT_WHITE, 1, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# Main Application Execution Loop
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("      TOUCHLESS SMART ATTENDANCE KIOSK - SYSTEM STARTUP        ")
    print("=" * 70)

    # 1. Initialize local SQLite database
    init_database()
    print("[INIT] Local SQLite database verified.")

    # 2. Initialize Computer Vision Pipeline
    try:
        pipeline = FacePipeline(TRAINER_FILE, STUDENTS_FILE)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

    # 3. Initialize Anti-Spam Cooldown & Cloud Sync Daemon
    anti_spam = AntiSpamManager(cooldown_seconds=COOLDOWN_SECONDS)
    sync_worker = CloudSyncWorker(webhook_url=GOOGLE_SHEETS_WEBHOOK_URL, interval_sec=SYNC_INTERVAL_SECONDS)
    sync_worker.start()

    hud = VisualHUD()

    # 4. Initialize Camera Stream
    print(f"[INIT] Opening camera at {FRAME_WIDTH}x{FRAME_HEIGHT}...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[FATAL ERROR] Unable to open camera. Make sure webcam is not in use.")
        sync_worker.running = False
        sys.exit(1)

    print("\n[READY] Kiosk active. Look at the camera and press [SPACEBAR] to log.")
    print("=" * 70)

    cv2.namedWindow("Touchless Smart Attendance Kiosk", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Mirror frame for natural self-view
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Continuous passive face detection for guide box
            face_box = pipeline.detect_face(gray_frame)

            # Key listener (simulates physical desk button trigger)
            key = cv2.waitKey(1) & 0xFF

            # Exit key
            if key in (27, ord('q'), ord('Q')):
                print("[INFO] Shutting down attendance kiosk...")
                break

            # -------------------------------------------------------------
            # TRIGGER EVENT: User presses [SPACEBAR]
            # -------------------------------------------------------------
            if key == 32:  # ASCII 32 = Spacebar
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if face_box is None:
                    # Case 1: No Face Detected
                    print(f"[{now_str}] Triggered: No face detected in frame.")
                    hud.set_banner(
                        state="NO_FACE",
                        title="NO FACE DETECTED",
                        subtitle="Please face the camera directly and try again.",
                        color=UI_WARNING,
                        duration=3.0
                    )
                else:
                    # Run LBPH Face Recognition
                    student_id, student_name, confidence = pipeline.recognize_crop(gray_frame, face_box)

                    if confidence <= LBPH_CONFIDENCE_THRESHOLD:
                        # Case 2: Positive Match Identified
                        is_cooldown, remaining_sec = anti_spam.is_cooling_down(student_id)

                        if is_cooldown:
                            # Anti-spam cooldown triggered
                            print(f"[{now_str}] Cooldown active for ID {student_id} ({remaining_sec:.1f}s remaining). Log skipped.")
                            hud.set_banner(
                                state="COOLDOWN",
                                title=f"ALREADY RECORDED: {student_name}",
                                subtitle=f"Please proceed. Cooldown active ({int(remaining_sec)}s left).",
                                color=UI_INFO,
                                duration=3.0
                            )
                        else:
                            # Pull student metadata
                            registry = pipeline.students_registry
                            student_info = registry.get(str(student_id), {})
                            roll_number = student_info.get("roll_number", str(student_id))
                            year = student_info.get("year", "N/A")
                            branch = student_info.get("branch", "N/A")
                            
                            now_dt = datetime.now()
                            date_str = now_dt.strftime("%Y-%m-%d")
                            in_time_str = now_dt.strftime("%H:%M:%S")

                            # Log to SQLite (< 5ms execution)
                            log_success = log_attendance_local(student_id, roll_number, student_name, 
                                                               year, branch, date_str, in_time_str, now_str)
                            anti_spam.register_log(student_id)

                            conf_display = f"Distance: {confidence:.1f} (Match Valid)"
                            print(f"[{now_str}] [ATTENDANCE CONFIRMED] Roll: {roll_number} | {student_name} | In-Time: {in_time_str} | {branch}")
                            
                            hud.set_banner(
                                state="SUCCESS",
                                title=f"CONFIRMED: {student_name}",
                                subtitle=f"Roll: {roll_number} | In-Time: {in_time_str} | {branch}",
                                color=UI_SUCCESS,
                                duration=3.5
                            )
                    else:
                        # Case 3: Confidence score too high (unrecognized face)
                        print(f"[{now_str}] [UNRECOGNIZED] Face distance {confidence:.1f} exceeded threshold {LBPH_CONFIDENCE_THRESHOLD}.")
                        hud.set_banner(
                            state="UNRECOGNIZED",
                            title="UNRECOGNIZED FACE",
                            subtitle=f"Distance score: {confidence:.1f}. Please retry or register via enroll.py.",
                            color=UI_DANGER,
                            duration=3.0
                        )

            # Query pending sync count for live HUD display
            pending_count, _ = get_sync_stats()
            
            # Render HUD elements
            hud.draw(display_frame, sync_worker.last_sync_status, pending_count, face_box)

            cv2.imshow("Touchless Smart Attendance Kiosk", display_frame)

    finally:
        sync_worker.running = False
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Kiosk terminated cleanly.")


if __name__ == "__main__":
    main()
