"""
====================================================================================================
TOUCHLESS SMART ATTENDANCE KIOSK - AUTOSCAN TEST ENGINE (kiosk_autoscan_test.py)
====================================================================================================
100% Hands-Free & Buttonless Attendance Pipeline for Bare Raspberry Pi / Laptop Testing.
Continuously scans webcam frames, recognizes registered students via OpenCV LBPH,
automatically logs attendance upon positive match, strictly enforces a 10-second anti-spam
cooldown, and provides large, high-visibility color-coded ANSI terminal feedback alongside
the live video window (with headless SSH fallback).

HARDWARE TARGET:
  - Bare Raspberry Pi 3B+ / Laptop
  - Standard USB or Built-in Webcam (cv2.VideoCapture(0))
  - No physical GPIO buttons, no OLED screen, no buzzers required
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

# In LBPH, confidence measures Chi-Square distance (0 = identical match).
# Values strictly below this threshold trigger an automatic attendance log.
LBPH_CONFIDENCE_THRESHOLD = 75.0

# Anti-spam cooldown per student in seconds
COOLDOWN_SECONDS = 10.0

# Cloud Synchronization polling interval in seconds
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

# -----------------------------------------------------------------------------
# ANSI Color Codes for Terminal Readouts
# -----------------------------------------------------------------------------
CLR_RESET   = "\033[0m"
CLR_BOLD    = "\033[1m"
CLR_DIM     = "\033[2m"
CLR_GREEN   = "\033[92m"
CLR_RED     = "\033[91m"
CLR_YELLOW  = "\033[93m"
CLR_BLUE    = "\033[94m"
CLR_MAGENTA = "\033[95m"
CLR_CYAN    = "\033[96m"
CLR_WHITE   = "\033[97m"
BG_GREEN    = "\033[42m"
BG_BLUE     = "\033[44m"
BG_RED      = "\033[41m"

# UI Theme Color Tokens for OpenCV Window (BGR)
UI_BG_HEADER = (20, 20, 24)
UI_BG_BANNER = (30, 30, 36)
UI_ACCENT = (255, 191, 0)
UI_TEXT_WHITE = (250, 250, 250)
UI_TEXT_MUTED = (160, 160, 165)
UI_SUCCESS = (76, 217, 100)
UI_DANGER = (50, 50, 235)
UI_WARNING = (0, 165, 255)
UI_INFO = (235, 180, 50)


# -----------------------------------------------------------------------------
# Haar Cascade XML Path Resolver
# -----------------------------------------------------------------------------
def get_cascade_path() -> str:
    """Resolves Haar Cascade XML from local project directory first, then OpenCV data."""
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        return local_path
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        opencv_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(opencv_path):
            return opencv_path
    return "haarcascade_frontalface_default.xml"


# -----------------------------------------------------------------------------
# Local SQLite Database Engine (WAL Mode, < 5ms latency)
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
                         year: str, branch: str, date_str: str, in_time_str: str, timestamp_str: str) -> tuple[bool, float]:
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
        return True, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"{CLR_RED}[DB ERROR] Failed to record attendance: {e}{CLR_RESET}")
        return False, elapsed_ms
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
    synced=1 on HTTP 200. Logs color-coded ANSI status updates to the terminal.
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
        print(f"{CLR_CYAN}[CLOUD SYNC] Daemon thread active. Interval: {self.interval_sec}s{CLR_RESET}")
        while self.running:
            try:
                self._sync_pending_records()
            except Exception as e:
                self.last_sync_status = f"Err: {str(e)[:20]}"
            
            # Responsive sleep loop
            for _ in range(int(self.interval_sec)):
                if not self.running:
                    break
                time.sleep(1.0)

    def _sync_pending_records(self):
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

            t0 = time.perf_counter()
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=25.0
            )
            roundtrip_ms = (time.perf_counter() - t0) * 1000.0

            if response.status_code == 200:
                placeholders = ",".join(["?"] * len(row_ids))
                cursor.execute(f"UPDATE attendance SET synced = 1 WHERE id IN ({placeholders});", row_ids)
                conn.commit()
                self.last_sync_time = datetime.now().strftime("%H:%M:%S")
                self.last_sync_status = f"Synced {len(records)} rows ({self.last_sync_time})"

                # Prominent ANSI Cloud Sync Banner in Terminal
                print(f"\n{CLR_BOLD}{CLR_CYAN}╔══════════════════════════════════════════════════════════════════════════════╗")
                print(f"║ ☁  GOOGLE SHEETS CLOUD SYNC SUCCESS                                          ║")
                print(f"╠══════════════════════════════════════════════════════════════════════════════╣{CLR_RESET}")
                print(f"{CLR_CYAN}║  • Records Synced   : {CLR_BOLD}{len(records)}{CLR_RESET}{CLR_CYAN} row(s) pushed to Google Sheet")
                print(f"{CLR_CYAN}║  • HTTP Status      : {CLR_GREEN}{response.status_code} OK{CLR_RESET}{CLR_CYAN} (Roundtrip: {roundtrip_ms:.1f}ms)")
                print(f"{CLR_CYAN}║  • Local DB Status  : Updated synced = 1 for IDs: {row_ids}")
                print(f"{CLR_CYAN}║  • Sync Timestamp   : {self.last_sync_time}")
                print(f"{CLR_BOLD}{CLR_CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{CLR_RESET}\n")
            else:
                self.last_sync_status = f"HTTP {response.status_code}"
                print(f"\n{CLR_YELLOW}[CLOUD SYNC WARN] Google Sheets Webhook returned HTTP {response.status_code}: {response.text[:80]}{CLR_RESET}\n")

        except requests.exceptions.RequestException as net_err:
            self.last_sync_status = "Offline (Retrying)"
            # Silent resilience, no console spam on offline network
        finally:
            conn.close()


# -----------------------------------------------------------------------------
# Anti-Spam Cooldown & Console Throttling Manager
# -----------------------------------------------------------------------------
class AntiSpamManager:
    """
    Manages in-memory cooldown timestamps per student ID to prevent continuous
    logging when a student remains in front of the camera.
    Also throttles console printouts so the terminal remains clean.
    """
    def __init__(self, cooldown_seconds: float = 10.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_logged: dict[int, float] = {}
        self.last_console_warn: dict[int, float] = {}
        self.last_unknown_warn: float = 0.0

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

    def should_print_cooldown_warn(self, student_id: int, interval_sec: float = 2.5) -> bool:
        """Throttles repeated cooldown messages in terminal."""
        now = time.time()
        last = self.last_console_warn.get(student_id, 0.0)
        if now - last >= interval_sec:
            self.last_console_warn[student_id] = now
            return True
        return False

    def should_print_unknown_warn(self, interval_sec: float = 2.5) -> bool:
        """Throttles repeated unrecognized face messages in terminal."""
        now = time.time()
        if now - self.last_unknown_warn >= interval_sec:
            self.last_unknown_warn = now
            return True
        return False


# -----------------------------------------------------------------------------
# Face Recognition & Detection Pipeline
# -----------------------------------------------------------------------------
class FacePipeline:
    """Encapsulates Haar Cascade face detector and LBPH recognizer."""
    def __init__(self, trainer_path: str, students_path: str):
        self.trainer_path = trainer_path
        self.students_path = students_path
        self.students_registry = self._load_registry()
        
        # Load Haar Cascade
        cascade_file = get_cascade_path()
        if not os.path.exists(cascade_file):
            raise FileNotFoundError(f"Haar Cascade XML not found at: {cascade_file}")
            
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
        print(f"{CLR_GREEN}[INFO] LBPH model successfully loaded from '{self.trainer_path}'.{CLR_RESET}")

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
        # Return the largest face detected
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        return faces[0]

    def recognize_crop(self, gray_frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, str, float]:
        """
        Takes face bounding box, normalizes with histogram equalization,
        and runs LBPH prediction. Returns (student_id, student_name, confidence_distance).
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
    """Renders GUI HUD banners onto the OpenCV frame for connected monitors."""
    def __init__(self):
        self.banner_state = "IDLE"
        self.banner_title = ""
        self.banner_subtitle = ""
        self.banner_expiry = 0.0
        self.banner_color = UI_BG_BANNER

    def set_banner(self, state: str, title: str, subtitle: str, color: tuple, duration: float = 2.5):
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
        cv2.putText(frame, "TOUCHLESS ATTENDANCE - AUTOSCAN MODE", (16, 22), cv2.FONT_HERSHEY_DUPLEX, 0.55, UI_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Continuous Scanning Active [Hands-Free]", (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, UI_TEXT_MUTED, 1, cv2.LINE_AA)

        # Cloud Sync Pill
        sync_txt = f"Cloud: {sync_status}"
        if pending_count > 0:
            sync_txt += f" ({pending_count} pending)"
        sync_color = UI_SUCCESS if "Synced" in sync_status or "Idle" in sync_status else UI_WARNING
        cv2.putText(frame, sync_txt, (w - 290, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sync_color, 1, cv2.LINE_AA)

        # 2. Bottom Instruction Bar
        cv2.rectangle(frame, (0, h - 38), (w, h), UI_BG_HEADER, -1)
        cv2.putText(frame, "LOOK AT CAMERA - ATTENDANCE LOGS AUTOMATICALLY", (16, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.44, UI_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "[Q] Exit", (w - 75, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, UI_TEXT_MUTED, 1, cv2.LINE_AA)

        # 3. Face Bounding Box
        if face_box is not None:
            x, y, fw, fh = face_box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), UI_ACCENT, 2)
            cv2.putText(frame, "SCANNING...", (x, max(y - 8, 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_ACCENT, 1, cv2.LINE_AA)

        # 4. Status Notification Banner
        if now < self.banner_expiry:
            banner_h = 74
            banner_y = h - 38 - banner_h
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (12, banner_y), (w - 12, banner_y + banner_h), self.banner_color, -1)
            cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
            
            cv2.rectangle(frame, (12, banner_y), (20, banner_y + banner_h), UI_TEXT_WHITE, -1)
            cv2.rectangle(frame, (12, banner_y), (w - 12, banner_y + banner_h), (255, 255, 255), 1)

            cv2.putText(frame, self.banner_title, (32, banner_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, UI_TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, self.banner_subtitle, (32, banner_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_TEXT_WHITE, 1, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# Terminal ANSI Print Helpers
# -----------------------------------------------------------------------------
def print_startup_banner():
    """Prints system startup banner with configuration details."""
    print(f"\n{CLR_BOLD}{CLR_CYAN}╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║       TOUCHLESS SMART ATTENDANCE KIOSK - AUTOSCAN TEST MODE                  ║")
    print(f"╠══════════════════════════════════════════════════════════════════════════════╣{CLR_RESET}")
    print(f"{CLR_WHITE}║  • Mode             : {CLR_GREEN}{CLR_BOLD}100% Hands-Free / Buttonless Auto-Scan{CLR_RESET}")
    print(f"{CLR_WHITE}║  • Video Pipeline   : {CLR_CYAN}cv2.VideoCapture(0) (640x480 resolution){CLR_RESET}")
    print(f"{CLR_WHITE}║  • Recognizer       : {CLR_CYAN}OpenCV LBPH (Threshold: distance < {LBPH_CONFIDENCE_THRESHOLD}){CLR_RESET}")
    print(f"{CLR_WHITE}║  • Anti-Spam Guard  : {CLR_YELLOW}{COOLDOWN_SECONDS:.0f} seconds cooldown per student ID{CLR_RESET}")
    print(f"{CLR_WHITE}║  • Database Engine  : {CLR_MAGENTA}Local SQLite ({DB_PATH}) via WAL mode (< 5ms){CLR_RESET}")
    print(f"{CLR_WHITE}║  • Cloud Webhook    : {CLR_CYAN}Google Sheets background thread (every {SYNC_INTERVAL_SECONDS:.0f}s){CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{CLR_RESET}\n")
    print(f"{CLR_GREEN}{CLR_BOLD}>>> SYSTEM READY. Facing camera triggers instant auto-logging... <<<{CLR_RESET}\n")


def print_attendance_logged(student_id: int, roll_number: str, student_name: str, 
                            year: str, branch: str, in_time: str, distance: float, latency_ms: float, timestamp: str):
    """Large, vibrant ANSI banner printed upon successful attendance recording."""
    match_pct = max(0.0, min(100.0, (1.0 - (distance / LBPH_CONFIDENCE_THRESHOLD)) * 100.0))
    print(f"\n{CLR_BOLD}{CLR_GREEN}╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  ✔  ATTENDANCE AUTOMATICALLY RECORDED                                        ║")
    print(f"╠══════════════════════════════════════════════════════════════════════════════╣{CLR_RESET}")
    print(f"{CLR_GREEN}║  • Roll Number    : {CLR_BOLD}{CLR_WHITE}{roll_number}{CLR_RESET}")
    print(f"{CLR_GREEN}║  • Student Name   : {CLR_BOLD}{CLR_WHITE}{student_name}{CLR_RESET}")
    print(f"{CLR_GREEN}║  • Year / Branch  : {CLR_WHITE}{year} | {branch}{CLR_RESET}")
    print(f"{CLR_GREEN}║  • In-Time        : {CLR_BOLD}{CLR_CYAN}{in_time}{CLR_RESET} ({timestamp.split(' ')[0]})")
    print(f"{CLR_GREEN}║  • LBPH Distance  : {CLR_WHITE}{distance:.1f}{CLR_RESET} (Match Quality: {match_pct:.0f}%)")
    print(f"{CLR_GREEN}║  • SQLite Latency : {CLR_WHITE}{latency_ms:.2f} ms{CLR_RESET} (< 5ms WAL transaction)")
    print(f"{CLR_GREEN}║  • Cloud Status   : {CLR_CYAN}Queued for Google Sheets sync{CLR_RESET}")
    print(f"{CLR_GREEN}║  • Anti-Spam      : {CLR_YELLOW}{COOLDOWN_SECONDS:.0f}s cooldown locked for {roll_number}{CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_GREEN}╚══════════════════════════════════════════════════════════════════════════════╝{CLR_RESET}\n")


# -----------------------------------------------------------------------------
# Main Auto-Scan Execution Loop
# -----------------------------------------------------------------------------
def main():
    # 1. Initialize local SQLite database
    init_database()

    # 2. Print Startup Banner
    print_startup_banner()

    # 3. Load Computer Vision Pipeline
    try:
        pipeline = FacePipeline(TRAINER_FILE, STUDENTS_FILE)
    except Exception as e:
        print(f"\n{CLR_RED}{CLR_BOLD}[FATAL ERROR] {e}{CLR_RESET}\n")
        sys.exit(1)

    # 4. Start Anti-Spam Manager & Cloud Sync Worker
    anti_spam = AntiSpamManager(cooldown_seconds=COOLDOWN_SECONDS)
    sync_worker = CloudSyncWorker(webhook_url=GOOGLE_SHEETS_WEBHOOK_URL, interval_sec=SYNC_INTERVAL_SECONDS)
    sync_worker.start()

    hud = VisualHUD()

    # 5. Initialize Camera Stream
    print(f"{CLR_CYAN}[CAMERA] Initializing cv2.VideoCapture(0) at {FRAME_WIDTH}x{FRAME_HEIGHT}...{CLR_RESET}")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"\n{CLR_RED}[FATAL ERROR] Unable to open camera. Check USB connection or permissions.{CLR_RESET}\n")
        sync_worker.running = False
        sys.exit(1)

    # Detect if display / GUI environment is available (handles headless SSH gracefully)
    gui_available = True
    try:
        cv2.namedWindow("Touchless Smart Attendance Kiosk - Auto-Scan Test", cv2.WINDOW_AUTOSIZE)
    except Exception:
        gui_available = False
        print(f"{CLR_YELLOW}[INFO] Running in headless mode (no X11 / GUI display detected). Terminal output active.{CLR_RESET}")

    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_counter += 1
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy() if gui_available else None
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Continuous Face Detection
            face_box = pipeline.detect_face(gray_frame)

            if face_box is not None:
                # Run LBPH Face Recognition automatically
                student_id, student_name, confidence = pipeline.recognize_crop(gray_frame, face_box)

                now_dt = datetime.now()
                now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

                if confidence <= LBPH_CONFIDENCE_THRESHOLD:
                    # Positive match identified
                    is_cooldown, remaining_sec = anti_spam.is_cooling_down(student_id)

                    if is_cooldown:
                        # Throttled console printout while cooling down
                        if anti_spam.should_print_cooldown_warn(student_id, interval_sec=3.0):
                            print(f"{CLR_BLUE}[COOLDOWN] {student_name} (ID: {student_id}) already logged. Cooldown: {remaining_sec:.1f}s remaining.{CLR_RESET}")
                        
                        hud.set_banner(
                            state="COOLDOWN",
                            title=f"ALREADY LOGGED: {student_name}",
                            subtitle=f"Cooldown active ({int(remaining_sec)}s remaining). Please proceed.",
                            color=UI_INFO,
                            duration=2.0
                        )
                    else:
                        # --- AUTOMATIC ATTENDANCE LOGGING ---
                        student_info = pipeline.students_registry.get(str(student_id), {})
                        roll_number = student_info.get("roll_number", str(student_id))
                        year = student_info.get("year", "N/A")
                        branch = student_info.get("branch", "N/A")

                        date_str = now_dt.strftime("%Y-%m-%d")
                        in_time_str = now_dt.strftime("%H:%M:%S")

                        success, latency_ms = log_attendance_local(student_id, roll_number, student_name, 
                                                                   year, branch, date_str, in_time_str, now_str)
                        anti_spam.register_log(student_id)

                        # Print large color-coded ANSI terminal banner
                        print_attendance_logged(student_id, roll_number, student_name, year, branch, in_time_str, confidence, latency_ms, now_str)

                        # Set GUI HUD Banner
                        hud.set_banner(
                            state="SUCCESS",
                            title=f"CONFIRMED: {student_name}",
                            subtitle=f"Roll: {roll_number} | In-Time: {in_time_str} | {branch}",
                            color=UI_SUCCESS,
                            duration=3.5
                        )
                else:
                    # Unrecognized face detected
                    if anti_spam.should_print_unknown_warn(interval_sec=3.0):
                        print(f"{CLR_RED}[UNRECOGNIZED FACE] Distance: {confidence:.1f} (Threshold: {LBPH_CONFIDENCE_THRESHOLD}). Registration required.{CLR_RESET}")
                    
                    hud.set_banner(
                        state="UNRECOGNIZED",
                        title="UNRECOGNIZED FACE",
                        subtitle=f"Distance score: {confidence:.1f}. Please enroll first.",
                        color=UI_DANGER,
                        duration=2.0
                    )

            # Render GUI window if display is available
            if gui_available:
                pending_count, _ = get_sync_stats()
                hud.draw(display_frame, sync_worker.last_sync_status, pending_count, face_box)
                cv2.imshow("Touchless Smart Attendance Kiosk - Auto-Scan Test", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print(f"\n{CLR_YELLOW}[INFO] Exit key pressed. Shutting down auto-scan kiosk...{CLR_RESET}")
                    break

    except KeyboardInterrupt:
        print(f"\n{CLR_YELLOW}[INFO] KeyboardInterrupt received. Shutting down...{CLR_RESET}")
    finally:
        sync_worker.running = False
        cap.release()
        if gui_available:
            cv2.destroyAllWindows()
        print(f"{CLR_GREEN}[INFO] Kiosk terminated cleanly.{CLR_RESET}\n")


if __name__ == "__main__":
    main()
