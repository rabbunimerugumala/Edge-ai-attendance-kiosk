"""
=============================================================================
TOUCHLESS SMART ATTENDANCE KIOSK - STUDENT ENROLLMENT (enroll.py)
=============================================================================
Edge AI & Computer Vision pipeline for capturing standardized face datasets.
Prompts for Student ID (integer) and Name, guides user alignment via on-screen
reticle, normalizes face crops (grayscale, 200x200, histogram equalization),
and saves samples for LBPH training.

Hardware Target: Laptop Webcam (Dev) -> Raspberry Pi 3B+ (Production)
=============================================================================
"""

import os
import sys
import json
import time
from datetime import datetime
import cv2
import numpy as np

# Configuration Constants
DATASET_DIR = "dataset"
STUDENTS_FILE = "students.json"
TOTAL_SAMPLES = 20
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FACE_CROP_SIZE = (200, 200)
CAPTURE_DELAY_SEC = 0.25  # Minimum interval between captures to capture micro-variations

# Color Palette (BGR for OpenCV)
COLOR_BG_DARK = (24, 24, 27)        # Dark Charcoal
COLOR_ACCENT = (255, 191, 0)        # Electric Amber
COLOR_SUCCESS = (76, 217, 100)      # Vivid Green
COLOR_WARNING = (0, 165, 255)       # Orange
COLOR_TEXT = (255, 255, 255)        # Pure White
COLOR_MUTED = (160, 160, 160)       # Light Slate


def load_students_registry() -> dict:
    """Loads existing students registry from students.json or creates a new one."""
    if os.path.exists(STUDENTS_FILE):
        try:
            with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Corrupted {STUDENTS_FILE}. Initializing empty registry.")
            return {}
    return {}


def save_student_to_registry(student_id: int, student_name: str) -> None:
    """Updates students.json with the new student record."""
    registry = load_students_registry()
    registry[str(student_id)] = {
        "student_id": int(student_id),
        "student_name": student_name.strip(),
        "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"[INFO] Successfully updated {STUDENTS_FILE} for ID: {student_id} ({student_name})")


def get_student_input() -> tuple[int, str]:
    """Prompts terminal user for valid numeric student ID and name."""
    print("=" * 65)
    print("      TOUCHLESS SMART ATTENDANCE KIOSK - ENROLLMENT MODULE      ")
    print("=" * 65)
    
    # Validate numeric Student ID
    while True:
        raw_id = input("\n[>] Enter Numeric Student ID (e.g. 101, 102): ").strip()
        if not raw_id.isdigit():
            print("[ERROR] Student ID must contain positive digits only (LBPH label requirement).")
            continue
        student_id = int(raw_id)
        if student_id <= 0:
            print("[ERROR] Student ID must be greater than 0.")
            continue
        
        # Check if already enrolled
        registry = load_students_registry()
        if str(student_id) in registry:
            existing_name = registry[str(student_id)].get("student_name", "Unknown")
            override = input(f"[?] ID {student_id} is already registered to '{existing_name}'. Re-enroll? (y/n): ").strip().lower()
            if override != 'y':
                continue
        break

    # Validate Student Name
    while True:
        student_name = input("[>] Enter Student Full Name: ").strip()
        if not student_name:
            print("[ERROR] Student name cannot be empty.")
            continue
        break

    return student_id, student_name


def draw_hud(frame: np.ndarray, count: int, total: int, student_name: str, student_id: int, status_msg: str, is_capturing: bool) -> None:
    """Renders a sleek, high-visibility edge AI HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # Top Banner Header
    cv2.rectangle(frame, (0, 0), (w, 55), COLOR_BG_DARK, -1)
    cv2.putText(frame, "STUDENT ENROLLMENT KIOSK", (16, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Candidate: {student_name} [ID: {student_id}]", (16, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MUTED, 1, cv2.LINE_AA)

    # Progress Calculation
    progress_ratio = min(count / total, 1.0)
    bar_width = 180
    bar_height = 14
    bar_x = w - bar_width - 20
    bar_y = 22
    
    # Progress Bar Background and Fill
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    fill_w = int(bar_width * progress_ratio)
    bar_color = COLOR_SUCCESS if progress_ratio >= 1.0 else COLOR_ACCENT
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_height), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), COLOR_TEXT, 1)
    
    cv2.putText(frame, f"Captured: {count}/{total}", (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1, cv2.LINE_AA)

    # Bottom Instructions Banner
    cv2.rectangle(frame, (0, h - 45), (w, h), COLOR_BG_DARK, -1)
    cv2.putText(frame, f"Status: {status_msg}", (16, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_SUCCESS if is_capturing else COLOR_WARNING, 1, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' or [ESC] to Abort", (w - 210, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MUTED, 1, cv2.LINE_AA)


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


def run_enrollment() -> None:
    """Main enrollment routine."""
    student_id, student_name = get_student_input()

    # Create dataset target directory: dataset/<student_id>/
    student_dir = os.path.join(DATASET_DIR, str(student_id))
    os.makedirs(student_dir, exist_ok=True)

    # Load OpenCV Haar Cascade
    cascade_path = get_cascade_path()
    if not os.path.exists(cascade_path):
        print(f"[FATAL] Haar Cascade file not found at: {cascade_path}")
        sys.exit(1)
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[FATAL] Failed to load Haar Cascade XML classifier.")
        sys.exit(1)

    # Initialize Camera
    print(f"\n[INFO] Starting camera stream at {FRAME_WIDTH}x{FRAME_HEIGHT}...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[FATAL] Unable to open camera. Check if another application is using webcam.")
        sys.exit(1)

    print("[INFO] Camera initialized. Please face the camera inside the green guide box.")
    print("[INFO] Tilt head slightly during capture to generate high-variance training features.")

    sample_count = 0
    last_capture_time = 0.0
    status_msg = "Align face in the center frame"

    # Define target reticle box in the frame center
    box_w, box_h = 240, 240
    box_x = (FRAME_WIDTH - box_w) // 2
    box_y = (FRAME_HEIGHT - box_h) // 2 + 10

    cv2.namedWindow("Enrollment Kiosk - Touchless Attendance", cv2.WINDOW_AUTOSIZE)

    try:
        while sample_count < TOTAL_SAMPLES:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Dropped camera frame.")
                continue

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Draw central guide reticle (dashed/bracketed aesthetic)
            reticle_color = COLOR_MUTED
            is_face_aligned = False

            # Haar detection restricted to scaleFactor=1.2, minNeighbors=5 for high precision
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(120, 120),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            current_time = time.time()

            if len(faces) == 0:
                status_msg = "Searching for face... Keep head upright."
            elif len(faces) > 1:
                status_msg = "Multiple faces detected! Only 1 person permitted."
                reticle_color = COLOR_WARNING
            else:
                x, y, w, h = faces[0]
                
                # Check if face is reasonably centered in the target zone
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                in_target_x = box_x <= face_center_x <= (box_x + box_w)
                in_target_y = box_y <= face_center_y <= (box_y + box_h)

                if in_target_x and in_target_y:
                    is_face_aligned = True
                    reticle_color = COLOR_SUCCESS

                    # Visual bounding box over the detected face
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), COLOR_SUCCESS, 2)
                    
                    # Capture check with rate limiter
                    if (current_time - last_capture_time) >= CAPTURE_DELAY_SEC:
                        sample_count += 1
                        last_capture_time = current_time

                        # Standardized preprocessing:
                        # 1. Grayscale crop
                        face_crop = gray[y:y + h, x:x + w]
                        # 2. Resize to 200x200
                        face_resized = cv2.resize(face_crop, FACE_CROP_SIZE, interpolation=cv2.INTER_AREA)
                        # 3. Histogram Equalization (normalizes lighting across all environments)
                        face_equalized = cv2.equalizeHist(face_resized)

                        # Save normalized image
                        img_filename = os.path.join(student_dir, f"sample_{sample_count:02d}.jpg")
                        cv2.imwrite(img_filename, face_equalized)

                        status_msg = f"Captured {sample_count}/{TOTAL_SAMPLES} - Keep holding steady..."
                    else:
                        status_msg = f"Capturing: {sample_count}/{TOTAL_SAMPLES} (Hold steady)"
                else:
                    status_msg = "Move face closer to the center target box"
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), COLOR_WARNING, 2)

            # Draw target guide brackets
            # Top-Left corner
            cv2.line(display_frame, (box_x, box_y), (box_x + 30, box_y), reticle_color, 2)
            cv2.line(display_frame, (box_x, box_y), (box_x, box_y + 30), reticle_color, 2)
            # Top-Right corner
            cv2.line(display_frame, (box_x + box_w, box_y), (box_x + box_w - 30, box_y), reticle_color, 2)
            cv2.line(display_frame, (box_x + box_w, box_y), (box_x + box_w, box_y + 30), reticle_color, 2)
            # Bottom-Left corner
            cv2.line(display_frame, (box_x, box_y + box_h), (box_x + 30, box_y + box_h), reticle_color, 2)
            cv2.line(display_frame, (box_x, box_y + box_h), (box_x, box_y + box_h - 30), reticle_color, 2)
            # Bottom-Right corner
            cv2.line(display_frame, (box_x + box_w, box_y + box_h), (box_x + box_w - 30, box_y + box_h), reticle_color, 2)
            cv2.line(display_frame, (box_x + box_w, box_y + box_h), (box_x + box_w, box_y + box_h - 30), reticle_color, 2)

            # Render HUD
            draw_hud(display_frame, sample_count, TOTAL_SAMPLES, student_name, student_id, status_msg, is_face_aligned)

            cv2.imshow("Enrollment Kiosk - Touchless Attendance", display_frame)

            # Keyboard listener for graceful exit
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("\n[INFO] Enrollment cancelled by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if sample_count >= TOTAL_SAMPLES:
        save_student_to_registry(student_id, student_name)
        print("\n" + "=" * 65)
        print(f"[SUCCESS] Enrollment Completed for {student_name} (ID: {student_id})!")
        print(f"[SUCCESS] Saved {sample_count} samples to '{student_dir}/'")
        print("[NEXT STEP] Run 'python train.py' to train/update the LBPH model.")
        print("=" * 65 + "\n")
    else:
        print(f"\n[WARN] Incomplete enrollment ({sample_count}/{TOTAL_SAMPLES} samples captured).")
        print("Please rerun 'python enroll.py' to complete the dataset.\n")


if __name__ == "__main__":
    run_enrollment()
