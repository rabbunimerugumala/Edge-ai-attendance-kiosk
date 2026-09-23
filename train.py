"""
=============================================================================
TOUCHLESS SMART ATTENDANCE KIOSK - MODEL TRAINER (train.py)
=============================================================================
Trains the OpenCV Local Binary Patterns Histograms (LBPH) Face Recognizer
using standardized face crops from dataset/<student_id>/. Outputs the
compact, lightweight mathematical model to 'face-trainer.yml'.

Hardware Target: Lightweight offline CPU execution (< 50MB RAM during training).
=============================================================================
"""

import os
import sys
import json
import time
import cv2
import numpy as np

DATASET_DIR = "dataset"
STUDENTS_FILE = "students.json"
TRAINER_OUTPUT_FILE = "face-trainer.yml"


def load_students_registry() -> dict:
    """Loads student name mapping from students.json."""
    if os.path.exists(STUDENTS_FILE):
        try:
            with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def prepare_training_data(dataset_path: str):
    """
    Parses the dataset directory and prepares face samples and numeric labels.
    Directory structure:
        dataset/
          ├── 101/
          │    ├── sample_01.jpg
          │    └── ...
          └── 102/
               └── ...
    """
    faces = []
    labels = []
    student_stats = {}

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset directory '{dataset_path}' does not exist.")
        print("[HINT] Run 'python enroll.py' first to enroll students.")
        return None, None, None

    registry = load_students_registry()
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    if not subdirs:
        print(f"[ERROR] No student subdirectories found inside '{dataset_path}'.")
        print("[HINT] Run 'python enroll.py' to register at least one student.")
        return None, None, None

    print(f"\n[INFO] Scanning '{dataset_path}' for training samples...")

    for subdir in subdirs:
        if not subdir.isdigit():
            print(f"[WARN] Skipping folder '{subdir}': Folder name must be a numeric Student ID.")
            continue

        student_id = int(subdir)
        student_info = registry.get(str(student_id), {})
        student_name = student_info.get("student_name", f"Student-{student_id}")

        student_folder = os.path.join(dataset_path, subdir)
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [f for f in os.listdir(student_folder) if f.lower().endswith(valid_extensions)]

        if not image_files:
            print(f"[WARN] No valid face images found in '{student_folder}'. Skipping.")
            continue

        loaded_for_student = 0
        for img_name in image_files:
            img_path = os.path.join(student_folder, img_name)
            # Read in grayscale mode
            face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if face_img is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            # Standardize crop size (200x200) and apply histogram equalization
            if face_img.shape != (200, 200):
                face_img = cv2.resize(face_img, (200, 200), interpolation=cv2.INTER_AREA)
            face_img = cv2.equalizeHist(face_img)

            faces.append(face_img)
            labels.append(student_id)
            loaded_for_student += 1

        student_stats[student_id] = {
            "name": student_name,
            "sample_count": loaded_for_student
        }
        print(f"  --> Student ID {student_id:4d} | Name: {student_name:<20} | Samples: {loaded_for_student:2d}")

    return faces, labels, student_stats


def train_model() -> None:
    """Trains the LBPH face recognizer and serializes it to face-trainer.yml."""
    print("=" * 68)
    print("       TOUCHLESS SMART ATTENDANCE KIOSK - LBPH TRAINER          ")
    print("=" * 68)

    start_time = time.time()
    faces, labels, student_stats = prepare_training_data(DATASET_DIR)

    if not faces or not labels:
        print("[FATAL] Training aborted: No valid training data found.")
        sys.exit(1)

    total_samples = len(faces)
    total_students = len(student_stats)

    print(f"\n[INFO] Initializing LBPH Face Recognizer...")
    # Verify opencv-contrib-python module availability
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print("\n[FATAL] cv2.face module not found!")
        print("You must install 'opencv-contrib-python' instead of standard 'opencv-python':")
        print("    pip uninstall opencv-python")
        print("    pip install opencv-contrib-python\n")
        sys.exit(1)

    # LBPH Hyperparameters tuned for edge stability
    # radius=1, neighbors=8, grid_x=8, grid_y=8 (OpenCV standard)
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    print(f"[INFO] Training model on {total_samples} samples across {total_students} student(s)...")
    recognizer.train(faces, np.array(labels, dtype=np.int32))

    print(f"[INFO] Serializing trained model to '{TRAINER_OUTPUT_FILE}'...")
    recognizer.write(TRAINER_OUTPUT_FILE)

    duration = time.time() - start_time
    file_size_kb = os.path.getsize(TRAINER_OUTPUT_FILE) / 1024.0

    print("\n" + "=" * 68)
    print("                   TRAINING SUMMARY                             ")
    print("=" * 68)
    print(f"  • Total Students Registered : {total_students}")
    print(f"  • Total Face Crops Trained  : {total_samples}")
    print(f"  • Trained Model File        : {TRAINER_OUTPUT_FILE} ({file_size_kb:.1f} KB)")
    print(f"  • Training Duration         : {duration:.2f} seconds")
    print("=" * 68)
    print("[SUCCESS] LBPH face recognizer successfully trained and ready for kiosk deployment.")
    print("[NEXT STEP] Launch the kiosk using: python kiosk_laptop.py\n")


if __name__ == "__main__":
    train_model()
