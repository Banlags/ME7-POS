import os
import threading
import time
from typing import Optional
from . import state  # NEW: to send detections to POS state

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from . import state  

# =========================
# Configuration
# =========================

# Camera index: 0 is usually the default webcam.
CAMERA_INDEX = 0

# Target frame size (width, height) for processing.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Path to your YOLO model.
# Assumes best.pt is in the project root (same level as app/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "best.pt")  # adjust if needed

# =========================
# Globals
# =========================

_model: Optional[YOLO] = None
_capture_thread: Optional[threading.Thread] = None
_stop_flag = False

_latest_frame_lock = threading.Lock()
_latest_frame: Optional[np.ndarray] = None

mp_hands = mp.solutions.hands
_hands = None

gesture_state = "idle"
last_open_time = 0.0
last_toggle_time = 0.0

# Tunable parameters (relaxed)
GESTURE_SEQUENCE_TIMEOUT = 2.0        # more time between open and closed (seconds)
GESTURE_TOGGLE_COOLDOWN = 2.0         # shorter cooldown between toggles

MIN_OPEN_FRAMES = 3                   # was 7; now quicker to confirm
MIN_CLOSED_FRAMES = 3                 # ditto

open_frame_count = 0
closed_frame_count = 0


# =========================
# Model + camera initialization
# =========================

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO model not found at {MODEL_PATH}. "
                f"Place your best.pt there or update MODEL_PATH in vision.py."
            )
        _model = YOLO(MODEL_PATH)
        print(f"[VISION] Loaded YOLO model from {MODEL_PATH}")

    global _hands
    if _hands is None:
        _hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[VISION] MediaPipe Hands initialized.")


def start_camera():
    """Initialize the YOLO model and start the camera capture thread."""
    global _capture_thread, _stop_flag

    if _capture_thread is not None and _capture_thread.is_alive():
        print("[VISION] Camera thread already running.")
        return

    load_model()

    _stop_flag = False
    _capture_thread = threading.Thread(target=_camera_loop, daemon=True)
    _capture_thread.start()
    print("[VISION] Camera capture thread started.")


def stop_camera():
    """Signal the camera thread to stop."""
    global _stop_flag
    _stop_flag = True
    print("[VISION] Camera capture stop requested.")


# =========================
# Camera capture + YOLO inference loop
# =========================

def _camera_loop():
    global _latest_frame

    print("[VISION] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[VISION] Failed to open camera.")
        return

    # Optionally set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("[VISION] Camera opened successfully.")

    try:
        while not _stop_flag:
            ret, frame = cap.read()
            if not ret:
                print("[VISION] Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            # Resize for consistency (YOLO can handle various sizes)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Run YOLO inference
            annotated = _run_yolo_and_annotate(frame)

            # Save the annotated frame as the latest frame
            with _latest_frame_lock:
                _latest_frame = annotated

            # Small sleep to avoid hogging CPU completely
            time.sleep(0.01)

    finally:
        cap.release()
        print("[VISION] Camera released.")


def _run_yolo_and_annotate(frame: np.ndarray) -> np.ndarray:
    """
    Run YOLO on the frame and draw bounding boxes with labels.
    Also sends the best detection (if any) to the POS state manager.
    Always runs hand gesture processing, even if no items are detected.
    Returns an annotated BGR frame.
    """
    global _model
    annotated = frame.copy()

    if _model is None:
        # No model loaded; still process gestures
        state.process_detection(None, None)
        _process_hand_gesture(annotated)
        return annotated

    results = _model(frame, verbose=False)

    # ---------- YOLO detection + POS logic ----------
    best_box = None
    best_conf = -1.0

    if results and len(results) > 0:
        res = results[0]

        if res.boxes is not None and len(res.boxes) > 0:
            # Find the highest-confidence box for POS logic
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

            if best_box is not None:
                cls_id = int(best_box.cls[0])
                class_name = _model.names.get(cls_id, str(cls_id))
                state.process_detection(cls_id, class_name)
            else:
                state.process_detection(None, None)

            # Draw ALL boxes for visualization
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = _model.names.get(cls_id, str(cls_id))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - baseline),
                    (x1 + tw, y1),
                    (0, 255, 0),
                    thickness=-1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
        else:
            # No boxes
            state.process_detection(None, None)
    else:
        # No results
        state.process_detection(None, None)

    # ---------- Hand gesture processing (runs EVERY frame) ----------
    _process_hand_gesture(annotated)

    return annotated
    

def _classify_hand_gesture(hand_landmarks, image_width: int, image_height: int) -> str:
    """
    Classify hand gesture as 'open', 'closed', or 'unknown' based on finger extension.
    Simple heuristic:
    - Count how many fingertips are above their PIP joints (for a typical upright hand).
    - If >= 3 fingers extended  -> 'open'
    - If <= 1 finger extended   -> 'closed'
    - Else                      -> 'unknown'
    """
    # Mediapipe landmark indices for fingertips and PIP joints
    # Index finger: tip 8, pip 6
    # Middle finger: tip 12, pip 10
    # Ring finger: tip 16, pip 14
    # Little finger: tip 20, pip 18
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    extended_fingers = 0

    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]

        # y is normalized [0,1]; smaller y = higher on image
        if tip.y < pip.y - 0.02:  # small margin
            extended_fingers += 1

    if extended_fingers >= 3:
        return "open"
    elif extended_fingers <= 1:
        return "closed"
    else:
        return "unknown"


def _process_hand_gesture(frame: np.ndarray) -> str:
    """
    Run MediaPipe Hands on the frame, classify gesture (open/closed/unknown),
    and detect a stable sequence:
        (1) Open Palm held for MIN_OPEN_FRAMES
        (2) Then Closed Fist held for MIN_CLOSED_FRAMES
    within GESTURE_SEQUENCE_TIMEOUT seconds, and respecting a cooldown.
    """
    global _hands, gesture_state, last_open_time, last_toggle_time
    global open_frame_count, closed_frame_count

    if _hands is None:
        return "none"

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(frame_rgb)

    gesture_label = "none"
    now = time.time()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        gesture_label = _classify_hand_gesture(hand_landmarks, w, h)

        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

        if gesture_label == "open":
            open_frame_count += 1
            closed_frame_count = 0

            if gesture_state == "idle" and open_frame_count >= MIN_OPEN_FRAMES:
                gesture_state = "open_confirmed"
                last_open_time = now

        elif gesture_label == "closed":
            closed_frame_count += 1

            if (
                gesture_state == "open_confirmed"
                and closed_frame_count >= MIN_CLOSED_FRAMES
                and (now - last_open_time) <= GESTURE_SEQUENCE_TIMEOUT
                and (now - last_toggle_time) >= GESTURE_TOGGLE_COOLDOWN
            ):
                if state.pos_session.active:
                    state.end_session()
                else:
                    state.start_session()

                last_toggle_time = now
                gesture_state = "idle"
                open_frame_count = 0
                closed_frame_count = 0

        else:
            open_frame_count = max(open_frame_count - 1, 0)
            closed_frame_count = max(closed_frame_count - 1, 0)

        if gesture_state == "open_confirmed" and (now - last_open_time) > GESTURE_SEQUENCE_TIMEOUT:
            gesture_state = "idle"
            open_frame_count = 0
            closed_frame_count = 0

        cv2.putText(
            frame,
            f"Gesture: {gesture_label}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    else:
        open_frame_count = max(open_frame_count - 1, 0)
        closed_frame_count = max(closed_frame_count - 1, 0)

        if gesture_state == "open_confirmed" and (now - last_open_time) > GESTURE_SEQUENCE_TIMEOUT:
            gesture_state = "idle"
            open_frame_count = 0
            closed_frame_count = 0

        gesture_label = "none"

    status_text = "SESSION ACTIVE" if state.pos_session.active else "SESSION INACTIVE"
    cv2.putText(
        frame,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return gesture_label

# =========================
# Helper for FastAPI video endpoint
# =========================

def get_latest_frame_jpeg() -> Optional[bytes]:
    """
    Returns the latest annotated frame encoded as JPEG bytes,
    or None if no frame is available yet.
    """
    with _latest_frame_lock:
        if _latest_frame is None:
            return None
        frame = _latest_frame.copy()

    # Encode as JPEG
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None

    return buffer.tobytes()