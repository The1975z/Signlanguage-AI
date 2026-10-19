from __future__ import annotations

import collections
import sys
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import CLASSES
from features import normalize

MODEL_PATH  = str(ROOT / "models" / "hand_landmarker.task")
FONT_PATH   = "/System/Library/Fonts/Supplemental/Tahoma.ttf"
FONT_SIZE   = 48
SMOOTH_WIN  = 8
CONF_THRESH = 0.6

CLASS_CHARS = {code: char for code, char, _ in CLASSES}
CLASS_CODES  = [code for code, _, _ in CLASSES]


def load_classifier(name: str = "best"):
    return joblib.load(ROOT / "models" / f"{name}.joblib")


def build_detector() -> HandLandmarker:
    opts = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(opts)


def draw_thai(frame: np.ndarray, text: str, pos: tuple, size: int = FONT_SIZE) -> np.ndarray:
    pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw  = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_landmarks(frame: np.ndarray, landmarks) -> np.ndarray:
    h, w = frame.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    return frame


def run(model_name: str = "best"):
    clf     = load_classifier(model_name)
    cap     = cv2.VideoCapture(0)
    history: collections.deque = collections.deque(maxlen=SMOOTH_WIN)
    ts      = 0

    with build_detector() as detector:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            ts   += 33

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, ts)

            label_text = ""
            if result.hand_landmarks:
                lm  = result.hand_landmarks[0]
                row = {f"x{i}": lm[i].x for i in range(21)}
                row.update({f"y{i}": lm[i].y for i in range(21)})
                row.update({f"z{i}": lm[i].z for i in range(21)})

                feat = normalize(row, w, h)
                if feat is not None:
                    proba = clf.predict_proba([feat])[0]
                    top3  = np.argsort(proba)[::-1][:3]
                    history.append(top3[0])

                    if len(history) == SMOOTH_WIN:
                        from collections import Counter
                        voted_idx = Counter(history).most_common(1)[0][0]
                        if proba[voted_idx] >= CONF_THRESH:
                            label_text = CLASS_CHARS.get(CLASS_CODES[voted_idx], "?")

                frame = draw_landmarks(frame, lm)

            fps_text = f"FPS: {cap.get(cv2.CAP_PROP_FPS):.0f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if label_text:
                frame = draw_thai(frame, label_text, (w // 2 - 30, 40), size=72)

            cv2.imshow("Thai Fingerspelling", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
