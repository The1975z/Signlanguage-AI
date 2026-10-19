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

from dataset import CLASSES, get_image_paths
from inference import predict_from_landmarks, top_label

MODEL_PATH  = str(ROOT / "models" / "hand_landmarker.task")
FONT_PATH   = "/System/Library/Fonts/Supplemental/Tahoma.ttf"
RAW_DIR     = ROOT / "data" / "raw" / "One-Stage-TFS Thai One-Stage Fingerspelling Dataset"

SMOOTH_WIN  = 8
CONF_THRESH = 0.55
HOLD_OK_SEC = 1.5

CLASS_CHARS  = {code: char for code, char, _ in CLASSES}
CLASS_CODES  = [code for code, _, _ in CLASSES]
CLASS_FOLDER = {code: folder for code, _, folder in CLASSES}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

font_big = ImageFont.truetype(FONT_PATH, 72)
font_med = ImageFont.truetype(FONT_PATH, 36)
font_sm  = ImageFont.truetype(FONT_PATH, 24)
font_xs  = ImageFont.truetype(FONT_PATH, 20)


def load_ref_image(code: str) -> np.ndarray:
    folder   = CLASS_FOLDER[code]
    imgs     = get_image_paths(RAW_DIR, "train", folder)
    opts_img = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE, num_hands=1,
    )
    with HandLandmarker.create_from_options(opts_img) as d:
        for path in imgs[5:30]:
            try:
                img    = Image.open(path).convert("RGB")
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img))
                res    = d.detect(mp_img)
                if not res.hand_landmarks:
                    continue
                lm     = res.hand_landmarks[0]
                W2, H2 = img.size
                xs     = [l.x * W2 for l in lm]
                ys     = [l.y * H2 for l in lm]
                cx, cy = int(sum(xs) / 21), int(sum(ys) / 21)
                r      = int(max(max(xs)-min(xs), max(ys)-min(ys)) * 0.72)
                r      = max(r, 60)
                crop   = img.crop((max(0, cx-r), max(0, cy-r),
                                   min(W2, cx+r), min(H2, cy+r)))
                return np.array(crop.resize((220, 220)))
            except Exception:
                continue
    return np.zeros((220, 220, 3), dtype=np.uint8)


def draw_skeleton(frame: np.ndarray, lm, w: int, h: int):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for i, (px, py) in enumerate(pts):
        color = (0, 255, 255) if i == 0 else (0, 255, 0)
        r     = 6 if i == 0 else 4
        cv2.circle(frame, (px, py), r, color, -1)


def draw_debug_panel(frame: np.ndarray, probs: dict,
                     target_code: str, predicted_code: str) -> np.ndarray:
    h, w    = frame.shape[:2]
    pw      = 300
    panel   = np.zeros((h, pw, 3), dtype=np.uint8)
    panel[:] = (25, 25, 35)

    pil  = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    draw.text((8, 6), "Confidence (ทุกคลาส)", font=font_xs, fill=(180, 180, 180))

    bar_x    = 60
    bar_maxw = pw - bar_x - 10
    row_h    = (h - 30) // len(CLASS_CODES)

    for i, code in enumerate(CLASS_CODES):
        char = CLASS_CHARS[code]
        p    = float(probs.get(code, 0.0))
        y0   = 28 + i * row_h

        is_target = (code == target_code)
        is_pred   = (code == predicted_code)

        if is_target:
            bg = (20, 60, 20)
        elif is_pred:
            bg = (60, 40, 10)
        else:
            bg = (25, 25, 35)
        draw.rectangle([0, y0, pw, y0 + row_h - 1], fill=bg)

        label_color = (80, 255, 80) if is_target else (255, 180, 60) if is_pred else (180, 180, 180)
        draw.text((4, y0 + 2), char, font=font_xs, fill=label_color)

        bar_w = int(p * bar_maxw)
        bar_color = (60, 220, 60) if is_target else (220, 160, 40) if is_pred else (80, 100, 160)
        if bar_w > 0:
            draw.rectangle([bar_x, y0 + 4, bar_x + bar_w, y0 + row_h - 4], fill=bar_color)

        pct_text = f"{p*100:.0f}%"
        draw.text((bar_x + bar_maxw - 34, y0 + 2), pct_text, font=font_xs, fill=(160, 160, 160))

    panel = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    combined = np.hstack([frame, panel])
    return combined


def draw_pil(frame: np.ndarray, fn) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fn(pil)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def run():
    clf   = joblib.load(ROOT / "models" / "best.joblib")
    order = list(range(len(CLASSES)))
    np.random.shuffle(order)

    current_idx  = 0
    correct      = 0
    total        = 0
    history: collections.deque = collections.deque(maxlen=SMOOTH_WIN)
    hold_correct = 0.0
    ts           = 0
    last_probs: dict = {}

    print("โหลด reference images ...")
    refs = {code: load_ref_image(code) for code, _, _ in CLASSES}
    print("พร้อม — Space=ข้าม  Q=ออก  ** อยู่คนเดียวในเฟรม **")

    opts = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    with HandLandmarker.create_from_options(opts) as detector:
        while cap.isOpened() and current_idx < len(order):
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            ts   += int(1000 / fps)

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, ts)

            target_code    = CLASS_CODES[order[current_idx]]
            target_char    = CLASS_CHARS[target_code]
            predicted_char = "?"
            predicted_code = ""
            is_correct     = False
            multi_warn     = len(result.hand_landmarks) > 1 if result.hand_landmarks else False

            if result.hand_landmarks:
                lm  = result.hand_landmarks[0]
                draw_skeleton(frame, lm, w, h)

                probs = predict_from_landmarks(clf, lm, w, h)
                if probs is not None:
                    last_probs = probs
                    history.append(top_label(probs)[0])
                    if len(history) == SMOOTH_WIN:
                        voted = collections.Counter(history).most_common(1)[0][0]
                        if probs[voted] >= CONF_THRESH:
                            predicted_code = voted
                            predicted_char = CLASS_CHARS[voted]
                            is_correct     = (voted == target_code)

            if is_correct:
                hold_correct += 1.0 / fps
            else:
                hold_correct  = 0.0

            if hold_correct >= HOLD_OK_SEC:
                correct      += 1
                total        += 1
                current_idx  += 1
                hold_correct  = 0.0
                history.clear()
                continue

            ref_bgr  = cv2.cvtColor(refs[target_code], cv2.COLOR_RGB2BGR)
            ref_size = 240
            ry1, ry2 = h - ref_size - 36, h - 36
            if ry1 >= 0:
                frame[ry1:ry2, 10:10+ref_size] = cv2.resize(ref_bgr, (ref_size, ref_size))
                bcolor = (0, 255, 0) if is_correct else (160, 160, 40)
                cv2.rectangle(frame, (10, ry1), (10+ref_size, ry2), bcolor, 3)
                cv2.putText(frame, "Ref", (14, ry1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 40), 2)

            def draw_ui(pil: Image.Image):
                d = ImageDraw.Draw(pil)
                d.text((w//2 - 44, 8),  target_char,        font=font_big, fill=(255, 255, 255))
                d.text((w//2 - 44, 88), f"ทำท่า: {target_char}", font=font_xs, fill=(200, 200, 200))
                if predicted_char != "?":
                    color  = (80, 220, 80) if is_correct else (80, 120, 230)
                    status = "OK  ถูก!" if is_correct else "XX  ผิด"
                    d.text((w//2 - 44, 114), f"โมเดลเห็น: {predicted_char}", font=font_sm, fill=color)
                    d.text((w//2 - 44, 154), status, font=font_med,
                           fill=(80, 220, 80) if is_correct else (80, 120, 230))
                    if is_correct:
                        bw = int((hold_correct / HOLD_OK_SEC) * 200)
                        d.rectangle([w//2 - 100, 200, w//2 - 100 + bw, 218], fill=(80, 220, 80))
                        d.rectangle([w//2 - 100, 200, w//2 + 100, 218], outline=(120, 120, 120))
                score = f"{correct}/{max(total,1)}  [{current_idx+1}/15]"
                d.text((10, 10), score, font=font_xs, fill=(100, 220, 255))
                if multi_warn:
                    d.text((10, h - 290), "!! อยู่คนเดียวในเฟรม !!", font=font_sm, fill=(60, 60, 255))

            frame = draw_pil(frame, draw_ui)
            frame = draw_debug_panel(frame, last_probs, target_code, predicted_code)

            cv2.imshow("Thai Fingerspelling Quiz", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                total       += 1
                current_idx += 1
                hold_correct = 0.0
                history.clear()

    cap.release()
    cv2.destroyAllWindows()
    if total > 0:
        print(f"\nผล: ถูก {correct} / {total}  ({correct/total*100:.1f}%)")
    else:
        print("ยังไม่ได้เทส")


if __name__ == "__main__":
    run()
