import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CLASSES, SPLITS, get_image_paths, get_raw_base

MODEL_PATH = str(ROOT / "models" / "hand_landmarker.task")
OUT_DIR    = ROOT / "data" / "processed"

LM_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]


def build_detector() -> HandLandmarker:
    opts = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(opts)


def extract_one(detector: HandLandmarker, img_path: Path, split: str, label: str) -> dict:
    mp_img  = mp.Image.create_from_file(str(img_path))
    result  = detector.detect(mp_img)
    img_h, img_w = mp_img.height, mp_img.width

    row: dict = {
        "split":             split,
        "label":             label,
        "file_path":         str(img_path),
        "img_w":             img_w,
        "img_h":             img_h,
        "detected":          0,
        "handedness":        "",
        "handedness_score":  np.nan,
    }

    if result.hand_landmarks:
        lm  = result.hand_landmarks[0]
        row["detected"]         = 1
        row["handedness"]       = result.handedness[0][0].display_name
        row["handedness_score"] = result.handedness[0][0].score
        for i, pt in enumerate(lm):
            row[f"x{i}"] = pt.x
            row[f"y{i}"] = pt.y
            row[f"z{i}"] = pt.z
    else:
        for col in LM_COLS:
            row[col] = np.nan

    return row


def extract_split(split_key: str) -> pd.DataFrame:
    raw   = get_raw_base(ROOT)
    rows  = []

    with build_detector() as detector:
        for code, _, folder in CLASSES:
            paths  = get_image_paths(raw, split_key, folder)
            before = len(rows)
            for p in paths:
                try:
                    rows.append(extract_one(detector, p, split_key, code))
                except Exception as e:
                    print(f"    skip {p.name}: {e}", flush=True)
            chunk = rows[before:]
            found = sum(1 for r in chunk if r["detected"])
            rate  = found / len(chunk) * 100 if chunk else 0
            print(f"  {folder:<15}  {found:>4}/{len(chunk)}  ({rate:.1f}%)", flush=True)

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_key in SPLITS:
        print(f"\n=== {split_key} ===")
        df = extract_split(split_key)
        out = OUT_DIR / f"landmarks_{split_key}.csv"
        df.to_csv(out, index=False)

        total    = len(df)
        detected = df["detected"].sum()
        print(f"  บันทึก: {out.name}  ({total} แถว  detected {detected}/{total}  {detected/total*100:.1f}%)")


if __name__ == "__main__":
    main()
