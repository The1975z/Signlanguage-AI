"""เส้นทาง inference ของ demo: landmarks จาก MediaPipe → feature → probability ต่อคลาส

demo ทุกตัวต้องทายผ่านโมดูลนี้ ผลจะได้ตรงกับ offline eval
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from features import normalize


def landmarks_to_row(landmarks) -> dict:
    """แปลง landmark 21 จุด (object ที่มี .x .y .z) เป็น dict คอลัมน์ x0..z20 แบบเดียวกับใน CSV ของ extract.py"""
    row = {}
    for i, lm in enumerate(landmarks):
        row[f"x{i}"] = lm.x
        row[f"y{i}"] = lm.y
        row[f"z{i}"] = lm.z
    return row


def class_probabilities(clf, feat: np.ndarray) -> dict:
    """คืน {ชื่อคลาส: probability} โดยผูกกับ clf.classes_

    sklearn เรียงคอลัมน์ของ predict_proba ตาม classes_ (a–z) ซึ่งไม่ใช่ลำดับใน dataset.CLASSES
    จึงห้ามเอา index ของ probability ไปเปิดใน list อื่นเอง
    """
    proba = clf.predict_proba(np.asarray(feat, dtype=np.float64).reshape(1, -1))[0]
    return dict(zip(clf.classes_, proba))


def predict_from_landmarks(clf, landmarks, img_w: int, img_h: int) -> Optional[dict]:
    """ทายจาก landmarks ของภาพขนาด img_w×img_h ที่ใช้ detect คืน {ชื่อคลาส: probability} หรือ None ถ้า landmark เสีย"""
    feat = normalize(landmarks_to_row(landmarks), img_w, img_h)
    if feat is None:
        return None
    return class_probabilities(clf, feat)


def top_label(probs: dict) -> tuple:
    """คืน (ชื่อคลาส, probability) ของคลาสที่ probability สูงสุด"""
    label = max(probs, key=probs.get)
    return label, float(probs[label])
