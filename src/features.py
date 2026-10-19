"""normalize พิกัด landmark ให้เป็น feature ที่ train, evaluate และ demo ใช้ร่วมกัน

ถ้าแก้ไฟล์นี้ต้อง train ใหม่ ไม่ต้อง extract ใหม่ เพราะ CSV เก็บพิกัดดิบไว้
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

LM_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
N_FEATURES = 63


def normalize(row: Union[pd.Series, dict], img_w: int, img_h: int) -> Optional[np.ndarray]:
    """แปลงพิกัดดิบของ MediaPipe (0–1 ตามความกว้าง/สูงของภาพ) เป็น feature 63 มิติ

    ผลลัพธ์ไม่ขึ้นกับตำแหน่งมือ ระยะห่างจากกล้อง และสัดส่วนภาพ: ข้อมืออยู่ที่จุดกำเนิด
    ใช้ระยะข้อมือถึงโคนนิ้วกลาง (จุด 9) เป็นหน่วยความยาว และคืน None ถ้าสองจุดนี้ทับกัน
    """
    aspect = img_w / img_h

    pts = np.array([[row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]] for i in range(21)], dtype=np.float64)

    pts[:, 0] *= aspect
    pts[:, 2] *= aspect

    pts -= pts[0]

    scale = np.linalg.norm(pts[9, :2])
    if scale < 1e-6:
        return None

    pts /= scale
    return pts.flatten()


def normalize_df(df: pd.DataFrame) -> tuple:
    """normalize ทุกแถวที่ detected=1 คืน (X, y, mask) โดย mask ยาวเท่า df และเป็น True เฉพาะแถวที่อยู่ใน X"""
    X_rows, y_rows, mask = [], [], []

    for _, row in df.iterrows():
        if not row["detected"]:
            mask.append(False)
            continue
        feat = normalize(row, int(row["img_w"]), int(row["img_h"]))
        if feat is None:
            mask.append(False)
            continue
        X_rows.append(feat)
        y_rows.append(row["label"])
        mask.append(True)

    return np.array(X_rows), np.array(y_rows), np.array(mask)


def mirror(X: np.ndarray) -> np.ndarray:
    """กลับซ้ายขวาบน feature ที่ normalize แล้ว (กลับเครื่องหมายแกน x) ใช้ทำ data augmentation"""
    X_m = X.copy()
    x_idx = [i * 3 for i in range(21)]
    X_m[:, x_idx] *= -1
    return X_m
