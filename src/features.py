from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

LM_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
N_FEATURES = 63


def normalize(row: Union[pd.Series, dict], img_w: int, img_h: int) -> Optional[np.ndarray]:
    """
    แปลง landmark 21 จุดเป็น feature vector 63 มิติที่ไม่ขึ้นกับตำแหน่งและขนาดในภาพ

    ขั้นตอน:
      1. แก้สัดส่วน: x *= W/H, z *= W/H  (y คงเดิม)
      2. ย้ายจุดกำเนิดไปที่ข้อมือ (จุด 0)
      3. หารด้วยระยะข้อมือ→โคนนิ้วกลาง (จุด 9, เฉพาะ x,y)

    คืน None ถ้า scale ≈ 0 (landmark ผิดปกติ)
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
    """
    normalize ทุกแถวใน DataFrame ที่ detected=1

    คืน:
      X      — (N, 63) feature matrix
      y      — (N,) label array (class code string)
      mask   — (len(df),) bool mask ของแถวที่ normalize สำเร็จ
    """
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
    """กลับซ้าย-ขวา: x ← -x  (ใช้กับ feature ที่ normalize แล้ว)"""
    X_m = X.copy()
    x_idx = [i * 3 for i in range(21)]
    X_m[:, x_idx] *= -1
    return X_m
