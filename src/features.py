"""Landmark normalization shared by training, evaluation and the demos.

Any change here requires retraining. Re-extraction is not needed because the CSVs keep the raw
coordinates.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

LM_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
N_FEATURES = 63


def normalize(row: Union[pd.Series, dict], img_w: int, img_h: int) -> Optional[np.ndarray]:
    """Convert raw MediaPipe landmarks into a 63-dimensional feature vector.

    The result does not depend on hand position, distance from the camera or image aspect ratio:
    the wrist is moved to the origin and the wrist-to-middle-finger-MCP distance (point 9) becomes
    the unit length.

    Args:
        row: Mapping with keys x0..z20 holding MediaPipe's normalized coordinates (fractions of
            the image width and height).
        img_w: Width in pixels of the image the landmarks were detected on.
        img_h: Height in pixels of that image.

    Returns:
        Flattened array of 21 points x (x, y, z), or None when the wrist and point 9 coincide.
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
    """Normalize every row with ``detected == 1``.

    Returns:
        ``(X, y, mask)`` where ``mask`` has one entry per row of ``df`` and is True for the rows kept in ``X``.
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
    """Flip normalized features left-right by negating every x coordinate; used for data augmentation."""
    X_m = X.copy()
    x_idx = [i * 3 for i in range(21)]
    X_m[:, x_idx] *= -1
    return X_m
