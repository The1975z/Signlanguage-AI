"""Inference path shared by the demos: MediaPipe landmarks -> features -> class probabilities.

Every demo should predict through this module so that live results match the offline evaluation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from features import normalize


def landmarks_to_row(landmarks) -> dict:
    """Convert 21 landmarks (objects with .x/.y/.z) into the x0..z20 columns written by extract.py."""
    row = {}
    for i, lm in enumerate(landmarks):
        row[f"x{i}"] = lm.x
        row[f"y{i}"] = lm.y
        row[f"z{i}"] = lm.z
    return row


def class_probabilities(clf, feat: np.ndarray) -> dict:
    """Return the probability of every class, keyed by class name.

    The columns of ``predict_proba`` follow ``clf.classes_`` (sorted labels), not the order of
    ``config.CLASSES``, so never map a column index to a class name by hand.

    Args:
        clf: Fitted classifier or pipeline that supports ``predict_proba``.
        feat: Feature vector produced by ``features.normalize``.

    Returns:
        Dict mapping class name to probability.
    """
    proba = clf.predict_proba(np.asarray(feat, dtype=np.float64).reshape(1, -1))[0]
    return dict(zip(clf.classes_, proba))


def predict_from_landmarks(clf, landmarks, img_w: int, img_h: int) -> Optional[dict]:
    """Predict class probabilities for one detected hand.

    Args:
        clf: Fitted classifier or pipeline that supports ``predict_proba``.
        landmarks: The 21 landmarks MediaPipe returned for the hand.
        img_w: Width in pixels of the image the hand was detected on.
        img_h: Height in pixels of that image.

    Returns:
        Dict mapping class name to probability, or None if the landmarks are degenerate.
    """
    feat = normalize(landmarks_to_row(landmarks), img_w, img_h)
    if feat is None:
        return None
    return class_probabilities(clf, feat)


def top_label(probs: dict) -> tuple:
    """Return ``(class_name, probability)`` for the most likely class."""
    label = max(probs, key=probs.get)
    return label, float(probs[label])
