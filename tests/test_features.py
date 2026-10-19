import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from features import N_FEATURES, mirror, normalize

RNG = np.random.default_rng(42)
IMG_W, IMG_H = 1280, 720


def _make_hand(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.2, 0.8, (21, 3))
    pts[0] = [0.5, 0.7, 0.0]
    pts[9] = [0.55, 0.5, 0.02]
    return pts


def _pts_to_row(pts: np.ndarray) -> dict:
    row = {"img_w": IMG_W, "img_h": IMG_H, "detected": 1}
    for i, (x, y, z) in enumerate(pts):
        row[f"x{i}"] = x
        row[f"y{i}"] = y
        row[f"z{i}"] = z
    return row


def test_output_shape():
    row = _pts_to_row(_make_hand())
    feat = normalize(row, IMG_W, IMG_H)
    assert feat is not None
    assert feat.shape == (N_FEATURES,)


def test_translation_invariant():
    pts = _make_hand()
    shifted = pts.copy()
    shifted[:, :2] += np.array([0.1, -0.15])

    f1 = normalize(_pts_to_row(pts), IMG_W, IMG_H)
    f2 = normalize(_pts_to_row(shifted), IMG_W, IMG_H)

    assert f1 is not None and f2 is not None
    np.testing.assert_allclose(f1, f2, atol=1e-6)


def test_scale_invariant():
    pts = _make_hand()
    scaled = pts.copy()
    wrist = pts[0].copy()
    scaled = wrist + (pts - wrist) * 1.8

    f1 = normalize(_pts_to_row(pts), IMG_W, IMG_H)
    f2 = normalize(_pts_to_row(scaled), IMG_W, IMG_H)

    assert f1 is not None and f2 is not None
    np.testing.assert_allclose(f1, f2, atol=1e-5)


def test_wrist_zero_after_normalize():
    pts  = _make_hand()
    feat = normalize(_pts_to_row(pts), IMG_W, IMG_H)
    assert feat is not None
    # จุด 0 (ข้อมือ) ต้องเป็น (0, 0, 0) หลัง normalize
    np.testing.assert_allclose(feat[:3], [0.0, 0.0, 0.0], atol=1e-9)


def test_degenerate_returns_none():
    pts = _make_hand()
    pts[9] = pts[0]  # scale = 0
    feat = normalize(_pts_to_row(pts), IMG_W, IMG_H)
    assert feat is None


def test_mirror_flips_x_only():
    X = RNG.standard_normal((4, N_FEATURES))
    Xm = mirror(X)
    x_idx = [i * 3 for i in range(21)]
    y_idx = [i * 3 + 1 for i in range(21)]
    np.testing.assert_array_equal(Xm[:, x_idx], -X[:, x_idx])
    np.testing.assert_array_equal(Xm[:, y_idx], X[:, y_idx])
