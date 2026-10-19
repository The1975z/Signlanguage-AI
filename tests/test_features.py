"""Property tests for normalize and mirror using synthetic hands; no dataset needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from features import N_FEATURES, mirror, normalize, normalize_df

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
    """Scaling the hand 1.8x around the wrist (moving it closer to the camera) must not change the features."""
    pts = _make_hand()
    scaled = pts.copy()
    wrist = pts[0].copy()
    scaled = wrist + (pts - wrist) * 1.8

    f1 = normalize(_pts_to_row(pts), IMG_W, IMG_H)
    f2 = normalize(_pts_to_row(scaled), IMG_W, IMG_H)

    assert f1 is not None and f2 is not None
    np.testing.assert_allclose(f1, f2, atol=1e-5)


def test_aspect_ratio_invariant():
    """The same hand at the same pixel positions in a wider image must give the same features.

    The live camera is 16:9 while the dataset mixes 3:2 and 16:9, and MediaPipe expresses x and z
    as fractions of the image width.
    """
    pts = _make_hand()
    wide_w = IMG_W + 400
    wide = pts.copy()
    wide[:, [0, 2]] *= IMG_W / wide_w

    f1 = normalize(_pts_to_row(pts), IMG_W, IMG_H)
    f2 = normalize(_pts_to_row(wide), wide_w, IMG_H)

    assert f1 is not None and f2 is not None
    np.testing.assert_allclose(f1, f2, atol=1e-9)


def test_wrist_zero_after_normalize():
    pts  = _make_hand()
    feat = normalize(_pts_to_row(pts), IMG_W, IMG_H)
    assert feat is not None
    np.testing.assert_allclose(feat[:3], [0.0, 0.0, 0.0], atol=1e-9)


def test_degenerate_returns_none():
    """A wrist lying on point 9 gives a zero scale and must return None instead of dividing by zero."""
    pts = _make_hand()
    pts[9] = pts[0]
    feat = normalize(_pts_to_row(pts), IMG_W, IMG_H)
    assert feat is None


def test_normalize_df_skips_undetected_and_degenerate_rows():
    """Undetected and degenerate rows must be left out of X, and mask must point back to the rows that were kept."""
    undetected, degenerate = 3, 7
    rows = []
    for i in range(10):
        pts = _make_hand(seed=i)
        if i == degenerate:
            pts[9] = pts[0]
        if i == undetected:
            pts = np.full((21, 3), np.nan)
        row = _pts_to_row(pts)
        row["label"] = f"class_{i}"
        row["detected"] = int(i != undetected)
        rows.append(row)

    X, y, mask = normalize_df(pd.DataFrame(rows))

    kept = [i for i in range(10) if i not in (undetected, degenerate)]
    assert X.shape == (len(kept), N_FEATURES)
    np.testing.assert_array_equal(mask, [i in kept for i in range(10)])
    np.testing.assert_array_equal(y, [f"class_{i}" for i in kept])


def test_mirror_flips_x_only():
    X = RNG.standard_normal((4, N_FEATURES))
    Xm = mirror(X)
    x_idx = [i * 3 for i in range(21)]
    y_idx = [i * 3 + 1 for i in range(21)]
    np.testing.assert_array_equal(Xm[:, x_idx], -X[:, x_idx])
    np.testing.assert_array_equal(Xm[:, y_idx], X[:, y_idx])
