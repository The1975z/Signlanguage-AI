"""Tests for the step-4 checks in clean.py using small synthetic tables."""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from clean import anomaly_summary, detection_by_class
from config import CLASSES


def _row(name: str, label: str = "ko_kai", detected: int = 1, shift: float = 0.0) -> dict:
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    pts = rng.uniform(0.3, 0.7, (21, 3))
    pts[9] = pts[0] + np.array([0.05, -0.2, 0.0])
    pts[:, 0] += shift
    row = {"file_path": f"/data/{name}.jpg", "label": label, "detected": detected, "img_w": 1080, "img_h": 720}
    for i, (x, y, z) in enumerate(pts):
        row[f"x{i}"], row[f"y{i}"], row[f"z{i}"] = (x, y, z) if detected else (np.nan, np.nan, np.nan)
    return row


def test_anomaly_summary_counts_each_problem_once():
    """Each synthetic problem row must be counted by exactly its own check."""
    normal, twin = _row("normal"), _row("twin_a")
    twin_b = {**twin, "file_path": "/data/twin_b.jpg"}
    nan_row = _row("nan")
    nan_row["x4"] = np.nan
    degenerate = _row("degenerate")
    degenerate["x9"], degenerate["y9"] = degenerate["x0"], degenerate["y0"]
    df = pd.DataFrame([normal, twin, twin_b, nan_row, degenerate,
                       _row("far", shift=1.2), _row("edge", shift=0.35), _row("missing", detected=0)])

    checks = anomaly_summary(df)

    assert checks["nan_rows"] == 1
    assert checks["far_outside"] == 1
    assert checks["slightly_outside"] == 1
    assert checks["duplicate_rows"] == 2
    assert checks["duplicate_files"] == ["twin_a.jpg", "twin_b.jpg"]
    assert checks["degenerate"] == 1


def test_detection_by_class_counts_rows_detections_and_unreadable_images():
    """Counts are per split and class, and images missing from the CSV are reported as unreadable."""
    df = pd.DataFrame([_row("a"), _row("b"), _row("c", detected=0), _row("d", label="do_dek")])
    image_counts = {("train", code): 0 for code, _, _ in CLASSES}
    image_counts[("train", "ko_kai")] = 4
    image_counts[("train", "do_dek")] = 1

    table = detection_by_class({"train": df}, image_counts).set_index("code")

    assert table.loc["ko_kai", ["images", "rows", "unreadable", "detected", "undetected"]].tolist() == [4, 3, 1, 2, 1]
    assert table.loc["ko_kai", "detection_rate"] == round(2 / 3, 4)
    assert table.loc["do_dek", ["rows", "detected"]].tolist() == [1, 1]
    assert np.isnan(table.loc["to_tao", "detection_rate"])
