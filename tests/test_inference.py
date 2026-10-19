"""Checks that the demo inference path (src/inference.py) agrees with the offline evaluation on labelled data.

Landmarks come from data/processed instead of a camera, so these tests run anywhere, including Docker.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CLASSES
from features import normalize_df
from inference import class_probabilities, predict_from_landmarks, top_label

CODES = [code for code, _, _ in CLASSES]
PROCESSED = ROOT / "data" / "processed"
BEST_MODEL = ROOT / "models" / "best.joblib"


def _load(split: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"landmarks_{split}.csv")


def _as_landmarks(row) -> list:
    """Rebuild MediaPipe-like landmark objects (.x/.y/.z) from a CSV row."""
    return [SimpleNamespace(x=row[f"x{i}"], y=row[f"y{i}"], z=row[f"z{i}"]) for i in range(21)]


def _demo_path(clf, df: pd.DataFrame) -> np.ndarray:
    """Predict row by row through inference.py, the same way the demos handle each frame."""
    preds = []
    for _, row in df.iterrows():
        if not row["detected"]:
            continue
        probs = predict_from_landmarks(clf, _as_landmarks(row), int(row["img_w"]), int(row["img_h"]))
        if probs is not None:
            preds.append(top_label(probs)[0])
    return np.array(preds)


@pytest.fixture(scope="module")
def knn():
    X, y, _ = normalize_df(_load("train"))
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))])
    return pipe.fit(X, y)


def test_probabilities_follow_classifier_class_order(knn):
    """``classes_`` is ordered differently from ``config.CLASSES``, so a wrong index-to-name mapping fails here."""
    assert list(knn.classes_) != CODES, "class orders match by coincidence; this test cannot catch an ordering bug"
    X, _, _ = normalize_df(_load("test").head(60))
    for x in X:
        probs = class_probabilities(knn, x)
        assert set(probs) == set(CODES)
        assert top_label(probs)[0] == knn.predict(x.reshape(1, -1))[0]


def test_demo_path_matches_offline_predict(knn):
    """Per-frame predictions must equal batch ``predict`` on every row of the test split."""
    df = _load("test")
    X, y, _ = normalize_df(df)
    preds = _demo_path(knn, df)
    assert len(preds) == len(y)
    assert (preds == knn.predict(X)).all()
    assert (preds == y).mean() > 0.90


@pytest.mark.skipif(not BEST_MODEL.exists(), reason="models/best.joblib is not tracked by git; run src/train.py first")
def test_best_model_demo_path_matches_offline_eval():
    """Same check against the real models/best.joblib.

    Up to 1% disagreement with ``predict`` is allowed: SVC computes ``predict_proba`` with Platt
    scaling, so its argmax can differ from ``predict`` on a few rows.
    """
    import joblib

    clf = joblib.load(BEST_MODEL)
    df = _load("test")
    X, y, _ = normalize_df(df)
    preds = _demo_path(clf, df)
    assert (preds == clf.predict(X)).mean() >= 0.99
    assert (preds == y).mean() >= 0.95
