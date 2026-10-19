"""ตรวจว่าเส้นทางที่ demo ใช้ทาย (src/inference.py) ให้ผลเท่ากับ offline eval บนข้อมูลที่มี label

ใช้ landmark จาก data/processed แทนกล้อง จึงรันได้ทุกเครื่อง รวมถึงใน Docker
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

from dataset import CLASSES
from features import normalize_df
from inference import class_probabilities, predict_from_landmarks, top_label

CODES = [code for code, _, _ in CLASSES]
PROCESSED = ROOT / "data" / "processed"
BEST_MODEL = ROOT / "models" / "best.joblib"


def _load(split: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"landmarks_{split}.csv")


def _as_landmarks(row) -> list:
    """สร้าง object ที่มี .x .y .z จากแถว CSV ให้หน้าตาเหมือน landmark ที่ MediaPipe คืนมา"""
    return [SimpleNamespace(x=row[f"x{i}"], y=row[f"y{i}"], z=row[f"z{i}"]) for i in range(21)]


def _demo_path(clf, df: pd.DataFrame) -> np.ndarray:
    """ทายทีละแถวผ่าน inference.py แบบเดียวกับที่ demo ทำกับแต่ละเฟรม"""
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
    """ลำดับ classes_ ต่างจาก dataset.CLASSES อยู่แล้ว ถ้ามีการแปลง index เป็นชื่อคลาสผิดลำดับ เทสนี้จะ fail"""
    assert list(knn.classes_) != CODES, "ลำดับตรงกันโดยบังเอิญ เทสนี้จะจับ bug ลำดับ label ไม่ได้"
    X, _, _ = normalize_df(_load("test").head(60))
    for x in X:
        probs = class_probabilities(knn, x)
        assert set(probs) == set(CODES)
        assert top_label(probs)[0] == knn.predict(x.reshape(1, -1))[0]


def test_demo_path_matches_offline_predict(knn):
    """ทายทีละเฟรมแบบ demo ต้องได้ผลเท่ากับ predict ทั้ง batch ทุกแถวของ test split"""
    df = _load("test")
    X, y, _ = normalize_df(df)
    preds = _demo_path(knn, df)
    assert len(preds) == len(y)
    assert (preds == knn.predict(X)).all()
    assert (preds == y).mean() > 0.90


@pytest.mark.skipif(not BEST_MODEL.exists(), reason="models/best.joblib ไม่อยู่ใน git ต้อง train ก่อน")
def test_best_model_demo_path_matches_offline_eval():
    """เหมือนเทสก่อนหน้าแต่ใช้ models/best.joblib ตัวจริง

    ยอมให้ต่างจาก predict ได้ไม่เกิน 1% เพราะ SVC คำนวณ predict_proba ด้วย Platt scaling
    ทำให้ argmax ของ probability ไม่ตรงกับ predict ได้บางแถว
    """
    import joblib

    clf = joblib.load(BEST_MODEL)
    df = _load("test")
    X, y, _ = normalize_df(df)
    preds = _demo_path(clf, df)
    assert (preds == clf.predict(X)).mean() >= 0.99
    assert (preds == y).mean() >= 0.95
