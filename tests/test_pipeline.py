import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from features import mirror

RNG = np.random.default_rng(42)
N_CLASSES = 15
CLASSES = [f"class_{i}" for i in range(N_CLASSES)]


def _make_dataset(n: int = 300):
    X = RNG.standard_normal((n, 63))
    y = np.array(CLASSES * (n // N_CLASSES + 1))[:n]
    return X, y


def test_mlp_string_labels_no_early_stopping():
    X, y = _make_dataset()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(32,), max_iter=50, random_state=42)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds).issubset(set(CLASSES))


def test_mirror_doubles_dataset():
    X, y = _make_dataset(60)
    Xm, ym = mirror(X), y.copy()
    X2 = np.vstack([X, Xm])
    y2 = np.concatenate([y, ym])
    assert X2.shape == (120, 63)
    assert y2.shape == (120,)


def test_normalize_df_shape():
    pytest.importorskip("pandas")
    import pandas as pd
    from features import LM_COLS, normalize_df

    rows = []
    for i in range(10):
        row = {"split": "train", "label": "ko_kai", "detected": 1,
               "img_w": 1280, "img_h": 720}
        pts = RNG.uniform(0.1, 0.9, (21, 3))
        pts[9] = pts[0] + np.array([0.1, 0.2, 0.01])
        for j in range(21):
            row[f"x{j}"] = pts[j, 0]
            row[f"y{j}"] = pts[j, 1]
            row[f"z{j}"] = pts[j, 2]
        rows.append(row)
    df = pd.DataFrame(rows)
    X, y, mask = normalize_df(df)
    assert X.shape == (10, 63)
    assert y.shape == (10,)
    assert mask.sum() == 10
