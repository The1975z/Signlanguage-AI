from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import CLASS_ID, CLASSES
from features import mirror, normalize_df

SEED      = 42
CV_FOLDS  = 5
OUT_DIR   = ROOT / "models"
RESULT_CSV = ROOT / "results" / "experiments.csv"


def load_split(split_key: str) -> tuple:
    path = ROOT / "data" / "processed" / f"landmarks_{split_key}.csv"
    df   = pd.read_csv(path)
    X, y, _ = normalize_df(df)
    return X, y


def augment(X: np.ndarray, y: np.ndarray) -> tuple:
    Xm = mirror(X)
    return np.vstack([X, Xm]), np.concatenate([y, y])


def build_pipelines() -> dict[str, Pipeline]:
    return {
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=5)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED)),
        ]),
        "rf": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                     random_state=SEED)),
        ]),
    }


PARAM_GRIDS: dict[str, dict] = {
    "knn": {"clf__n_neighbors": [1, 3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
    "svm": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
    "rf":  {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 20]},
    "mlp": {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-3],
            "clf__max_iter": [300]},
}


def tune(name: str, pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    gs  = GridSearchCV(pipe, PARAM_GRIDS[name], cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
    gs.fit(X, y)
    print(f"  best params: {gs.best_params_}  cv-macro-F1: {gs.best_score_:.4f}")
    return gs.best_estimator_


def save_model(pipe: Pipeline, name: str, feature_info: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT_DIR / f"{name}.joblib")
    meta = {
        "model":       name,
        "classes":     [code for code, _, _ in CLASSES],
        "class_order": CLASS_ID,
        "features":    feature_info,
        "sklearn":     __import__("sklearn").__version__,
        "seed":        SEED,
    }
    (OUT_DIR / f"{name}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def append_result(row: dict):
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if RESULT_CSV.exists():
        df_new.to_csv(RESULT_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RESULT_CSV, index=False)


def main():
    from sklearn.metrics import f1_score, accuracy_score

    print("โหลด train / test ...")
    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")
    print(f"  train: {X_tr.shape}  test: {X_te.shape}")

    best_f1   = -1.0
    best_pipe = None
    best_name = ""

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["experiment","model","augment","tuned","split",
                           "accuracy","macro_f1","train_sec"]).to_csv(RESULT_CSV, index=False)

    print("\n=== E2: Baseline (normalized, no augment) ===")
    for name, pipe in build_pipelines().items():
        t0 = perf_counter()
        pipe.fit(X_tr, y_tr)
        elapsed = perf_counter() - t0
        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, average="macro")
        print(f"  {name:<4}  acc={acc:.4f}  macro-F1={mf1:.4f}  ({elapsed:.1f}s)")
        append_result({
            "experiment": "E2_baseline", "model": name,
            "augment": False, "tuned": False,
            "split": "test", "accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
            "train_sec": round(elapsed, 2),
        })
        if mf1 > best_f1:
            best_f1, best_pipe, best_name = mf1, pipe, f"E2_{name}"

    print("\n=== E4: Mirror augmentation ===")
    X_aug, y_aug = augment(X_tr, y_tr)
    print(f"  train after augment: {X_aug.shape}")

    for name, pipe in build_pipelines().items():
        t0 = perf_counter()
        pipe.fit(X_aug, y_aug)
        elapsed = perf_counter() - t0
        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, average="macro")
        print(f"  {name:<4}  acc={acc:.4f}  macro-F1={mf1:.4f}  ({elapsed:.1f}s)")
        append_result({
            "experiment": "E4_mirror_aug", "model": name,
            "augment": True, "tuned": False,
            "split": "test", "accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
            "train_sec": round(elapsed, 2),
        })
        if mf1 > best_f1:
            best_f1, best_pipe, best_name = mf1, pipe, f"E4_{name}"

    print(f"\nบันทึก best model: {best_name}  macro-F1={best_f1:.4f}")
    save_model(best_pipe, "best", {
        "n_features": 63, "landmark_pts": 21,
        "axes": ["x", "y", "z"],
        "normalization": "wrist_origin_scale_by_pt9_xy",
        "aspect_correction": "x_z_multiplied_by_W_H",
        "experiment": best_name,
    })


if __name__ == "__main__":
    main()
