"""Train and select the classifier.

Runs experiments E2 (baseline), E3 (GridSearchCV) and E4 (mirror augmentation) for KNN, SVM, RF
and MLP, scores every model on the test split, and saves the one with the best macro-F1 as
models/best.joblib. All scores go to results/experiments.csv, which is overwritten on each run.
The unseen split is never used here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import mirror, normalize_df

SEED       = 42
CV_FOLDS   = 5
OUT_DIR    = ROOT / "models"
RESULT_CSV = ROOT / "results" / "experiments.csv"
RESULT_COLUMNS = ["experiment", "model", "augment", "tuned", "split", "accuracy", "macro_f1", "train_sec"]


class Experiment(NamedTuple):
    tag: str
    use_mirror: bool
    use_grid_search: bool


EXPERIMENTS = [
    Experiment("E2_baseline", use_mirror=False, use_grid_search=False),
    Experiment("E3_tuned", use_mirror=False, use_grid_search=True),
    Experiment("E4_mirror_aug", use_mirror=True, use_grid_search=False),
]


def load_split(split_key: str) -> tuple:
    """Load data/processed/landmarks_<split_key>.csv and return normalized ``(X, y)`` for the usable rows."""
    path = ROOT / "data" / "processed" / f"landmarks_{split_key}.csv"
    df   = pd.read_csv(path)
    X, y, _ = normalize_df(df)
    return X, y


def augment(X: np.ndarray, y: np.ndarray) -> tuple:
    """Append a left-right mirrored copy of ``X`` after the original rows, repeating ``y`` in the same order."""
    Xm = mirror(X)
    return np.vstack([X, Xm]), np.concatenate([y, y])


def build_pipelines() -> dict[str, Pipeline]:
    """Return untrained baseline pipelines keyed by model name.

    The MLP keeps ``early_stopping`` off because sklearn 1.6.1 crashes when it is combined with
    string labels.
    """
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
    "mlp": {"clf__hidden_layer_sizes": [(64,), (128, 64), (256, 128)], "clf__alpha": [1e-4, 1e-3],
            "clf__max_iter": [300]},
}


def tune(name: str, pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Grid-search ``PARAM_GRIDS[name]`` with stratified k-fold CV on the given training data only.

    Returns:
        The best pipeline by mean CV macro-F1, refitted on all of ``X``.
    """
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    gs  = GridSearchCV(pipe, PARAM_GRIDS[name], cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
    gs.fit(X, y)
    print(f"  best params: {gs.best_params_}  cv-macro-F1: {gs.best_score_:.4f}")
    return gs.best_estimator_


def save_model(pipe: Pipeline, name: str, experiment: str):
    """Save ``pipe`` to models/<name>.joblib with <name>_meta.json next to it.

    ``classes`` in the metadata is ``pipe.classes_``, which is the column order of ``predict_proba``.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT_DIR / f"{name}.joblib")
    meta = {
        "experiment": experiment,
        "estimator":  type(pipe.named_steps["clf"]).__name__,
        "params":     pipe.named_steps["clf"].get_params(),
        "classes":    [str(c) for c in pipe.classes_],
        "features":   {"n_features": 63, "normalization": "features.normalize"},
        "sklearn":    sklearn.__version__,
        "seed":       SEED,
    }
    (OUT_DIR / f"{name}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))


def append_result(row: dict):
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if RESULT_CSV.exists():
        df_new.to_csv(RESULT_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RESULT_CSV, index=False)


def main():
    print("โหลด train / test ...")
    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")
    X_aug, y_aug = augment(X_tr, y_tr)
    print(f"  train: {X_tr.shape}  train+mirror: {X_aug.shape}  test: {X_te.shape}")

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=RESULT_COLUMNS).to_csv(RESULT_CSV, index=False)

    best_f1, best_pipe, best_name = -1.0, None, ""
    for exp in EXPERIMENTS:
        print(f"\n=== {exp.tag} ===")
        X_fit, y_fit = (X_aug, y_aug) if exp.use_mirror else (X_tr, y_tr)
        for name, pipe in build_pipelines().items():
            t0 = perf_counter()
            model = tune(name, pipe, X_fit, y_fit) if exp.use_grid_search else pipe.fit(X_fit, y_fit)
            elapsed = perf_counter() - t0
            y_pred = model.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            mf1 = f1_score(y_te, y_pred, average="macro")
            print(f"  {name:<4}  acc={acc:.4f}  macro-F1={mf1:.4f}  ({elapsed:.1f}s)")
            append_result({
                "experiment": exp.tag, "model": name,
                "augment": exp.use_mirror, "tuned": exp.use_grid_search,
                "split": "test", "accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
                "train_sec": round(elapsed, 2),
            })
            if mf1 > best_f1:
                best_f1, best_pipe, best_name = mf1, model, f"{exp.tag.split('_')[0]}_{name}"

    print(f"\nบันทึก best model: {best_name}  macro-F1={best_f1:.4f}")
    save_model(best_pipe, "best", best_name)


if __name__ == "__main__":
    main()
