from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import CLASSES
from features import normalize_df

CLASS_CODES = [code for code, _, _ in CLASSES]
CLASS_CHARS = {code: char for code, char, _ in CLASSES}


def load_model(name: str = "best"):
    path = ROOT / "models" / f"{name}.joblib"
    return joblib.load(path)


def load_split(split_key: str) -> tuple:
    path = ROOT / "data" / "processed" / f"landmarks_{split_key}.csv"
    df   = pd.read_csv(path)
    total  = len(df)
    detected = int(df["detected"].sum())
    X, y, _ = normalize_df(df)
    return X, y, total, detected


def evaluate(model, X: np.ndarray, y: np.ndarray, total_images: int, detected: int, split_name: str):
    y_pred  = model.predict(X)
    acc_det = accuracy_score(y, y_pred)

    correct = int((y_pred == y).sum())
    e2e_acc = correct / total_images

    macro_p  = precision_score(y, y_pred, average="macro", zero_division=0)
    macro_r  = recall_score(y, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*50}")
    print(f"ผลบน {split_name}")
    print(f"  ภาพทั้งหมด:         {total_images}")
    print(f"  MediaPipe detected: {detected}/{total_images}  ({detected/total_images*100:.1f}%)")
    print(f"  Accuracy (detected only): {acc_det:.4f}")
    print(f"  End-to-end accuracy:      {e2e_acc:.4f}  ({correct}/{total_images})")
    print(f"  Macro-Precision: {macro_p:.4f}")
    print(f"  Macro-Recall:    {macro_r:.4f}")
    print(f"  Macro-F1:        {macro_f1:.4f}")

    print("\n--- Per-class ---")
    report = classification_report(y, y_pred, labels=CLASS_CODES,
                                   target_names=[CLASS_CHARS[c] for c in CLASS_CODES],
                                   zero_division=0)
    print(report)

    cm = confusion_matrix(y, y_pred, labels=CLASS_CODES)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"cm_{split_name}.npy", cm)
    print(f"  confusion matrix → results/cm_{split_name}.npy")

    return {
        "split":        split_name,
        "total":        total_images,
        "detected":     detected,
        "acc_detected": round(acc_det, 4),
        "e2e_acc":      round(e2e_acc, 4),
        "macro_p":      round(macro_p, 4),
        "macro_r":      round(macro_r, 4),
        "macro_f1":     round(macro_f1, 4),
    }


def main():
    model = load_model("best")

    results = []
    for split_key, split_name in [("test", "test"), ("unseen", "unseen")]:
        X, y, total, detected = load_split(split_key)
        results.append(evaluate(model, X, y, total, detected, split_name))

    df = pd.DataFrame(results)
    df.to_csv(ROOT / "results" / "final_results.csv", index=False)
    print("\nบันทึก results/final_results.csv")


if __name__ == "__main__":
    main()
