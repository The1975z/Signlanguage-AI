"""Step 7: final metrics, per-class tables and confusion matrices for the saved model.

``--splits`` picks what to score and defaults to Test and Unseen. Unseen is the final hold-out:
score it once, after the model has been chosen on Test.

Writes, for each split, results/eval_<split>_metrics.csv, eval_<split>_per_class.csv,
eval_<split>_confusion.csv, eval_<split>_confusion.png and eval_<split>.md, plus
results/experiments_macro_f1.png from the experiment log. CSV files are UTF-8 with a BOM so that
Excel shows the Thai letters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CLASSES
from features import normalize_df
from plotting import class_labels, plt

CLASS_CODES = [code for code, _, _ in CLASSES]
CLASS_CHARS = {code: char for code, char, _ in CLASSES}
RESULTS = ROOT / "results"


def load_model(name: str = "best") -> tuple:
    """Return the saved pipeline and the experiment it came from, read from <name>_meta.json."""
    model = joblib.load(ROOT / "models" / f"{name}.joblib")
    meta = json.loads((ROOT / "models" / f"{name}_meta.json").read_text())
    return model, meta["experiment"]


def load_split(split_key: str) -> tuple:
    """Return ``(X, y, images, detected)`` for one split; ``X`` and ``y`` cover the detected rows only."""
    df = pd.read_csv(ROOT / "data" / "processed" / f"landmarks_{split_key}.csv")
    X, y, _ = normalize_df(df)
    return X, y, len(df), int(df["detected"].sum())


def metrics_row(split: str, experiment: str, y_true, y_pred, images: int, detected: int) -> dict:
    """Headline metrics; ``end_to_end_accuracy`` also counts every undetected image as a wrong answer."""
    correct = int((np.asarray(y_pred) == np.asarray(y_true)).sum())
    return {
        "split": split, "model": experiment, "images": images, "detected": detected,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "end_to_end_accuracy": round(correct / images, 4),
        "macro_p": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_r": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    """Precision, recall, F1 and support for every class, in ``config.CLASSES`` order.

    ``labels`` is always passed: without it scikit-learn orders the rows by sorted label, and names
    attached by position end up on the wrong classes.
    """
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_CODES, zero_division=0)
    return pd.DataFrame({
        "code": CLASS_CODES, "letter": [CLASS_CHARS[c] for c in CLASS_CODES],
        "precision": p.round(4), "recall": r.round(4), "f1": f.round(4), "support": s,
    })


def confusion_table(y_true, y_pred) -> pd.DataFrame:
    """Counts with true classes as rows and predicted classes as columns, labelled by Thai letter."""
    letters = [CLASS_CHARS[c] for c in CLASS_CODES]
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_CODES)
    return pd.DataFrame(cm, index=pd.Index(letters, name="true"), columns=pd.Index(letters, name="predicted"))


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table without the optional ``tabulate`` dependency."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def plot_confusion(cm: pd.DataFrame, path: Path, title: str):
    """Save the confusion matrix as a heatmap with the non-zero counts written in the cells."""
    labels = class_labels(CLASS_CODES)
    counts = cm.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(counts, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(counts):
        if value:
            ax.text(j, i, value, ha="center", va="center", fontsize=8,
                    color="white" if value > counts.max() / 2 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_experiments(log: Path, path: Path):
    """Save a grouped bar chart of Test macro-F1 per model and experiment from the experiment log."""
    table = pd.read_csv(log).pivot(index="model", columns="experiment", values="macro_f1")
    ax = table.plot.bar(figsize=(10, 4), rot=0, ylim=(max(0.0, table.min().min() - 0.05), 1.0))
    ax.set_ylabel("macro-F1 (Test)")
    ax.set_title("Experiments E1–E4")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def evaluate_split(model, experiment: str, split: str) -> dict:
    """Score one split, write its CSV, markdown and heatmap files, and return the headline metrics."""
    X, y, images, detected = load_split(split)
    y_pred = model.predict(X)
    metrics = metrics_row(split, experiment, y, y_pred, images, detected)
    per_class = per_class_table(y, y_pred)
    cm = confusion_table(y, y_pred)

    stem = RESULTS / f"eval_{split}"
    pd.DataFrame([metrics]).to_csv(f"{stem}_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(f"{stem}_per_class.csv", index=False, encoding="utf-8-sig")
    cm.to_csv(f"{stem}_confusion.csv", encoding="utf-8-sig")
    plot_confusion(cm, Path(f"{stem}_confusion.png"), f"{split} — {experiment}")
    Path(f"{stem}.md").write_text(
        f"# {split} — {experiment}\n\n{markdown_table(pd.DataFrame([metrics]))}\n\n"
        f"## Per class\n\n{markdown_table(per_class)}\n",
        encoding="utf-8",
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", choices=["test", "unseen"], default=["test", "unseen"])
    args = parser.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    model, experiment = load_model()
    for split in args.splits:
        m = evaluate_split(model, experiment, split)
        print(f"{split:<6} acc={m['accuracy']:.4f}  end-to-end={m['end_to_end_accuracy']:.4f}  "
              f"macro-F1={m['macro_f1']:.4f}  ({m['detected']}/{m['images']} detected)")
    plot_experiments(RESULTS / "experiments.csv", RESULTS / "experiments_macro_f1.png")
    print("บันทึก results/eval_<split>_* และ results/experiments_macro_f1.png")


if __name__ == "__main__":
    main()
