"""Tests for the labelled result tables written by evaluate.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from evaluate import confusion_table, markdown_table, metrics_row, per_class_table


def test_per_class_table_attaches_scores_to_the_right_class():
    """Rows follow config.CLASSES and carry their own scores, whatever order scikit-learn sorts labels in."""
    y_true = np.array(["to_tao", "to_tao", "bo_baimai", "bo_baimai"])
    y_pred = np.array(["to_tao", "bo_baimai", "bo_baimai", "bo_baimai"])

    table = per_class_table(y_true, y_pred).set_index("code")

    assert list(table["letter"][:5]) == ["ก", "ด", "ต", "น", "บ"]
    assert table.loc["to_tao", ["precision", "recall", "support"]].tolist() == [1.0, 0.5, 2]
    assert table.loc["bo_baimai", ["precision", "recall", "support"]].tolist() == [round(2 / 3, 4), 1.0, 2]
    assert table.loc["ko_kai", "support"] == 0


def test_confusion_table_counts_true_rows_against_predicted_columns():
    """Cell (true ก, predicted ด) holds the ก images that were predicted as ด."""
    y_true = np.array(["ko_kai", "ko_kai", "do_dek"])
    y_pred = np.array(["ko_kai", "do_dek", "do_dek"])

    cm = confusion_table(y_true, y_pred)

    assert list(cm.index[:2]) == ["ก", "ด"] and list(cm.columns[:2]) == ["ก", "ด"]
    assert (cm.loc["ก", "ก"], cm.loc["ก", "ด"], cm.loc["ด", "ก"], cm.loc["ด", "ด"]) == (1, 1, 0, 1)
    assert cm.to_numpy().sum() == 3


def test_metrics_row_counts_undetected_images_as_wrong():
    """Perfect predictions on 10 detected images out of 12 give accuracy 1.0 but end-to-end 10/12."""
    y = np.array(["ko_kai"] * 9 + ["do_dek"])

    metrics = metrics_row("test", "E3_svm", y, y.copy(), images=12, detected=10)

    assert metrics["accuracy"] == 1.0
    assert metrics["end_to_end_accuracy"] == round(10 / 12, 4)


def test_markdown_table_renders_header_rule_and_rows():
    md = markdown_table(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))

    assert md.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |", "| 2 | y |"]
