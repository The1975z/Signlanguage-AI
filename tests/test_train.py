"""Tests for the data preparation and model setup in train.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from train import augment, build_pipelines


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_mlp_pipeline_fits_string_labels():
    """sklearn 1.6.1 crashes when early_stopping is used with string labels, so the MLP in train.py must keep it off."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 63))
    y = np.array([f"class_{i % 15}" for i in range(150)])
    pipe = build_pipelines()["mlp"].set_params(clf__max_iter=30)

    assert not pipe.named_steps["clf"].early_stopping
    pipe.fit(X, y)
    assert set(pipe.predict(X)) <= set(y)


def test_augment_appends_mirrored_copy_with_same_labels():
    """augment must append an x-mirrored copy after the original rows, keeping y, z and the label order intact."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 63))
    y = np.array([f"class_{i % 5}" for i in range(20)])
    x_idx = list(range(0, 63, 3))
    yz_idx = [i for i in range(63) if i % 3]

    Xa, ya = augment(X, y)

    assert Xa.shape == (40, 63)
    np.testing.assert_array_equal(Xa[:20], X)
    np.testing.assert_array_equal(Xa[20:, x_idx], -X[:, x_idx])
    np.testing.assert_array_equal(Xa[20:, yz_idx], X[:, yz_idx])
    np.testing.assert_array_equal(ya, np.concatenate([y, y]))
