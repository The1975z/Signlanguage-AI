"""Tests for the dataset layout helpers in config.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from config import SPLITS, get_image_paths


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_get_image_paths_matches_jpg_in_any_case(tmp_path):
    """The dataset mixes .jpg and .JPG; both must be found and other files ignored."""
    images = tmp_path / SPLITS["test"] / "KO_KAI" / "Images (JPEG)"
    for name in ("a.jpg", "b.JPG", "c.xml"):
        _touch(images / name)

    assert [p.name for p in get_image_paths(tmp_path, "test", "KO_KAI")] == ["a.jpg", "b.JPG"]


def test_get_image_paths_reads_the_flat_unseen_layout(tmp_path):
    """Unseen keeps its images directly in the class folder, without an Images (JPEG) subfolder."""
    folder = tmp_path / SPLITS["unseen"] / "KO_KAI"
    for name in ("k1.jpg", "k2.JPG"):
        _touch(folder / name)

    assert [p.name for p in get_image_paths(tmp_path, "unseen", "KO_KAI")] == ["k1.jpg", "k2.JPG"]
