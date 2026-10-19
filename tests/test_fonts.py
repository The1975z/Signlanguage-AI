"""Tests for the cross-platform Thai font lookup in fonts.py."""
import sys
from pathlib import Path

import matplotlib
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import fonts

BUNDLED_TTF = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")


@pytest.fixture(autouse=True)
def fresh_cache():
    fonts.thai_font.cache_clear()
    yield
    fonts.thai_font.cache_clear()


def test_thai_font_uses_the_first_installed_candidate(monkeypatch):
    """Missing candidates are skipped and the first file that exists is loaded."""
    monkeypatch.setattr(fonts, "THAI_FONT_FILES", ("/missing/Tahoma.ttf", BUNDLED_TTF))

    assert fonts.thai_font(20).path == BUNDLED_TTF


def test_thai_font_falls_back_when_no_candidate_exists(monkeypatch):
    """A machine with none of the fonts (for example Windows before) must still get a usable font."""
    monkeypatch.setattr(fonts, "THAI_FONT_FILES", ("/missing/Tahoma.ttf",))

    assert fonts.thai_font(20) is not None
