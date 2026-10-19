"""Matplotlib setup shared by the report scripts.

Importing this module selects the non-interactive Agg backend and the first installed Thai-capable
font, before any figure is created, so tick labels can show Thai letters.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import CLASSES

THAI_FONTS = ("Tahoma", "Thonburi", "Noto Sans Thai", "Loma")


def _use_thai_font() -> bool:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in THAI_FONTS:
        if name in installed:
            plt.rcParams["font.family"] = name
            return True
    return False


HAS_THAI_FONT = _use_thai_font()


def class_labels(codes) -> list:
    """Return Thai letters for the class codes when a Thai font is installed, otherwise the codes."""
    letters = {code: letter for code, letter, _ in CLASSES}
    return [letters[c] for c in codes] if HAS_THAI_FONT else list(codes)
