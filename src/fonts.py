"""Find a Thai-capable TrueType font for Pillow on macOS, Windows or Linux."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

THAI_FONT_FILES = (
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Thonburi.ttc",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/LeelawUI.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
)


@lru_cache(maxsize=None)
def thai_font(size: int):
    """Return the first installed font from ``THAI_FONT_FILES`` at ``size`` pixels.

    Falls back to Pillow's default font, which cannot draw Thai, so the demos still start on a
    machine without any of these fonts. Fonts are cached per size because the demos draw every frame.
    """
    for path in THAI_FONT_FILES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)
