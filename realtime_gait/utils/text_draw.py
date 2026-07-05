"""Draw Unicode (Chinese) text on OpenCV BGR images via Pillow."""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _find_cjk_font() -> str | None:
    candidates = [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyhbd.ttc"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "simhei.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


_FONT_PATH = _find_cjk_font()
_FONTS: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size not in _FONTS:
        if _FONT_PATH:
            _FONTS[size] = ImageFont.truetype(_FONT_PATH, size)
        else:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def has_cjk(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def put_text(
    frame_bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    *,
    font_size: int = 22,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw text; uses Pillow for CJK, OpenCV for ASCII-only."""
    if not text:
        return frame_bgr

    if not has_cjk(text):
        cv2.putText(
            frame_bgr,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size / 32.0,
            color_bgr,
            max(1, thickness),
            cv2.LINE_AA,
        )
        return frame_bgr

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(org, text, font=font, fill=color_rgb)
    painted = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(frame_bgr, painted)
    return frame_bgr
