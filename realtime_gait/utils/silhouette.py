"""In-memory silhouette crop / gait input packing (no disk IO)."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def crop_and_pad(
    frame_bgr: np.ndarray,
    bbox_xyxy: Sequence[float],
    pad_ratio: float = 0.1,
    out_size: int = 192,
) -> np.ndarray:
    """Crop person region, square-pad white background, resize to out_size."""
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError("Invalid bbox for crop")

    x1n = max(0, int(x1 - pad_ratio * w))
    x2n = min(w_img, int(x2 + pad_ratio * w))
    y1n = max(0, int(y1 - pad_ratio * h))
    y2n = min(h_img, int(y2 + pad_ratio * h))
    crop = frame_bgr[y1n:y2n, x1n:x2n]
    if crop.size == 0:
        raise ValueError("Empty crop")

    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    ox = (side - cw) // 2
    oy = (side - ch) // 2
    canvas[oy : oy + ch, ox : ox + cw] = crop
    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def mask_to_sil64(mask: np.ndarray, img_size: int = 64) -> np.ndarray:
    """Binary/gray mask -> uint8 silhouette for GaitBase (GREW-style, 64px height)."""
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask
    gray = gray.astype(np.uint8)
    ratio = gray.shape[1] / max(gray.shape[0], 1)
    resized = cv2.resize(
        gray,
        (max(1, int(img_size * ratio)), img_size),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized


def silhouettes_to_gait_input(
    sils: Sequence[np.ndarray],
    track_key: str,
    sample_frames: int = 30,
) -> Tuple[list, list, list, list, np.ndarray]:
    """
    Pack silhouette list into OpenGait demo inputs tuple.

    Returns:
        ([[seq_array]], [lab], [typ], [view], seqL)
    """
    if len(sils) < 5:
        raise ValueError(f"Need >=5 silhouettes, got {len(sils)}")

    if len(sils) > sample_frames:
        idx = np.linspace(0, len(sils) - 1, sample_frames, dtype=int)
        picked = [sils[i] for i in idx]
    else:
        picked = list(sils)

    seq = np.asarray(picked, dtype=np.uint8)
    lab = track_key.split("-")[0] if "-" in track_key else track_key
    return (
        [[seq]],
        [lab],
        ["undefined"],
        ["undefined"],
        np.array([[len(picked)]]),
    )
