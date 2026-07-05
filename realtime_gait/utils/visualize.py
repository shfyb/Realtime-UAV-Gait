"""Draw tracking + recognition overlay on BGR frame."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from ..core.types import FrameResult, TrackResult
from .person_registry import PersonRegistry
from .text_draw import put_text


def _bbox_label(tr: TrackResult) -> str:
    """On-frame label: ASCII gallery id or T{track_id} (always visible on video)."""
    if tr.gallery_id:
        return tr.gallery_id
    return f"T{tr.track_id}"


def _track_label(tr: TrackResult, registry: Optional[PersonRegistry] = None) -> str:
    if tr.display_name:
        return tr.display_name
    if tr.gallery_id and registry is not None:
        display = registry.get_display_name(tr.gallery_id)
        if display:
            return display
    if tr.gallery_id:
        return tr.gallery_id
    return f"T{tr.track_id}"


def draw_results(
    frame_bgr: np.ndarray,
    frame_result: FrameResult,
    id_colors: Optional[Dict[str, tuple]] = None,
    registry: Optional[PersonRegistry] = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    id_colors = id_colors or {}

    for tr in frame_result.tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
        label = _bbox_label(tr)
        color_key = tr.gallery_id or label
        color = id_colors.get(color_key, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = label
        if tr.distance is not None:
            txt += f" d={tr.distance:.1f}"
        y_text = max(22, y1 - 6)
        put_text(out, txt, (x1, y_text), font_size=22, color_bgr=color, thickness=2)
    return out


def color_for_id(gid: str) -> tuple:
    h = abs(hash(gid)) % 255
    return (37 * h % 255, 17 * h % 255, 29 * h % 255)
