"""Per-track gait recognition state + lightweight silhouette similarity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


def silhouette_similarity(
    current: List[np.ndarray],
    reference: List[np.ndarray],
    sample: int = 8,
) -> float:
    """
    Fast similarity in [0, 1]. 1 = identical, 0 = very different.

    Compares evenly sampled silhouettes via normalized mean absolute error.
    """
    if not current or not reference:
        return 0.0

    def _pick(sils: List[np.ndarray], n: int) -> List[np.ndarray]:
        if len(sils) <= n:
            return list(sils)
        idx = np.linspace(0, len(sils) - 1, n, dtype=int)
        return [sils[i] for i in idx]

    cur = _pick(current, sample)
    ref = _pick(reference, sample)
    n = min(len(cur), len(ref))
    if n == 0:
        return 0.0

    scores = []
    for i in range(n):
        a = cur[i].astype(np.float32)
        b = ref[i].astype(np.float32)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        diff = np.abs(a - b).mean() / 255.0
        scores.append(1.0 - float(diff))

    return float(np.mean(scores))


@dataclass
class TrackRecognitionState:
    track_id: int
    recognized_once: bool = False
    reference_sils: List[np.ndarray] = field(default_factory=list)
    last_gallery_id: Optional[str] = None
    last_distance: Optional[float] = None
    last_recognition_ts: float = 0.0
    last_similarity_check_ts: float = 0.0


class TrackRecognitionManager:
    def __init__(self):
        self._states: Dict[int, TrackRecognitionState] = {}

    def get(self, track_id: int) -> TrackRecognitionState:
        if track_id not in self._states:
            self._states[track_id] = TrackRecognitionState(track_id=track_id)
        return self._states[track_id]

    def prune(self, keep_ids: List[int]) -> None:
        keep = set(keep_ids)
        for tid in [t for t in self._states if t not in keep]:
            del self._states[tid]

    def clear(self) -> None:
        self._states.clear()
