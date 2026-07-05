from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class SilEntry:
    ts: float
    sil: np.ndarray


class TrackSilhouetteBuffer:
    """Per track_id time-stamped silhouette buffer."""

    def __init__(self, max_duration_sec: float = 6.0):
        self.max_duration_sec = max_duration_sec
        self._entries: Deque[SilEntry] = deque()
        self._last_seen_ts: float = 0.0

    def append(self, sil: np.ndarray, ts: Optional[float] = None) -> None:
        ts = ts if ts is not None else time.perf_counter()
        self._last_seen_ts = ts
        self._entries.append(SilEntry(ts=ts, sil=sil))
        self._prune_old(ts)

    def touch(self, ts: float) -> None:
        """Refresh last-seen time (track active, e.g. between seg frames)."""
        self._last_seen_ts = ts
        if self._entries:
            self._entries[-1] = SilEntry(ts=ts, sil=self._entries[-1].sil)

    def _prune_old(self, now: float) -> None:
        while self._entries and (now - self._entries[0].ts) > self.max_duration_sec:
            self._entries.popleft()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def silhouettes(self) -> List[np.ndarray]:
        return [e.sil for e in self._entries]

    @property
    def last_seen_ts(self) -> float:
        return self._last_seen_ts

    @property
    def span_sec(self) -> float:
        if len(self._entries) < 2:
            return 0.0
        return self._entries[-1].ts - self._entries[0].ts

    def ready(self, min_count: int, min_duration_sec: float) -> bool:
        if len(self._entries) < min_count:
            return False
        return self.span_sec >= min_duration_sec

    def silhouettes_since(self, since_ts: float) -> List[np.ndarray]:
        return [e.sil for e in self._entries if e.ts >= since_ts]

    def stats_since(self, since_ts: float) -> tuple[int, float]:
        entries = [e for e in self._entries if e.ts >= since_ts]
        if not entries:
            return 0, 0.0
        span = entries[-1].ts - entries[0].ts if len(entries) >= 2 else 0.0
        return len(entries), span

    def ready_since(self, since_ts: float, min_count: int, min_duration_sec: float) -> bool:
        count, span = self.stats_since(since_ts)
        return count >= min_count and span >= min_duration_sec


class TrackBufferManager:
    """Keep buffers for briefly lost tracks (grace aligned with ByteTrack lost window)."""

    def __init__(self, max_duration_sec: float = 6.0, grace_sec: float = 1.0):
        self.max_duration_sec = max_duration_sec
        self.grace_sec = grace_sec
        self._buffers: Dict[int, TrackSilhouetteBuffer] = {}

    def get(self, track_id: int) -> TrackSilhouetteBuffer:
        if track_id not in self._buffers:
            self._buffers[track_id] = TrackSilhouetteBuffer(self.max_duration_sec)
        return self._buffers[track_id]

    def touch_active(self, track_ids: List[int], now: float) -> None:
        for tid in track_ids:
            if tid in self._buffers:
                self._buffers[tid].touch(now)

    def ids_in_grace(self, now: float) -> List[int]:
        """Track ids whose buffer should be kept (active or within grace)."""
        kept = []
        for tid, buf in self._buffers.items():
            if now - buf.last_seen_ts <= self.grace_sec:
                kept.append(tid)
        return kept

    def prune(self, active_ids: List[int], now: float) -> None:
        active = set(active_ids)
        for tid in list(self._buffers.keys()):
            if tid in active:
                continue
            buf = self._buffers[tid]
            if now - buf.last_seen_ts > self.grace_sec:
                del self._buffers[tid]
