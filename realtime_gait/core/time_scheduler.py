"""Wall-clock scheduler — independent of ingest FPS / network jitter."""

from __future__ import annotations

import time
from typing import Optional


class TimeScheduler:
    """
    Gate pipeline stages by elapsed time, not frame count.

    Ingest FPS may vary (RTSP jitter, PyAV burst decode); only wall clock matters.
    """

    def __init__(
        self,
        process_interval_ms: float = 33.3,
        seg_interval_ms: float = 66.7,
    ):
        self.process_interval_ms = max(1.0, process_interval_ms)
        self.seg_interval_ms = max(1.0, seg_interval_ms)
        self._last_process_ts: Optional[float] = None
        self._last_seg_ts: Optional[float] = None
        self._ingest_count = 0
        self._process_count = 0

    def on_ingest(self, now: Optional[float] = None) -> tuple[bool, str]:
        """
        Returns:
            (should_process, skip_reason)
        """
        now = now if now is not None else time.perf_counter()
        self._ingest_count += 1

        if self._last_process_ts is None:
            self._last_process_ts = now
            self._process_count += 1
            return True, ""

        elapsed_ms = (now - self._last_process_ts) * 1000.0
        if elapsed_ms < self.process_interval_ms:
            return False, f"wait_process ({elapsed_ms:.0f}/{self.process_interval_ms:.0f}ms)"

        self._last_process_ts = now
        self._process_count += 1
        return True, ""

    def should_segment(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.perf_counter()
        if self._last_seg_ts is None:
            self._last_seg_ts = now
            return True
        elapsed_ms = (now - self._last_seg_ts) * 1000.0
        if elapsed_ms >= self.seg_interval_ms:
            self._last_seg_ts = now
            return True
        return False

    @property
    def ingest_count(self) -> int:
        return self._ingest_count

    @property
    def process_count(self) -> int:
        return self._process_count

    @property
    def target_process_hz(self) -> float:
        return 1000.0 / self.process_interval_ms

    def reset(self) -> None:
        self._last_process_ts = None
        self._last_seg_ts = None
        self._ingest_count = 0
        self._process_count = 0
