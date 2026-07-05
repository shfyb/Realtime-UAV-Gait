"""
Low-latency RTSP/SRT stream reader (PyAV).

Outputs numpy BGR uint8 frames via get_latest().
"""

from __future__ import annotations

import os
import threading
import time
from typing import List, Optional, Tuple

import av
import numpy as np

DEFAULT_STREAM_URL = os.getenv("STREAM_URL", "rtsp://127.0.0.1:8554/home")
RTSP_FALLBACK_URL = os.getenv("RTSP_FALLBACK_URL", "rtsp://127.0.0.1:8554/home")

FFMPEG_OPTIONS = {
    "fflags": "nobuffer",
    "flags": "low_delay",
    "analyzeduration": "0",
    "probesize": "32768",
    "rw_timeout": "5000000",
    "max_delay": "0",
    "reorder_queue_size": "0",
}


class LatestFrameReader:
    """Background thread decodes stream; main thread polls latest BGR frame."""

    def __init__(self, url: str, fallback_url: Optional[str] = None):
        self.url = url
        self.url_candidates: List[str] = [url]
        fb = fallback_url if fallback_url is not None else RTSP_FALLBACK_URL
        if fb and fb != url:
            self.url_candidates.append(fb)
        self.active_url_idx = 0

        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_ts = 0.0
        self.running = True
        self.total_read = 0
        self.reconnect_count = 0
        self.consecutive_failures = 0
        self.last_error = ""

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while self.running:
            active_url = self.url_candidates[self.active_url_idx]
            try:
                with av.open(active_url, mode="r", options=FFMPEG_OPTIONS) as container:
                    video_stream = next((s for s in container.streams if s.type == "video"), None)
                    if video_stream is None:
                        raise RuntimeError("流中没有视频轨道")

                    self.consecutive_failures = 0
                    print(f"[stream_reader] opened: {active_url}")

                    for frame in container.decode(video=0):
                        if not self.running:
                            break
                        img = frame.to_ndarray(format="bgr24")
                        now = time.time()
                        with self.lock:
                            self.latest_frame = img
                            self.latest_ts = now
                            self.total_read += 1

                if self.running:
                    raise RuntimeError("解码循环中断")

            except Exception as exc:
                self.consecutive_failures += 1
                self.reconnect_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                wait_s = min(0.2 * self.consecutive_failures, 1.5)
                if len(self.url_candidates) > 1:
                    self.active_url_idx = (self.active_url_idx + 1) % len(self.url_candidates)
                    next_url = self.url_candidates[self.active_url_idx]
                    print(f"[stream_reader] reconnect: {self.last_error}; switch to {next_url}")
                else:
                    print(f"[stream_reader] reconnect: {self.last_error}")
                time.sleep(wait_s)

    def get_latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        """
        Returns:
            frame_bgr: HxWx3 uint8 copy, or None if not ready
            frame_ts: host time when frame was received
            seq: monotonic frame sequence (for dedup)
        """
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0, 0
            return self.latest_frame.copy(), self.latest_ts, self.total_read

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
