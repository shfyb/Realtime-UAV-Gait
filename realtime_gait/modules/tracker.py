"""ByteTrack with adaptive dt / frame_rate for unstable process intervals."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from ..config.settings import TrackerConfig
from ..utils.paths import setup_import_paths

setup_import_paths()

from tracker.byte_tracker import BYTETracker, STrack  # noqa: E402


def _set_kalman_dt(kf, dt: float) -> None:
    """Scale constant-velocity Kalman step (ByteTrack assumes dt=1 per frame)."""
    ndim = 4
    dt = float(max(0.05, min(dt, 8.0)))
    kf._motion_mat = np.eye(2 * ndim)
    for i in range(ndim):
        kf._motion_mat[i, ndim + i] = dt


class ByteTrackEngine:
    """
    ByteTrack wrapper with wall-clock adaptive parameters.

    - Kalman predict step uses dt_kf = dt_sec / ema_dt (≈1 when steady, >1 when gap is long)
    - max_time_lost derived from max_time_lost_sec / ema_dt (real seconds, not fixed frame count)
    """

    def __init__(self, cfg: TrackerConfig, nominal_hz: Optional[float] = None):
        self.cfg = cfg
        hz = nominal_hz or float(cfg.frame_rate or 30)
        self._nominal_hz = max(1.0, hz)
        self._nominal_dt = 1.0 / self._nominal_hz

        self._tracker = BYTETracker(frame_rate=int(round(self._nominal_hz)))
        self._tid_offset: Optional[int] = None

        self._last_ts: Optional[float] = None
        self._ema_dt: float = self._nominal_dt
        self._effective_fps: float = self._nominal_hz
        self._last_dt_kf: float = 1.0
        self._last_track_ms: float = 0.0

        logger.info(
            f"ByteTrack adaptive | nominal {self._nominal_hz:.1f}Hz, "
            f"max_lost={cfg.max_time_lost_sec}s"
        )

    def reset(self) -> None:
        self._tracker = BYTETracker(frame_rate=int(round(self._nominal_hz)))
        self._tid_offset = None
        self._last_ts = None
        self._ema_dt = self._nominal_dt
        self._effective_fps = self._nominal_hz
        self._last_dt_kf = 1.0

    def _adapt_timing(self, wall_ts: float) -> None:
        if self._last_ts is None:
            dt_sec = self._nominal_dt
        else:
            dt_sec = wall_ts - self._last_ts
        self._last_ts = wall_ts

        dt_sec = float(np.clip(dt_sec, self.cfg.dt_min_sec, self.cfg.dt_max_sec))

        alpha = self.cfg.dt_ema_alpha
        self._ema_dt = alpha * dt_sec + (1.0 - alpha) * self._ema_dt
        self._ema_dt = max(self._ema_dt, self.cfg.dt_min_sec)

        self._effective_fps = 1.0 / self._ema_dt
        self._last_dt_kf = dt_sec / self._ema_dt

        lost_frames = int(round(self.cfg.max_time_lost_sec / self._ema_dt))
        lost_frames = max(1, lost_frames)
        self._tracker.max_time_lost = lost_frames
        self._tracker.buffer_size = lost_frames

        dt_kf = self._last_dt_kf
        _set_kalman_dt(self._tracker.kalman_filter, dt_kf)
        _set_kalman_dt(STrack.shared_kalman, dt_kf)

    def update(
        self,
        dets: Optional[np.ndarray],
        img_h: int,
        img_w: int,
        wall_ts: Optional[float] = None,
    ) -> List[Tuple[int, List[float]]]:
        """
        Returns list of (track_id, bbox_xyxy).
        """
        t0 = time.perf_counter()
        now = wall_ts if wall_ts is not None else t0
        self._adapt_timing(now)

        if dets is None or len(dets) == 0:
            filtered = np.empty((0, 5), dtype=np.float32)
        else:
            filtered = dets.astype(np.float32)

        online_targets = self._tracker.update(
            filtered,
            [img_h, img_w],
            [img_h, img_w],
        )

        results: List[Tuple[int, List[float]]] = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = int(t.track_id)
            if self._tid_offset is None:
                self._tid_offset = tid - 1
            tid = tid - self._tid_offset

            w, h = float(tlwh[2]), float(tlwh[3])
            area = w * h
            if area < self.cfg.min_area:
                continue
            ar = w / max(h, 1e-6)
            if ar > self.cfg.max_aspect_ratio or ar < self.cfg.min_aspect_ratio:
                continue

            x1, y1 = float(tlwh[0]), float(tlwh[1])
            x2, y2 = x1 + w, y1 + h
            results.append((tid, [x1, y1, x2, y2]))

        self._last_track_ms = (time.perf_counter() - t0) * 1000.0
        return results

    @property
    def last_track_ms(self) -> float:
        return self._last_track_ms

    @property
    def effective_fps(self) -> float:
        return self._effective_fps

    @property
    def ema_dt_sec(self) -> float:
        return self._ema_dt

    @property
    def last_dt_kf(self) -> float:
        return self._last_dt_kf
