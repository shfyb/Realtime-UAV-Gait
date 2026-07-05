"""Save annotated pipeline frames and session performance summary."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

import cv2

from ..core.types import FrameResult


class SnapshotSaver:
    """Auto-save visualization frames when tracks or gallery IDs appear."""

    def __init__(
        self,
        save_dir: Path,
        *,
        interval_sec: float = 2.0,
        save_on_track: bool = True,
        save_on_gallery: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.interval_sec = max(0.0, interval_sec)
        self.save_on_track = save_on_track
        self.save_on_gallery = save_on_gallery
        self._last_save_ts = 0.0
        self._saved_count = 0
        self._seen_gallery: Set[str] = set()
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default_session_dir(cls, base: Optional[Path] = None) -> Path:
        root = base or Path("output/stream_screenshots")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / stamp

    @property
    def saved_count(self) -> int:
        return self._saved_count

    def maybe_save(self, vis, result: FrameResult, *, reader_seq: int = 0) -> Optional[Path]:
        if not result.processed:
            return None

        gallery_ids = [t.gallery_id for t in result.tracks if t.gallery_id]
        new_gallery = [gid for gid in gallery_ids if gid not in self._seen_gallery]
        has_track = len(result.tracks) > 0
        now = time.time()

        force = self.save_on_gallery and bool(new_gallery)
        if force:
            for gid in new_gallery:
                self._seen_gallery.add(gid)

        if not force:
            if not (self.save_on_track and has_track):
                return None
            if self.interval_sec > 0 and (now - self._last_save_ts) < self.interval_sec:
                return None

        labels = [t.gallery_id or f"T{t.track_id}" for t in result.tracks]
        tag = "recognized" if force else "track"
        if new_gallery:
            tag = f"recognized_{new_gallery[0]}"
        elif labels:
            tag = f"track_{labels[0]}"

        fname = f"{self._saved_count:06d}_{tag}_f{result.frame_index}_seq{reader_seq}.jpg"
        path = self.save_dir / fname
        cv2.imwrite(str(path), vis)
        self._saved_count += 1
        self._last_save_ts = now
        return path


class PerfCollector:
    """Aggregate pipeline timings for end-of-session report."""

    def __init__(self):
        self._n = 0
        self._det: list[float] = []
        self._trk: list[float] = []
        self._seg: list[float] = []
        self._gait: list[float] = []
        self._total: list[float] = []
        self._trk_fps: list[float] = []
        self._lag_ms: list[float] = []
        self._show_fps: list[float] = []
        self._track_counts: list[int] = []

    def record(
        self,
        result: FrameResult,
        *,
        stream_lag_ms: float = 0.0,
        show_fps: float = 0.0,
    ) -> None:
        if not result.processed:
            return
        self._n += 1
        t = result.timings_ms
        self._det.append(float(t.get("detect", 0)))
        self._trk.append(float(t.get("track", 0)))
        self._seg.append(float(t.get("segment", 0)))
        self._gait.append(float(t.get("gait", 0)))
        self._total.append(float(t.get("total", 0)))
        self._trk_fps.append(float(t.get("track_fps", 0)))
        self._lag_ms.append(stream_lag_ms)
        if show_fps > 0:
            self._show_fps.append(show_fps)
        self._track_counts.append(len(result.tracks))

    def _avg(self, xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def summary(self) -> dict:
        proc_hz = self._avg(self._trk_fps)
        return {
            "processed_frames": self._n,
            "avg_detect_ms": round(self._avg(self._det), 1),
            "avg_track_ms": round(self._avg(self._trk), 1),
            "avg_segment_ms": round(self._avg(self._seg), 1),
            "avg_gait_ms": round(self._avg(self._gait), 1),
            "avg_total_ms": round(self._avg(self._total), 1),
            "avg_track_hz": round(proc_hz, 2),
            "avg_stream_lag_ms": round(self._avg(self._lag_ms), 1),
            "avg_display_fps": round(self._avg(self._show_fps), 2),
            "avg_tracks": round(self._avg(self._track_counts), 2),
            "effective_process_hz": round(1000.0 / self._avg(self._total), 2) if self._avg(self._total) > 0 else 0,
        }

    def write_report(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=False)
        return path
