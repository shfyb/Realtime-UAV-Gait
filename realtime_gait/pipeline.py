"""
RealtimeGaitPipeline — wall-clock scheduled (FPS/jitter tolerant).

Architecture:
  DroneYOLO -> ByteTrack -> PP-HumanSeg -> GaitBase (per track_id buffer)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

from .config.settings import PipelineConfig, load_config
from .core.time_scheduler import TimeScheduler
from .core.track_buffer import TrackBufferManager
from .core.track_recognition import TrackRecognitionManager, silhouette_similarity
from .core.types import FrameResult, TrackResult
from .modules.detector import DroneYoloDetector
from .modules.recognizer import GaitBaseRecognizer, GalleryStore
from .modules.segmentor import PPHumanSegSegmentor
from .modules.tracker import ByteTrackEngine
from .utils.paths import setup_import_paths
from .utils.gallery_io import DEFAULT_GALLERY_DIR, get_registry
from .utils.visualize import draw_results

setup_import_paths()


class RealtimeGaitPipeline:
    """
    Stateful streaming pipeline.

    Scheduling uses wall-clock intervals, NOT frame count — safe for unstable RTSP FPS.

    Gait trigger:
      - First: span >= 1.5s AND count >= 15 silhouettes
      - Then every 1s: silhouette similarity vs last reference; if sim < threshold
        OR track_id changed → full GaitBase + gallery match
    """

    def __init__(self, config: Optional[Union[PipelineConfig, str, Path]] = None):
        if config is None:
            self.cfg = load_config()
        elif isinstance(config, PipelineConfig):
            self.cfg = config
        else:
            self.cfg = load_config(config)

        self._timing = self.cfg.timing
        self._recog_cfg = self.cfg.recognizer

        device = self.cfg.device
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA unavailable, falling back to CPU")
                device = "cpu"

        logger.info("Initializing realtime gait pipeline...")
        det = self.cfg.detector
        self.detector = DroneYoloDetector(
            self.cfg.drone_yolo,
            device=device,
            conf=det.conf,
            iou=det.iou,
            pedestrian_class_id=det.pedestrian_class_id,
        )
        self.tracker = ByteTrackEngine(
            self.cfg.tracker,
            nominal_hz=self._timing.target_process_hz,
        )
        self.segmentor = PPHumanSegSegmentor(self.cfg.seg_model, self.cfg.segmentor)
        self.recognizer = GaitBaseRecognizer(self.cfg, self._recog_cfg)

        self.scheduler = TimeScheduler(
            process_interval_ms=self._timing.process_interval_ms,
            seg_interval_ms=self._timing.seg_interval_ms,
        )
        grace = self.cfg.tracker.max_time_lost_sec
        self.buffers = TrackBufferManager(
            self._timing.sil_buffer_duration_sec,
            grace_sec=grace,
        )
        self._recog_mgr = TrackRecognitionManager()
        self._frame_index = 0
        self.gallery_dir = DEFAULT_GALLERY_DIR
        self._registry = get_registry(self.gallery_dir)
        self._gallery_recognition_enabled = False

        logger.info(
            f"Pipeline ready | time-based: process ~{self._timing.target_process_hz:.1f}Hz, "
            f"seg ~{self._timing.target_seg_hz:.1f}Hz, "
            f"first gait: {self._timing.min_sil_count} sils / {self._timing.min_sil_duration_sec}s, "
            f"re-check every {self._timing.recognition_interval_sec}s "
            f"(sim<{self._recog_cfg.sil_similarity_threshold})"
        )

    @property
    def gallery(self) -> GalleryStore:
        return self.recognizer.gallery

    def reset(self) -> None:
        self.tracker.reset()
        self.scheduler.reset()
        grace = self.cfg.tracker.max_time_lost_sec
        self.buffers = TrackBufferManager(
            self._timing.sil_buffer_duration_sec,
            grace_sec=grace,
        )
        self._recog_mgr.clear()
        self._frame_index = 0

    def register_gallery_embedding(self, person_id: str, embedding) -> None:
        self.gallery.register_embedding(person_id, embedding)
        self.invalidate_gallery_recognition()

    def load_gallery_from_features(self, gallery_feat: dict) -> None:
        self.gallery.register_from_probe_dict(gallery_feat)
        self.invalidate_gallery_recognition()
        logger.info(f"Gallery loaded: {len(self.gallery)} identities")

    def invalidate_gallery_recognition(self) -> None:
        """Clear per-track gait state so gallery changes trigger a fresh 1:N match."""
        self._recog_mgr.clear()

    def set_gallery_recognition_enabled(self, enabled: bool) -> None:
        """Preview=False skips GaitBase 1:N; Recognize=True runs gallery matching."""
        self._gallery_recognition_enabled = enabled
        if enabled:
            self.invalidate_gallery_recognition()

    @property
    def gallery_recognition_enabled(self) -> bool:
        return self._gallery_recognition_enabled

    def _full_recognize(
        self,
        track_id: int,
        buf,
        now: float,
        state,
        probe_key: str,
    ) -> Tuple[Optional[str], Optional[float]]:
        emb = self.recognizer.extract_embedding(
            buf.silhouettes,
            probe_key,
            sample_frames=self._timing.gait_sample_frames,
        )
        gid, dist = self.recognizer.match(emb, probe_key)
        state.track_id = track_id
        state.recognized_once = True
        state.reference_sils = list(buf.silhouettes)
        state.last_gallery_id = gid
        state.last_distance = dist
        state.last_recognition_ts = now
        state.last_similarity_check_ts = now
        return gid, dist

    def _evaluate_recognition(
        self,
        track_id: int,
        buf,
        now: float,
    ) -> Tuple[Optional[str], Optional[float], bool]:
        """
        Returns (gallery_id, distance, did_full_gait).
        """
        t = self._timing
        if not buf.ready(t.min_sil_count, t.min_sil_duration_sec):
            state = self._recog_mgr.get(track_id)
            return state.last_gallery_id, state.last_distance, False

        state = self._recog_mgr.get(track_id)
        probe_key = f"probe-{track_id:03d}"

        # First recognition: 1.5s + >=15 frames
        if not state.recognized_once:
            gid, dist = self._full_recognize(track_id, buf, now, state, probe_key)
            return gid, dist, True

        # track_id changed since last full recognition → force re-recognize
        if state.track_id != track_id:
            gid, dist = self._full_recognize(track_id, buf, now, state, probe_key)
            return gid, dist, True

        # Every recognition_interval_sec: lightweight similarity check
        if now - state.last_similarity_check_ts < t.recognition_interval_sec:
            return state.last_gallery_id, state.last_distance, False

        state.last_similarity_check_ts = now
        sim = silhouette_similarity(buf.silhouettes, state.reference_sils)
        if sim < self._recog_cfg.sil_similarity_threshold:
            logger.debug(f"track {track_id} sim={sim:.3f} < threshold, re-recognize")
            gid, dist = self._full_recognize(track_id, buf, now, state, probe_key)
            return gid, dist, True

        return state.last_gallery_id, state.last_distance, False

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[float] = None,
        wall_ts: Optional[float] = None,
    ) -> FrameResult:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be HxWx3 BGR uint8 array")

        now = wall_ts if wall_ts is not None else time.perf_counter()
        self._frame_index += 1

        should_process, skip_reason = self.scheduler.on_ingest(now)
        if not should_process:
            return FrameResult(
                frame_index=self._frame_index,
                processed=False,
                skipped_reason=skip_reason,
            )

        timings: Dict[str, float] = {}
        t_total = time.perf_counter()

        dets, img_info = self.detector.detect(frame_bgr)
        timings["detect"] = img_info.get("detect_ms", 0.0)

        tracks_raw = self.tracker.update(
            dets,
            int(img_info["height"]),
            int(img_info["width"]),
            wall_ts=now,
        )
        timings["track"] = self.tracker.last_track_ms
        timings["track_fps"] = self.tracker.effective_fps
        timings["track_dt_kf"] = self.tracker.last_dt_kf

        active_ids = [tid for tid, _ in tracks_raw]

        do_seg = self.scheduler.should_segment(now)
        t = self._timing
        track_results: List[TrackResult] = []

        for track_id, bbox in tracks_raw:
            buf = self.buffers.get(track_id)
            buf.touch(now)
            tr = TrackResult(
                track_id=track_id,
                bbox_xyxy=bbox,
                sil_count=len(buf),
                ready=buf.ready(t.min_sil_count, t.min_sil_duration_sec),
            )

            if do_seg:
                try:
                    sil = self.segmentor.segment_crop(frame_bgr, bbox)
                    buf.append(sil, ts=now)
                    tr.sil_count = len(buf)
                    tr.ready = buf.ready(t.min_sil_count, t.min_sil_duration_sec)
                except (ValueError, cv2.error) as e:
                    logger.debug(f"seg skip track {track_id}: {e}")

            if self._gallery_recognition_enabled and len(self.gallery) > 0:
                try:
                    gid, dist, _ = self._evaluate_recognition(track_id, buf, now)
                    tr.gallery_id = gid
                    tr.distance = dist
                    if gid:
                        tr.display_name = self._registry.get_display_name(gid)
                except Exception as e:
                    logger.warning(f"gait failed track {track_id}: {e}")

            track_results.append(tr)

        # Keep buffers + recognition state during ByteTrack lost grace window
        keep_ids = self.buffers.ids_in_grace(now)
        self.buffers.prune(active_ids, now)
        self._recog_mgr.prune(keep_ids)

        timings["segment"] = self.segmentor.last_seg_ms if do_seg else 0.0
        timings["gait"] = self.recognizer.last_gait_ms
        timings["total"] = (time.perf_counter() - t_total) * 1000.0

        return FrameResult(
            frame_index=self._frame_index,
            processed=True,
            tracks=track_results,
            timings_ms=timings,
        )

    def process_frame_visualized(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[float] = None,
        wall_ts: Optional[float] = None,
    ) -> tuple[np.ndarray, FrameResult]:
        result = self.process_frame(frame_bgr, timestamp_ms, wall_ts=wall_ts)
        vis = draw_results(frame_bgr, result, registry=self._registry)
        return vis, result
