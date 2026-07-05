"""Background stream + pipeline session for the web UI."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..config import load_config
from ..core.types import FrameResult
from ..pipeline import RealtimeGaitPipeline
from ..stream_reader import DEFAULT_STREAM_URL, LatestFrameReader
from ..utils.gallery_io import (
    DEFAULT_GALLERY_DIR,
    delete_gallery_person,
    enroll_from_silhouettes,
    get_gallery_dir,
    get_registry,
    load_gallery_directory,
    load_gallery_pickle,
    list_gallery_people,
    list_enroll_candidates,
    resolve_enroll_track,
    save_all_person_pickles,
)
from ..utils.text_draw import put_text
from ..utils.visualize import draw_results


class AppMode(str, Enum):
    PREVIEW = "preview"          # 仅显示检测跟踪
    RECOGNIZE = "recognize"      # 加载 gallery 后识别
    ENROLLING = "enrolling"      # 用户点击后开始采集轮廓


@dataclass
class EnrollState:
    active: bool = False
    person_name: str = ""
    started_ts: float = 0.0
    track_id: Optional[int] = None
    message: str = ""


@dataclass
class SessionStatus:
    running: bool = False
    pipeline_ready: bool = False
    stream_url: str = ""
    mode: str = AppMode.PREVIEW.value
    gallery_count: int = 0
    gallery_path: str = ""
    fps: float = 0.0
    lag_ms: float = 0.0
    reader_seq: int = 0
    reconnects: int = 0
    tracks: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    enroll: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    last_error: str = ""
    recognition_enabled: bool = False
    gallery_people: list[dict[str, Any]] = field(default_factory=list)
    frame_index: int = 0
    processed: bool = False


class StreamSession:
    """Thread-safe wrapper around pipeline + RTSP reader for HTTP/MJPEG."""

    def __init__(
        self,
        *,
        config_path: str = "",
        gallery_path: str = "output/gallery",
        default_stream: str = DEFAULT_STREAM_URL,
    ):
        self._config_path = config_path
        self._gallery_dir = get_gallery_dir(gallery_path)
        self._default_stream = default_stream

        self._lock = threading.RLock()
        self._pipeline: Optional[RealtimeGaitPipeline] = None
        self._reader: Optional[LatestFrameReader] = None
        self._worker: Optional[threading.Thread] = None
        self._running = False

        self._mode = AppMode.PREVIEW
        self._enroll = EnrollState()
        self._message = ""
        self._last_error = ""

        self._latest_jpeg: Optional[bytes] = None
        self._latest_result = FrameResult(frame_index=0, processed=False)
        self._last_processed_seq = 0
        self._show_fps = 0.0
        self._fps_counter = 0
        self._fps_window_start = time.time()
        self._stream_lag_ms = 0.0

    @property
    def gallery_dir(self) -> Path:
        return self._gallery_dir

    @property
    def default_stream(self) -> str:
        return self._default_stream

    def _ensure_pipeline(self) -> RealtimeGaitPipeline:
        if self._pipeline is None:
            cfg = load_config(self._config_path if self._config_path else None)
            self._pipeline = RealtimeGaitPipeline(cfg)
            self._pipeline.gallery_dir = self._gallery_dir
            self._pipeline._registry = get_registry(self._gallery_dir)
            n = load_gallery_directory(self._pipeline, self._gallery_dir)
            if n:
                self._message = f"已加载 gallery（{n} 人）"
                self._mode = AppMode.RECOGNIZE
                self._pipeline.set_gallery_recognition_enabled(True)
        return self._pipeline

    def start(self, stream_url: str = "") -> str:
        with self._lock:
            if self._running:
                return "图传已在运行"

            url = (stream_url or self._default_stream).strip()
            if not url:
                raise ValueError("请填写图传地址")

            self._ensure_pipeline()
            self._reader = LatestFrameReader(url)
            self._running = True
            self._last_processed_seq = 0
            self._latest_result = FrameResult(frame_index=0, processed=False)
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()
            self._message = f"正在连接图传: {url}"
            return self._message

    def stop(self) -> str:
        with self._lock:
            self._running = False
            reader = self._reader
            self._reader = None
            if reader is not None:
                reader.release()
            if self._worker is not None:
                self._worker.join(timeout=2.0)
                self._worker = None
            self._cancel_enroll_locked()
            self._message = "图传已停止"
            return self._message

    def set_mode_recognize(self) -> str:
        with self._lock:
            self._cancel_enroll_locked()
            pipe = self._ensure_pipeline()
            if len(pipe.gallery) == 0:
                raise RuntimeError(
                    "步态档案为空：请先「完成注册」或点击「重新加载」加载 output/gallery 下的 pkl"
                )
            pipe.set_gallery_recognition_enabled(True)
            self._mode = AppMode.RECOGNIZE
            names = list(pipe._registry.all_display_names().values())
            self._message = f"识别模式：与档案比对（{len(pipe.gallery)} 人：{', '.join(names) or '—'}）"
            return self._message

    def set_mode_preview(self) -> str:
        with self._lock:
            self._cancel_enroll_locked()
            if self._pipeline is not None:
                self._pipeline.set_gallery_recognition_enabled(False)
            self._mode = AppMode.PREVIEW
            self._message = "预览模式：仅检测+跟踪（绿框 T1），不做步态身份比对"
            return self._message

    def start_enroll(self, person_name: str, track_id: Optional[int] = None) -> str:
        with self._lock:
            if not self._running or self._pipeline is None:
                raise RuntimeError("请先连接图传（点击右侧「连接图传」，看到 LIVE 后再注册）")
            name = person_name.strip()
            if not name:
                raise ValueError("请填写注册姓名/ID")

            if self._pipeline is not None:
                self._pipeline.set_gallery_recognition_enabled(False)

            self._mode = AppMode.ENROLLING
            self._enroll = EnrollState(
                active=True,
                person_name=name,
                started_ts=time.perf_counter(),
                track_id=track_id,
                message="正在采集步态，请让人正常行走约 1.5 秒…",
            )
            if track_id is not None:
                self._message = f"正在采集 T{track_id}（{name}）的步态，请该目标正常行走约 1.5 秒…"
            else:
                self._message = (
                    f"正在采集「{name}」的步态；多人时请在下拉框或左侧识别卡片点选目标 T 编号"
                )
            self._enroll.message = self._message
            return self._message

    def set_enroll_track(self, track_id: int) -> str:
        with self._lock:
            if not self._enroll.active:
                raise RuntimeError("当前未在注册采集中")
            if track_id < 0:
                raise ValueError("无效的跟踪 ID")
            self._enroll.track_id = track_id
            name = self._enroll.person_name
            self._message = f"注册目标已切换为 T{track_id}（{name}）"
            return self._message

    def finish_enroll(self, *, allow_partial: bool = False) -> str:
        with self._lock:
            if not self._enroll.active or self._pipeline is None:
                raise RuntimeError("当前未在注册采集中")

            result = self._latest_result
            track_id, hint = resolve_enroll_track(
                self._pipeline,
                result,
                self._enroll.track_id,
                since_ts=self._enroll.started_ts,
            )
            if track_id is None:
                raise RuntimeError(hint)

            buf = self._pipeline.buffers.get(track_id)
            sils = buf.silhouettes_since(self._enroll.started_ts)
            timing = self._pipeline.cfg.timing
            count, span = buf.stats_since(self._enroll.started_ts)
            ready = buf.ready_since(
                self._enroll.started_ts,
                timing.min_sil_count,
                timing.min_sil_duration_sec,
            )
            if not ready and not allow_partial:
                raise RuntimeError(
                    f"轮廓不足：{count}/{timing.min_sil_count} 帧，"
                    f"时长 {span:.1f}/{timing.min_sil_duration_sec}s，请继续行走"
                )

            ok, msg, english_id, saved = enroll_from_silhouettes(
                self._pipeline,
                self._enroll.person_name,
                sils,
                gallery_dir=self._gallery_dir,
                allow_partial=allow_partial or not ready,
            )
            if not ok:
                raise RuntimeError(msg)

            self._pipeline.invalidate_gallery_recognition()
            self._pipeline._registry.load()
            self._pipeline.set_gallery_recognition_enabled(True)
            self._cancel_enroll_locked()
            self._mode = AppMode.RECOGNIZE
            self._message = f"{msg}，已保存 {saved.name}，识别已刷新"
            return self._message

    def cancel_enroll(self) -> str:
        with self._lock:
            self._cancel_enroll_locked()
            if self._pipeline is not None and len(self._pipeline.gallery) > 0:
                self._mode = AppMode.RECOGNIZE
            else:
                self._mode = AppMode.PREVIEW
            self._message = "已取消注册采集"
            return self._message

    def save_gallery(self, path: str = "") -> str:
        with self._lock:
            if self._pipeline is None or len(self._pipeline.gallery) == 0:
                raise RuntimeError("gallery 为空，无法保存")
            out_dir = get_gallery_dir(path) if path else self._gallery_dir
            saved = save_all_person_pickles(self._pipeline, out_dir)
            self._gallery_dir = out_dir
            names = ", ".join(p.name for p in saved)
            self._message = f"Gallery 已保存: {names}"
            return self._message

    def reload_gallery(self, path: str = "") -> str:
        with self._lock:
            pipe = self._ensure_pipeline()
            p = get_gallery_dir(path.strip()) if path.strip() else self._gallery_dir
            self._gallery_dir = p
            pipe.gallery_dir = p
            pipe._registry = get_registry(p)
            pipe.gallery.clear()
            if p.is_file():
                n = load_gallery_pickle(pipe, p)
            else:
                n = load_gallery_directory(pipe, p)
            pipe.invalidate_gallery_recognition()
            pipe._registry.load()
            if n > 0:
                pipe.set_gallery_recognition_enabled(True)
                self._mode = AppMode.RECOGNIZE
            else:
                pipe.set_gallery_recognition_enabled(False)
                self._mode = AppMode.PREVIEW
            self._message = f"已加载 gallery: {p}（{n} 人）"
            return self._message

    def delete_gallery_person(self, person_id: str) -> str:
        with self._lock:
            pipe = self._ensure_pipeline()
            pid = person_id.strip()
            if not pid:
                raise ValueError("请先在列表中选择要删除的人员")

            english_id, display = delete_gallery_person(pipe, pid, self._gallery_dir)
            remaining = len(pipe.gallery)

            if remaining > 0:
                pipe.invalidate_gallery_recognition()
                if self._mode != AppMode.ENROLLING:
                    pipe.set_gallery_recognition_enabled(True)
                    self._mode = AppMode.RECOGNIZE
            else:
                pipe.set_gallery_recognition_enabled(False)
                if self._mode != AppMode.ENROLLING:
                    self._mode = AppMode.PREVIEW

            self._message = f"已删除 {display}（{english_id}），档案库剩余 {remaining} 人"
            return self._message

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_status(self) -> SessionStatus:
        with self._lock:
            pipe = self._pipeline
            result = self._latest_result
            reader = self._reader

            enroll_info: dict[str, Any] = {
                "active": self._enroll.active,
                "person_name": self._enroll.person_name,
                "track_id": self._enroll.track_id,
                "selected_track_id": self._enroll.track_id,
                "sil_count": 0,
                "sil_span_sec": 0.0,
                "sil_required": 15,
                "duration_required_sec": 1.5,
                "ready": False,
                "message": self._enroll.message,
                "candidates": [],
                "track_hint": "",
            }
            if self._enroll.active and pipe is not None and result.processed:
                enroll_info["candidates"] = list_enroll_candidates(
                    pipe, result, self._enroll.started_ts
                )
                track_id, hint = resolve_enroll_track(
                    pipe,
                    result,
                    self._enroll.track_id,
                    since_ts=self._enroll.started_ts,
                )
                enroll_info["track_id"] = track_id
                enroll_info["selected_track_id"] = track_id
                enroll_info["track_hint"] = hint
                if track_id is not None:
                    buf = pipe.buffers.get(track_id)
                    count, span = buf.stats_since(self._enroll.started_ts)
                    timing = pipe.cfg.timing
                    enroll_info["sil_count"] = count
                    enroll_info["sil_span_sec"] = round(span, 2)
                    enroll_info["sil_required"] = timing.min_sil_count
                    enroll_info["duration_required_sec"] = timing.min_sil_duration_sec
                    enroll_info["ready"] = buf.ready_since(
                        self._enroll.started_ts,
                        timing.min_sil_count,
                        timing.min_sil_duration_sec,
                    )

            tracks = []
            for tr in result.tracks:
                tracks.append({
                    "track_id": tr.track_id,
                    "gallery_id": tr.gallery_id,
                    "display_name": tr.display_name or tr.gallery_id,
                    "distance": tr.distance,
                    "sil_count": tr.sil_count,
                    "ready": tr.ready,
                    "bbox": tr.bbox_xyxy,
                })

            people: list[dict[str, Any]] = []
            if pipe is not None:
                people = list_gallery_people(pipe, self._gallery_dir)

            return SessionStatus(
                running=self._running,
                pipeline_ready=pipe is not None,
                stream_url=reader.url if reader else "",
                mode=self._mode.value,
                gallery_count=len(pipe.gallery) if pipe else 0,
                gallery_path=str(self._gallery_dir),
                fps=round(self._show_fps, 1),
                lag_ms=round(self._stream_lag_ms, 1),
                reader_seq=reader.total_read if reader else 0,
                reconnects=reader.reconnect_count if reader else 0,
                tracks=tracks,
                timings_ms=dict(result.timings_ms) if result.processed else {},
                enroll=enroll_info,
                message=self._message,
                last_error=self._last_error,
                recognition_enabled=pipe.gallery_recognition_enabled if pipe else False,
                gallery_people=people,
                frame_index=result.frame_index,
                processed=result.processed,
            )

    def _cancel_enroll_locked(self) -> None:
        self._enroll = EnrollState()

    def _draw_overlay(self, vis: np.ndarray, result: FrameResult, lag_ms: float) -> None:
        pipe = self._pipeline
        if pipe is None:
            return

        mode_labels = {
            AppMode.PREVIEW.value: "PREVIEW",
            AppMode.RECOGNIZE.value: "RECOGNIZE",
            AppMode.ENROLLING.value: "ENROLLING",
        }
        lines = [
            f"Mode: {mode_labels.get(self._mode.value, self._mode.value)}",
            f"Gallery: {len(pipe.gallery)}  FPS: {self._show_fps:.1f}  Lag: {lag_ms:.0f}ms",
        ]
        if self._enroll.active and result.processed:
            enroll_sil = 0
            enroll_span = 0.0
            enroll_ready = False
            enroll_req = 15
            track_id = None
            timing = pipe.cfg.timing
            enroll_req = timing.min_sil_count
            track_id, _ = resolve_enroll_track(
                pipe,
                result,
                self._enroll.track_id,
                since_ts=self._enroll.started_ts,
            )
            if track_id is not None:
                buf = pipe.buffers.get(track_id)
                enroll_sil, enroll_span = buf.stats_since(self._enroll.started_ts)
                enroll_ready = buf.ready_since(
                    self._enroll.started_ts,
                    timing.min_sil_count,
                    timing.min_sil_duration_sec,
                )
            target = f"T{track_id}" if track_id is not None else "?"
            lines.append(
                f"注册 {self._enroll.person_name} @ {target} "
                f"sil {enroll_sil}/{enroll_req} ({enroll_span:.1f}s)"
            )
            if enroll_ready:
                lines.append(">>> 可点击「完成注册」<<<")
        elif self._mode == AppMode.RECOGNIZE:
            lines.append("识别模式 · 步态比对已开启")
        else:
            lines.append("预览模式 · 仅 T1 跟踪框")

        enroll_target: Optional[int] = None
        if self._enroll.active and result.processed:
            enroll_target, _ = resolve_enroll_track(
                pipe,
                result,
                self._enroll.track_id,
                since_ts=self._enroll.started_ts,
            )

        if result.processed:
            for tr in result.tracks:
                label = tr.display_name or tr.gallery_id or f"T{tr.track_id}"
                lines.append(f"  {label} sil={tr.sil_count} ready={tr.ready}")
                if enroll_target is not None and tr.track_id == enroll_target:
                    x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
                    cv2.rectangle(vis, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 255, 255), 3)

        y = 28
        for line in lines[:8]:
            color = (0, 255, 255) if line.startswith("注册") or "完成注册" in line else (0, 255, 0)
            put_text(vis, line, (12, y), font_size=20, color_bgr=color, thickness=2)
            y += 24

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
                reader = self._reader
                pipeline = self._pipeline
            if reader is None or pipeline is None:
                time.sleep(0.05)
                continue

            frame, frame_ts, seq = reader.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            process_new = seq != self._last_processed_seq
            if process_new:
                self._last_processed_seq = seq
                try:
                    vis, result = pipeline.process_frame_visualized(
                        frame,
                        timestamp_ms=(time.time() - frame_ts) * 1000.0,
                        wall_ts=time.perf_counter(),
                    )
                except Exception as exc:
                    self._last_error = str(exc)
                    time.sleep(0.01)
                    continue
            else:
                result = self._latest_result
                vis = frame.copy()
                if result.processed and result.tracks and pipeline is not None:
                    vis = draw_results(vis, result, registry=pipeline._registry)

            lag_ms = (time.time() - frame_ts) * 1000.0
            self._fps_counter += 1
            now = time.time()
            if now - self._fps_window_start >= 1.0:
                show_fps = self._fps_counter / (now - self._fps_window_start)
                self._fps_counter = 0
                self._fps_window_start = now
            else:
                show_fps = self._show_fps

            self._draw_overlay(vis, result, lag_ms)

            ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                jpeg = buf.tobytes()
                with self._lock:
                    self._latest_result = result
                    self._stream_lag_ms = lag_ms
                    self._show_fps = show_fps
                    self._latest_jpeg = jpeg

            time.sleep(0.001)
