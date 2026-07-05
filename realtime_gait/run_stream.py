#!/usr/bin/env python3
"""
Full pipeline: RTSP/SRT stream (input) -> RealtimeGaitPipeline -> display.

Run from OpenGait root:
  python -m realtime_gait.run_stream
  python -m realtime_gait.run_stream --stream rtsp://host:8554/home
  python -m realtime_gait.run_stream --gallery gallery.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_OPENGAIT = Path(__file__).resolve().parent.parent
if str(_OPENGAIT) not in sys.path:
    sys.path.insert(0, str(_OPENGAIT))

from realtime_gait.config import load_config
from realtime_gait.core.types import FrameResult
from realtime_gait.pipeline import RealtimeGaitPipeline
from realtime_gait.stream_reader import DEFAULT_STREAM_URL, LatestFrameReader
from realtime_gait.utils.gallery_io import (
    enroll_track,
    get_gallery_dir,
    get_registry,
    load_gallery_directory,
    load_gallery_pickle,
    next_person_id,
    pick_enroll_track,
    save_all_person_pickles,
)
from realtime_gait.utils.text_draw import put_text
from realtime_gait.utils.snapshot import PerfCollector, SnapshotSaver


def draw_hud(
    frame_bgr,
    *,
    display_fps: float,
    stream_lag_ms: float,
    result: FrameResult,
    reader_seq: int,
    reconnects: int,
    gallery_size: int,
    enroll_hint: str = "",
) -> None:
    """In-place overlay for stream + pipeline stats."""
    lines = [
        f"Stream FPS: {display_fps:.1f}",
        f"Local Lag: {stream_lag_ms:.1f} ms",
        f"Frames in: {reader_seq}  pipeline: {result.frame_index}",
        f"Gallery: {gallery_size}",
        f"Reconnects: {reconnects}",
    ]
    if result.processed:
        trk_fps = result.timings_ms.get("track_fps", 0)
        dt_kf = result.timings_ms.get("track_dt_kf", 1.0)
        lines.append(
            f"GPU ms  det={result.timings_ms.get('detect', 0):.0f} "
            f"trk={result.timings_ms.get('track', 0):.0f} "
            f"seg={result.timings_ms.get('segment', 0):.0f} "
            f"gait={result.timings_ms.get('gait', 0):.0f} "
            f"| trkHz={trk_fps:.1f} dt_kf={dt_kf:.2f}"
        )
        for tr in result.tracks:
            label = tr.gallery_id or f"T{tr.track_id}"
            lines.append(f"  {label} sil={tr.sil_count} ready={tr.ready}")
    else:
        lines.append(f"skipped ({result.skipped_reason})")

    y = 28
    for line in lines[:8]:
        cv2.putText(
            frame_bgr, line, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
        y += 22
    cv2.putText(
        frame_bgr, "q=quit  e=enroll  s=save gallery", (12, y + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if enroll_hint:
        put_text(
            frame_bgr, enroll_hint, (12, y + 32),
            font_size=20, color_bgr=(0, 255, 255), thickness=2,
        )


def _load_gallery(pipeline: RealtimeGaitPipeline, path: str) -> None:
    p = Path(path)
    pipeline.gallery_dir = get_gallery_dir(p)
    pipeline._registry = get_registry(pipeline.gallery_dir)
    if p.is_file():
        n = load_gallery_pickle(pipeline, p)
    else:
        n = load_gallery_directory(pipeline, p)
    if n:
        print(f"Loaded gallery from {p} ({n} ids)")


def run_stream_loop(
    pipeline: RealtimeGaitPipeline,
    reader: LatestFrameReader,
    *,
    display: bool = True,
    window_name: str = "Realtime Gait (Stream)",
    snapshot: SnapshotSaver | None = None,
    perf: PerfCollector | None = None,
    gallery_path: Path | None = None,
    enroll_prefix: str = "user",
    enroll_name: str = "",
) -> None:
    last_processed_seq = 0
    last_show_time = time.time()
    show_fps_counter = 0
    show_fps = 0.0

    # Reuse last visualization when stride skips processing
    last_vis = None
    last_result = FrameResult(frame_index=0, processed=False)

    print("Stream + gait pipeline running. Press q to quit.")

    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    last_status_print = 0.0
    enroll_hint = ""
    enroll_hint_until = 0.0
    fixed_enroll_name = enroll_name.strip()

    try:
        while True:
            frame, frame_ts, seq = reader.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            # Only feed pipeline when decoder produced a new frame
            if seq != last_processed_seq:
                last_processed_seq = seq
                vis, result = pipeline.process_frame_visualized(
                    frame,
                    timestamp_ms=(time.time() - frame_ts) * 1000.0,
                    wall_ts=time.perf_counter(),
                )
                last_vis = vis
                last_result = result
            elif last_vis is not None:
                vis = last_vis.copy()
                result = last_result
            else:
                vis = frame
                result = FrameResult(frame_index=0, processed=False)

            now = time.time()
            if now - last_show_time >= 1.0:
                show_fps = show_fps_counter / (now - last_show_time)
                show_fps_counter = 0
                last_show_time = now

            stream_lag_ms = (now - frame_ts) * 1000.0
            if perf is not None and result.processed:
                perf.record(result, stream_lag_ms=stream_lag_ms, show_fps=show_fps)

            if now - last_status_print >= 2.0:
                ids = [t.gallery_id or f"T{t.track_id}" for t in result.tracks]
                tms = result.timings_ms if result.processed else {}
                print(
                    f"[status] seq={seq} tracks={len(result.tracks)} ids={ids} "
                    f"det={tms.get('detect', 0):.0f} seg={tms.get('segment', 0):.0f} "
                    f"gait={tms.get('gait', 0):.0f} lag={stream_lag_ms:.0f}ms show_fps={show_fps:.1f}"
                )
                last_status_print = now

            draw_hud(
                vis,
                display_fps=show_fps,
                stream_lag_ms=stream_lag_ms,
                result=result,
                reader_seq=seq,
                reconnects=reader.reconnect_count,
                gallery_size=len(pipeline.gallery),
                enroll_hint=enroll_hint if time.time() < enroll_hint_until else "",
            )

            if snapshot is not None:
                saved = snapshot.maybe_save(vis, result, reader_seq=seq)
                if saved is not None:
                    print(f"[snapshot] saved {saved}")

            if display:
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("e"):
                    track_id, hint = pick_enroll_track(pipeline, result)
                    if track_id is None:
                        enroll_hint = hint
                    else:
                        person_id = fixed_enroll_name or next_person_id(pipeline, enroll_prefix)
                        ok, enroll_hint = enroll_track(
                            pipeline,
                            track_id,
                            person_id,
                            gallery_dir=gallery_path,
                            allow_partial=bool(hint),
                        )
                        if hint and ok:
                            enroll_hint = f"{enroll_hint}（{hint}）"
                    enroll_hint_until = time.time() + 4.0
                    print(f"[gallery] {enroll_hint}")
                if key == ord("s") and gallery_path is not None:
                    paths = save_all_person_pickles(pipeline, gallery_path)
                    enroll_hint = f"保存 -> {len(paths)} 个 pkl（{len(pipeline.gallery)}人）"
                    enroll_hint_until = time.time() + 4.0
                    print(f"[gallery] {enroll_hint}")
            else:
                time.sleep(0.001)

            show_fps_counter += 1

    except KeyboardInterrupt:
        pass
    finally:
        reader.release()
        if display:
            cv2.destroyAllWindows()
        if snapshot is not None:
            print(f"[snapshot] total saved: {snapshot.saved_count} -> {snapshot.save_dir}")
        if perf is not None and snapshot is not None and perf.summary()["processed_frames"] > 0:
            report = perf.write_report(snapshot.save_dir / "performance_report.json")
            print(f"[perf] report saved: {report}")
            print(f"[perf] {perf.summary()}")
        if gallery_path is not None and len(pipeline.gallery) > 0:
            paths = save_all_person_pickles(pipeline, gallery_path)
            print(f"[gallery] auto-saved on exit: {len(paths)} files ({len(pipeline.gallery)} ids)")


def main() -> None:
    parser = argparse.ArgumentParser(description="RTSP stream + realtime gait pipeline")
    parser.add_argument("--config", type=str, default="", help="YAML config path")
    parser.add_argument("--stream", type=str, default=DEFAULT_STREAM_URL, help="RTSP/SRT URL")
    parser.add_argument("--fallback", type=str, default="", help="Fallback stream URL")
    parser.add_argument("--gallery", type=str, default="", help="Gallery pickle (load + auto-save)")
    parser.add_argument(
        "--enroll-name",
        type=str,
        default="",
        help="Fixed person ID when pressing e in stream window",
    )
    parser.add_argument(
        "--enroll-prefix",
        type=str,
        default="user",
        help="Auto ID prefix: user_001, user_002... when --enroll-name omitted",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Screenshot dir (default: output/stream_screenshots/<timestamp>)",
    )
    parser.add_argument("--no-save", action="store_true", help="Disable auto screenshots")
    parser.add_argument("--save-interval", type=float, default=2.0)
    args = parser.parse_args()

    cfg = load_config(args.config if args.config else None)
    print(f"Stream URL: {args.stream}")
    print(
        f"Pipeline: process={cfg.timing.process_interval_ms}ms, "
        f"seg={cfg.timing.seg_interval_ms}ms, "
        f"min_sil={cfg.timing.min_sil_count}/{cfg.timing.min_sil_duration_sec}s"
    )

    reader = LatestFrameReader(
        args.stream,
        fallback_url=args.fallback or None,
    )

    print("Loading gait models (DroneYOLO + ByteTrack + PP-HumanSeg + GaitBase)...")
    pipeline = RealtimeGaitPipeline(cfg)

    gallery_path: Path | None = None
    if args.gallery:
        gallery_path = Path(args.gallery)
        _load_gallery(pipeline, str(gallery_path))
    elif args.enroll_name:
        gallery_path = Path("output/gallery")

    if gallery_path is not None and not gallery_path.parent.exists():
        gallery_path.parent.mkdir(parents=True, exist_ok=True)

    # Wait for first frame
    t0 = time.time()
    while reader.get_latest()[0] is None:
        if time.time() - t0 > 15:
            reader.release()
            raise RuntimeError("No frame received within 15s. Check STREAM_URL.")
        time.sleep(0.05)
    print("First frame received, starting pipeline.")

    snapshot = None
    perf = PerfCollector()
    if not args.no_save:
        save_dir = Path(args.save_dir) if args.save_dir else SnapshotSaver.default_session_dir()
        snapshot = SnapshotSaver(save_dir, interval_sec=args.save_interval)
        print(f"Auto screenshot save: {save_dir.resolve()} (interval={args.save_interval}s)")

    run_stream_loop(
        pipeline,
        reader,
        display=not args.no_display,
        snapshot=snapshot,
        perf=perf,
        gallery_path=gallery_path,
        enroll_prefix=args.enroll_prefix,
        enroll_name=args.enroll_name,
    )


if __name__ == "__main__":
    main()
