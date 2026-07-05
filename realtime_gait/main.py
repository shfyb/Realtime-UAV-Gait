#!/usr/bin/env python3
"""
CLI entry for realtime_gait pipeline.

Run from OpenGait root:
  python -m realtime_gait.main --video /path/to/test.mp4
  python -m realtime_gait.main --video test.mp4 --build-gallery
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure OpenGait is cwd or on path
_OPENGAIT = Path(__file__).resolve().parent.parent
if str(_OPENGAIT) not in sys.path:
    sys.path.insert(0, str(_OPENGAIT))

from realtime_gait.config import load_config
from realtime_gait.pipeline import RealtimeGaitPipeline
from realtime_gait.utils.gallery_io import enroll_track, load_gallery_pickle, save_gallery_pickle


def run_video(
    pipeline: RealtimeGaitPipeline,
    video_path: str,
    display: bool = False,
    max_frames: int = 0,
    output_path: str = "",
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    writer = None
    if output_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or pipeline.cfg.timing.target_process_hz
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    n = 0
    t0 = time.perf_counter()
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if max_frames > 0 and n > max_frames:
            break

        vis, result = pipeline.process_frame_visualized(frame)
        if result.processed:
            processed += 1
            if processed % 30 == 0:
                ids = [t.gallery_id or f"T{t.track_id}" for t in result.tracks]
                print(
                    f"frame={result.frame_index} tracks={len(result.tracks)} "
                    f"ids={ids} timings={result.timings_ms}"
                )

        if writer is not None and result.processed:
            writer.write(vis)
        if display and result.processed:
            cv2.imshow("realtime_gait", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    print(f"Done: ingest={n} processed={processed} time={elapsed:.2f}s")


def build_gallery_from_video(
    pipeline: RealtimeGaitPipeline,
    video_path: str,
    person_id: str = "001",
) -> bool:
    """Offline enroll: run pipeline until silhouettes ready, register one person."""
    cap = cv2.VideoCapture(video_path)
    pipeline.reset()
    person_id = person_id.strip() or "001"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame)
        if not result.processed:
            continue
        for tr in result.tracks:
            buf = pipeline.buffers.get(tr.track_id)
            if buf.ready(
                pipeline.cfg.timing.min_sil_count,
                pipeline.cfg.timing.min_sil_duration_sec,
            ):
                ok, msg = enroll_track(pipeline, tr.track_id, person_id)
                print(msg)
                cap.release()
                return ok
    cap.release()
    print("Warning: gallery enrollment failed (not enough silhouettes)")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime Drone Gait Pipeline")
    parser.add_argument("--config", type=str, default="", help="YAML config path")
    parser.add_argument("--video", type=str, default="", help="Test video path")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--build-gallery",
        action="store_true",
        help="Enroll gallery from --video before recognition loop",
    )
    parser.add_argument("--enroll-id", type=str, default="001", help="Person ID for --build-gallery")
    parser.add_argument(
        "--save-gallery",
        type=str,
        default="",
        help="Save gallery.pkl after --build-gallery (e.g. output/gallery.pkl)",
    )
    parser.add_argument("--gallery", type=str, default="", help="Load gallery.pkl before run")
    args = parser.parse_args()

    cfg_path = args.config if args.config else None
    cfg = load_config(cfg_path)
    pipeline = RealtimeGaitPipeline(cfg)

    if args.gallery:
        n = load_gallery_pickle(pipeline, args.gallery)
        print(f"Loaded gallery: {args.gallery} ({n} ids)")

    if args.build_gallery and args.video:
        if build_gallery_from_video(pipeline, args.video, args.enroll_id):
            if args.save_gallery:
                path = save_gallery_pickle(pipeline, args.save_gallery)
                print(f"Gallery saved: {path}")

    if args.video:
        run_video(
            pipeline,
            args.video,
            display=args.display,
            max_frames=args.max_frames,
            output_path=args.output,
        )
    else:
        print("Pipeline initialized. Import RealtimeGaitPipeline in your video-link callback.")
        print("See realtime_gait/examples/feed_frames.py")


if __name__ == "__main__":
    main()
