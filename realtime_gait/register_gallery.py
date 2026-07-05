#!/usr/bin/env python3
"""
Gallery enrollment — 从视频或图传流注册步态到 gallery.pkl。

用法示例（OpenGait 根目录）:

  # 全自动：等人走够 1.5s 后注册并保存（最接近「一键」）
  python -m realtime_gait.register_gallery --video walk.mp4 --name 张三 --out output/gallery.pkl --auto

  # RTSP 全自动注册
  python -m realtime_gait.register_gallery --stream rtsp://127.0.0.1:8554/home --name user01 --auto

  # 交互：实时窗口按 e 注册当前人，s 保存，q 退出
  python -m realtime_gait.register_gallery --stream rtsp://... --out output/gallery.pkl --interactive
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
from realtime_gait.pipeline import RealtimeGaitPipeline
from realtime_gait.stream_reader import DEFAULT_STREAM_URL, LatestFrameReader
from realtime_gait.utils.gallery_io import (
    DEFAULT_GALLERY_DIR,
    enroll_track,
    get_gallery_dir,
    load_gallery_directory,
    load_gallery_pickle,
    next_person_id,
    pick_enroll_track,
    save_all_person_pickles,
)


def _open_capture(video: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    return cap


def auto_enroll_from_video(
    pipeline: RealtimeGaitPipeline,
    video_path: str,
    display_name: str,
    *,
    gallery_dir: Path,
    allow_partial: bool = False,
    timeout_sec: float = 120.0,
) -> tuple[bool, str]:
    cap = _open_capture(video_path)
    deadline = time.time() + timeout_sec
    last_msg = ""

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame, wall_ts=time.perf_counter())
        if not result.processed:
            continue
        track_id, hint = pick_enroll_track(pipeline, result)
        if track_id is None:
            last_msg = hint
            continue
        if hint and not allow_partial:
            last_msg = hint
            continue
        ok, msg = enroll_track(
            pipeline,
            track_id,
            display_name,
            gallery_dir=gallery_dir,
            allow_partial=allow_partial or bool(hint),
        )
        cap.release()
        return ok, msg

    cap.release()
    return False, last_msg or f"超时（{timeout_sec}s）内未能完成注册"


def auto_enroll_from_stream(
    pipeline: RealtimeGaitPipeline,
    stream_url: str,
    display_name: str,
    *,
    gallery_dir: Path,
    allow_partial: bool = False,
    timeout_sec: float = 120.0,
) -> tuple[bool, str]:
    reader = LatestFrameReader(stream_url)
    deadline = time.time() + timeout_sec
    last_seq = -1
    last_msg = ""

    try:
        while time.time() < deadline:
            frame, _, seq = reader.get_latest()
            if frame is None or seq == last_seq:
                time.sleep(0.02)
                continue
            last_seq = seq
            result = pipeline.process_frame(frame, wall_ts=time.perf_counter())
            if not result.processed:
                continue
            track_id, hint = pick_enroll_track(pipeline, result)
            if track_id is None:
                last_msg = hint
                continue
            if hint and not allow_partial:
                last_msg = hint
                continue
            ok, msg = enroll_track(
                pipeline,
                track_id,
                display_name,
                gallery_dir=gallery_dir,
                allow_partial=allow_partial or bool(hint),
            )
            return ok, msg
    finally:
        reader.release()

    return False, last_msg or f"超时（{timeout_sec}s）内未能完成注册"


def interactive_enroll_stream(
    pipeline: RealtimeGaitPipeline,
    stream_url: str,
    gallery_dir: Path,
    *,
    enroll_prefix: str = "user",
    enroll_name: str = "",
) -> None:
    reader = LatestFrameReader(stream_url)
    window = "Gallery Enrollment (e=注册 s=保存 q=退出)"
    enroll_msg = ""
    enroll_msg_until = 0.0
    fixed_name = enroll_name.strip()

    print("交互注册：让人在镜头前正常走 1.5s，按 e 注册，s 保存 gallery，q 退出")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    try:
        last_seq = -1
        while True:
            frame, _, seq = reader.get_latest()
            if frame is None:
                time.sleep(0.02)
                continue

            if seq != last_seq:
                last_seq = seq
                vis, result = pipeline.process_frame_visualized(
                    frame, wall_ts=time.perf_counter()
                )
            else:
                vis = frame.copy()
                from realtime_gait.core.types import FrameResult

                result = FrameResult(frame_index=0, processed=False)

            hud = [
                f"Gallery: {len(pipeline.gallery)}  |  {gallery_dir}",
                "e=注册当前人  s=保存  q=退出",
            ]
            if fixed_name:
                hud.append(f"注册名: {fixed_name}")
            else:
                hud.append(f"自动 ID 前缀: {enroll_prefix}")
            if enroll_msg and time.time() < enroll_msg_until:
                hud.append(enroll_msg)
            for tr in result.tracks if result.processed else []:
                hud.append(f"  T{tr.track_id} sil={tr.sil_count} ready={tr.ready}")

            y = 28
            for line in hud[:6]:
                color = (0, 255, 255) if line.startswith("已") or line.startswith("保存") else (0, 255, 0)
                if line.startswith("注册失败"):
                    color = (0, 0, 255)
                cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                y += 26

            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                track_id, hint = pick_enroll_track(pipeline, result)
                if track_id is None:
                    enroll_msg = hint
                else:
                    person_id = fixed_name or next_person_id(pipeline, enroll_prefix)
                    ok, enroll_msg = enroll_track(
                        pipeline,
                        track_id,
                        person_id,
                        gallery_dir=gallery_dir,
                        allow_partial=bool(hint),
                    )
                    if hint and ok:
                        enroll_msg = f"{enroll_msg}（{hint}）"
                enroll_msg_until = time.time() + 4.0
                print(enroll_msg)
            if key == ord("s"):
                paths = save_all_person_pickles(pipeline, gallery_dir)
                enroll_msg = f"保存成功 -> {len(paths)} 个文件（{len(pipeline.gallery)} 人）"
                enroll_msg_until = time.time() + 4.0
                print(enroll_msg)
    finally:
        reader.release()
        cv2.destroyAllWindows()
        if len(pipeline.gallery) > 0:
            paths = save_all_person_pickles(pipeline, gallery_dir)
            print(f"退出前自动保存: {len(paths)} 个 pkl（{len(pipeline.gallery)} 人）")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register gait gallery (one-click friendly)")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--video", type=str, default="", help="Enrollment video path")
    parser.add_argument("--stream", type=str, default="", help="RTSP/SRT URL")
    parser.add_argument("--name", type=str, default="", help="Person ID / display name")
    parser.add_argument(
        "--out",
        type=str,
        default="output/gallery",
        help="Gallery directory (per-person English pkl, e.g. Suhui.pkl)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="user",
        help="Auto ID prefix when --name omitted (user_001, user_002...)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto enroll when silhouettes ready, then save and exit",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open window: e=enroll, s=save, q=quit",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--allow-partial", action="store_true", help="Allow enroll before min sil count")
    args = parser.parse_args()

    if not args.video and not args.stream:
        if not args.interactive:
            args.stream = DEFAULT_STREAM_URL
        else:
            parser.error("需要 --video 或 --stream")

    cfg = load_config(args.config if args.config else None)
    pipeline = RealtimeGaitPipeline(cfg)

    out_dir = get_gallery_dir(args.out)
    pipeline.gallery_dir = out_dir
    loaded = load_gallery_directory(pipeline, out_dir)
    if loaded:
        print(f"已加载已有 gallery: {out_dir}（{loaded} 人）")

    if args.interactive:
        url = args.stream or DEFAULT_STREAM_URL
        interactive_enroll_stream(
            pipeline,
            url,
            out_dir,
            enroll_prefix=args.prefix,
            enroll_name=args.name,
        )
        return

    display_name = args.name.strip() or next_person_id(pipeline, args.prefix)
    if not args.auto:
        parser.error("请指定 --auto（全自动注册）或 --interactive（按键注册）")

    print(f"自动注册 [{display_name}]，请让人在镜头前正常行走约 {cfg.timing.min_sil_duration_sec}s ...")

    if args.video:
        ok, msg = auto_enroll_from_video(
            pipeline,
            args.video,
            display_name,
            gallery_dir=out_dir,
            allow_partial=args.allow_partial,
            timeout_sec=args.timeout,
        )
    else:
        ok, msg = auto_enroll_from_stream(
            pipeline,
            args.stream or DEFAULT_STREAM_URL,
            display_name,
            gallery_dir=out_dir,
            allow_partial=args.allow_partial,
            timeout_sec=args.timeout,
        )

    print(msg)
    if ok:
        paths = save_all_person_pickles(pipeline, out_dir)
        print(f"Gallery 已保存: {', '.join(p.name for p in paths)}（共 {len(pipeline.gallery)} 人）")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
