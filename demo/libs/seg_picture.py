import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger

from track_yolov8 import Predictor, BYTETracker, Timer, YOLO
from track_id_merge import force_merge_all_track_id_dict

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "paddle"))
from seg_demo import load_seg_model, predict_seg

# === 配置参数 ===
track_cfgs = {
    "model": {
        "ckpt": "/home/liaoqi/Code_09/code/Drone-YOLO/runs/dronegait_droneyolo_s/train/weights/best.pt",
    },
    "device": "gpu",
    "save_result": "True",
}

seg_cfgs = {
    "model": {
        "seg_model": "./demo/checkpoints/seg_model/human_pp_humansegv2_lite_192x192_inference_model_with_softmax/deploy.yaml",
    }
}

SEG_INPUT_SIZE = (192, 192)
MIN_FRAMES_PER_ID = 10

model = None
predictor = None
seg_predictor = None
tracker = BYTETracker(frame_rate=30)
frame_id_global = 0
tid_offset = 0
mark_tid = True


def init_models(track_gpu_id=0, seg_gpu_id=None):
    """seg_gpu_id=None 时分割走 CPU，避免 Paddle 与 PyTorch 同卡冲突。"""
    global model, predictor, seg_predictor

    track_device = f"cuda:{track_gpu_id}" if track_cfgs["device"] == "gpu" else "cpu"
    model = YOLO(track_cfgs["model"]["ckpt"])
    predictor = Predictor(model=model, device=track_device)
    logger.info(f"Loaded YOLOv8 model on {track_device}")

    use_seg_gpu = seg_gpu_id is not None and seg_gpu_id >= 0
    seg_predictor = load_seg_model(
        seg_cfgs["model"]["seg_model"],
        gpu_id=seg_gpu_id if use_seg_gpu else 0,
        use_gpu=use_seg_gpu,
    )
    if use_seg_gpu:
        logger.info(f"Loaded PP-HumanSeg on GPU {seg_gpu_id}")
    else:
        logger.info("Loaded PP-HumanSeg on CPU (Paddle+PyTorch 同卡易冲突，默认 CPU)")


def reset_track_state():
    """每个视频独立跟踪，避免跨视频 ID 漂移。"""
    global tracker, frame_id_global, tid_offset, mark_tid
    tracker = BYTETracker(frame_rate=30)
    frame_id_global = 0
    tid_offset = 0
    mark_tid = True


def track(frame):
    global frame_id_global, tid_offset, mark_tid

    timer = Timer()
    outputs, img_info = predictor.inference(frame, timer)

    if isinstance(outputs[0], torch.Tensor):
        outputs[0] = outputs[0].cpu().numpy()

    person_dets = []
    if outputs[0] is not None:
        for det in outputs[0]:
            person_dets.append(det[:5])

    filtered_outputs = np.array(person_dets) if person_dets else np.empty((0, 5))

    online_targets = tracker.update(
        filtered_outputs,
        [img_info['height'], img_info['width']],
        [img_info['height'], img_info['width']]
    )

    bboxes, track_ids = [], []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id

        if mark_tid:
            tid_offset = tid - 1
            mark_tid = False
        tid = tid - tid_offset

        aspect_ratio = tlwh[2] / tlwh[3]
        area = tlwh[2] * tlwh[3]
        if area < 100 or aspect_ratio > 3.0 or aspect_ratio < 0.3:
            continue

        x1, y1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h
        bboxes.append([x1, y1, x2, y2])
        track_ids.append(tid)

    frame_id_global += 1
    return bboxes, track_ids


def prepare_crop(frame, bbox):
    """裁剪 + 10% padding + 白底正方形画布 + resize，与原版逻辑一致。"""
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    x1_new = max(0, int(x1 - 0.1 * w))
    x2_new = min(frame.shape[1], int(x2 + 0.1 * w))
    y1_new = max(0, int(y1 - 0.1 * h))
    y2_new = min(frame.shape[0], int(y2 + 0.1 * h))

    crop = frame[y1_new:y2_new, x1_new:x2_new]
    if crop.size == 0:
        return None

    side = max(crop.shape[:2])
    canvas = np.ones((side, side, 3), dtype=np.uint8) * 255
    offset_x = (side - crop.shape[1]) // 2
    offset_y = (side - crop.shape[0]) // 2
    canvas[offset_y:offset_y + crop.shape[0], offset_x:offset_x + crop.shape[1]] = crop

    return cv2.resize(canvas, SEG_INPUT_SIZE)


def collect_videos(input_root):
    videos = []
    for dirpath, _, filenames in os.walk(input_root):
        for filename in sorted(filenames):
            if filename.endswith('.mp4'):
                videos.append(osp.join(dirpath, filename))
    return sorted(videos)


def process_video(input_path, output_root, input_root):
    reset_track_state()
    if track_cfgs["device"] == "gpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rel_path = osp.relpath(input_path, input_root)
    rel_path_no_ext = osp.splitext(rel_path)[0]
    video_name = rel_path_no_ext.replace("\\", "/")

    try:
        subject_id, seq_name, base_name = video_name.split('/')
    except ValueError:
        print(f"[错误] 无法从视频名解析出 subject_id 和 seq_name: {video_name}")
        cap.release()
        return

    frame_idx = 0
    track_id_dict = {}

    with tqdm(total=total_frames, desc=f"追踪视频: {video_name}", unit="帧") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bboxes, ids = track(frame)

            if bboxes:
                for bbox, tid in zip(bboxes, ids):
                    tid_str = f"id{tid}"
                    track_id_dict.setdefault(tid_str, []).append((frame_idx, bbox.copy()))

            frame_idx += 1
            pbar.update(1)

    cap.release()

    track_id_dict, merge_groups = force_merge_all_track_id_dict(track_id_dict)
    for primary, absorbed in merge_groups:
        for sec_tid, reason in absorbed:
            logger.info(f"{video_name}: 合并 {sec_tid} -> {primary} ({reason})")

    valid_ids = {
        tid: frames for tid, frames in track_id_dict.items()
        if len(frames) >= MIN_FRAMES_PER_ID
    }
    if len(valid_ids) == 0:
        print(f"[跳过] 视频 {video_name} 无有效ID（均需≥{MIN_FRAMES_PER_ID}帧），跳过分割")
        return

    save_dir = osp.join(output_root, subject_id, seq_name, base_name)
    os.makedirs(save_dir, exist_ok=True)

    frame_tasks = {}
    seg_total = 0
    for tid_str, frames_info in valid_ids.items():
        for fidx, bbox in frames_info:
            frame_tasks.setdefault(fidx, []).append((tid_str, bbox))
            seg_total += 1

    cap = cv2.VideoCapture(input_path)
    frame_idx = 0
    with tqdm(total=seg_total, desc=f"分割视频: {video_name}", unit="张") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_tasks:
                for tid_str, bbox in frame_tasks[frame_idx]:
                    resized = prepare_crop(frame, bbox)
                    if resized is None:
                        pbar.update(1)
                        continue
                    save_name = f"{frame_idx:05d}.png"
                    save_path = osp.join(save_dir, save_name)
                    predict_seg(seg_predictor, resized, save_path)
                    pbar.update(1)

            frame_idx += 1

    cap.release()
    print(f"[完成] 分割完成 -> {save_dir}")


def process_dataset(input_root, output_root, shard_id=0, num_shards=1):
    videos = collect_videos(input_root)
    if num_shards > 1:
        videos = [v for i, v in enumerate(videos) if i % num_shards == shard_id]
        logger.info(f"分片 {shard_id}/{num_shards}，本进程处理 {len(videos)} 个视频")

    for input_path in videos:
        process_video(input_path, output_root, input_root)


def parse_args():
    parser = argparse.ArgumentParser(description="批量跟踪 + 人像分割（Drone-YOLO + PP-HumanSeg）")
    parser.add_argument(
        "--input_root",
        default="/data/liaoqi/dronegait1/DroneGait_video/high",
    )
    parser.add_argument(
        "--output_root",
        default="/data/liaoqi/dronegait1/dronegait_droneyolo_LITE_192*192",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="当前进程分片编号（从 0 开始）",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="总分片数，多卡时设为 GPU 数量",
    )
    parser.add_argument(
        "--track-gpu-id",
        type=int,
        default=0,
        help="YOLO 跟踪使用的 GPU 编号",
    )
    parser.add_argument(
        "--seg-gpu-id",
        type=int,
        default=-1,
        help="分割 GPU 编号；-1 表示 CPU（单卡推荐，避免与 YOLO 冲突）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seg_gpu_id = args.seg_gpu_id if args.seg_gpu_id >= 0 else None
    init_models(track_gpu_id=args.track_gpu_id, seg_gpu_id=seg_gpu_id)
    process_dataset(
        args.input_root,
        args.output_root,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
