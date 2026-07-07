import argparse
import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger

from track_yolov8 import Predictor, BYTETracker, Timer, YOLO
from track_id_merge import force_merge_all_track_id_dict

LIBS_DIR = osp.dirname(osp.abspath(__file__))
OPENGAIT_ROOT = osp.abspath(osp.join(LIBS_DIR, "..", ".."))
DEFAULT_SEG_CKPT = osp.join(
    OPENGAIT_ROOT, "demo/checkpoints/yolov8_seg/yolov8n-seg.pt"
)
MIN_SEG_CKPT_BYTES = 5_000_000

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
        "ckpt": DEFAULT_SEG_CKPT,
    },
    "person_cls": 0,
    "conf": 0.25,
    "iou": 0.6,
    "mask_threshold": 127,
}

SEG_INPUT_SIZE = (192, 192)
MIN_FRAMES_PER_ID = 10

track_model = None
track_predictor = None
seg_model = None
tracker = BYTETracker(frame_rate=30)
frame_id_global = 0
tid_offset = 0
mark_tid = True
_seg_device = "cuda:0"


def ensure_seg_ckpt(ckpt_path):
    """校验权重完整性，损坏时尝试重新下载。"""
    ckpt_path = osp.abspath(ckpt_path)
    os.makedirs(osp.dirname(ckpt_path), exist_ok=True)

    def _is_valid(path):
        return osp.isfile(path) and osp.getsize(path) >= MIN_SEG_CKPT_BYTES

    if _is_valid(ckpt_path):
        return ckpt_path

    if osp.isfile(ckpt_path):
        logger.warning(f"分割权重不完整，删除后重新下载: {ckpt_path}")
        os.remove(ckpt_path)

    logger.info("正在下载 YOLOv8-Seg 权重 yolov8n-seg.pt ...")
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = attempt_download_asset("yolov8n-seg.pt")
    if not _is_valid(downloaded):
        raise RuntimeError(
            f"YOLOv8-Seg 权重下载失败或损坏: {downloaded}\n"
            "请手动下载到 demo/checkpoints/yolov8_seg/yolov8n-seg.pt:\n"
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt"
        )

    if osp.abspath(downloaded) != ckpt_path:
        import shutil
        shutil.copy2(downloaded, ckpt_path)

    return ckpt_path


def init_models(gpu_id=0):
    """跟踪与分割均使用 Ultralytics/YOLO，可安全共用同一张 GPU。"""
    global track_model, track_predictor, seg_model, _seg_device

    if track_cfgs["device"] == "gpu" and torch.cuda.is_available():
        track_device = f"cuda:{gpu_id}"
        _seg_device = track_device
    else:
        track_device = "cpu"
        _seg_device = "cpu"

    track_model = YOLO(track_cfgs["model"]["ckpt"])
    track_predictor = Predictor(model=track_model, device=track_device)
    logger.info(f"Loaded Drone-YOLO tracker on {track_device}")

    seg_ckpt = ensure_seg_ckpt(seg_cfgs["model"]["ckpt"])
    seg_model = YOLO(seg_ckpt)
    logger.info(f"Loaded YOLOv8-Seg on {_seg_device}: {seg_ckpt}")


def reset_track_state():
    global tracker, frame_id_global, tid_offset, mark_tid
    tracker = BYTETracker(frame_rate=30)
    frame_id_global = 0
    tid_offset = 0
    mark_tid = True


def track(frame):
    global frame_id_global, tid_offset, mark_tid

    timer = Timer()
    outputs, img_info = track_predictor.inference(frame, timer)

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
    """裁剪 + 10% padding + 白底正方形画布 + resize，与 seg_picture.py 一致。"""
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


def predict_yolov8_seg(crop_bgr, save_path):
    """对裁剪图做 YOLOv8-Seg，输出 192x192 二值轮廓 PNG。"""
    h, w = crop_bgr.shape[:2]
    out_mask = np.zeros((h, w), dtype=np.uint8)

    predict_kwargs = {
        "conf": seg_cfgs["conf"],
        "iou": seg_cfgs["iou"],
        "imgsz": SEG_INPUT_SIZE[0],
        "device": _seg_device,
        "half": False,
        "verbose": False,
    }
    person_cls = seg_cfgs.get("person_cls")
    if person_cls is not None:
        predict_kwargs["classes"] = [person_cls]

    results = seg_model.predict(crop_bgr, **predict_kwargs)
    result = results[0]

    if result.masks is not None and len(result.masks) > 0:
        masks = result.masks.data.cpu().numpy()
        best_mask = None
        best_area = 0
        for mask in masks:
            area = float(mask.sum())
            if area > best_area:
                best_area = area
                best_mask = mask

        if best_mask is not None:
            mask_u8 = (best_mask * 255).astype(np.uint8)
            if mask_u8.shape != (h, w):
                mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            thr = seg_cfgs["mask_threshold"]
            out_mask = np.where(mask_u8 >= thr, 255, 0).astype(np.uint8)

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, out_mask)


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
                    predict_yolov8_seg(resized, save_path)
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
    parser = argparse.ArgumentParser(
        description="批量跟踪 + 人像分割（Drone-YOLO + YOLOv8-Seg）"
    )
    parser.add_argument(
        "--input_root",
        default="/data/liaoqi/dronegait1/DroneGait_video/high",
    )
    parser.add_argument(
        "--output_root",
        default="/data/liaoqi/dronegait1/dronegait_yolov8seg",
    )
    parser.add_argument(
        "--seg-ckpt",
        default=DEFAULT_SEG_CKPT,
        help="YOLOv8-Seg 权重路径",
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
        "--gpu-id",
        type=int,
        default=0,
        help="跟踪与分割共用的 GPU 编号",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seg_cfgs["model"]["ckpt"] = args.seg_ckpt
    init_models(gpu_id=args.gpu_id)
    process_dataset(
        args.input_root,
        args.output_root,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

#cd /home/liaoqi/Code_09/code/All-in-One-Gait/OpenGait && for i in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=$i python demo/libs/seg_picture_yolov8seg.py --shard-id $i --num-shards 5 & done; wait