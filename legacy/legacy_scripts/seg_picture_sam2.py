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

LIBS_DIR = osp.dirname(osp.abspath(__file__))
OPENGAIT_ROOT = osp.abspath(osp.join(LIBS_DIR, "..", ".."))
DEFAULT_SAM2_CKPT = osp.join(
    OPENGAIT_ROOT, "demo/checkpoints/sam2/sam2.1_hiera_small.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
MIN_SAM2_CKPT_BYTES = 100_000_000

SAM2_CKPT_URLS = {
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

# === 配置参数 ===
track_cfgs = {
    "model": {
        "ckpt": "/home/liaoqi/Code_09/code/Drone-YOLO/runs/dronegait_droneyolo_s/train/weights/best.pt",
    },
    "device": "gpu",
    "save_result": "True",
}

sam2_cfgs = {
    "model": {
        "config": DEFAULT_SAM2_CONFIG,
        "ckpt": DEFAULT_SAM2_CKPT,
    },
    "mask_threshold": 127,
}

SEG_INPUT_SIZE = (192, 192)
MIN_FRAMES_PER_ID = 10
PAD_RATIO = 0.1

track_model = None
track_predictor = None
sam2_predictor = None
_device = "cuda:0"
tracker = BYTETracker(frame_rate=30)
frame_id_global = 0
tid_offset = 0
mark_tid = True


def ensure_sam2_installed():
    try:
        import sam2  # noqa: F401
        return
    except ImportError as exc:
        raise RuntimeError(
            "未安装 SAM2。请先执行：\n"
            "  git clone https://github.com/facebookresearch/sam2.git\n"
            "  cd sam2 && pip install -e .\n"
            "或在 gaitfusion 环境中：pip install SAM-2\n"
            "详见 https://github.com/facebookresearch/sam2"
        ) from exc


def ensure_sam2_ckpt(ckpt_path):
    ckpt_path = osp.abspath(ckpt_path)
    os.makedirs(osp.dirname(ckpt_path), exist_ok=True)

    if osp.isfile(ckpt_path) and osp.getsize(ckpt_path) >= MIN_SAM2_CKPT_BYTES:
        return ckpt_path

    if osp.isfile(ckpt_path):
        logger.warning(f"SAM2 权重不完整，删除后重新下载: {ckpt_path}")
        os.remove(ckpt_path)

    ckpt_name = osp.basename(ckpt_path)
    url = SAM2_CKPT_URLS.get(ckpt_name)
    if not url:
        raise RuntimeError(
            f"未知 SAM2 权重文件名: {ckpt_name}\n"
            f"支持: {list(SAM2_CKPT_URLS.keys())}"
        )

    logger.info(f"正在下载 SAM2 权重: {url}")
    import urllib.request
    urllib.request.urlretrieve(url, ckpt_path)

    if osp.getsize(ckpt_path) < MIN_SAM2_CKPT_BYTES:
        raise RuntimeError(f"SAM2 权重下载失败或损坏: {ckpt_path}")

    return ckpt_path


def init_models(gpu_id=0):
    global track_model, track_predictor, sam2_predictor, _device

    ensure_sam2_installed()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if track_cfgs["device"] == "gpu" and torch.cuda.is_available():
        _device = f"cuda:{gpu_id}"
    else:
        _device = "cpu"

    track_model = YOLO(track_cfgs["model"]["ckpt"])
    track_predictor = Predictor(model=model, device=_device)
    logger.info(f"Loaded Drone-YOLO tracker on {_device}")

    ckpt_path = ensure_sam2_ckpt(sam2_cfgs["model"]["ckpt"])
    config_path = sam2_cfgs["model"]["config"]
    sam2_model = build_sam2(config_path, ckpt_path, device=_device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    logger.info(f"Loaded SAM2 on {_device}: {config_path} | {ckpt_path}")


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


def padded_bbox(bbox, frame_shape):
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    h_img, w_img = frame_shape[:2]
    x1_new = max(0, int(x1 - PAD_RATIO * w))
    x2_new = min(w_img, int(x2 + PAD_RATIO * w))
    y1_new = max(0, int(y1 - PAD_RATIO * h))
    y2_new = min(h_img, int(y2 + PAD_RATIO * h))
    if x2_new <= x1_new or y2_new <= y1_new:
        return None
    return x1_new, y1_new, x2_new, y2_new


def mask_crop_to_silhouette(mask_bool, x1, y1, x2, y2):
    crop = mask_bool[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    mask_u8 = np.where(crop, 255, 0).astype(np.uint8)
    side = max(mask_u8.shape[:2])
    canvas = np.zeros((side, side), dtype=np.uint8)
    offset_x = (side - mask_u8.shape[1]) // 2
    offset_y = (side - mask_u8.shape[0]) // 2
    canvas[offset_y:offset_y + mask_u8.shape[0], offset_x:offset_x + mask_u8.shape[1]] = mask_u8
    return cv2.resize(canvas, SEG_INPUT_SIZE, interpolation=cv2.INTER_NEAREST)


def predict_sam2_frame(frame_bgr, bbox, save_path):
    """整帧 SAM2 box prompt，输出 192x192 二值轮廓。"""
    pad = padded_bbox(bbox, frame_bgr.shape)
    if pad is None:
        return False

    x1, y1, x2, y2 = pad
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_box = np.array([x1, y1, x2, y2], dtype=np.float32)

    with torch.inference_mode():
        sam2_predictor.set_image(frame_rgb)
        masks, scores, _ = sam2_predictor.predict(
            box=input_box,
            multimask_output=False,
        )

    if masks is None or len(masks) == 0:
        out_mask = np.zeros(SEG_INPUT_SIZE, dtype=np.uint8)
    else:
        mask_bool = masks[0].astype(bool)
        sil = mask_crop_to_silhouette(mask_bool, x1, y1, x2, y2)
        if sil is None:
            return False
        thr = sam2_cfgs["mask_threshold"]
        out_mask = np.where(sil >= thr, 255, 0).astype(np.uint8)

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, out_mask)
    return True


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
    for frames_info in valid_ids.values():
        for fidx, bbox in frames_info:
            frame_tasks.setdefault(fidx, []).append(bbox)
            seg_total += 1

    cap = cv2.VideoCapture(input_path)
    frame_idx = 0
    with tqdm(total=seg_total, desc=f"SAM2分割: {video_name}", unit="张") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_tasks:
                for bbox in frame_tasks[frame_idx]:
                    save_name = f"{frame_idx:05d}.png"
                    save_path = osp.join(save_dir, save_name)
                    predict_sam2_frame(frame, bbox, save_path)
                    pbar.update(1)

            frame_idx += 1

    cap.release()
    print(f"[完成] SAM2 分割完成 -> {save_dir}")


def process_dataset(input_root, output_root, shard_id=0, num_shards=1):
    videos = collect_videos(input_root)
    if num_shards > 1:
        videos = [v for i, v in enumerate(videos) if i % num_shards == shard_id]
        logger.info(f"分片 {shard_id}/{num_shards}，本进程处理 {len(videos)} 个视频")

    for input_path in videos:
        process_video(input_path, output_root, input_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量跟踪 + 人像分割（Drone-YOLO + SAM2）"
    )
    parser.add_argument(
        "--input_root",
        default="/data/liaoqi/dronegait1/DroneGait_video/high",
    )
    parser.add_argument(
        "--output_root",
        default="/data/liaoqi/dronegait1/dronegait_sam2",
    )
    parser.add_argument(
        "--sam2-config",
        default=DEFAULT_SAM2_CONFIG,
        help="SAM2 配置文件，如 configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument(
        "--sam2-ckpt",
        default=DEFAULT_SAM2_CKPT,
        help="SAM2 权重路径",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sam2_cfgs["model"]["config"] = args.sam2_config
    sam2_cfgs["model"]["ckpt"] = args.sam2_ckpt
    init_models(gpu_id=args.gpu_id)
    process_dataset(
        args.input_root,
        args.output_root,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
