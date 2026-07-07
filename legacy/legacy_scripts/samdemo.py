import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

sys.path.append(os.path.abspath('.') + "/demo/libs/")

from fair_track_uav.lib.opts import opts
from fair_track_uav.lib.tracker.multitracker import JDETracker
from fair_track_uav.lib.tracking_utils.timer import Timer
from fair_track_uav.lib.datasets.dataset.jde import letterbox

# === SAM 相关 ===
from segment_anything import sam_model_registry, SamPredictor


# ========== 配置 ==========
track_cfgs = {
    "model": {
        "fair_ckpt": "./demo/checkpoints/fairmot_model/fairmot_dla34.pth"
    },
    "device": "gpu",
    "save_result": True,
}

sam_cfgs = {
    "model": {
        "type": "vit_l",
        "checkpoint": "/home/liaoqi/Code_09/code/segment-anything-main/segment-anything-main/sam_vit_l_0b3195.pth"
    }
}


# ========== 追踪 ==========
def load_tracker():
    opt = opts().init()
    opt.load_model = track_cfgs["model"]["fair_ckpt"]
    tracker = JDETracker(opt, frame_rate=30)
    return tracker, opt


def prep_image(img, opt):
    img_size = (opt.input_w, opt.input_h)
    img, ratio, padw, padh = letterbox(img, height=img_size[1], width=img_size[0])
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    return img.unsqueeze(0), img_size


def track(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    tracker, opt = load_tracker()
    timer = Timer()
    raw_track_results = {}
    mark = True
    diff = 0

    pbar = tqdm(total=total_frames, desc=f"Tracking {Path(video_path).name}", ncols=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break
        blobs, _ = prep_image(frame, opt)
        blobs = blobs.to(device)
        timer.tic()
        online_targets = tracker.update(blobs, frame)

        for t in online_targets:
            tid = t.track_id
            tlwh = t.tlwh
            if mark:
                mark = False
                diff = tid - 1
            tid -= diff
            raw_track_results.setdefault(frame_id, []).append([tid, *tlwh])

        frame_id += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    # 主ID选择：保留帧数最多的轨迹
    id_count = defaultdict(int)
    for objs in raw_track_results.values():
        for obj in objs:
            id_count[obj[0]] += 1

    if not id_count:
        return {}

    main_id = max(id_count.items(), key=lambda x: x[1])[0]
    track_results = {}
    for fid, objs in raw_track_results.items():
        for obj in objs:
            if obj[0] == main_id:
                track_results.setdefault(fid, []).append([1, *obj[1:]])

    return track_results


# ========== SAM ==========
def load_sam_model():
    sam = sam_model_registry[sam_cfgs["model"]["type"]](
        checkpoint=sam_cfgs["model"]["checkpoint"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor, device


# ========== 输出路径 ==========
def get_output_path(video_path, output_root, input_folder):
    relative_path = Path(video_path).relative_to(input_folder)
    parent_dir = relative_path.parent   # e.g. 037/nm-01
    cam = Path(video_path).stem         # e.g. 6 from 6.mp4
    return os.path.join(output_root, str(parent_dir), cam)


# ========== 分割 + 保存 ==========
def imageflow_demo(video_path, track_result, output_root, input_folder, predictor, output_size=192):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    output_path = get_output_path(video_path, output_root, input_folder)
    os.makedirs(output_path, exist_ok=True)

    pbar = tqdm(total=total_frames, desc=f"Segmenting {Path(video_path).name}", ncols=100)

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        if frame_id in track_result:
            predictor.set_image(frame)  # 每帧只 set 一次

            for tidxywh in track_result[frame_id]:
                tid = tidxywh[0]
                x, y, w, h = tidxywh[1:]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

                # pad bbox，避免裁剪太紧
                pad_w = int(0.1 * (x2 - x1))
                pad_h = int(0.1 * (y2 - y1))
                x1_new = max(0, x1 - pad_w)
                y1_new = max(0, y1 - pad_h)
                x2_new = min(frame.shape[1] - 1, x2 + pad_w)
                y2_new = min(frame.shape[0] - 1, y2 + pad_h)

                if x2_new <= x1_new or y2_new <= y1_new:
                    continue  # 无效 bbox

                # SAM 分割
                mask = predictor.predict(
                    box=np.array([x1_new, y1_new, x2_new, y2_new]),
                    multimask_output=False
                )[0]

                mask = np.squeeze(mask).astype(np.uint8)
                mask_img = (mask * 255).astype(np.uint8)
                mask_crop = mask_img[y1_new:y2_new, x1_new:x2_new]

                if mask_crop.size == 0:
                    continue

                h, w = mask_crop.shape
                if h == 0 or w == 0:
                    continue

                # resize 到 output_size×output_size，居中
                scale = output_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h <= 0 or new_w <= 0:
                    continue

                mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                out_img = np.zeros((output_size, output_size), dtype=np.uint8)
                y_offset = (output_size - new_h) // 2
                x_offset = (output_size - new_w) // 2
                out_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized

                # 保存
                save_name = os.path.join(output_path, f"{frame_id:06d}.png")
                cv2.imwrite(save_name, out_img)

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()


# ========== 单视频处理 ==========
def full_video_process(video_path, output_root, input_folder, predictor):
    output_path = get_output_path(video_path, output_root, input_folder)
    if os.path.exists(output_path) and len(list(Path(output_path).glob("*.png"))) > 0:
        print(f"⏩ Skipping already segmented video: {video_path}")
        return

    print(f"\n🔍 Processing: {video_path}")
    track_result = track(video_path)

    if not track_result:
        print(f"⚠️ No valid track result for: {video_path}")
        return

    imageflow_demo(video_path, track_result, output_root, input_folder, predictor)
    print(f"✅ Saved: {output_path}")


# ========== 批量处理入口 ==========
def batch_process_all(input_folder, output_root, predictor):
    video_paths = list(Path(input_folder).rglob("*.mp4"))
    if not video_paths:
        print("❌ No videos found.")
        return

    pbar = tqdm(video_paths, desc="🚀 Processing Videos", ncols=100, file=sys.stdout, dynamic_ncols=True)

    for video_path in pbar:
        full_video_process(str(video_path), output_root, input_folder, predictor)


# ========== MAIN ==========
if __name__ == "__main__":
    input_folder = "/data/liaoqi/dronegait1/DroneGait_video/high"
    output_root = "/data/liaoqi/dronegait1/dronegait_sam"

    predictor, _ = load_sam_model()
    batch_process_all(input_folder, output_root, predictor)
