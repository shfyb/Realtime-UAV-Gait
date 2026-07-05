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
from demo.libs.paddle.seg_demo import predict_seg, load_seg_model

# ========== 配置 ==========
track_cfgs = {
    "model": {
        "fair_ckpt": "./demo/checkpoints/fairmot_model/fairmot_dla34.pth"
    },
    "device": "gpu",
    "save_result": "True",
}

seg_cfgs = {
    "model": {
        "seg_model": "./demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml"
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

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break
        blobs, _ = prep_image(frame, opt)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    return raw_track_results

# ========== 分割 + 保存 ==========
def imageflow_demo(video_path, track_result, output_folder, seg_model):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    pbar = tqdm(total=total_frames, desc=f"Segmenting {Path(video_path).name}", ncols=100)

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        if frame_id in track_result:
            for tidxywh in track_result[frame_id]:
                tid = tidxywh[0]
                x, y, w, h = tidxywh[1:]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

                # 边界裁剪
                x1_new = max(0, int(x1 - 0.1 * w))
                x2_new = min(int(cap.get(3)), int(x2 + 0.1 * w))
                y1_new = max(0, int(y1 - 0.1 * h))
                y2_new = min(int(cap.get(4)), int(y2 + 0.1 * h))
                new_w, new_h = x2_new - x1_new, y2_new - y1_new

                if new_w <= 0 or new_h <= 0:
                    # 无效框，直接跳过
                    continue

                crop = frame[y1_new:y2_new, x1_new:x2_new]
                side = max(new_w, new_h)
                tmp_img = 255 * np.ones((side, side, 3), dtype=np.uint8)
                dw = (side - new_w) // 2
                dh = (side - new_h) // 2

                # 修正防止越界
                h_crop, w_crop = crop.shape[:2]
                if h_crop == 0 or w_crop == 0:
                    continue
                target_h = min(new_h, side - dh)
                target_w = min(new_w, side - dw)
                crop = crop[:target_h, :target_w]

                tmp_img[dh:dh + target_h, dw:dw + target_w] = crop
                tmp_resized = cv2.resize(tmp_img, (192, 192))

                # 每个ID单独文件夹
                id_folder = os.path.join(output_folder, f"id{tid}")
                os.makedirs(id_folder, exist_ok=True)
                save_name = os.path.join(id_folder, f"{frame_id:06d}.png")
                predict_seg(seg_model, tmp_resized, save_name)

        frame_id += 1
        pbar.update(1)
    pbar.close()
    cap.release()



# ========== 单视频处理 ==========
def full_video_process(video_path, output_root, seg_model):
    video_name = Path(video_path).stem
    output_folder = os.path.join(output_root, video_name)

    # 递归查找 PNG 文件，确保能检测到子文件夹里的分割结果
    if os.path.exists(output_folder) and len(list(Path(output_folder).rglob("*.png"))) > 0:
        print(f"⏩ Skipping already segmented video: {video_path}")
        return

    print(f"\n🔍 Processing: {video_path}")
    track_result = track(video_path)

    if not track_result:
        print(f"⚠️ No valid track result for: {video_path}")
        return

    imageflow_demo(video_path, track_result, output_folder, seg_model)
    print(f"✅ Saved: {output_folder}")


# ========== 批量处理入口 ==========
def batch_process_all(input_folder, output_root):
    # 一次性加载分割模型
    seg_model = load_seg_model(seg_cfgs["model"]["seg_model"])

    video_paths = list(Path(input_folder).rglob("*.MP4"))
    if not video_paths:
        print("❌ No videos found.")
        return

    for video_path in tqdm(video_paths, desc="🚀 Processing Videos", ncols=100):
        full_video_process(str(video_path), output_root, seg_model)

# ========== MAIN ==========
if __name__ == "__main__":
    input_folder = "/data/liaoqi/Drone/NewVideo_26"
    output_root = "/data/liaoqi/Drone/DroneSeg"
    batch_process_all(input_folder, output_root)
