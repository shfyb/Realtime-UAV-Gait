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
from demo.libs.paddle.seg_demo import seg_image

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

    # 🚀 直接返回所有 ID 的轨迹
    return raw_track_results


# ========== 获取输出路径结构 ==========
def get_output_path(video_path, output_root, input_folder):
    relative_path = Path(video_path).relative_to(input_folder)
    parent_dir = relative_path.parent  # e.g. 037/nm-01
    cam = Path(video_path).stem       # e.g. 6 from 6.mp4
    return os.path.join(output_root, str(parent_dir), cam)
'''
# ========== 分割 + 保存 ==========
def imageflow_demo(video_path, track_result, output_root, input_folder):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    base_output_path = get_output_path(video_path, output_root, input_folder)
    os.makedirs(base_output_path, exist_ok=True)

    pbar = tqdm(total=total_frames, desc=f"Segmenting {Path(video_path).name}", ncols=100)

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        if frame_id in track_result:
            for tidxywh in track_result[frame_id]:
                tid = tidxywh[0]  # 多ID保留
                x, y, w, h = tidxywh[1:]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                w, h = x2 - x1, y2 - y1

                x1_new = max(0, int(x1 - 0.1 * w))
                x2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(x2 + 0.1 * w))
                y1_new = max(0, int(y1 - 0.1 * h))
                y2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(y2 + 0.1 * h))
                new_w, new_h = x2_new - x1_new, y2_new - y1_new

                crop = frame[y1_new:y2_new, x1_new:x2_new]

                # 🛡️ 跳过无效框
                if crop.size == 0 or new_w <= 0 or new_h <= 0:
                    continue

                # 居中填充
                side = max(new_w, new_h)
                tmp_img = 255 * np.ones((side, side, 3), dtype=np.uint8)
                dw = (side - new_w) // 2
                dh = (side - new_h) // 2
                tmp_img[dh:dh + new_h, dw:dw + new_w] = crop
                tmp_resized = cv2.resize(tmp_img, (192, 192))

                # 统计每个 ID 的帧数
                id_frame_counts = defaultdict(int)
                for frame_id, objs in track_result.items():
                    for obj in objs:
                        id_frame_counts[obj[0]] += 1

                # 判断是否为多 ID
                if len(id_frame_counts) > 1:
                    # 多 ID 情况：保留帧率最多的 ID 在原路径，其他 ID 创建子文件夹
                    max_frame_id = max(id_frame_counts, key=id_frame_counts.get)
                    if tid == max_frame_id:
                        # 帧率最多的 ID 保存在原路径
                        save_name = f"{frame_id:06d}.png"
                        seg_image(tmp_resized, seg_cfgs["model"]["seg_model"], save_name, base_output_path)
                    else:
                        # 其他 ID 创建子文件夹
                        id_output_path = os.path.join(base_output_path, f"id{tid}")
                        # 检查是否已经处理过该子文件夹
                        if not os.path.exists(id_output_path) or len(list(Path(id_output_path).glob("*.png"))) == 0:
                            os.makedirs(id_output_path, exist_ok=True)
                            save_name = f"{frame_id:06d}.png"
                            seg_image(tmp_resized, seg_cfgs["model"]["seg_model"], save_name, id_output_path)
                else:
                    # 单 ID 情况：直接保存在原路径
                    save_name = f"{frame_id:06d}.png"
                    seg_image(tmp_resized, seg_cfgs["model"]["seg_model"], save_name, base_output_path)

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

# ========== 单视频处理 ==========
def full_video_process(video_path, output_root, input_folder):
    output_path = get_output_path(video_path, output_root, input_folder)
    if os.path.exists(output_path) and len(list(Path(output_path).glob("*.png"))) > 0:
        print(f"⏩ Skipping already segmented video: {video_path}")
        return

    print(f"\n🔍 Processing: {video_path}")
    track_result = track(video_path)

    if not track_result:
        print(f"⚠️ No valid track result for: {video_path}")
        return

    imageflow_demo(video_path, track_result, output_root, input_folder)
    print(f"✅ Saved: {output_path}")

# ========== 批量处理入口 ==========
def batch_process_all(input_folder, output_root):
    video_paths = list(Path(input_folder).rglob("*.mp4"))
    if not video_paths:
        print("❌ No videos found.")
        return

    pbar = tqdm(video_paths, desc="🚀 Processing Videos", ncols=100, file=sys.stdout, dynamic_ncols=True)

    for video_path in pbar:
        full_video_process(str(video_path), output_root, input_folder)
'''

# 临时修改版：
import shutil

def imageflow_demo(video_path, track_result, output_root, input_folder, custom_output_path=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    # 使用默认路径或自定义路径
    if custom_output_path is None:
        base_output_path = get_output_path(video_path, output_root, input_folder)
    else:
        base_output_path = custom_output_path

    os.makedirs(base_output_path, exist_ok=True)

    # --- 预先统计每个 id 的出现帧数（只做一次） ---
    id_frame_counts = defaultdict(int)
    for fid, objs in track_result.items():
        for obj in objs:
            id_frame_counts[obj[0]] += 1

    unique_ids = list(id_frame_counts.keys())
    multiple_ids = len(unique_ids) > 1
    max_id = None
    if multiple_ids:
        max_id = max(id_frame_counts, key=id_frame_counts.get)

    pbar = tqdm(total=total_frames, desc=f"Segmenting {Path(video_path).name}", ncols=100)

    # 读取视频帧并保存
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        if frame_id in track_result:
            for tidxywh in track_result[frame_id]:
                tid = tidxywh[0]
                x, y, w, h = tidxywh[1:]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                w_box, h_box = x2 - x1, y2 - y1

                x1_new = max(0, int(x1 - 0.1 * w_box))
                x2_new = min(frame_width, int(x2 + 0.1 * w_box))
                y1_new = max(0, int(y1 - 0.1 * h_box))
                y2_new = min(frame_height, int(y2 + 0.1 * h_box))
                new_w, new_h = x2_new - x1_new, y2_new - y1_new

                crop = frame[y1_new:y2_new, x1_new:x2_new]

                # 跳过无效框
                if crop.size == 0 or new_w <= 0 or new_h <= 0:
                    continue

                # 居中填充并缩放到 192x192
                side = max(new_w, new_h)
                tmp_img = 255 * np.ones((side, side, 3), dtype=np.uint8)
                dw = (side - new_w) // 2
                dh = (side - new_h) // 2
                tmp_img[dh:dh + new_h, dw:dw + new_w] = crop
                tmp_resized = cv2.resize(tmp_img, (192, 192))

                # 决定保存路径：
                # - 单 id：直接保存到 base_output_path
                # - 多 id：帧数最多的 id 保存在 base_output_path，其他 id 保存在 base_output_path/id{tid}
                '''
                if not multiple_ids:
                    save_dir = base_output_path
                else:
                    if tid == max_id:
                        save_dir = base_output_path
                    else:
                        save_dir = os.path.join(base_output_path, f"id{tid}")
                '''
                save_dir = base_output_path
                # 如果目标文件夹不存在则创建
                os.makedirs(save_dir, exist_ok=True)

                # 如果你希望跳过已经存在的单张图片（避免覆盖），可以在此判断存在性；目前直接覆盖/写入
                save_name = f"{frame_id:06d}.png"
                seg_image(tmp_resized, seg_cfgs["model"]["seg_model"], save_name, save_dir)

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()


def full_video_process(video_path, output_root, input_folder):
    base_output_path = get_output_path(video_path, output_root, input_folder)

    # 如果已经存在 png 文件，则认为已处理过并跳过（保留原有行为）
    if os.path.exists(base_output_path) and len(list(Path(base_output_path).glob("*.png"))) > 0:
        print(f"⏩ Skipping already processed video: {video_path}")
        return "unchanged"

    print(f"\n🔍 Processing: {video_path}")

    # 重新追踪
    track_result = track(video_path)

    if not track_result:
        print(f"⚠️ No valid track result for: {video_path}")
        return "skipped"

    # 分割并保存：根据每个 id 出现帧数决定是否创建子文件夹
    print(f"🟢 Generating segmentation for all IDs...")
    imageflow_demo(video_path, track_result, output_root, input_folder)
    print(f"✅ Saved: {base_output_path}")
    return "regenerated"




    
def batch_process_all(input_folder, output_root):
    video_paths = list(Path(input_folder).rglob("*.MP4"))
    if not video_paths:
        print("❌ No videos found.")
        return

    stats = {
        "total": len(video_paths),
        "unchanged": 0,   # 已处理跳过
        "regenerated": 0, # 新生成
        "skipped": 0      # 无效
    }

    pbar = tqdm(video_paths, desc="🚀 Processing Videos", ncols=100, file=sys.stdout, dynamic_ncols=True)

    for video_path in pbar:
        video_path = str(video_path)
        result = full_video_process(video_path, output_root, input_folder)

        if result == "unchanged":
            stats["unchanged"] += 1
        elif result == "regenerated":
            stats["regenerated"] += 1
        elif result == "skipped":
            stats["skipped"] += 1

    print("\n📊 Summary:")
    print(f"  总视频数: {stats['total']}")
    print(f"  已处理跳过: {stats['unchanged']}")
    print(f"  重新生成: {stats['regenerated']}")
    print(f"  跳过无效: {stats['skipped']}")





# ========== MAIN ==========
if __name__ == "__main__":
    input_folder = "/data/liaoqi/理想数据new"
    output_root = "/data/liaoqi/理想数据new/LX_PKL"
    batch_process_all(input_folder, output_root)
