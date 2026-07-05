import os
import os.path as osp
import sys
import cv2
from pathlib import Path
import shutil
import torch
print("[Debug] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[Debug] torch sees device count:", torch.cuda.device_count())
print("[Debug] Current logical device:", torch.cuda.current_device())
print("[Debug] Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from ultralytics import YOLO  # YOLOv8官方库

from tracking_utils.predictor_yolov8 import Predictor
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from tracking_utils.visualize import plot_tracking, plot_track

# 如果你有自己写的预处理模块，请自行导入，否则删掉下面两行
from pretreatment import pretreat, imgs2inputs
sys.path.append((os.path.dirname(os.path.abspath(__file__))) + "/paddle/")
from seg_demo import seg_image

from loguru import logger

track_cfgs = {
    "model": {
        # yolov8n默认权重路径，如果你用自训练模型，改成对应路径
        #"ckpt": "./demo/checkpoints/bytetrack_model/yolov8n_uav.pt",
        "ckpt": "./demo/checkpoints/Drone-YOLO/best.pt",
    },
    "gait": {
        "dataset": "GREW",
    },
    "device": "gpu",
    "save_result": "True",
}

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]


def get_color(idx):
    if idx <= 4:
        color = colors[idx - 1]
    else:
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def load_yolov8_model():
    device = "cuda" if track_cfgs["device"] == "gpu" else "cpu"
    model_path = track_cfgs["model"]["ckpt"]
    model = YOLO(model_path)
    logger.info(f"Loaded YOLOv8 model on {device}")
    return model



model = load_yolov8_model()
"单目标追踪"
'''
def track(video_path, video_save_folder):
    """Tracks person in the input video

    Args:
        video_path (str): Path of input video
        video_save_folder (str): Tracking video storage root path after processing
    Returns:
        track_results (dict): Track information
    """
    device = "cuda" if track_cfgs["device"] == "gpu" else "cpu"
    predictor = Predictor(model=model, device=device)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = BYTETracker(frame_rate=30)
    timer = Timer()
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(video_save_folder, exist_ok=True)
    save_video_name = os.path.basename(video_path)
    save_video_path = osp.join(video_save_folder, save_video_name)
    print(f"video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    save_video_name = save_video_name.split(".")[0]
    results_all = []  # 全部轨迹记录
    raw_track_results = {}  # 用于后续筛选最多ID

    mark = True
    diff = 0
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if isinstance(outputs[0], torch.Tensor):
                outputs[0] = outputs[0].cpu().numpy()

            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs[0], [img_info['height'], img_info['width']], [img_info['height'], img_info['width']]
                )
                online_tlwhs = []
                online_ids = []
                online_scores = []

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    if mark:
                        mark = False
                        diff = tid - 1
                    tid = tid - diff

                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                        # 记录原始轨迹
                        if frame_id not in raw_track_results:
                            raw_track_results[frame_id] = []
                        raw_track_results[frame_id].append([tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])

                        results_all.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']

            if track_cfgs["save_result"] == "True":
                vid_writer.write(online_im)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    cap.release()
    vid_writer.release()

    # ========= 新增逻辑：找出出现次数最多的 ID 并重命名为001 ==========
    from collections import defaultdict

    id_counts = defaultdict(int)
    for frame_data in raw_track_results.values():
        for obj in frame_data:
            id_counts[obj[0]] += 1

    if not id_counts:
        print("⚠️ 没有检测到任何目标")
        return {}

    # 找出出现最多的 ID
    best_id = max(id_counts.items(), key=lambda x: x[1])[0]

    track_results = {}
    results = []

    for frame_id, objs in raw_track_results.items():
        for obj in objs:
            if obj[0] == best_id:
                # 重命名为001
                new_obj = [1, obj[1], obj[2], obj[3], obj[4]]
                if frame_id not in track_results:
                    track_results[frame_id] = []
                track_results[frame_id].append(new_obj)

                results.append(
                    f"{frame_id},001,{new_obj[1]:.2f},{new_obj[2]:.2f},{new_obj[3]:.2f},{new_obj[4]:.2f},-1,-1,-1\n"
                )

    if track_cfgs["save_result"] == "True":
        res_file = osp.join(video_save_folder, f"{save_video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

    return track_results
'''




'''
def track(video_path, video_save_folder):
    """Tracks person in the input video

    Args:
        video_path (str): Path of input video
        video_save_folder (str): Tracking video storage root path after processing
    Returns:
        track_results (dict): Track information
    """
    device = "cuda" if track_cfgs["device"] == "gpu" else "cpu"
    predictor = Predictor(model=model, device=device)


    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = BYTETracker(frame_rate=30)
    timer = Timer()
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(video_save_folder, exist_ok=True)
    save_video_name = os.path.basename(video_path)
    save_video_path = osp.join(video_save_folder, save_video_name)
    print(f"video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    save_video_name = save_video_name.split(".")[0]
    results = []
    track_results = {}
    mark = True
    diff = 0
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if isinstance(outputs[0], torch.Tensor):
                outputs[0] = outputs[0].cpu().numpy()
            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs[0], [img_info['height'], img_info['width']], [img_info['height'], img_info['width']]
                )
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    if mark:
                        mark = False
                        diff = tid - 1
                    tid = tid - diff
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        if frame_id not in track_results:
                            track_results[frame_id] = []
                        track_results[frame_id].append([tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if track_cfgs["save_result"] == "True":
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if track_cfgs["save_result"] == "True":
        res_file = osp.join(video_save_folder, f"{save_video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    return track_results

'''
from tqdm import tqdm

def track(video_path, video_save_folder): 
    """Tracks person in the input video

    Args:
        video_path (str): Path of input video
        video_save_folder (str): Tracking video storage root path after processing
    Returns:
        track_results (dict): Track information
    """
    device = "cuda" if track_cfgs["device"] == "gpu" else "cpu"
    
    predictor = Predictor(model=model, device=device)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = BYTETracker(frame_rate=30)
    timer = Timer()
    frame_id = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(video_save_folder, exist_ok=True)
    save_video_name = os.path.basename(video_path)
    save_video_path = osp.join(video_save_folder, save_video_name)
    print(f"video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    save_video_name = save_video_name.split(".")[0]
    results = []
    raw_track_results = {}  # 用于保存所有ID的检测结果
    mark = True
    diff = 0
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)

            if isinstance(outputs[0], torch.Tensor):
                outputs[0] = outputs[0].cpu().numpy()

            # ====== 只保留类别为行人的检测框 ======
            if outputs[0] is not None:
                detections = outputs[0]
                person_dets = []
                for det in detections:
                    class_id = int(det[5])
                    if class_id == 0 or class_id == 1:  # 行人类别一般是0,1
                        person_dets.append(det[:5])  # x1, y1, x2, y2, score
                if len(person_dets) > 0:
                    filtered_outputs = np.array(person_dets)
                else:
                    filtered_outputs = np.empty((0, 5))
            else:
                filtered_outputs = np.empty((0, 5))

            online_targets = tracker.update(
                filtered_outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']]
            )

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if mark:
                    mark = False
                    diff = tid - 1
                tid = tid - diff

                aspect_ratio = tlwh[2] / tlwh[3]
                area = tlwh[2] * tlwh[3]

                # 空中视角优化后的过滤条件
                if area < 100:
                    continue  # 太小
                if aspect_ratio > 3.0 or aspect_ratio < 0.3:
                    continue  # 太长或太扁
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

                if frame_id not in raw_track_results:
                    raw_track_results[frame_id] = []
                raw_track_results[frame_id].append([tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])

            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
            )

            if track_cfgs["save_result"] == "True":
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    cap.release()
    vid_writer.release()

    # ✅ 修改：直接保存所有 ID 的追踪结果（不做最多 ID 筛选与重命名）
    track_results = {}
    results = []
    for frame_id, objs in raw_track_results.items():
        for obj in objs:
            tid = obj[0]
            new_obj = [tid, obj[1], obj[2], obj[3], obj[4]]  # 保持原 ID 不变
            if frame_id not in track_results:
                track_results[frame_id] = []
            track_results[frame_id].append(new_obj)
            results.append(
                f"{frame_id},{tid:03d},{new_obj[1]:.2f},{new_obj[2]:.2f},{new_obj[3]:.2f},{new_obj[4]:.2f},-1,-1,-1\n"
            )

    if track_cfgs["save_result"] == "True":
        res_file = osp.join(video_save_folder, f"{save_video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

    return track_results



def writeresult(track_results, pgdict, video_path, video_save_folder):
    """
    使用已有的 tracking 结果和识别映射 pgdict 可视化识别结果到视频中

    Args:
        track_results (dict): {frame_id: [[tid, x, y, w, h], ...]}
        pgdict (dict): probe ID -> gallery ID 的匹配字典（如 {"probe-001": "gallery-123"}）
        video_path (str): 输入视频路径
        video_save_folder (str): 保存可视化视频的路径
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(video_save_folder, exist_ok=True)
    video_name = osp.basename(video_path).split(".")[0]
    first_key = next(iter(pgdict))
    gallery_name = pgdict[first_key].split("-")[0]
    save_video_name = f"G-{gallery_name}_P-{video_name}.mp4"
    save_video_path = osp.join(video_save_folder, save_video_name)
    print(f"Video save path: {save_video_path}")

    vid_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_id = 0
    results = []

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        online_tlwhs, online_ids, online_colors = [], [], []

        if frame_id in track_results:
            for obj in track_results[frame_id]:
                tid, x, y, w, h = obj
                pid = f"{video_name}-{tid:03d}"
                if pid not in pgdict:
                    continue
                gid = pgdict[pid]
                colorid = int(gid.split("-")[1])

                aspect_ratio = w / h
                area = w * h
                if area < 100 or aspect_ratio > 3.0 or aspect_ratio < 0.3:
                    continue

                online_tlwhs.append([x, y, w, h])
                online_ids.append(gid)
                online_colors.append(colorid)
                results.append(f"{frame_id},{gid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},-1,-1,-1\n")

        online_im = plot_track(
            frame, online_tlwhs, online_ids, online_colors,
            frame_id=frame_id + 1, fps=fps
        )

        if track_cfgs["save_result"] == "True":
            vid_writer.write(online_im)

        frame_id += 1

    cap.release()
    vid_writer.release()

    # 保存 txt
    if track_cfgs["save_result"] == "True":
        res_file = osp.join(video_save_folder, f"{save_video_name.split('.')[0]}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
