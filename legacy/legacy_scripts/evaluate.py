import os
import os.path as osp
import numpy as np
import json
import logging
from datetime import datetime
import time
import shutil

from track_yolov8 import track
from segment import seg
from recognise_uav import extract_sil, compare, compute_mAP, cuda_dist


# ========== 初始化日志 ==========
log_filename = f"demo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log(msg):
    print(msg)
    logging.info(msg)

# ========== 路径配置 ==========
DATASET_ROOT = '/data/liaoqi/dronegait1/DroneGait_video/high'
SAVE_ROOT = './demo/output/'
GALLERY_NAMES = ['nm-01', 'nm-02']
CONFIG_PATH = './datasets/CASIA-B/dronegait.json'

def is_video_file(filename):
    return filename.endswith('.mp4')

def safe_remove_if_exists(file_path):
    if osp.exists(file_path):
        os.remove(file_path)
        log(f"[File] Removed existing file: {file_path}")

def load_test_subjects_from_json(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config['TEST_SET']

def extract_view_name(filename):
    name, _ = osp.splitext(filename)
    return name

def process_video(video_path, track_save_dir, seg_save_dir, feat_save_dir):
    basename = osp.splitext(osp.basename(video_path))[0]
    seg_result_dir = osp.join(seg_save_dir, basename)

    if osp.exists(seg_result_dir):
        log(f"[Remove] Existing segmentation folder found, deleting: {seg_result_dir}")
        shutil.rmtree(seg_result_dir)

    track_file = osp.join(track_save_dir, f"{basename}.txt")
    seg_file = osp.join(seg_save_dir, f"{basename}.npz")
    safe_remove_if_exists(track_file)
    safe_remove_if_exists(seg_file)

    

    log(f"[Track] Start tracking: {video_path}")
    t_start = time.time()

    track_result = track(video_path, track_save_dir)
    
    t_end = time.time()
    log(f"[Track] Finished tracking: {video_path} (Time: {t_end - t_start:.2f}s)")

    log(f"[Segment] Start segmentation: {video_path}")
    s_start = time.time()

    inputs = seg(video_path, track_result, seg_save_dir)

    s_end = time.time()

    if inputs is None or len(inputs) == 0:
        log(f"[Skip] No valid segmentation inputs for: {video_path}")
    else:
        log(f"[Segment] Segmentation done: {video_path} (Time: {s_end - s_start:.2f}s)")

    return inputs

def evaluate_by_view(feature, label, seq_type, view):
    probe_seq_dict = {'NM': ['nm-03', 'nm-04'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']}
    gallery_seq_dict = ['nm-01', 'nm-02']
    view_list = sorted(np.unique(view))
    num_rank = 1
    acc = {}
    map_ = {}

    for type_, probe_seqs in probe_seq_dict.items():
        acc[type_] = np.zeros((len(view_list), len(view_list), num_rank)) - 1.
        map_[type_] = np.zeros(len(view_list)) - 1.

        for v1, probe_view in enumerate(view_list):
            probe_mask = np.isin(seq_type, probe_seqs) & (view == probe_view)
            probe_x = feature[probe_mask]
            probe_y = label[probe_mask]

            if len(probe_y) == 0:
                continue
            probe_x = np.array(probe_x).astype(np.float32)

            for v2, gallery_view in enumerate(view_list):
                gallery_mask = np.isin(seq_type, gallery_seq_dict) & (view == gallery_view)
                gallery_x = feature[gallery_mask]
                gallery_y = label[gallery_mask]

                if len(gallery_y) == 0:
                    continue
                gallery_x = np.array(gallery_x).astype(np.float32)

                dist = cuda_dist(probe_x, gallery_x, 'cosine')
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                matches = np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, :num_rank]]
                acc_val = np.sum(np.cumsum(matches, axis=1) > 0, axis=0) * 100. / dist.shape[0]
                acc[type_][v1, v2, :] = np.round(acc_val, 2)

            gallery_mask_all = np.isin(seq_type, gallery_seq_dict)
            gallery_all_x = np.array(feature[gallery_mask_all]).astype(np.float32)
            dist = cuda_dist(probe_x, gallery_all_x, 'cosine')
            map_val = compute_mAP(dist.cpu().numpy(), probe_y, label[gallery_mask_all],
                                  view[probe_mask], view[gallery_mask_all])
            map_[type_][v1] = map_val

    log("\n=== Evaluation Results ===")
    for type_ in probe_seq_dict:
        acc_diag = acc[type_]
        mean_acc = np.mean(de_diag(acc_diag[:, :, 0], each_angle=True))
        mean_map = np.mean(map_[type_])
        log(f"{type_}@Rank-1: {mean_acc:.2f}%\t{type_}@mAP: {mean_map:.2f}%")


def de_diag(acc_matrix, each_angle=False):
    assert acc_matrix.shape[0] == acc_matrix.shape[1]
    res = []
    for i in range(acc_matrix.shape[0]):
        valid = np.delete(acc_matrix[i, :], i)
        if each_angle:
            res.append(np.mean(valid))
        else:
            res.extend(valid)
    return np.array(res)

def evaluate():
    test_subjects = load_test_subjects_from_json(CONFIG_PATH)
    log("Loaded test subjects: " + ", ".join(test_subjects))

    # 全局累计数据，用于逐步全评估
    all_features = []
    all_labels = []
    all_seq_types = []
    all_views = []

    for subject in test_subjects:
        subject_path = osp.join(DATASET_ROOT, subject)
        if not osp.isdir(subject_path):
            log(f"[Skip] Subject path not found: {subject_path}")
            continue

        for seq_folder in os.listdir(subject_path):
            seq_path = osp.join(subject_path, seq_folder)
            if not osp.isdir(seq_path):
                continue

            for video_name in os.listdir(seq_path):
                if not is_video_file(video_name):
                    log(f"[Skip] Not a video file: {video_name}")
                    continue

                video_path = osp.join(seq_path, video_name)
                log(f"\n{'='*50}\n[Start] Processing video: {video_path}\n{'='*50}")

                basename = osp.splitext(osp.basename(video_path))[0]
                feat_file = osp.join(SAVE_ROOT + 'GaitFeatures/', f"{basename}.npy")
                safe_remove_if_exists(feat_file)

                total_start = time.time()
                inputs = process_video(video_path,
                                       SAVE_ROOT + 'tracks/',
                                       SAVE_ROOT + 'try_seg/',
                                       SAVE_ROOT + 'GaitFeatures/')

                if inputs is None or len(inputs) == 0:
                    log(f"[Skip] No inputs from seg for video: {video_path}")
                    log(f"{'-'*50}\n{'-'*50}\n")
                    continue

                log(f"[Feature] Start extracting gait features: {video_path}")
                f_start = time.time()
                feat = extract_sil(inputs, SAVE_ROOT + 'GaitFeatures/')
                f_end = time.time()
                log(f"[Feature] Finished extracting gait features: {video_path} (Time: {f_end - f_start:.2f}s)")

                total_end = time.time()
                log(f"[Process] Total processing time for video {basename}: {total_end - total_start:.2f}s")

                all_features.append(feat)
                all_labels.append(subject)
                all_seq_types.append(seq_folder)
                all_views.append(extract_view_name(video_name))

                log(f"{'-'*50}\n{'-'*50}\n")

        # 当前 subject 所有序列处理完毕，进行累计评估
        if len(all_features) > 0:
            log(f"\n{'='*50}\n[Evaluate] Evaluating cumulative subjects (up to {subject})\n{'='*50}")
            all_features_arr = np.array(all_features, dtype=object)
            all_labels_arr = np.array(all_labels)
            all_seq_types_arr = np.array(all_seq_types)
            all_views_arr = np.array(all_views)

            evaluate_by_view(all_features_arr, all_labels_arr, all_seq_types_arr, all_views_arr)
            log("[Evaluate] Finished evaluating cumulative subjects\n" + '='*50)


if __name__ == '__main__':
    evaluate()