import os
import os.path as osp
import time
import sys
import argparse
sys.path.append(os.path.abspath('.') + "/demo/libs/")
from track_uav import *
from segment import *
from recognise import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 指定使用单卡

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_fps", action="store_true", help="是否打印整个模型的总FPS")
    args = parser.parse_args()

    output_dir = "./demo/output/OutputVideos/"
    os.makedirs(output_dir, exist_ok=True)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    video_save_folder = osp.join(output_dir, timestamp)
    
    save_root = './demo/output/'

    gallery_video_path = "/data/liaoqi/dronegait1/DroneGait_video/high/055/bg-01/8.mp4"
    probe1_video_path  = "/data/liaoqi/dronegait1/DroneGait_video/high/055/cl-01/3.mp4"

    # ===== 开始计时 =====
    start_time = time.time()

    # tracking
    gallery_track_result = track(gallery_video_path, video_save_folder)
    probe1_track_result  = track(probe1_video_path, video_save_folder)

    gallery_video_name = gallery_video_path.split("/")[-1]
    gallery_video_name = save_root+'/try_seg/'+gallery_video_name.split(".")[0]

    probe1_video_name  = probe1_video_path.split("/")[-1]
    probe1_video_name  = save_root+'/try_seg/'+probe1_video_name.split(".")[0]
    
    exist = os.path.exists(gallery_video_name) and os.path.exists(probe1_video_name)
    
    # segmentation
    gallery_silhouette = seg(gallery_video_path, gallery_track_result, save_root+'/try_seg/')
    probe1_silhouette  = seg(probe1_video_path , probe1_track_result , save_root+'/try_seg/')

    # feature extraction
    gallery_feat = extract_sil(gallery_silhouette, save_root+'/GaitFeatures/') 
    probe1_feat  = extract_sil(probe1_silhouette , save_root+'/GaitFeatures/')
    
    # matching
    gallery_probe1_result = compare(probe1_feat, gallery_feat)
        # ===== 结束计时 =====
    end_time = time.time()
    total_time = end_time - start_time
    if args.show_fps:
        # 估算总帧数（两段视频的总帧数）
        total_frames = 0
        import cv2
        for video_path in [gallery_video_path, probe1_video_path]:
            cap = cv2.VideoCapture(video_path)
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        fps = total_frames / total_time if total_time > 0 else 0
        print(f"\n===== 模型整体性能统计 =====")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"总帧数: {total_frames} 帧")
        print(f"平均FPS: {fps:.2f} 帧/秒")
    writeresult(gallery_probe1_result, probe1_video_path, video_save_folder)



    

if __name__ == "__main__":
    main()
