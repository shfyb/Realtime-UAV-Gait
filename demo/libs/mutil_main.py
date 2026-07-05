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

    # 更新gallery路径
    gallery_video_path = "/data/liaoqi/12.11测试/12.11测试/45°/4510倍.MP4"
    
    # 定义多个probe视频路径
    probe_video_paths = [
        "/data/liaoqi/12.11测试/12.11测试/45°/4520倍.MP4",                # probe6
        "/data/liaoqi/12.11测试/12.11测试/45°/4528倍.MP4",                # probe7
        "/data/liaoqi/12.11测试/12.11测试/45°/4532倍.MP4",                # probe8
    ]

    # ===== 开始计时 =====
    start_time = time.time()

    # 先检查所有视频文件是否存在
    print("检查视频文件是否存在...")
    all_videos = [gallery_video_path] + probe_video_paths
    for video_path in all_videos:
        if not os.path.exists(video_path):
            print(f"警告: 视频文件不存在: {video_path}")
        else:
            print(f"找到视频: {video_path}")

    # tracking - gallery
    print("\n正在处理gallery视频...")
    try:
        gallery_track_result = track(gallery_video_path, video_save_folder)
        print(f"Gallery跟踪完成，结果长度: {len(gallery_track_result)}")
        
        # 检查gallery跟踪结果是否为空
        if not gallery_track_result or len(gallery_track_result) == 0:
            print("错误: Gallery跟踪结果为空，无法继续处理")
            return
            
    except Exception as e:
        print(f"Gallery跟踪失败: {e}")
        return

    # segmentation - gallery
    try:
        gallery_silhouette = seg(gallery_video_path, gallery_track_result, save_root+'/12.11seg/')
        print(f"Gallery分割完成，轮廓数量: {len(gallery_silhouette)}")
        
        # 检查gallery分割结果是否为空
        if not gallery_silhouette or len(gallery_silhouette) == 0:
            print("错误: Gallery分割结果为空，无法继续处理")
            return
            
    except Exception as e:
        print(f"Gallery分割失败: {e}")
        return

    # feature extraction - gallery
    try:
        gallery_feat = extract_sil(gallery_silhouette, save_root+'/GaitFeatures/')
        print(f"Gallery特征提取完成，特征维度: {len(gallery_feat)}")
        
        # 检查gallery特征是否为空
        if not gallery_feat or len(gallery_feat) == 0:
            print("错误: Gallery特征提取结果为空，无法继续处理")
            return
            
    except Exception as e:
        print(f"Gallery特征提取失败: {e}")
        return

    # 处理每个probe视频
    successful_probes = 0
    for i, probe_video_path in enumerate(probe_video_paths, 1):
        print(f"\n{'='*50}")
        print(f"正在处理第{i}个probe视频: {os.path.basename(probe_video_path)}")
        print(f"{'='*50}")
        
        if not os.path.exists(probe_video_path):
            print(f"视频文件不存在，跳过: {probe_video_path}")
            continue
            
        try:
            # tracking - probe
            print("正在进行目标跟踪...")
            probe_track_result = track(probe_video_path, video_save_folder)
            print(f"Probe跟踪完成，结果长度: {len(probe_track_result)}")
            
            # 检查probe跟踪结果是否为空
            if not probe_track_result or len(probe_track_result) == 0:
                print("警告: Probe跟踪结果为空，跳过此视频")
                continue

            # segmentation - probe
            print("正在进行轮廓分割...")
            probe_silhouette = seg(probe_video_path, probe_track_result, save_root+'/12.11seg/')
            print(f"Probe分割完成，轮廓数量: {len(probe_silhouette)}")
            
            # 检查probe分割结果是否为空
            if not probe_silhouette or len(probe_silhouette) == 0:
                print("警告: Probe分割结果为空，跳过此视频")
                continue

            # feature extraction - probe
            print("正在进行特征提取...")
            probe_feat = extract_sil(probe_silhouette, save_root+'/GaitFeatures/')
            print(f"Probe特征提取完成，特征维度: {len(probe_feat)}")
            
            # 检查probe特征是否为空
            if not probe_feat or len(probe_feat) == 0:
                print("警告: Probe特征提取结果为空，跳过此视频")
                continue

            # matching
            print("正在进行特征匹配...")
            gallery_probe_result = compare(probe_feat, gallery_feat)
            
            # 保存结果
            print("正在保存结果...")
            writeresult(gallery_probe_result, probe_video_path, video_save_folder)
            print(f"✅ 第{i}个probe视频处理完成")
            successful_probes += 1
            
        except Exception as e:
            print(f"❌ 处理第{i}个probe视频时发生错误: {e}")
            continue

    # ===== 结束计时 =====
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"处理完成！成功处理 {successful_probes}/{len(probe_video_paths)} 个probe视频")
    print(f"{'='*60}")
    
    if args.show_fps:
        # 估算总帧数（gallery + 所有probe视频的总帧数）
        total_frames = 0
        import cv2
        all_video_paths = [gallery_video_path] + probe_video_paths
        for video_path in all_video_paths:
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

        fps = total_frames / total_time if total_time > 0 else 0
        print(f"\n===== 模型整体性能统计 =====")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"总帧数: {total_frames} 帧")
        print(f"平均FPS: {fps:.2f} 帧/秒")
        print(f"成功处理的probe视频数量: {successful_probes}个")
        print(f"跳过的probe视频数量: {len(probe_video_paths) - successful_probes}个")

if __name__ == "__main__":
    main()