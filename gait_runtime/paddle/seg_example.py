import os
import cv2
import numpy as np
import paddle
from infer import Predictor_opengait  # 你自定义的分割类

# ========== 参数 ==========
video_path = '/home/liaoqi/Code_09/code/All-in-One-Gait/OpenGait/demo/output/InputVideos/6.mp4'
output_dir = 'output_silhouettes'
os.makedirs(output_dir, exist_ok=True)

# ========== 加载分割模型 ==========
seg_cfgs = {
    "model": {
        "seg_model": "./checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml",
    },
    "gait": {
        "dataset": "GREW",
    }
}
predictor = Predictor_opengait(seg_cfgs['model']['seg_model'])

# ========== 处理视频 ==========
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    bg = np.zeros((h, w, 3), dtype=np.uint8)  # 黑色背景

    # 预测：返回的是 (分割后图像, alpha)
    out_img, alpha = predictor.run(frame, bg)  # alpha.shape: (H, W, 1)

    # 生成剪影图
    alpha_gray = alpha.squeeze()  # (H, W)
    silhouette = (alpha_gray > 0.5).astype(np.uint8) * 255  # 黑底白人

    # 保存
    save_path = os.path.join(output_dir, f"{frame_idx:05d}.png")
    cv2.imwrite(save_path, silhouette)
    print(f"[保存] {save_path}")

    frame_idx += 1

cap.release()
