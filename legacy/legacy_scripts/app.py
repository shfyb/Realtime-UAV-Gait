import os
import cv2
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# ========= 你的模块 =========
from track_uav import track
from segment import seg
from recognise import extract_sil, compare

app = Flask(__name__)

# ===============================
# 绝对路径设置
# ===============================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
TRACK_FOLDER = os.path.join(BASE_DIR, 'static/tracking_frames')
SEG_FOLDER = os.path.join(BASE_DIR, 'static/seg_frames')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static/result_videos')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRACK_FOLDER, exist_ok=True)
os.makedirs(SEG_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ==========================================================
# 自动读取 seg 生成的文件夹并抽连续帧
# ==========================================================

def get_seg_samples(video_id, num_frames=10):

    base_dir = os.path.join(SEG_FOLDER, video_id)

    if not os.path.exists(base_dir):
        print("Seg folder not found:", base_dir)
        return []

    # 第一层目录（如 001）
    level1_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    if not level1_dirs:
        print("No level1 directory")
        return []

    level1_path = os.path.join(base_dir, level1_dirs[0])

    # 第二层目录（如 undefined）
    level2_dirs = sorted([
        d for d in os.listdir(level1_path)
        if os.path.isdir(os.path.join(level1_path, d))
    ])

    if not level2_dirs:
        print("No level2 directory")
        return []

    final_path = os.path.join(level1_path, level2_dirs[0])

    png_files = sorted([
        f for f in os.listdir(final_path)
        if f.endswith(".png")
    ])

    if not png_files:
        print("No png found")
        return []

    total = len(png_files)

    if total <= num_frames:
        selected = png_files
    else:
        start = max(0, total // 2 - num_frames // 2)
        selected = png_files[start:start + num_frames]

    img_urls = [
        url_for(
            'static',
            filename=f"seg_frames/{video_id}/{level1_dirs[0]}/{level2_dirs[0]}/{f}"
        )
        for f in selected
    ]

    print("Loaded seg frames:", len(img_urls))
    return img_urls


# ==========================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():

    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    video_uuid = str(uuid.uuid4())

    video_path = os.path.join(
        UPLOAD_FOLDER,
        video_uuid + "_" + filename
    )
    file.save(video_path)

    # ==================================================
    # 1️⃣ Tracking
    # ==================================================

    track_result = track(video_path, RESULT_FOLDER)

    cap = cv2.VideoCapture(video_path)
    tracking_images = []

    for count in range(5):
        ret, frame = cap.read()
        if not ret:
            break

        save_path = os.path.join(
            TRACK_FOLDER,
            f"{video_uuid}_{count}.jpg"
        )
        cv2.imwrite(save_path, frame)

        tracking_images.append(
            url_for(
                'static',
                filename=f"tracking_frames/{video_uuid}_{count}.jpg"
            )
        )

    cap.release()

    # ==================================================
    # 2️⃣ Segmentation（直接读取生成的文件夹）
    # ==================================================

    silhouettes = seg(video_path, track_result, SEG_FOLDER)

    seg_folder_name = video_uuid + "_5"

    seg_images = get_seg_samples(seg_folder_name, num_frames=10)

    # ==================================================
    # 3️⃣ Recognition
    # ==================================================

    try:
        feat = extract_sil(silhouettes, "static/")
        result_score = compare(feat, feat)
    except Exception as e:
        print("Recognition error:", e)
        result_score = "Error"

    # ==================================================
    # 4️⃣ 抽取结果视频 5 帧
    # ==================================================

    result_images = []

    result_video_path = os.path.join(
        RESULT_FOLDER,
        video_uuid + "_5.mp4"
    )

    if os.path.exists(result_video_path):

        cap = cv2.VideoCapture(result_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:

            frame_indices = np.linspace(
                0,
                total_frames - 1,
                5,
                dtype=int
            )

            for idx in frame_indices:

                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                save_path = os.path.join(
                    RESULT_FOLDER,
                    f"{video_uuid}_frame_{idx}.jpg"
                )

                cv2.imwrite(save_path, frame)

                result_images.append(
                    url_for(
                        'static',
                        filename=f"result_videos/{video_uuid}_frame_{idx}.jpg"
                    )
                )

        cap.release()

    # ==================================================

    return render_template(
        'index.html',
        tracking_images=tracking_images,
        seg_images=seg_images,
        result_images=result_images,
        result_score=result_score
    )


if __name__ == '__main__':
    app.run(debug=True)
