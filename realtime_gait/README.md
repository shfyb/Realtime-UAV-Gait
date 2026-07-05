# realtime_gait

实时无人机步态识别流水线（面向 **numpy BGR** 图传帧，约 **120fps** 输入）。

## 架构

```
图传 frame (BGR uint8, 120fps)
        │
        ▼
┌───────────────────┐
│  FrameScheduler   │  process_stride=4 → 约 30Hz 进入 GPU 流水线
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   DroneYOLO       │  demo/checkpoints/Drone-YOLO/best.pt
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   ByteTrack       │  多目标、有状态
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  PP-HumanSeg v2   │  按 track_id 裁剪 → 64px 轮廓序列（内存环形缓冲）
└─────────┬─────────┘
          ▼
┌───────────────────┐
│   GaitBase        │  gaitbase_da_dronegait1 → 1:N 比对 gallery
└───────────────────┘
```

## 目录结构

```
realtime_gait/
├── config/default.yaml      # 默认配置（路径、stride、阈值）
├── core/                    # 调度器、每目标轮廓缓冲、数据结构
├── modules/                 # 四段模型封装（各只加载一次）
├── utils/                   # 路径、轮廓预处理、可视化
├── pipeline.py              # RealtimeGaitPipeline 主入口
├── main.py                  # 命令行测试（视频文件）
└── examples/feed_frames.py  # 图传回调接入示例
```

## 依赖

图传解码额外需要：`pip install av`（PyAV）

与主工程相同，需已安装：

- `demo/checkpoints/Drone-YOLO/best.pt`
- `demo/checkpoints/seg_model/.../deploy.yaml`
- `demo/checkpoints/gait_model/GaitBase_DronGait1-60000.pt`
- `ultralytics`, `paddlepaddle-gpu`, PyTorch

## 快速使用

在 **仓库根目录**（包含 `realtime_gait/`、`demo/`、`configs/` 的目录）执行：

```bash
cd /path/to/All-in-One-Gait/OpenGait

# 用本地视频模拟图传
python -m realtime_gait.main --video /data/.../test.mp4 --display

# 先注册 gallery 再识别
python -m realtime_gait.main --video gallery.mp4 --build-gallery
python -m realtime_gait.main --video probe.mp4 --output out.mp4
```

## 接入图传（RTSP / SRT，numpy BGR）

图传读取在 `stream_reader.py`（PyAV 低延迟解码），全流程入口：

```bash
cd path/to/realtime-gait

# 默认 RTSP（或环境变量 STREAM_URL）
python -m realtime_gait.input

# 指定地址 + 加载 gallery
python -m realtime_gait.run_stream --stream rtsp://127.0.0.1:8554/home --gallery gallery.pkl
```

数据流：

```text
RTSP/SRT → LatestFrameReader (后台线程 BGR) → process_frame() → 画框+ID 显示
```

手动接入示例见 `examples/feed_frames.py`。

## 配置说明（`config/default.yaml`）

| 参数 | 含义 |
|------|------|
| `input_fps` | 图传帧率（120） |
| `process_stride` | 每 N 帧跑一次检测+跟踪（4 → 约 30Hz） |
| `seg_stride` | 在已处理帧中，每 N 帧分割一次 |
| `min_sil_frames` | 至少多少张轮廓才做步态识别 |
| `gait_sample_frames` | 送入 GaitBase 的采样帧数 |
| `recognition_interval` | 同一 track 两次识别的最小间隔（轮廓帧数） |

120fps 下建议保持 `process_stride>=4`，否则 GPU 很难跟上。

## Gallery 注册（推荐流程）

| 场景 | 命令 |
|------|------|
| **全自动一键**（等人走 1.5s 后自动注册并保存） | `python -m realtime_gait.register_gallery --stream rtsp://... --name 张三 --out output/gallery.pkl --auto` |
| **图传窗口按键**（e 注册 / s 保存 / q 退出） | `python -m realtime_gait.run_stream --stream rtsp://... --gallery output/gallery.pkl` |
| **离线视频注册** | `python -m realtime_gait.register_gallery --video walk.mp4 --name 张三 --out output/gallery.pkl --auto` |
| **识别时加载** | `python -m realtime_gait.run_stream --gallery output/gallery.pkl` |

窗口快捷键：`e` 注册当前行人，`s` 保存 gallery，`q` 退出（退出时也会自动保存）。

## Web 控制台（图传 + 一键注册）

浏览器操作界面，适合地面准备与现场演示：

```bash
pip install fastapi "uvicorn[standard]"
python -m realtime_gait.web --port 7860 --stream rtsp://127.0.0.1:8554/home
```

打开 http://127.0.0.1:7860/

| 按钮 | 作用 |
|------|------|
| 连接图传 | 开始拉 RTSP 并显示 MJPEG 画面 |
| 识别模式 | 与 gallery.pkl 比对，显示姓名 |
| **开始注册** | **从此刻起**采集步态轮廓（之前不计入） |
| 完成注册 | 提取特征并写入 gallery.pkl |
| 保存 Gallery | 手动保存档案 |

```python
# 方式 1：代码注册 embedding
pipeline.register_gallery_embedding("055", embedding_tensor)

# 方式 2：从离线 extract_sil 的 dict 加载
import pickle
gallery = pickle.load(open("gallery.pkl", "rb"))
pipeline.load_gallery_from_features(gallery)
```

## 文档

- 详细更新历史见 **[CHANGELOG.md](./CHANGELOG.md)**（后续改动请追加到该文件）

- 旧流程：`main.py` 三遍读 mp4 + 落盘 PNG  
- 新流程：`realtime_gait` 单帧 API、模型只加载一次、轮廓在内存按 `track_id` 累积  

旧代码未删除，可并行对比调试。
