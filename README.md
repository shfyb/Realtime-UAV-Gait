# Realtime Gait · 实时无人机步态识别

面向 **俯视航拍 / RTSP 图传** 的端到端行人步态识别系统。  
从视频流中完成：检测 → 跟踪 → 轮廓分割 → 步态特征提取 → 档案库 1:N 身份比对。

```
图传 RTSP (BGR)
      │
      ▼
 TimeScheduler (~30 Hz)
      │
      ▼
 DroneYOLO → ByteTrack → PP-HumanSeg → GaitBase
      │                                    │
      └──────── Web 控制台 / CLI ──────────┘
```

**Web 控制台**：图传预览、模式切换、一键注册、档案库管理、实时 KPI 与模块耗时面板。

---

## 目录

1. [环境要求](#1-环境要求)
2. [获取代码与模型权重](#2-获取代码与模型权重)
3. [Python 环境安装](#3-python-环境安装)
4. [MediaMTX 图传服务（RTSP）](#4-mediamtx-图传服务rtsp)
5. [启动识别系统](#5-启动识别系统)
6. [Web 控制台使用流程](#6-web-控制台使用流程)
7. [命令行用法](#7-命令行用法)
8. [配置说明](#8-配置说明)
9. [常见问题](#9-常见问题)
10. [仓库结构](#10-仓库结构)

---

## 1. 环境要求

### 1.1 计算与软件

| 项目 | 建议 |
|------|------|
| 操作系统 | Windows 10/11 x64（Linux 亦可，步骤类似） |
| Python | 3.9 ~ 3.10 |
| GPU | NVIDIA，8 GB 显存以上（RTX 3060 / 4060 级别） |
| CUDA | 11.8（与 PyTorch / Paddle 版本对应） |
| 图传 | RTSP 地址可达；本机调试可用手机 / 摄像头 + MediaMTX |

### 1.2 无人机与图传设备

本系统面向 **无人机俯视航拍画面** 设计（Drone-YOLO / GaitBase 均在航拍视角数据上训练）。真机部署建议满足：

| 项目 | 建议 |
|------|------|
| 机型 | 支持 **自定义 RTMP 推流** 的消费级/行业无人机（如大疆 DJI 系列，需 App 或第三方图传支持 RTMP） |
| 相机视角 | **俯视 / 斜俯视**，能清晰看到行人全身及行走姿态；避免纯侧视或过低高度 |
| 飞行高度 | 约 **8 ~ 30 m**（视场需覆盖目标全身，且轮廓分辨率足够；过高会导致人形过小） |
| 图传方式 | 无人机 / 手机 App → **RTMP** 推至地面站 → **MediaMTX** 转 **RTSP** → 本系统拉流 |
| 分辨率 / 帧率 | **720p ~ 1080p**，**25 ~ 60 fps**；本流水线有效处理约 **30 Hz** |
| 网络 | 无人机与地面站在 **同一局域网** 或稳定链路；推流地址示例：`rtmp://<地面站IP>:1935/live/home` |
| 地面站 | 运行 MediaMTX + 本识别程序的 PC（见第 4 节）；与无人机之间延迟 < 200 ms 为佳 |

**图传链路示意：**

```
[无人机相机] → [App / 图传模块 RTMP 推流]
        → [地面站 MediaMTX :1935]
        → [RTSP :8554/home]
        → [realtime_gait 识别]
```

> **调试说明**：无无人机时，可用 **手机 / 本机摄像头 + OBS / ffmpeg** 向 MediaMTX 推流做功能验证；但检测与步态模型针对 **航拍俯视域** 训练，室内 webcam 效果可能偏弱，不代表真机表现。

---

## 2. 获取代码与模型权重

### 2.1 克隆仓库

```powershell
git clone https://github.com/你的用户名/realtime-gait.git
cd realtime-gait
```

### 2.2 放置模型权重

权重文件体积较大，**不包含在 Git 仓库中**。请将以下文件放入对应目录：

```
demo/checkpoints/
├── Drone-YOLO/
│   └── best.pt                              # 行人检测（Drone-YOLO）
├── gait_model/
│   └── GaitBase_DronGait1-60000.pt          # 步态识别（GaitBase）
└── seg_model/
    └── human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/
        ├── deploy.yaml                      # 仓库已含
        ├── model.pdmodel                    # 需自行下载
        └── model.pdiparams                  # 需自行下载
```

详细说明见 [`demo/checkpoints/README.md`](demo/checkpoints/README.md)。

**快速校验：**

```powershell
python -c "
from pathlib import Path
for p in [
    'demo/checkpoints/Drone-YOLO/best.pt',
    'demo/checkpoints/gait_model/GaitBase_DronGait1-60000.pt',
    'demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/model.pdiparams',
]:
    print('OK   ' if Path(p).is_file() else '缺失 ', p)
"
```

---

## 3. Python 环境安装

推荐使用 **Conda** 管理环境（以下以环境名 `realtime-gait` 为例）。

### 3.1 创建环境

```powershell
conda create -n realtime-gait python=3.9 -y
conda activate realtime-gait
cd D:\path\to\realtime-gait
```

### 3.2 安装 PyTorch（GPU / CUDA 11.8）

```powershell
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

验证：

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 3.3 安装 PaddlePaddle GPU

Windows 请使用 [Paddle 官方安装页](https://www.paddlepaddle.org.cn/install/quick) 给出的 **Windows + GPU** 命令，例如：

```powershell
pip install paddlepaddle-gpu==2.6.2 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

> **cuDNN 提示（Windows）**：若 Paddle GPU 初始化失败，可将 PyTorch 安装目录下 `torch\lib\cudnn*.dll` 复制到 `paddle\libs\`，再重试。

### 3.4 安装项目依赖

```powershell
pip install -r requirements.txt
```

或使用一键脚本（会创建 `.venv` 并安装 PyTorch + 依赖）：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

---

## 4. MediaMTX 图传服务（RTSP）

本系统从 **RTSP** 读取 BGR 视频帧。无人机、手机或本地摄像头通常先推流到 **MediaMTX**，再由识别程序拉流。

### 4.1 下载 MediaMTX

1. 打开 [MediaMTX Releases](https://github.com/bluenviron/mediamtx/releases)
2. 下载 Windows 版本，例如 `mediamtx_v1.18.2_windows_amd64.zip`
3. 解压到固定目录，例如：

```
D:\tools\mediamtx\
├── mediamtx.exe
└── mediamtx.yml
```

### 4.2 最小配置示例

编辑 `mediamtx.yml`，保留默认端口并增加路径（按需调整）：

```yaml
# RTSP 默认 :8554，RTMP 默认 :1935

paths:
  # 推流端 publish 到此路径（RTMP / RTSP / WebRTC 等）
  live/home:
    source: publisher

  # 识别程序读取此路径（从 live/home 转发，按需拉流）
  home:
    source: rtmp://127.0.0.1/live/home
    sourceOnDemand: yes
```

**数据流关系：**

```
[手机 / 无人机 / OBS / ffmpeg] ──RTMP 1935──► live/home
                                              │
                                              ▼
                                    MediaMTX 转发
                                              │
                                              ▼
                         realtime_gait ◄── RTSP 8554 /home
```

### 4.3 启动 MediaMTX

```powershell
cd D:\tools\mediamtx
.\mediamtx.exe .\mediamtx.yml
```

看到类似日志即表示服务就绪：

```
[RTSP] listener opened on :8554
[RTMP] listener opened on :1935
```

### 4.4 向 MediaMTX 推流

将画面推到 `live/home`：
手机 / 无人机 App  `rtmp://<电脑IP>:1935/live/home`  在 App 自定义 RTMP 推流中填写 



### 4.5 验证 RTSP 是否可用

```powershell
# ffplay 预览（需安装 ffmpeg）
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/home
```

或在 VLC：**媒体 → 打开网络串流** → 输入 `rtsp://127.0.0.1:8554/home`。

> 若识别程序与 MediaMTX 不在同一台机器，将 `127.0.0.1` 换为 MediaMTX 所在机器的局域网 IP，并放行防火墙 **8554 / 1935** 端口。

---

## 5. 启动识别系统

确保：**MediaMTX 已运行且 RTSP 有画面**、**模型权重已就位**、**Conda 环境已激活**。

### 5.1 Web 控制台（推荐）

```powershell
conda activate realtime-gait
cd D:\path\to\realtime-gait

python -m realtime_gait.web --port 7860 --stream rtsp://127.0.0.1:8554/home
```

浏览器打开：**http://127.0.0.1:7860/**（更新前端后请 `Ctrl+F5` 强刷）

或使用快捷脚本：

```powershell
.\run_web.bat
```

### 5.2 端口被占用

```powershell
# 查看占用 7860 的进程
netstat -ano | findstr :7860
# 结束进程（将 PID 替换为实际值）
taskkill /PID <PID> /F
```

---

## 6. Web 控制台使用流程

```
连接图传 → 预览模式 → 注册档案 → 识别模式 → 查看结果
```

### Step 1 · 连接图传

1. 右侧 **图传连接** 填写 RTSP 地址（默认 `rtsp://127.0.0.1:8554/home`）
2. 点击 **连接图传**，左上角出现 **LIVE** 与 FPS
3. 左侧视频区应显示检测框（绿框 + T1 跟踪编号）

### Step 2 · 预览模式

- 点击 **预览模式**：仅检测 + 跟踪，**不做**身份比对
- 画面显示 `T1`、`T2`… 表示跟踪 ID，不代表姓名

### Step 3 · 注册步态档案

1. **一键注册** 填写姓名（支持中文，如 `张三`）
2. 点击 **开始注册** → 让人正常行走约 **1.5 秒**
3. 观察采集进度（轮廓帧数 / 时长），足够后点击 **完成注册**
4. 系统生成 `output/gallery/Zhangsan.pkl` 并更新 `registry.json`

> 多人场景：可在 **注册目标** 下拉框或左侧识别卡片中点选 `T` 编号。

### Step 4 · 识别模式

- 点击 **识别模式**：加载 `output/gallery/` 下全部档案，进行 1:N 比对
- 匹配成功显示 **中文姓名**；未匹配仍显示 `T1`（陌生人或档案库无此人）

### Step 5 · 档案库管理

- **重新加载档案**：从磁盘重新读取 pkl
- **删除选中**：移除对应 pkl 与 registry 条目

---

## 7. 命令行用法

### RTSP 实时流

```powershell
python -m realtime_gait.run_stream --stream rtsp://127.0.0.1:8554/home --gallery output/gallery
```

### 本地视频（无需 MediaMTX）

```powershell
python -m realtime_gait.main --video D:\data\test.mp4 --display
```

### 命令行注册

```powershell
python -m realtime_gait.register_gallery --stream rtsp://127.0.0.1:8554/home --name 张三 --out output/gallery --auto
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STREAM_URL` | `rtsp://127.0.0.1:8554/home` | 未指定 `--stream` 时使用 |
| `RTSP_FALLBACK_URL` | 同上 | 断流重连备用地址 |

---

## 8. 配置说明

主配置文件：[`realtime_gait/config/default.yaml`](realtime_gait/config/default.yaml)

```yaml
device: cuda                 # cuda / cpu

segmentor:
  use_gpu: true                # GPU 分割约 12 ms；CPU 约 130 ms

detector:
  conf: 0.25                   # 检测置信度；室内 webcam 可适当降低

timing:
  min_sil_count: 15            # 注册 / 识别最少轮廓帧数
  min_sil_duration_sec: 1.5    # 最少有效时长（秒）

recognizer:
  distance_threshold: 100.0    # Gallery 比对距离阈值
```

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| Web 启动报 `No module named uvicorn` | `pip install uvicorn fastapi`，确认 conda 环境已激活 |
| `请先连接图传` / 注册无反应 | 先点 **连接图传** 看到 LIVE；或重启 Web 后强刷页面 |
| `7860 端口被占用` | 见 [5.2 节](#52-web-控制台推荐) 结束旧进程 |
| RTSP 无画面 | 确认 MediaMTX 运行中、推流成功、`ffplay` 能预览 |
| 一直显示 T1 | 未开识别模式 / 档案库为空 / 未匹配到 gallery |
| CUDA 不可用 | 检查 PyTorch CUDA 版本与驱动；`nvidia-smi` 是否正常 |
| Paddle GPU 失败 | 按 3.3 节安装对应 wheel；必要时复制 cuDNN DLL |
| 检测框很少 | 室内摄像头与 Drone-YOLO 训练域不同，可降低 `detector.conf` |
| 有效帧率 ~28 Hz | 正常；步态模块周期性运行，不影响跟踪与分割 |

---

## 10. 仓库结构

```
realtime-gait/
├── realtime_gait/       # 流水线、Web UI、CLI
├── opengait/            # GaitBase 推理框架
├── configs/             # 步态模型 YAML
├── demo/
│   ├── libs/            # 检测 / 跟踪 / 分割 / 步态封装
│   └── checkpoints/     # 模型权重（本地放置）
├── output/              # 运行时 gallery、截图（gitignore）
├── requirements.txt
├── run_web.bat          # 快捷启动 Web
└── docs/
```

---


