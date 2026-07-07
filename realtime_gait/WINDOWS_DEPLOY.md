# Windows 本地部署指南（realtime_gait）

本指南用于在 Windows 本机部署你当前仓库中的图传无人机步态识别流水线。

## 1. 推荐环境

- Windows 10/11 x64
- Python 3.10（推荐）
- NVIDIA GPU + CUDA（推荐 11.8，需与 PyTorch / Paddle 版本匹配）
- 可用 PowerShell

## 2. 代码准备

将整个仓库拷贝到 Windows，例如：

`D:\All-in-One-Gait`

> `realtime_gait` 依赖 `gait_runtime`、`configs` 以及模型权重目录，不能只单独复制 `realtime_gait` 子目录。

## 3. 建立虚拟环境

在 PowerShell 中执行：

```powershell
cd D:\All-in-One-Gait\OpenGait
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 4. 安装 PyTorch（先装）

按你的 CUDA 版本安装。CUDA 11.8 示例：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果只用 CPU，请改成 CPU 轮子源。

## 5. 安装 realtime_gait 依赖

```powershell
pip install -r realtime_gait\requirements-windows.txt
```

安装 Paddle（示例，实际请以 Paddle 官方 Windows GPU 轮子为准）：

```powershell
pip install paddlepaddle-gpu
```

## 6. 权重与配置检查

确保以下文件存在：

- `checkpoints/Drone-YOLO/best.pt`
- `checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml`
- `checkpoints/gait_model/GaitBase_DronGait1-60000.pt`
- `OpenGait/configs/gaitbase/gaitbase_da_dronegait1.yaml`

默认读取配置为：

- `OpenGait/realtime_gait/config/default.yaml`

如果你路径不同，可复制该文件并改成绝对路径后通过 `--config` 指定。

## 7. 运行方式

### 7.1 RTSP 图传实时识别

```powershell
cd D:\All-in-One-Gait\OpenGait
python -m realtime_gait.run_stream --stream rtsp://你的地址:8554/home
```

可加 gallery：

```powershell
python -m realtime_gait.run_stream --stream rtsp://你的地址:8554/home --gallery D:\path\gallery.pkl
```

### 7.2 本地视频验证

```powershell
python -m realtime_gait.main --video D:\data\test.mp4 --display
```

先建 gallery 再识别：

```powershell
python -m realtime_gait.main --video D:\data\gallery.mp4 --build-gallery
python -m realtime_gait.main --video D:\data\probe.mp4 --output D:\data\out.mp4
```

## 8. 常见问题

- `No module named xxx`：先确认虚拟环境已激活，再执行依赖安装。
- `CUDA unavailable, falling back to CPU`：PyTorch CUDA 版本与驱动/CUDA 不匹配。
- `No frame received within 15s`：图传地址不可达，或被防火墙拦截。
- `paddle` 安装失败：请使用 Paddle 官方给出的 Windows 对应版本安装命令。

## 9. 一键脚本

本目录已附带：

- `realtime_gait/scripts/setup_windows.ps1`：初始化环境（含 PyTorch CUDA11.8 默认安装）
- `realtime_gait/scripts/run_stream.bat`：快速启动图传识别
- `realtime_gait/scripts/run_video.bat`：快速启动本地视频验证

# python -m realtime_gait.main --video D:\py\py\realtime_gait_windows_needed_bundle\OpenGait\test\1.mp4 --display
#python -m realtime_gait.run_stream --stream rtsp://127.0.0.1:8554/home
