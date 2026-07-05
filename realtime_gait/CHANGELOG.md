# realtime_gait 更新记录

本文档记录 `OpenGait/realtime_gait/` 模块的演进历史，便于汇报、交接与回溯。

---

## 如何追加新更新

每次有较大改动时，在 **「更新条目」** 最上方（本段说明下方）新增一节，格式如下：

```markdown
## [YYYY-MM-DD] 简短标题

### 背景 / 动机
（为什么要改）

### 主要变更
- 变更点 1
- 变更点 2

### 涉及文件
- `path/to/file.py`

### 配置 / 行为变化
（如有）

### 使用方式变化
（如有）
```

---

## 更新条目

---

## [2026-06-02] 规格对齐：相似度门控识别 + 轨迹缓冲保持 + 行人类别过滤

按最新设计规格核对并实现：**墙钟调度**、**ByteTrack 自适应 dt**、**首次 1.5s/≥15 帧**、**每 1s 相似度复检**、**短时丢失保持 track 缓冲**。

### 规格对照

| 需求 | 状态 | 实现位置 |
|------|------|----------|
| 调度改为时间控制 | ✅ 已有 | `core/time_scheduler.py` |
| 首次识别：≥1.5s 且 ≥15 帧 | ✅ 已有 | `TrackSilhouetteBuffer.ready()` |
| 每 1s 做相似度计算，低于阈值才重跑 GaitBase | ✅ **本次实现** | `core/track_recognition.py` + `pipeline._evaluate_recognition()` |
| track_id 改变 → 重新识别 | ✅ **本次实现** | `TrackRecognitionState.track_id` 比对 |
| 短时丢失保持 track / 缓冲 | ✅ **本次实现** | `TrackBufferManager` grace = `max_time_lost_sec` |
| ByteTrack 动态 dt + 按秒 lost | ✅ 已有 | `modules/tracker.py` |
| Drone-YOLO 仅 pedestrian | ✅ **本次实现** | `detector.pedestrian_class_id=0` |

### 步态识别逻辑（当前）

```
ready? (≥15 帧 且 跨度 ≥1.5s)
   ├─ 否 → 仅显示已有缓存 ID
   └─ 是
        ├─ 首次 → 全量 GaitBase + gallery，保存 reference_sils
        ├─ track_id 与上次识别不一致 → 全量重识别
        └─ 每 1s
             ├─ silhouette_similarity(当前, reference) ≥ 0.75 → 沿用上次 ID
             └─ 相似度 < 0.75 → 全量 GaitBase + gallery
```

### 配置

```yaml
timing:
  min_sil_count: 15
  min_sil_duration_sec: 1.5
  recognition_interval_sec: 1.0

recognizer:
  sil_similarity_threshold: 0.75

tracker:
  max_time_lost_sec: 1.0   # 缓冲 grace 与之对齐

detector:
  pedestrian_class_id: 0
```

### 涉及文件

- `pipeline.py` — 识别状态机、grace prune
- `core/track_recognition.py` — 新增
- `core/track_buffer.py` — grace 延迟清理、`last_seen_ts`
- `modules/detector.py` — class 过滤
- `config/default.yaml` / `settings.py`

---

## [2026-06-02] Drone-YOLO 类别过滤 + ByteTrack 短时丢失轨迹保持

在已有「墙钟调度 + 自适应 ByteTrack」基础上，进一步收紧检测输入、延长轨迹与轮廓缓冲的生命周期，使 **检测 → 跟踪 → 分割 → 步态缓冲 → 识别** 在遮挡、漏检、FPS 抖动时更连贯。

### 背景 / 动机

1. **Drone-YOLO 类别过滤**  
   DroneGait 检测模型为 **单类别行人**（`DroneGait.yaml` 中 `names: 0: pedestrian`）。若推理结果未按类别过滤，误检框会进入 ByteTrack，引发多余 track_id、轮廓缓冲被污染、步态识别干扰。

2. **ByteTrack 短时丢失轨迹保持**  
   原逻辑在每一帧用 `active_ids` 立即 `buffers.prune()`：目标一旦被 ByteTrack 标为 **Lost**（遮挡、漏检、短暂出画），**track_id 与已攒轮廓序列会被立刻删除**，导致：
   - 步态缓冲从头清零，`min_sil_count` / `min_sil_duration_sec` 难以满足；
   - ID switch 后同一人轮廓被拆到多个 track；
   - 识别结果闪烁或长时间无法输出。

   步态识别依赖 **连续轮廓时间序列**，需要在 ByteTrack 允许的丢失窗口内 **保留 track_id 与 silhouette buffer**，而不是「丢失即删」。

---

### 端到端流程（当前设计）

```
RTSP 图传帧 (numpy BGR)
        │
        ▼
┌───────────────────────────────────────┐
│ TimeScheduler（墙钟）                  │
│  process_interval_ms ≈ 33ms → 检测+跟踪 │
│  seg_interval_ms ≈ 67ms → 分割         │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ ① Drone-YOLO 检测                      │
│    仅保留 pedestrian 类（class_id=0）   │
│    输出 (x1,y1,x2,y2,score)            │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ ② ByteTrack 跟踪（自适应 dt）           │
│    稳定 track_id；短时 Lost 不立刻删缓冲  │
│    max_time_lost_sec 内可 re-activate   │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ ③ PP-HumanSeg 分割                     │
│    按 bbox 裁剪 → 192² → 64px 轮廓      │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ ④ 写入 gait 轮廓序列（按 track_id）     │
│    TrackSilhouetteBuffer：带时间戳 deque │
│    保留最近 sil_buffer_duration_sec     │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ ⑤ 等待足够 silhouette + 时间跨度        │
│    min_sil_count ≥ 15                   │
│    min_sil_duration_sec ≥ 1.5s         │
│    （短时丢失期间缓冲仍保留，可继续累积）  │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ ⑥ 时间控制后再触发识别                  │
│    recognition_interval_sec ≥ 1.0s     │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ ⑦ GaitBase 步态识别                    │
│    均匀采样 gait_sample_frames=30       │
│    与 gallery 1:N 比对 → 输出 ID       │
└───────────────────┬───────────────────┘
                    ▼
           画框 + gallery ID + HUD
```

**要点**：第 ④~⑦ 步依赖 **同一 track_id 下连续的轮廓时间序列**；第 ② 步的「短时丢失保持」是为第 ⑤ 步服务，避免偶发 Lost 打断步态累积。

---

### 主要变更

#### 1. Drone-YOLO 类别过滤

| 项目 | 说明 |
|------|------|
| 过滤规则 | Ultralytics 输出 `boxes.data` 为 `[x1,y1,x2,y2,conf,cls]` 时，**仅保留 `cls == 0`（pedestrian）** |
| 数据流 | 过滤后再送入 ByteTrack，减少误检 track |
| 配置 | 可与 `detector.conf` / `detector.iou` 配合；单类模型下 cls 恒为 0 |

**预期效果**：

- 减少非行人误检进入跟踪链；
- 降低多余 track_id 与无效轮廓；
- 提升步态缓冲「有效帧」比例。

#### 2. ByteTrack 短时丢失轨迹保持

| 项目 | 原行为 | 优化后 |
|------|--------|--------|
| 轮廓缓冲 `prune` | 仅当前帧 `online_targets` 的 id 保留，**Lost 即删** | 在 `max_time_lost_sec`（及可配置 grace）内，**保留 track_id 与已攒轮廓** |
| track_id | Lost 后 ByteTrack 可能 re-activate 同 id，但缓冲已被删 | 丢失窗口内缓冲连续，re-activate 后步态序列不中断 |
| 识别缓存 | `_recognition_cache` 随 track 被 prune 丢失 | 与 buffer 同生命周期延长，减少 ID 闪烁 |
| 与自适应 dt 关系 | `max_time_lost_sec` 已换算为帧数 | 缓冲 grace **与 ByteTrack lost 窗口对齐**，时间语义一致 |

**逻辑关系**：

```
ByteTrack Lost 状态（≤ max_time_lost_sec）
        ║  并行
        ║  TrackSilhouetteBuffer 不立即 prune
        ║  recognition_cache 保留
        ▼
Re-activate 或 超时后真正清除
```

#### 3. 与时间调度的配合

- **检测/跟踪/分割**仍由 `TimeScheduler` 按毫秒触发，不依赖稳定 FPS；
- **步态触发**仍由 `min_sil_count` + `min_sil_duration_sec` + `recognition_interval_sec` 控制；
- 短时丢失保持使「等待足够 silhouette」在 **真实飞行遮挡场景** 下仍可完成，而不因几帧漏检重置计数。

---

### 涉及文件

| 文件 | 变更 |
|------|------|
| `modules/detector.py` | Drone-YOLO 输出按 `class_id` 过滤，仅 pedestrian |
| `modules/tracker.py` | 暴露 lost / 全活跃 id 集合（或与 buffer grace 对齐） |
| `core/track_buffer.py` | `prune()` 改为带 **grace 时间** 的延迟清理，而非每帧硬删 |
| `pipeline.py` | 合并 online + 短时 lost 的 track_id 再管理 buffer；识别缓存同步 |
| `config/default.yaml` | 可选 `buffer_grace_sec` / `detector.pedestrian_class_id` |
| `config/settings.py` | 对应配置项 dataclass |

---

### 配置 / 行为变化（建议值）

```yaml
detector:
  conf: 0.25
  iou: 0.6
  pedestrian_class_id: 0    # DroneGait 单类行人

tracker:
  max_time_lost_sec: 1.0    # ByteTrack 丢失容忍；buffer grace 与之对齐

timing:
  sil_buffer_duration_sec: 6.0
  min_sil_count: 15
  min_sil_duration_sec: 1.5
  recognition_interval_sec: 1.0
```

| 参数 | 作用 |
|------|------|
| `pedestrian_class_id: 0` | 检测阶段类别门槛 |
| `max_time_lost_sec` | ByteTrack 丢失判定 + 缓冲延迟清理的上限（墙钟秒） |
| `min_sil_duration_sec` | 步态触发要求轮廓序列至少跨越的真实时间 |

---

### 使用方式变化

- **运行命令不变**：`python -m realtime_gait.input`
- **现象变化**：
  - 误检框、幽灵 track 减少；
  - 短暂遮挡后 **同一 track 的 `sil_count` 不再归零**；
  - 首次 / 更新识别更稳定，HUD 上 ID 闪烁减轻。

---

### 调试建议

| 现象 | 可调参数 |
|------|----------|
| 仍有非行人误检 | 提高 `detector.conf`；确认权重为 DroneGait 单类模型 |
| 遮挡后仍丢 ID | 增大 `max_time_lost_sec` → `1.5` |
| 幽灵 track 残留过久 | 减小 buffer grace 或 `sil_buffer_duration_sec` |
| 识别仍慢 | 略降 `min_sil_duration_sec`（可能略降准确率） |

HUD 仍可通过 `trkHz`、`dt_kf`、`sil=`、`ready=` 观察跟踪与缓冲状态。

---

### 一句话总结（汇报用）

> 在实时步态流水线中增加 Drone-YOLO 行人类别过滤，并在 ByteTrack 短时丢失窗口内保持 track_id 与轮廓缓冲不删除，使「检测→跟踪→分割→轮廓累积→时间门控→GaitBase 识别」在遮挡与 FPS 抖动下仍能保持序列连续、减少 ID 切换与识别中断。

---

## [2026-06-01] 模块初建 + 图传接入 + 墙钟调度 + 自适应 ByteTrack

本次为 `realtime_gait` 从 0 到可跑通 RTSP 图传全流程的完整迭代，包含四次逻辑演进（架构 → 图传 → 时间调度 → 跟踪自适应）。

### 背景 / 动机

原 `demo/libs` 为 **离线批处理**：两段 mp4 → 全片跟踪 → 再读一遍视频分割 → 落盘 PNG → 提特征 → 比对。无法对接无人机 **RTSP 图传**，且假设帧率稳定，在真实网络 jitter 下调度与跟踪会失真。

目标：新建独立模块，实现 **DroneYOLO + ByteTrack + PP-HumanSeg + GaitBase** 实时流水线，输入为 **numpy BGR**，并容忍 **FPS 不稳定**。

---

### 一、总体定位

| 维度 | 原 `demo/libs` | 新 `realtime_gait` |
|------|----------------|---------------------|
| 输入 | 视频文件路径 | numpy BGR 帧 / RTSP |
| 处理模式 | 整段视频批处理 | 逐帧有状态流式 |
| 中间结果 | 大量 PNG + 多遍读视频 | 内存缓冲，不落盘 |
| 模型加载 | 识别阶段易重复加载 | 四段模型启动时各加载一次 |
| 调度 | 按帧序号（如 `%4`） | **墙钟毫秒间隔** |
| 跟踪 | 固定 `frame_rate=30` | **动态 dt + 按秒 lost** |
| Gallery | 两段视频 offline 比对 | 预加载 pkl 或运行时注册 |

旧代码 **未删除**，可与新模块并行调试。

---

### 二、目录结构（26 个文件）

```
realtime_gait/
├── input.py                 # 图传+识别一键入口
├── run_stream.py            # 全流程主循环（读流→推理→显示）
├── stream_reader.py         # PyAV 低延迟 RTSP 解码
├── pipeline.py              # RealtimeGaitPipeline 核心
├── main.py                  # 本地 mp4 测试入口
├── CHANGELOG.md             # 本文件
├── README.md
├── config/
│   ├── default.yaml
│   └── settings.py
├── core/
│   ├── time_scheduler.py    # 墙钟时间调度
│   ├── frame_scheduler.py   # 旧帧计数调度（保留，已弃用）
│   ├── track_buffer.py      # 带时间戳的轮廓缓冲
│   └── types.py
├── modules/
│   ├── detector.py          # DroneYOLO
│   ├── tracker.py           # 自适应 ByteTrack
│   ├── segmentor.py         # PP-HumanSeg
│   └── recognizer.py        # GaitBase + GalleryStore
├── utils/
│   ├── paths.py
│   ├── silhouette.py
│   └── visualize.py
└── examples/
    └── feed_frames.py
```

---

### 三、技术架构

```
RTSP/SRT 图传
    ↓
LatestFrameReader（后台线程，BGR uint8，只保留最新帧）
    ↓
RealtimeGaitPipeline.process_frame(frame_bgr, wall_ts)
    ↓
① DroneYOLO 检测（demo/checkpoints/Drone-YOLO/best.pt）
    ↓
② ByteTrack 多目标跟踪（自适应 dt）
    ↓
③ PP-HumanSeg 分割（按 track_id 裁剪 → 轮廓）
    ↓
④ GaitBase 步态识别（gaitbase_da_dronegait1，与 gallery 1:N 比对）
    ↓
OpenCV 窗口：检测框 + 身份 ID + HUD 状态信息
```

---

### 四、迭代 1：流式 pipeline 架构

**新增** `RealtimeGaitPipeline` 与四段 `modules/` 封装：

- **DroneYOLO**：`tracking_utils/predictor_yolov8.Predictor` + Ultralytics 权重
- **ByteTrack**：`demo/libs/tracker/byte_tracker.py`
- **PP-HumanSeg**：`load_seg_model()` 只加载一次（修复原 `seg_image()` 每帧重建 predictor）
- **GaitBase**：`gaitbase_da_dronegait1.yaml`，ckpt hint 60000

**单帧 API**：`process_frame(frame_bgr)` / `process_frame_visualized()`

**轮廓处理**：`utils/silhouette.py` 内存裁剪、打包为 OpenGait 输入，无需 PNG 落盘。

**Gallery**：`GalleryStore` 支持 `register_embedding()` 与从 `recognise.extract_sil` 的 pkl 导入。

---

### 五、迭代 2：图传接入

| 文件 | 作用 |
|------|------|
| `stream_reader.py` | 从原 `input.py` 抽出 `LatestFrameReader` |
| `run_stream.py` | 读流 → pipeline → HUD 显示 |
| `input.py` | 统一入口，`python -m realtime_gait.input` |

**LatestFrameReader 要点**：

- PyAV + FFmpeg 低延迟：`nobuffer`、`low_delay`、`reorder_queue_size=0` 等
- 后台线程解码，主线程 `get_latest()` 取 `(frame_bgr, frame_ts, total_read)`
- `total_read` 去重，避免同一帧重复喂 pipeline
- 自动重连、备用 URL；环境变量 `STREAM_URL`、`RTSP_FALLBACK_URL`

**运行方式**：

```bash
cd OpenGait
python -m realtime_gait.input
STREAM_URL=rtsp://ip:8554/home python -m realtime_gait.input
python -m realtime_gait.run_stream --stream rtsp://... --gallery gallery.pkl
```

额外依赖：`pip install av`（PyAV）

---

### 六、迭代 3：帧计数 → 墙钟时间调度

**动机**：RTSP 帧率波动、PyAV burst 解码、网络 jitter 使「每 N 帧处理一次」不可预测。

**新增** `core/time_scheduler.py`，替代 `FrameScheduler`（后者保留但弃用）。

**默认配置**（`config/default.yaml` → `timing:`）：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `process_interval_ms` | 33.3 | 检测+跟踪最短间隔（~30Hz） |
| `seg_interval_ms` | 66.7 | 分割最短间隔（~15Hz） |
| `min_sil_count` | 15 | 首次识别最少轮廓张数 |
| `min_sil_duration_sec` | 1.5 | 轮廓时间跨度至少 1.5 秒 |
| `sil_buffer_duration_sec` | 6.0 | 每 track 只保留最近 6 秒轮廓 |
| `recognition_interval_sec` | 1.0 | 同一人两次识别至少间隔 1 秒 |
| `gait_sample_frames` | 30 | 送入 GaitBase 的采样帧数 |

**轮廓缓冲**（`core/track_buffer.py`）：

- 每条 `SilEntry(ts, sil)` 带时间戳
- 识别触发：**张数 ≥ min_sil_count 且 时间跨度 ≥ min_sil_duration_sec**
- 之后每 `recognition_interval_sec` 可再识别

**废弃字段**（`settings.py` 中 `_apply_legacy_frame_config()` 可自动换算）：

- `input_fps`、`process_stride`、`seg_stride`
- `min_sil_frames`、`max_sil_buffer`、`recognition_interval`

---

### 七、迭代 4：ByteTrack 自适应 frame_rate / dt

**动机**：即使调度约 30Hz，实际帧间隔仍不稳定；固定 `frame_rate=30` 导致：

- 卡尔曼假设每步 `dt=1`，预测位移不准 → IOU 匹配失败 → **ID switch**
- `max_time_lost` 按固定帧数，真实丢失容忍时间漂移

**实现**（`modules/tracker.py` → `ByteTrackEngine`）：

每次 `update(..., wall_ts=now)` 前 `_adapt_timing(wall_ts)`：

1. **测量间隔**：`dt_sec = now - last_ts`，钳位 `[dt_min_sec, dt_max_sec]`
2. **EMA 平滑**：`ema_dt`，`effective_fps = 1/ema_dt`
3. **卡尔曼 dt**：`dt_kf = dt_sec / ema_dt`，更新 `motion_mat`（实例 + `STrack.shared_kalman`）
4. **丢失判定**：`max_time_lost_frames = round(max_time_lost_sec / ema_dt)`

**新增 tracker 配置**：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `frame_rate` | 30 | 仅初始化 nominal |
| `max_time_lost_sec` | 1.0 | 丢失容忍（墙钟秒） |
| `dt_ema_alpha` | 0.2 | 间隔 EMA 平滑 |
| `dt_min_sec` | 0.005 | 最小间隔钳位 |
| `dt_max_sec` | 0.5 | 最大间隔钳位 |

**可观测性**：HUD 显示 `trkHz`、`dt_kf`；`timings_ms` 含 `track_fps`、`track_dt_kf`。

---

### 八、开启图传后的处理时序（当前默认参数）

以 **墙钟时间** 为准，与 ingest FPS 无关：

| 时刻 | 事件 |
|------|------|
| 0s | 连接 RTSP，加载四段模型，等待首帧 |
| 持续 | 后台解码；主循环取新帧 |
| 每 ≥33ms | DroneYOLO + 自适应 ByteTrack |
| 每 ≥67ms | PP-HumanSeg，轮廓写入 track 缓冲 |
| ~1.5s+ | 轮廓 ≥15 且跨度 ≥1.5s → 首次 GaitBase 识别 |
| 之后每 1s | 同 track 可再次识别 |
| 丢失 1s | ByteTrack 删除该 track |
| 6s | 超过 6 秒的旧轮廓从缓冲清除 |

**屏幕输出**：

- 每人：矩形框 + `T1` 或 `055-undefined d=23.5`
- 左上角：流 FPS、本地延迟、GPU 耗时、trkHz、dt_kf、轮廓数、ready 状态

**当前不自动落盘**结果视频/PNG（可后续扩展）。

---

### 九、涉及文件清单

| 文件 | 变更类型 |
|------|----------|
| `realtime_gait/` 整目录 | 新建 |
| `pipeline.py` | 核心流水线 |
| `stream_reader.py` | 图传解码 |
| `run_stream.py` | 全流程入口 |
| `input.py` | 快捷入口 |
| `main.py` | 本地视频测试 |
| `core/time_scheduler.py` | 墙钟调度 |
| `core/track_buffer.py` | 时间戳轮廓缓冲 |
| `core/frame_scheduler.py` | 遗留 |
| `modules/tracker.py` | 自适应 ByteTrack |
| `modules/detector.py` | DroneYOLO |
| `modules/segmentor.py` | PP-HumanSeg |
| `modules/recognizer.py` | GaitBase |
| `config/default.yaml` | 全部默认参数 |
| `config/settings.py` | 配置 dataclass + 旧 yaml 兼容 |
| `utils/*` | 路径、轮廓、可视化 |
| `examples/feed_frames.py` | 接入示例 |
| `README.md` | 使用说明 |

---

### 十、运行命令汇总

```bash
cd OpenGait

# 图传实时全流程
python -m realtime_gait.input

conda activate pytorch
cd D:\py\py\realtime_gait_windows_needed_bundle\OpenGait
python -m realtime_gait.web --port 7860 --stream rtsp://127.0.0.1:8554/home
http://127.0.0.1:7860/

cd D:\DJI\mediamtx_v1.18.2_windows_amd64
.\mediamtx.exe .\mediamtx.yml

# 指定流 + gallery
python -m realtime_gait.run_stream --stream rtsp://... --gallery gallery.pkl

# 本地 mp4 模拟
python -m realtime_gait.main --video test.mp4 --display

# 离线注册 gallery
python -m realtime_gait.main --video gallery.mp4 --build-gallery
```

---

### 十一、已知限制与后续方向

- [x] Gallery Web 控制台（图传 MJPEG + 一键注册按钮）
- [ ] 识别结果落盘（视频/日志）
- [ ] TensorRT / 模型量化
- [ ] 多人 ID 显示平滑（连续 K 次同 ID 才切换）
- [ ] ByteTrack 更深层 per-track dt（当前为全局 dt）

---

### 十二、一句话总结（汇报用）

> 新建 `realtime_gait` 实时步态识别模块，将 RTSP 图传（numpy BGR）接入 DroneYOLO→ByteTrack→PP-HumanSeg→GaitBase 四段流水线；调度由帧计数改为墙钟时间，轮廓缓冲与识别触发基于真实秒数；ByteTrack 增加动态卡尔曼 dt 与按秒丢失判定，以适配无人机图传 FPS 不稳定与网络抖动，实现从图传接入到实时检测、跟踪、分割、步态识别的端到端闭环。

---

<!-- 以下为后续更新追加区域，新条目请写在「更新条目」下本注释上方 -->
