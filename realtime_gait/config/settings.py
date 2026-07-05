from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from ..utils.paths import resolve_path


@dataclass
class TimingConfig:
    process_interval_ms: float = 33.3
    seg_interval_ms: float = 66.7
    min_sil_count: int = 15
    min_sil_duration_sec: float = 1.5
    sil_buffer_duration_sec: float = 6.0
    recognition_interval_sec: float = 1.0
    gait_sample_frames: int = 30

    @property
    def target_process_hz(self) -> float:
        return 1000.0 / self.process_interval_ms

    @property
    def target_seg_hz(self) -> float:
        return 1000.0 / self.seg_interval_ms


@dataclass
class TrackerConfig:
    frame_rate: int = 30
    max_time_lost_sec: float = 1.0
    dt_ema_alpha: float = 0.2
    dt_min_sec: float = 0.005
    dt_max_sec: float = 0.5
    min_area: float = 100.0
    max_aspect_ratio: float = 3.0
    min_aspect_ratio: float = 0.3


@dataclass
class DetectorConfig:
    conf: float = 0.25
    iou: float = 0.6
    pedestrian_class_id: int = 0


@dataclass
class SegmentorConfig:
    crop_pad_ratio: float = 0.1
    input_size: int = 192
    mask_threshold: int = 80
    use_gpu: bool = False


@dataclass
class RecognizerConfig:
    distance_threshold: float = 100.0
    sil_similarity_threshold: float = 0.75
    dataset: str = "GREW"


@dataclass
class PipelineConfig:
    drone_yolo: Path = field(default_factory=lambda: resolve_path("demo/checkpoints/Drone-YOLO/best.pt"))
    seg_model: Path = field(
        default_factory=lambda: resolve_path(
            "demo/checkpoints/seg_model/"
            "human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml"
        )
    )
    gait_cfg: Path = field(default_factory=lambda: resolve_path("configs/gaitbase/gaitbase_da_dronegait1.yaml"))
    gait_ckpt_hint: int = 60000
    device: str = "cuda"
    timing: TimingConfig = field(default_factory=TimingConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    segmentor: SegmentorConfig = field(default_factory=SegmentorConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)

    # Legacy frame-based keys (optional; converted to timing if present in old yaml)
    input_fps: int = 0
    process_stride: int = 0
    seg_stride: int = 0
    min_sil_frames: int = 0
    max_sil_buffer: int = 0
    gait_sample_frames: int = 0
    recognition_interval: int = 0


def _merge_dataclass(obj: Any, data: dict) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _apply_legacy_frame_config(cfg: PipelineConfig) -> None:
    """Convert old frame-count yaml to timing if user still has legacy keys."""
    t = cfg.timing
    if cfg.process_stride > 0 and cfg.input_fps > 0:
        hz = cfg.input_fps / cfg.process_stride
        t.process_interval_ms = 1000.0 / hz
    if cfg.seg_stride > 0 and t.process_interval_ms > 0:
        seg_hz = (1000.0 / t.process_interval_ms) / cfg.seg_stride
        t.seg_interval_ms = 1000.0 / seg_hz
    if cfg.min_sil_frames > 0:
        t.min_sil_count = cfg.min_sil_frames
    if cfg.gait_sample_frames > 0:
        t.gait_sample_frames = cfg.gait_sample_frames
    if cfg.recognition_interval > 0 and t.seg_interval_ms > 0:
        t.recognition_interval_sec = cfg.recognition_interval * t.seg_interval_ms / 1000.0
    if cfg.max_sil_buffer > 0 and t.seg_interval_ms > 0:
        t.sil_buffer_duration_sec = cfg.max_sil_buffer * t.seg_interval_ms / 1000.0


def load_config(yaml_path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    cfg = PipelineConfig()
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "default.yaml"
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        return cfg

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paths = raw.pop("paths", {})
    if paths:
        if "drone_yolo" in paths:
            cfg.drone_yolo = resolve_path(paths["drone_yolo"])
        if "seg_model" in paths:
            cfg.seg_model = resolve_path(paths["seg_model"])
        if "gait_cfg" in paths:
            cfg.gait_cfg = resolve_path(paths["gait_cfg"])
        if "gait_ckpt_hint" in paths:
            cfg.gait_ckpt_hint = int(paths["gait_ckpt_hint"])

    flat_keys = {
        "device",
        "input_fps", "process_stride", "seg_stride",
        "min_sil_frames", "max_sil_buffer", "gait_sample_frames", "recognition_interval",
    }
    nested = {}
    for key, value in raw.items():
        if key in flat_keys:
            setattr(cfg, key, value)
        else:
            nested[key] = value

    _merge_dataclass(cfg, nested)
    _apply_legacy_frame_config(cfg)

    if not cfg.tracker.frame_rate:
        cfg.tracker.frame_rate = max(1, int(round(cfg.timing.target_process_hz)))
    return cfg
