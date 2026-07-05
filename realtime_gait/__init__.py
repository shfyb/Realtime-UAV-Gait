"""Real-time drone gait recognition: DroneYOLO + ByteTrack + PP-HumanSeg + GaitBase."""

from .config import PipelineConfig, load_config

__all__ = [
    "RealtimeGaitPipeline",
    "PipelineConfig",
    "load_config",
    "LatestFrameReader",
    "DEFAULT_STREAM_URL",
]


def __getattr__(name: str):
    if name == "RealtimeGaitPipeline":
        from .pipeline import RealtimeGaitPipeline
        return RealtimeGaitPipeline
    if name in ("LatestFrameReader", "DEFAULT_STREAM_URL"):
        from . import stream_reader
        return getattr(stream_reader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
