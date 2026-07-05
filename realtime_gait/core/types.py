from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackResult:
    track_id: int
    bbox_xyxy: List[float]
    gallery_id: Optional[str] = None
    display_name: Optional[str] = None
    distance: Optional[float] = None
    sil_count: int = 0
    ready: bool = False


@dataclass
class FrameResult:
    frame_index: int
    processed: bool
    tracks: List[TrackResult] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    skipped_reason: Optional[str] = None
