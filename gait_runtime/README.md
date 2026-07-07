# Gait Runtime

This directory contains the algorithm/runtime layer that used to live under
`demo/libs/`. It is part of the main program, not the demo surface.

## Required by `realtime_gait`

| Path | Used by | Purpose |
|------|---------|---------|
| `tracking_utils/` | `realtime_gait.modules.detector` | DroneYOLO prediction wrapper and timer |
| `tracker/` | `realtime_gait.modules.tracker` | ByteTrack implementation |
| `paddle/seg_demo.py` and `paddle/paddleseg/` | `realtime_gait.modules.segmentor` | PP-HumanSeg inference |
| `model/` | `realtime_gait.modules.recognizer` | GaitBase demo model wrapper |
| `gait_compare.py` | `realtime_gait.modules.recognizer` | Gallery distance matching |
| `cython_bbox.py` | `tracker/` fallback path | Pure-Python bbox overlap compatibility |
| `ultralytics/` | local compatibility fallback | Vendored YOLO runtime; prefer the pinned package when possible |
| `mmseg/` | legacy segmentation compatibility | Kept for compatibility with older adapters |

## What moved

Historical offline scripts such as `track.py`, `segment.py`, `recognise.py`,
`seg_picture_*.py`, and `samdemo.py` live in `legacy/legacy_scripts/`.
They are useful for reference, but the release path should not import them.

The long-term cleanup target is to replace vendored third-party trees with
pinned package dependencies and keep only small project-specific adapters here.
