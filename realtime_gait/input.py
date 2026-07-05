"""
图传输入 + 实时步态识别 全流程入口。

用法（在 OpenGait 根目录）:
  python -m realtime_gait.input
  STREAM_URL=rtsp://ip:8554/home python -m realtime_gait.input
  python -m realtime_gait.input --gallery /path/to/gallery.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

_OPENGAIT = Path(__file__).resolve().parent.parent
if str(_OPENGAIT) not in sys.path:
    sys.path.insert(0, str(_OPENGAIT))

from realtime_gait.run_stream import main

if __name__ == "__main__":
    main()
