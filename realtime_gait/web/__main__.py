#!/usr/bin/env python3
"""
Launch the realtime gait web console.

  python -m realtime_gait.web
  python -m realtime_gait.web --port 7860 --stream rtsp://127.0.0.1:8554/home
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_OPENGAIT = Path(__file__).resolve().parent.parent.parent
if str(_OPENGAIT) not in sys.path:
    sys.path.insert(0, str(_OPENGAIT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime gait web console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", default="")
    parser.add_argument("--gallery", default="output/gallery")
    parser.add_argument("--stream", default="", help="Default RTSP URL pre-filled in UI")
    args = parser.parse_args()

    import uvicorn

    from realtime_gait.web.server import create_app

    application = create_app(
        gallery_path=args.gallery,
        default_stream=args.stream,
        config_path=args.config,
    )
    print(f"Open http://{args.host}:{args.port}/ in your browser")
    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
