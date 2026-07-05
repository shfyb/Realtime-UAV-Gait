"""
Example: wire stream reader to RealtimeGaitPipeline.

Production entry: python -m realtime_gait.run_stream
This file shows the minimal integration pattern.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_OPENGAIT = Path(__file__).resolve().parents[2]
if str(_OPENGAIT) not in sys.path:
    sys.path.insert(0, str(_OPENGAIT))

from realtime_gait.config import load_config
from realtime_gait.pipeline import RealtimeGaitPipeline
from realtime_gait.stream_reader import DEFAULT_STREAM_URL, LatestFrameReader


def main() -> None:
    reader = LatestFrameReader(DEFAULT_STREAM_URL)
    pipeline = RealtimeGaitPipeline(load_config())

    last_seq = 0
    print("Waiting for stream frames...")
    try:
        while True:
            frame, frame_ts, seq = reader.get_latest()
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq

            result = pipeline.process_frame(frame, timestamp_ms=(time.time() - frame_ts) * 1000)
            if result.processed:
                for tr in result.tracks:
                    label = tr.gallery_id or f"T{tr.track_id}"
                    print(f"[{result.frame_index}] {label} sil={tr.sil_count}")
    finally:
        reader.release()


if __name__ == "__main__":
    main()
