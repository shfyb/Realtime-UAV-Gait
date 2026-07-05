"""Schedule 120fps ingest against slower GPU pipeline stages."""

from __future__ import annotations


class FrameScheduler:
    def __init__(
        self,
        process_stride: int = 4,
        seg_stride: int = 2,
    ):
        self.process_stride = max(1, process_stride)
        self.seg_stride = max(1, seg_stride)
        self._ingest_count = 0
        self._process_count = 0

    def on_ingest(self) -> tuple[bool, str]:
        """
        Returns:
            (should_process, reason_if_skipped)
        """
        self._ingest_count += 1
        if self._ingest_count % self.process_stride != 0:
            return False, "process_stride"
        self._process_count += 1
        return True, ""

    def should_segment(self) -> bool:
        return self._process_count % self.seg_stride == 0

    @property
    def ingest_count(self) -> int:
        return self._ingest_count

    @property
    def process_count(self) -> int:
        return self._process_count

    def reset(self) -> None:
        self._ingest_count = 0
        self._process_count = 0
