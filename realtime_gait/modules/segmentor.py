"""PP-HumanSeg v2 (Paddle) — model loaded once, in-memory inference."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from ..config.settings import SegmentorConfig
from ..utils.paths import setup_import_paths
from ..utils.silhouette import crop_and_pad, mask_to_sil64

setup_import_paths()

from seg_demo import load_seg_model  # noqa: E402


class PPHumanSegSegmentor:
    def __init__(self, deploy_yaml: Path, cfg: SegmentorConfig):
        self.cfg = cfg
        device_label = "GPU" if cfg.use_gpu else "CPU"
        logger.info(f"Loading PP-HumanSeg ({device_label}): {deploy_yaml}")
        self._predictor = load_seg_model(str(deploy_yaml), use_gpu=cfg.use_gpu)

    def segment_crop(self, frame_bgr: np.ndarray, bbox_xyxy: list) -> np.ndarray:
        """BGR frame + bbox -> 64xH uint8 silhouette (no disk)."""
        t0 = time.perf_counter()
        crop = crop_and_pad(
            frame_bgr,
            bbox_xyxy,
            pad_ratio=self.cfg.crop_pad_ratio,
            out_size=self.cfg.input_size,
        )
        bg = np.ones_like(crop, dtype=np.uint8) * 255
        out_img, out_mask = self._predictor.run(crop, bg)
        del out_img
        out_mask = np.squeeze(out_mask)
        mask = np.where(out_mask < self.cfg.mask_threshold, 0, 255).astype(np.uint8)
        sil = mask_to_sil64(mask)
        self._last_seg_ms = (time.perf_counter() - t0) * 1000.0
        return sil

    @property
    def last_seg_ms(self) -> float:
        return getattr(self, "_last_seg_ms", 0.0)
