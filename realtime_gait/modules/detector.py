"""DroneYOLO detector (Ultralytics YOLO weights)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger

from ..config.settings import DetectorConfig
from ..utils.paths import setup_import_paths

setup_import_paths()

from tracking_utils.predictor_yolov8 import Predictor  # noqa: E402
from tracking_utils.timer import Timer  # noqa: E402
from ultralytics import YOLO  # noqa: E402


class DroneYoloDetector:
    def __init__(
        self,
        weights: Path,
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.6,
        pedestrian_class_id: int = 0,
    ):
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.conf = conf
        self.iou = iou
        self.pedestrian_class_id = pedestrian_class_id
        self._timer = Timer()
        logger.info(f"Loading DroneYOLO: {weights} (class={pedestrian_class_id})")
        self._yolo = YOLO(str(weights))
        self._predictor = Predictor(model=self._yolo, device=self.device)
        self._predictor.confidence_threshold = conf
        self._predictor.nms_threshold = iou

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
        """
        Returns:
            dets: (N, 5) x1,y1,x2,y2,score — pedestrian class only
            img_info: height, width, ...
        """
        t0 = time.perf_counter()
        outputs, img_info = self._predictor.inference(frame_bgr, self._timer)
        if outputs[0] is None:
            dets = None
        else:
            arr = outputs[0]
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            rows = []
            for det in arr:
                if det.shape[0] >= 6:
                    cls_id = int(det[5])
                    if cls_id != self.pedestrian_class_id:
                        continue
                rows.append(det[:5].tolist())
            dets = np.array(rows, dtype=np.float32) if rows else None
        img_info["detect_ms"] = (time.perf_counter() - t0) * 1000.0
        return dets, img_info
