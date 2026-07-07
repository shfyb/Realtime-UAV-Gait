"""GaitBase embedding + gallery 1:N matching."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from ..config.settings import RecognizerConfig, PipelineConfig
from ..utils.paths import setup_import_paths
from ..utils.person_id import normalize_gallery_person_id
from ..utils.silhouette import silhouettes_to_gait_input

setup_import_paths()

import model.baselineDemo as baseline_demo  # noqa: E402
from opengait.utils import config_loader  # noqa: E402
import gait_compare as gc  # noqa: E402


class GalleryStore:
    """In-memory gallery: person_id -> OpenGait-style feature nest."""

    def __init__(self):
        self._feat: Dict[str, list] = {}

    def register_embedding(self, person_id: str, embedding: torch.Tensor) -> None:
        entry = {
            "undefined": {
                "undefined": embedding,
            }
        }
        if person_id not in self._feat:
            self._feat[person_id] = [entry]
        else:
            self._feat[person_id].append(entry)

    def register_from_probe_dict(self, probe_feat: dict) -> None:
        """Import OpenGait-style features from an offline extraction script."""
        for pid, items in probe_feat.items():
            for item in items:
                for typ, views in item.items():
                    for view, emb in views.items():
                        self.register_embedding(pid, emb)

    @property
    def feature_dict(self) -> dict:
        return self._feat

    def __len__(self) -> int:
        return len(self._feat)

    def clear(self) -> None:
        self._feat.clear()

    def remove(self, person_id: str) -> bool:
        if person_id not in self._feat:
            return False
        del self._feat[person_id]
        return True


class GaitBaseRecognizer:
    def __init__(self, cfg: PipelineConfig, recog_cfg: RecognizerConfig):
        self.cfg = cfg
        self.recog_cfg = recog_cfg
        logger.info(f"Loading GaitBase config: {cfg.gait_cfg}")
        cfgs = config_loader(str(cfg.gait_cfg))
        cfgs["evaluator_cfg"]["restore_hint"] = cfg.gait_ckpt_hint
        self._model = getattr(baseline_demo, "BaselineDemo")(cfgs, training=False)
        self._model.requires_grad_(False)
        self._model.eval()
        self.gallery = GalleryStore()
        self._last_gait_ms = 0.0

    def extract_embedding(
        self,
        silhouettes: List[np.ndarray],
        track_key: str,
        sample_frames: Optional[int] = None,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        n = sample_frames or self.cfg.timing.gait_sample_frames
        inputs = silhouettes_to_gait_input(silhouettes, track_key, sample_frames=n)
        ipts = self._model.inputs_pretreament(inputs)
        with torch.no_grad():
            retval, _ = self._model.forward(ipts)
        emb = retval["inference_feat"]["embeddings"]
        self._last_gait_ms = (time.perf_counter() - t0) * 1000.0
        return emb

    def match(
        self,
        embedding: torch.Tensor,
        probe_key: str,
    ) -> Tuple[Optional[str], Optional[float]]:
        if len(self.gallery) == 0:
            return None, None
        gid, dic_sort = gc.comparefeat(
            embedding,
            self.gallery.feature_dict,
            probe_key,
            self.recog_cfg.distance_threshold,
        )
        dist = None
        if dic_sort:
            dist = float(dic_sort[0][1])
        return normalize_gallery_person_id(gid), dist

    @property
    def last_gait_ms(self) -> float:
        return self._last_gait_ms
