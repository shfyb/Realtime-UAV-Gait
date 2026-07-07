"""Pure NumPy fallback for cython_bbox.bbox_overlaps (Windows: pip install often fails)."""

from __future__ import annotations

import numpy as np


def bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """IoU matrix for axis-aligned boxes in [x1, y1, x2, y2] format."""
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    query_boxes = np.ascontiguousarray(query_boxes, dtype=np.float64)

    n = boxes.shape[0]
    k = query_boxes.shape[0]
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=np.float64)

    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    query_area = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])

    overlaps = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        x1 = np.maximum(boxes[i, 0], query_boxes[:, 0])
        y1 = np.maximum(boxes[i, 1], query_boxes[:, 1])
        x2 = np.minimum(boxes[i, 2], query_boxes[:, 2])
        y2 = np.minimum(boxes[i, 3], query_boxes[:, 3])
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        inter = w * h
        union = box_area[i] + query_area - inter
        overlaps[i, :] = np.where(union > 0, inter / union, 0.0)

    return overlaps
