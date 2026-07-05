"""Load/save gallery and enroll tracks from the live pipeline."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from ..core.types import FrameResult
from .person_id import to_english_person_id, unique_english_id
from .person_registry import PersonRegistry

if TYPE_CHECKING:
    from ..pipeline import RealtimeGaitPipeline

_SAFE_ASCII_RE = re.compile(r"[^A-Za-z0-9_.-]+")

DEFAULT_GALLERY_DIR = Path("output/gallery")
_REGISTRY_CACHE: dict[str, PersonRegistry] = {}


def get_gallery_dir(path_or_dir: str | Path | None = None) -> Path:
    if path_or_dir is None or str(path_or_dir).strip() == "":
        return DEFAULT_GALLERY_DIR
    p = Path(path_or_dir)
    if p.suffix.lower() == ".pkl":
        return p.parent
    return p


def get_registry(gallery_dir: str | Path | None = None) -> PersonRegistry:
    path = str(get_gallery_dir(gallery_dir).resolve())
    if path not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[path] = PersonRegistry(get_gallery_dir(gallery_dir))
    return _REGISTRY_CACHE[path]


def resolve_person_identity(
    display_name: str,
    pipeline: RealtimeGaitPipeline,
    gallery_dir: Path,
) -> Tuple[str, str]:
    """Return (english_id, display_name). English ID used as gallery key and filename."""
    display = display_name.strip()
    if not display:
        raise ValueError("姓名不能为空")

    base = to_english_person_id(display)
    existing = set(pipeline.gallery.feature_dict.keys())
    english_id = unique_english_id(base, gallery_dir, existing)
    return english_id, display


def load_gallery_pickle(pipeline: RealtimeGaitPipeline, path: str | Path) -> int:
    path = Path(path)
    if not path.is_file():
        return 0
    with open(path, "rb") as f:
        gallery_feat = pickle.load(f)
    pipeline.load_gallery_from_features(gallery_feat)
    return len(pipeline.gallery)


def load_gallery_directory(
    pipeline: RealtimeGaitPipeline,
    gallery_dir: str | Path,
) -> int:
    """Load all *.pkl in directory and merge into pipeline."""
    gallery_dir = Path(gallery_dir)
    pipeline.gallery.clear()
    if not gallery_dir.is_dir():
        return 0
    count = 0
    for pkl in sorted(gallery_dir.glob("*.pkl")):
        load_gallery_pickle(pipeline, pkl)
        count = len(pipeline.gallery)
    get_registry(gallery_dir).load()
    return count


def save_gallery_pickle(pipeline: RealtimeGaitPipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline.gallery.feature_dict, f)
    return path


def save_person_pickle(
    pipeline: RealtimeGaitPipeline,
    english_id: str,
    gallery_dir: str | Path,
) -> Path:
    """Save one person to output/gallery/Suhui.pkl."""
    gallery_dir = Path(gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    feat = pipeline.gallery.feature_dict.get(english_id)
    if not feat:
        raise ValueError(f"Gallery 中无 ID: {english_id}")
    path = gallery_dir / f"{english_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump({english_id: feat}, f)
    return path


def save_all_person_pickles(
    pipeline: RealtimeGaitPipeline,
    gallery_dir: str | Path,
) -> list[Path]:
    gallery_dir = Path(gallery_dir)
    saved = []
    for english_id in pipeline.gallery.feature_dict:
        saved.append(save_person_pickle(pipeline, english_id, gallery_dir))
    return saved


def sanitize_ascii_id(name: str) -> str:
    cleaned = _SAFE_ASCII_RE.sub("", name.strip())
    return cleaned or "User"


def next_person_id(pipeline: RealtimeGaitPipeline, prefix: str = "user") -> str:
    prefix = sanitize_ascii_id(prefix) or "User"
    existing = set(pipeline.gallery.feature_dict.keys())
    n = 1
    while f"{prefix}{n}" in existing:
        n += 1
    return f"{prefix}{n}"


def pick_enroll_track(
    pipeline: RealtimeGaitPipeline,
    result: FrameResult,
) -> Tuple[Optional[int], str]:
    if not result.tracks:
        return None, "画面中无人，请让人走进镜头后再注册"

    timing = pipeline.cfg.timing
    candidates: list[tuple[int, int, bool]] = []
    for tr in result.tracks:
        buf = pipeline.buffers.get(tr.track_id)
        ready = buf.ready(timing.min_sil_count, timing.min_sil_duration_sec)
        candidates.append((tr.track_id, len(buf), ready))

    ready = [c for c in candidates if c[2]]
    pool = ready if ready else candidates
    track_id, sil_count, is_ready = max(pool, key=lambda x: x[1])

    if not is_ready:
        need = timing.min_sil_count
        sec = timing.min_sil_duration_sec
        return track_id, (
            f"轮廓不足（{sil_count}/{need} 帧，需连续走 {sec}s），可继续走或仍尝试注册"
        )
    return track_id, ""


def list_enroll_candidates(
    pipeline: RealtimeGaitPipeline,
    result: FrameResult,
    since_ts: float,
) -> list[dict]:
    """Per-track silhouette stats since enrollment started (for multi-person UI)."""
    timing = pipeline.cfg.timing
    rows: list[dict] = []
    for tr in result.tracks:
        buf = pipeline.buffers.get(tr.track_id)
        count, span = buf.stats_since(since_ts)
        rows.append({
            "track_id": tr.track_id,
            "sil_count": count,
            "sil_total": len(buf),
            "sil_span_sec": round(span, 2),
            "ready": buf.ready_since(
                since_ts,
                timing.min_sil_count,
                timing.min_sil_duration_sec,
            ),
        })
    rows.sort(key=lambda r: (-r["sil_count"], r["track_id"]))
    return rows


def resolve_enroll_track(
    pipeline: RealtimeGaitPipeline,
    result: FrameResult,
    selected_track_id: Optional[int] = None,
    *,
    since_ts: Optional[float] = None,
) -> Tuple[Optional[int], str]:
    """
    Pick enrollment target: explicit track_id if given, else auto (most silhouettes).
    When since_ts is set, readiness is measured from enrollment start for that track.
    """
    if not result.tracks:
        return None, "画面中无人，请让人走进镜头后再注册"

    timing = pipeline.cfg.timing

    if selected_track_id is not None:
        active_ids = {tr.track_id for tr in result.tracks}
        if selected_track_id not in active_ids:
            return None, f"目标 T{selected_track_id} 已离开画面，请重新选择"

        buf = pipeline.buffers.get(selected_track_id)
        if since_ts is not None:
            sil_count, span = buf.stats_since(since_ts)
            ready = buf.ready_since(
                since_ts,
                timing.min_sil_count,
                timing.min_sil_duration_sec,
            )
        else:
            sil_count = len(buf)
            span = 0.0
            ready = buf.ready(timing.min_sil_count, timing.min_sil_duration_sec)

        if not ready:
            need = timing.min_sil_count
            sec = timing.min_sil_duration_sec
            return selected_track_id, (
                f"T{selected_track_id} 轮廓不足（{sil_count}/{need} 帧，"
                f"时长 {span:.1f}/{sec}s），请该目标继续正常行走"
            )
        return selected_track_id, ""

    return pick_enroll_track(pipeline, result)


def enroll_from_silhouettes(
    pipeline: RealtimeGaitPipeline,
    display_name: str,
    silhouettes: list,
    *,
    gallery_dir: str | Path = DEFAULT_GALLERY_DIR,
    allow_partial: bool = False,
) -> Tuple[bool, str, str, Path]:
    """
    Register person; returns (ok, message, english_id, saved_pkl_path).
    Gallery key is English ID; display name stored in registry.json.
    """
    gallery_dir = get_gallery_dir(gallery_dir)
    english_id, display = resolve_person_identity(display_name, pipeline, gallery_dir)
    timing = pipeline.cfg.timing
    sil_n = len(silhouettes)
    if sil_n < 3:
        return False, f"注册失败：轮廓过少（{sil_n}）", english_id, gallery_dir / f"{english_id}.pkl"
    if not allow_partial and sil_n < timing.min_sil_count:
        return False, (
            f"注册失败：轮廓 {sil_n}/{timing.min_sil_count}，"
            f"需至少走 {timing.min_sil_duration_sec}s"
        ), english_id, gallery_dir / f"{english_id}.pkl"

    emb = pipeline.recognizer.extract_embedding(
        silhouettes,
        english_id,
        sample_frames=timing.gait_sample_frames,
    )
    pipeline.register_gallery_embedding(english_id, emb)
    registry = get_registry(gallery_dir)
    registry.register(english_id, display)
    saved = save_person_pickle(pipeline, english_id, gallery_dir)
    return True, f"已注册 {display}（{english_id}，{sil_n} 帧）", english_id, saved


def list_gallery_people(pipeline: "RealtimeGaitPipeline", gallery_dir: str | Path) -> list[dict[str, str]]:
    gallery_dir = get_gallery_dir(gallery_dir)
    registry = get_registry(gallery_dir)
    ids = list(pipeline.gallery.feature_dict.keys())
    return registry.list_people(ids)


def delete_gallery_person(
    pipeline: RealtimeGaitPipeline,
    english_id: str,
    gallery_dir: str | Path,
) -> Tuple[str, str]:
    """
    Remove one person from memory, registry.json, and {id}.pkl on disk.
    Returns (english_id, display_name).
    """
    english_id = english_id.strip()
    if not english_id:
        raise ValueError("请选择要删除的人员")

    gallery_dir = get_gallery_dir(gallery_dir)
    registry = get_registry(gallery_dir)
    in_memory = english_id in pipeline.gallery.feature_dict
    pkl_path = gallery_dir / f"{english_id}.pkl"
    on_disk = pkl_path.is_file()
    in_registry = english_id in registry.all_display_names()

    if not in_memory and not on_disk and not in_registry:
        raise ValueError(f"档案库中不存在: {english_id}")

    display = registry.get_display_name(english_id) or english_id

    if in_memory:
        pipeline.gallery.remove(english_id)
        pipeline.invalidate_gallery_recognition()

    if on_disk:
        pkl_path.unlink()

    registry.unregister(english_id)

    return english_id, display


def enroll_track(
    pipeline: RealtimeGaitPipeline,
    track_id: int,
    display_name: str,
    *,
    gallery_dir: str | Path = DEFAULT_GALLERY_DIR,
    allow_partial: bool = False,
) -> Tuple[bool, str]:
    buf = pipeline.buffers.get(track_id)
    timing = pipeline.cfg.timing
    sil_n = len(buf)
    if not allow_partial and not buf.ready(timing.min_sil_count, timing.min_sil_duration_sec):
        return False, (
            f"注册失败：track T{track_id} 轮廓 {sil_n}/{timing.min_sil_count}，"
            f"需至少走 {timing.min_sil_duration_sec}s"
        )
    ok, msg, _, _ = enroll_from_silhouettes(
        pipeline,
        display_name,
        buf.silhouettes,
        gallery_dir=gallery_dir,
        allow_partial=allow_partial,
    )
    if ok:
        msg = f"{msg}（来自 T{track_id}）"
    return ok, msg
