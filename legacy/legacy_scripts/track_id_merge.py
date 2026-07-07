"""合并因跟踪中断产生的重复 track id。"""

from __future__ import annotations

import os
import os.path as osp
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

FrameSet = Set[int]
FramesDict = Dict[str, FrameSet]
MergeGroup = Tuple[str, List[Tuple[str, str]]]


def _frame_set_from_track_entries(entries: Iterable) -> FrameSet:
    return {item[0] for item in entries}


def plan_merges(
    frames_dict: FramesDict,
    max_gap: int = 30,
    max_overlap: int = 3,
    dup_ratio: float = 0.5,
    dup_min_overlap: int = 10,
) -> List[MergeGroup]:
    """
    根据帧时间线规划 id 合并。

    规则 1（跟踪中断）: 两段轨迹帧几乎不重叠，且间隔 <= max_gap。
    规则 2（重复跟踪）: 两段轨迹大量重叠（同一人的重复 id）。
    """
    if not frames_dict:
        return []

    items = sorted(frames_dict.items(), key=lambda kv: min(kv[1]))
    groups: List[MergeGroup] = []
    primary = items[0][0]
    primary_set = set(items[0][1])
    absorbed: List[Tuple[str, str]] = []

    for tid, frames in items[1:]:
        sf = set(frames)
        overlap = len(primary_set & sf)
        gap = min(sf) - max(primary_set) - 1
        denom = min(len(primary_set), len(sf))
        ratio = overlap / denom if denom else 0.0

        reason = None
        if overlap <= max_overlap and 0 <= gap <= max_gap:
            reason = f"track_break gap={gap}"
        elif overlap >= dup_min_overlap and ratio >= dup_ratio:
            reason = f"duplicate overlap={overlap} ratio={ratio:.2f}"

        if reason:
            primary_set |= sf
            absorbed.append((tid, reason))
        else:
            groups.append((primary, absorbed))
            primary = tid
            primary_set = sf
            absorbed = []

    groups.append((primary, absorbed))
    return groups


def merge_track_id_dict(
    track_id_dict: Dict[str, list],
    max_gap: int = 30,
    max_overlap: int = 3,
    dup_ratio: float = 0.5,
    dup_min_overlap: int = 10,
) -> Tuple[Dict[str, list], List[MergeGroup]]:
    """合并内存中的 track 结果，返回新 dict 与合并记录。"""
    frames_dict = {
        tid: _frame_set_from_track_entries(entries)
        for tid, entries in track_id_dict.items()
        if entries
    }
    groups = plan_merges(
        frames_dict,
        max_gap=max_gap,
        max_overlap=max_overlap,
        dup_ratio=dup_ratio,
        dup_min_overlap=dup_min_overlap,
    )

    merged: Dict[str, list] = {}
    for primary, absorbed in groups:
        entries = list(track_id_dict.get(primary, []))
        frame_seen = {item[0] for item in entries}
        for sec_tid, _ in absorbed:
            for item in track_id_dict.get(sec_tid, []):
                if item[0] not in frame_seen:
                    entries.append(item)
                    frame_seen.add(item[0])
        entries.sort(key=lambda x: x[0])
        merged[primary] = entries

    active_groups = [(p, a) for p, a in groups if a]
    return merged, active_groups


def load_frames_from_dir(video_dir: Path) -> FramesDict:
    frames_dict: FramesDict = {}
    for child in video_dir.iterdir():
        if child.is_dir() and child.name.startswith("id"):
            frames = sorted(int(p.stem) for p in child.glob("*.png"))
            if frames:
                frames_dict[child.name] = set(frames)
    return frames_dict


def apply_merge_to_video_dir(
    video_dir: Path,
    dry_run: bool = True,
    max_gap: int = 30,
    max_overlap: int = 3,
    dup_ratio: float = 0.5,
    dup_min_overlap: int = 10,
) -> dict:
    """对已生成的 id 目录执行合并。"""
    video_dir = Path(video_dir)
    frames_dict = load_frames_from_dir(video_dir)
    if len(frames_dict) <= 1:
        return {"path": str(video_dir), "merged": False, "groups": []}

    groups = plan_merges(
        frames_dict,
        max_gap=max_gap,
        max_overlap=max_overlap,
        dup_ratio=dup_ratio,
        dup_min_overlap=dup_min_overlap,
    )
    absorbed_total = sum(len(a) for _, a in groups)
    if absorbed_total == 0:
        return {"path": str(video_dir), "merged": False, "groups": groups}

    for primary, absorbed in groups:
        if not absorbed:
            continue
        primary_dir = video_dir / primary
        for sec_tid, reason in absorbed:
            sec_dir = video_dir / sec_tid
            if not sec_dir.is_dir():
                continue
            for png in sec_dir.glob("*.png"):
                dst = primary_dir / png.name
                if dst.exists():
                    if not dry_run:
                        png.unlink()
                elif not dry_run:
                    shutil.move(str(png), str(dst))
            if not dry_run and sec_dir.is_dir():
                sec_dir.rmdir() if not any(sec_dir.iterdir()) else shutil.rmtree(sec_dir)

    remaining_ids = [
        d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("id")
    ]
    renamed = False
    if len(remaining_ids) == 1 and video_dir.name.endswith("_mutil"):
        new_path = video_dir.parent / video_dir.name[: -len("_mutil")]
        if not dry_run:
            if new_path.exists():
                raise FileExistsError(f"无法重命名，目标已存在: {new_path}")
            video_dir.rename(new_path)
            video_dir = new_path
        renamed = True

    return {
        "path": str(video_dir),
        "merged": True,
        "groups": groups,
        "remaining_ids": len(remaining_ids),
        "renamed_to_single": renamed,
    }


def plan_force_merge_all(frames_dict: FramesDict) -> List[MergeGroup]:
    """将所有 id 强制合并到帧数最多的主 id。"""
    if len(frames_dict) <= 1:
        primary = next(iter(frames_dict), None)
        return [(primary, [])] if primary else []

    primary = max(frames_dict.keys(), key=lambda k: len(frames_dict[k]))
    absorbed = [
        (tid, f"force_all -> {primary}")
        for tid in sorted(frames_dict.keys())
        if tid != primary
    ]
    return [(primary, absorbed)]


def force_merge_all_track_id_dict(
    track_id_dict: Dict[str, list],
) -> Tuple[Dict[str, list], List[MergeGroup]]:
    frames_dict = {
        tid: _frame_set_from_track_entries(entries)
        for tid, entries in track_id_dict.items()
        if entries
    }
    groups = plan_force_merge_all(frames_dict)
    if not groups or not groups[0][0]:
        return track_id_dict, []

    primary, absorbed = groups[0]
    if not absorbed:
        return track_id_dict, []

    entries = list(track_id_dict.get(primary, []))
    frame_seen = {item[0] for item in entries}
    for sec_tid, _ in absorbed:
        for item in track_id_dict.get(sec_tid, []):
            if item[0] not in frame_seen:
                entries.append(item)
                frame_seen.add(item[0])
    entries.sort(key=lambda x: x[0])
    return {primary: entries}, [groups[0]]


def _move_id_pngs_to_primary(video_dir: Path, primary: str, sec_tid: str, dry_run: bool):
    primary_dir = video_dir / primary
    sec_dir = video_dir / sec_tid
    if not sec_dir.is_dir():
        return
    for png in sec_dir.glob("*.png"):
        dst = primary_dir / png.name
        if dst.exists():
            if not dry_run:
                png.unlink()
        elif not dry_run:
            shutil.move(str(png), str(dst))
    if not dry_run and sec_dir.is_dir():
        if not any(sec_dir.iterdir()):
            sec_dir.rmdir()
        else:
            shutil.rmtree(sec_dir)


def flatten_single_id_dir(video_dir: Path, dry_run: bool = True) -> bool:
    """仅一个 id 子目录时，把 png 提到上一层并删除 id 目录。"""
    video_dir = Path(video_dir)
    id_dirs = sorted(
        d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("id")
    )
    if len(id_dirs) != 1:
        return False

    id_dir = id_dirs[0]
    for png in id_dir.glob("*.png"):
        dst = video_dir / png.name
        if dst.exists():
            if not dry_run:
                png.unlink()
        elif not dry_run:
            shutil.move(str(png), str(dst))

    if not dry_run and id_dir.is_dir():
        if not any(id_dir.iterdir()):
            id_dir.rmdir()
        else:
            shutil.rmtree(id_dir)
    return True


def strip_mutil_suffix(video_dir: Path, dry_run: bool = True) -> Path:
    video_dir = Path(video_dir)
    if not video_dir.name.endswith("_mutil"):
        return video_dir
    new_path = video_dir.parent / video_dir.name[: -len("_mutil")]
    if not dry_run:
        if new_path.exists():
            raise FileExistsError(f"无法重命名，目标已存在: {new_path}")
        video_dir.rename(new_path)
        return new_path
    return new_path


def apply_force_merge_and_flatten(video_dir: Path, dry_run: bool = True) -> dict:
    """强制合并全部 id，去掉 _mutil，并展平单层 id 目录。"""
    video_dir = Path(video_dir)
    report = {
        "path": str(video_dir),
        "force_merged": False,
        "flattened": False,
        "renamed": False,
        "groups": [],
    }

    frames_dict = load_frames_from_dir(video_dir)
    if len(frames_dict) > 1:
        groups = plan_force_merge_all(frames_dict)
        primary, absorbed = groups[0]
        report["groups"] = groups
        for sec_tid, reason in absorbed:
            _move_id_pngs_to_primary(video_dir, primary, sec_tid, dry_run)
        report["force_merged"] = True

    if video_dir.name.endswith("_mutil"):
        video_dir = strip_mutil_suffix(video_dir, dry_run=dry_run)
        report["renamed"] = True
        report["path"] = str(video_dir)

    if flatten_single_id_dir(video_dir, dry_run=dry_run):
        report["flattened"] = True
        report["path"] = str(video_dir)

    return report


def iter_all_video_dirs(root: Path) -> Iterable[Path]:
    root = Path(root)
    for subject in sorted(root.iterdir()):
        if not subject.is_dir():
            continue
        for seq in sorted(subject.iterdir()):
            if not seq.is_dir():
                continue
            for video_dir in sorted(seq.iterdir()):
                if not video_dir.is_dir():
                    continue
                has_id = any(
                    d.is_dir() and d.name.startswith("id")
                    for d in video_dir.iterdir()
                )
                has_png = any(video_dir.glob("*.png"))
                if has_id or has_png or video_dir.name.endswith("_mutil"):
                    yield video_dir


def iter_video_dirs(root: Path) -> Iterable[Path]:
    """仅含多个 id 的目录。"""
    for video_dir in iter_all_video_dirs(root):
        id_dirs = [
            d for d in video_dir.iterdir()
            if d.is_dir() and d.name.startswith("id")
        ]
        if len(id_dirs) > 1:
            yield video_dir


def process_dataset_root(
    root: str,
    dry_run: bool = True,
    **kwargs,
) -> List[dict]:
    reports = []
    for video_dir in iter_video_dirs(Path(root)):
        report = apply_merge_to_video_dir(video_dir, dry_run=dry_run, **kwargs)
        if report.get("merged"):
            reports.append(report)
    return reports
