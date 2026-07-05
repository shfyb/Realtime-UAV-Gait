"""Convert display names (e.g. Chinese) to ASCII gallery IDs (e.g. Suhui)."""

from __future__ import annotations

import re
from pathlib import Path

_ASCII_ID_RE = re.compile(r"[^A-Za-z0-9]+")


def _capitalize_part(part: str) -> str:
    part = part.strip()
    if not part:
        return ""
    return part[0].upper() + part[1:].lower()


def to_english_person_id(display_name: str) -> str:
    """
    苏辉 -> Suhui, 张三 -> Zhangsan.
    Uses pypinyin when available; otherwise keeps existing ASCII tokens.
    """
    name = display_name.strip()
    if not name:
        return "User"

    if name.isascii() and re.match(r"^[A-Za-z0-9_.-]+$", name):
        parts = re.split(r"[_\-.]+", name)
        return "".join(_capitalize_part(p) for p in parts if p) or "User"

    try:
        from pypinyin import lazy_pinyin, Style

        syllables = lazy_pinyin(name, style=Style.NORMAL, errors="ignore")
        merged = "".join(s.lower() for s in syllables if s.isalnum())
        if merged:
            return merged[0].upper() + merged[1:]
    except ImportError:
        pass

    ascii_only = _ASCII_ID_RE.sub("", name)
    if ascii_only:
        return _capitalize_part(ascii_only)

    return "User"


def normalize_gallery_person_id(match_id: str | None) -> str | None:
    """
    comparefeat returns keys like 'Suhui-undefined' (OpenGait type/view placeholders).
    Gallery registry and pkl files use bare person ids ('Suhui').
    """
    if not match_id:
        return None
    return match_id.split("-", 1)[0]


def unique_english_id(base_id: str, gallery_dir: Path, in_memory_ids: set[str]) -> str:
    """Avoid file / gallery key collision: Suhui -> Suhui2."""
    gallery_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_id
    if candidate not in in_memory_ids and not (gallery_dir / f"{candidate}.pkl").is_file():
        return candidate
    n = 2
    while True:
        candidate = f"{base_id}{n}"
        if candidate not in in_memory_ids and not (gallery_dir / f"{candidate}.pkl").is_file():
            return candidate
        n += 1
