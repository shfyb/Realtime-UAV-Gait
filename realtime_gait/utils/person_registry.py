"""Display name registry: English gallery ID <-> Chinese name."""

from __future__ import annotations

import json
from pathlib import Path


class PersonRegistry:
    """Maps gallery keys (Suhui) to UI labels (苏辉)."""

    def __init__(self, gallery_dir: Path):
        self.gallery_dir = Path(gallery_dir)
        self.registry_path = self.gallery_dir / "registry.json"
        self._display_by_id: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        self._display_by_id = {}
        if self.registry_path.is_file():
            with open(self.registry_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._display_by_id = {str(k): str(v) for k, v in data.items()}

    def save(self) -> None:
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._display_by_id, f, ensure_ascii=False, indent=2)

    def register(self, english_id: str, display_name: str) -> None:
        self._display_by_id[english_id] = display_name.strip()
        self.save()

    def unregister(self, english_id: str) -> bool:
        if english_id not in self._display_by_id:
            return False
        del self._display_by_id[english_id]
        self.save()
        return True

    def get_display_name(self, english_id: str | None) -> str | None:
        if not english_id:
            return None
        if english_id in self._display_by_id:
            return self._display_by_id[english_id]
        base = english_id.split("-", 1)[0]
        return self._display_by_id.get(base, base)

    def all_display_names(self) -> dict[str, str]:
        return dict(self._display_by_id)

    def list_people(self, english_ids: list[str]) -> list[dict[str, str]]:
        """Build UI list: display name + file + id."""
        rows: list[dict[str, str]] = []
        for english_id in sorted(english_ids):
            display = self._display_by_id.get(english_id, english_id)
            rows.append({
                "id": english_id,
                "display_name": display,
                "file": f"{english_id}.pkl",
            })
        return rows
