"""Path bootstrap: wire realtime_gait to OpenGait demo/libs."""

from __future__ import annotations

import sys
from pathlib import Path

REALTIME_ROOT = Path(__file__).resolve().parents[1]
OPENGAIT_ROOT = REALTIME_ROOT.parent
REPO_ROOT = OPENGAIT_ROOT.parent
DEMO_LIBS = OPENGAIT_ROOT / "demo" / "libs"


def setup_import_paths() -> None:
    paddle_libs = DEMO_LIBS / "paddle"
    for p in (str(DEMO_LIBS), str(paddle_libs), str(OPENGAIT_ROOT), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def resolve_path(rel: str | Path) -> Path:
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    return (OPENGAIT_ROOT / rel).resolve()
