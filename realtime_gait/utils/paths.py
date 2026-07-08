"""Path bootstrap: wire realtime_gait to the repository runtime layer."""

from __future__ import annotations

import sys
from pathlib import Path

REALTIME_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REALTIME_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
GAIT_RUNTIME = PROJECT_ROOT / "gait_runtime"
OPENGAIT_RUNTIME = PROJECT_ROOT / "opengait"

# Backward-compatible aliases for older imports/documentation.
OPENGAIT_ROOT = PROJECT_ROOT
DEMO_LIBS = GAIT_RUNTIME


def setup_import_paths() -> None:
    paddle_libs = GAIT_RUNTIME / "paddle"
    for p in (
        str(GAIT_RUNTIME),
        str(paddle_libs),
        str(OPENGAIT_RUNTIME),
        str(PROJECT_ROOT),
        str(REPO_ROOT),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def resolve_path(rel: str | Path) -> Path:
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    return (PROJECT_ROOT / rel).resolve()
