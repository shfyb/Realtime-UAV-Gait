#!/usr/bin/env python3
"""
Build a Windows runnable zip for realtime_gait (minimal required files only).

Default output:
    /data/liaoqi/realtime_gait_windows_needed_bundle.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

IGNORE_PARTS = {"__pycache__", ".git", ".idea", ".vscode", ".pytest_cache", ".mypy_cache"}
IGNORE_SUFFIX = {".pyc", ".pyo", ".tmp", ".swp", ".log"}


def _repo_root_from_script(script_file: Path) -> Path:
    # .../All-in-One-Gait/OpenGait/realtime_gait/scripts/build_windows_full_bundle.py
    return script_file.resolve().parents[3]


def _iter_files(base: Path):
    if base.is_file():
        yield base
        return
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if any(part in IGNORE_PARTS for part in p.parts):
            continue
        if p.suffix.lower() in IGNORE_SUFFIX:
            continue
        yield p


def _add_path(zf: ZipFile, src: Path, arc_prefix: Path, missing: list[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    for p in _iter_files(src):
        arcname = (arc_prefix / p.relative_to(src)).as_posix()
        zf.write(p, arcname)


def _add_text(zf: ZipFile, arcname: str, text: str) -> None:
    zf.writestr(arcname, text)


def main() -> None:
    script_file = Path(__file__).resolve()
    repo_root = _repo_root_from_script(script_file)

    parser = argparse.ArgumentParser(description="Build minimal Windows bundle for realtime_gait")
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="All-in-One-Gait root path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/liaoqi/realtime_gait_windows_needed_bundle.zip"),
        help="Output zip path",
    )
    parser.add_argument(
        "--seg-model-dir",
        type=Path,
        default=None,
        help="Optional external seg model dir, will be packed to OpenGait/demo/checkpoints/seg_model/...",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any required item is missing")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_zip = args.output.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # Required runtime code for realtime_gait
    required_items = [
        ("OpenGait/realtime_gait", "OpenGait/realtime_gait"),
        ("OpenGait/opengait", "OpenGait/opengait"),
        ("OpenGait/configs", "OpenGait/configs"),
        ("OpenGait/demo/libs", "OpenGait/demo/libs"),
        ("requirements.txt", "requirements.txt"),
    ]

    # Required model/checkpoint files for current realtime_gait pipeline
    model_items = [
        ("OpenGait/demo/checkpoints/Drone-YOLO/best.pt", "OpenGait/demo/checkpoints/Drone-YOLO/best.pt"),
        ("OpenGait/demo/checkpoints/gait_model/GaitBase_DronGait1-60000.pt", "OpenGait/demo/checkpoints/gait_model/GaitBase_DronGait1-60000.pt"),
        (
            "OpenGait/demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax",
            "OpenGait/demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax",
        ),
    ]
    if args.seg_model_dir is not None:
        model_items[-1] = (
            str(args.seg_model_dir.resolve()),
            "OpenGait/demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax",
        )

    missing: list[str] = []
    if output_zip.exists():
        output_zip.unlink()

    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for src_rel, arc_rel in required_items + model_items:
            src = repo_root / src_rel
            _add_path(zf, src, Path(arc_rel), missing)

        _add_text(
            zf,
            "run_setup.ps1",
            '$ErrorActionPreference = "Stop"\n'
            'Set-Location -Path "$PSScriptRoot\\OpenGait"\n'
            ". .\\realtime_gait\\scripts\\setup_windows.ps1\n",
        )
        _add_text(
            zf,
            "run_stream.bat",
            "@echo off\nsetlocal\ncd /d %~dp0\\OpenGait\ncall realtime_gait\\scripts\\run_stream.bat %*\nendlocal\n",
        )
        _add_text(
            zf,
            "run_video.bat",
            "@echo off\nsetlocal\ncd /d %~dp0\\OpenGait\ncall realtime_gait\\scripts\\run_video.bat %*\nendlocal\n",
        )
        _add_text(
            zf,
            "START_HERE.txt",
            "Realtime Gait Windows bundle (needed files only)\n\n"
            "1) powershell -ExecutionPolicy Bypass -File .\\run_setup.ps1\n"
            "2) .\\run_stream.bat rtsp://YOUR_IP:8554/home\n"
            "3) .\\run_video.bat D:\\path\\test.mp4\n\n"
            "Guide: OpenGait\\realtime_gait\\WINDOWS_DEPLOY.md\n",
        )
        _add_text(
            zf,
            "BUNDLE_MANIFEST.txt",
            "Bundle Type: needed files only\n"
            f"Output: {output_zip}\n"
            f"Missing Count: {len(missing)}\n"
            + ("\n".join(f"- {m}" for m in missing) if missing else "- none\n"),
        )

    if missing and args.strict:
        raise FileNotFoundError("Missing required items:\n" + "\n".join(f"- {m}" for m in missing))

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[OK] Bundle: {output_zip}")
    print(f"[OK] Size: {size_mb:.2f} MB")
    if missing:
        print("[WARN] Missing paths:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
