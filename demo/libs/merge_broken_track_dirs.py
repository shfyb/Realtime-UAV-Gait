"""后处理：强制合并全部 id、去掉 _mutil、展平单层 id 目录。"""

import argparse
from pathlib import Path

from loguru import logger

from track_id_merge import apply_force_merge_and_flatten, iter_all_video_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="强制合并所有 id，去掉 _mutil，单人目录展平（png 直接放上层）"
    )
    parser.add_argument(
        "--root",
        default="/data/liaoqi/dronegait1/dronegait_droneyolo_LITE_192*192",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行（默认仅预览）",
    )
    parser.add_argument(
        "--report",
        default="",
        help="报告输出路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    dry_run = not args.apply

    if dry_run:
        logger.info("预览模式（加 --apply 才会真正修改）")

    reports = []
    for video_dir in iter_all_video_dirs(root):
        rep = apply_force_merge_and_flatten(video_dir, dry_run=dry_run)
        if rep["force_merged"] or rep["flattened"] or rep["renamed"]:
            reports.append(rep)
            rel = video_dir.relative_to(root)
            actions = []
            if rep["force_merged"]:
                actions.append("合并全部id")
            if rep["renamed"]:
                actions.append("去掉_mutil")
            if rep["flattened"]:
                actions.append("展平id目录")
            suffix = " [将处理]" if dry_run else " [已处理]"
            logger.info(f"{rel}{suffix}: {', '.join(actions)}")

    report_path = (
        Path(args.report)
        if args.report
        else root.parent / f"{root.name}_force_flatten_report.txt"
    )
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"root: {root}\n")
        f.write(f"mode: {'apply' if args.apply else 'dry-run'}\n")
        f.write(f"changed_dirs: {len(reports)}\n\n")
        for rep in reports:
            f.write(f"{rep['path']}\n")
            if rep["force_merged"]:
                for pri, absorbed in rep.get("groups", []):
                    for sec, reason in absorbed:
                        f.write(f"  {sec} -> {pri} ({reason})\n")
            if rep["renamed"]:
                f.write("  strip _mutil suffix\n")
            if rep["flattened"]:
                f.write("  flatten id dir -> parent\n")
            f.write("\n")

    logger.info(f"处理目录数: {len(reports)}")
    logger.info(f"报告: {report_path}")
    if dry_run:
        logger.info("确认后执行: python demo/libs/merge_broken_track_dirs.py --force-all --apply")


if __name__ == "__main__":
    import sys
    if "--force-all" in sys.argv:
        sys.argv.remove("--force-all")
        main()
    else:
        # 兼容旧用法：默认也走 force-all 流程（用户本次需求）
        main()
