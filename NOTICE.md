# Third-Party Notices

Realtime Gait combines project-specific integration code with several upstream
research/runtime components. Before publishing a public release, verify the
license terms of every bundled third-party component and keep their original
license notices with the source.

Known bundled or adapted components include:

- OpenGait-style gait recognition code under `opengait/`.
- MMSegmentation-related code under `gait_runtime/mmseg/`.
- PaddleSeg-related code under `gait_runtime/paddle/paddleseg/`.
- Ultralytics YOLO-related code under `gait_runtime/ultralytics/`.
- ByteTrack-style tracking utilities under `gait_runtime/tracker/` and
  `gait_runtime/tracking_utils/`.

Older offline demo scripts are archived under `legacy/legacy_scripts/`.
They are retained for reference and should not be treated as supported release
entry points.

Model weights are intentionally not part of the source release. Store them as
private artifacts, GitHub Release assets, or Git LFS objects according to their
own licenses and redistribution rules.
