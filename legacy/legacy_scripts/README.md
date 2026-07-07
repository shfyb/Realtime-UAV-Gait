# Legacy Demo Scripts

These files are older offline demos, experiments, and batch-processing helpers.
They were moved out of `demo/libs/` to keep the release runtime surface clear.
The supported runtime layer now lives in `gait_runtime/`; `demo/` is reserved
for runnable demonstrations.

The current supported entry points are:

- Web console: `python -m realtime_gait.web`
- RTSP stream CLI: `python -m realtime_gait.run_stream`
- Local video CLI: `python -m realtime_gait.main`
- Gallery registration: `python -m realtime_gait.register_gallery`

Legacy scripts may still be useful for comparison or data preparation, but they
are not maintained as public release APIs.
