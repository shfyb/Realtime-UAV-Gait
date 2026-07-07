# Demo

`demo/` is the public demonstration layer for the project. It should stay small:
launchers, scenario notes, and assets that explain how to run the real-time
system.

## Launch

From the repository root:

```powershell
.\demo\run_web.bat
```

or:

```powershell
python -m realtime_gait.web --port 7860 --stream rtsp://127.0.0.1:8554/home
```

For command-line stream testing:

```powershell
.\demo\run_stream.bat
```

The model/runtime code lives outside this folder:

- `gait_runtime/`: detection, tracking, segmentation, and recognition adapters.
- `checkpoints/`: local model-weight locations.
- `realtime_gait/`: real-time pipeline, CLI, and Web integration.
