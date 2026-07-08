# Model Checkpoints

The realtime pipeline requires model bundles under this directory. **Checkpoint
weights are stored in this repository via [Git LFS](https://git-lfs.com/).**

After cloning, fetch the weights:

```bash
git lfs install
git lfs pull
```

Expected layout:

```
checkpoints/
├── Drone-YOLO/
│   ├── best.pt                          # Drone-YOLO pedestrian detector
│   └── best1.pt                         # optional/alternate detector checkpoint
├── gait_model/
│   └── GaitBase_DronGait1-60000.pt      # GaitBase recognizer (DroneGait1)
└── seg_model/
    └── human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/
        ├── deploy.yaml                  # Paddle inference config
        ├── model.pdmodel                # PaddleSeg export
        └── model.pdiparams              # PaddleSeg export
```

## Segmentation (PP-HumanSeg v2)

The mobile inference bundle is included under `seg_model/.../`.
`deploy.yaml` is tracked as plain Git text; `model.pdmodel` and `model.pdiparams` are Git LFS objects.

If you need to refresh this bundle, export from [PaddleSeg PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9/contrib/PP-HumanSeg/README.md) and replace the files above.

## Detection (Drone-YOLO)

`best.pt` is the Drone-YOLO / Ultralytics detector used by the realtime pipeline.
Path: `realtime_gait/config/default.yaml` → `drone_yolo`.

`best1.pt` is kept as an alternate detector checkpoint. The default pipeline does
not load it unless the config path is changed.

## Gait recognition (GaitBase)

`GaitBase_DronGait1-60000.pt` comes from OpenGait training on DroneGait1.
Config: `configs/gaitbase/gaitbase_da_dronegait1.yaml`.

## Verify

From the **repository root**:

```bash
python -c "
from pathlib import Path
root = Path('.')
checks = [
    root / 'checkpoints/Drone-YOLO/best.pt',
    root / 'checkpoints/gait_model/GaitBase_DronGait1-60000.pt',
    root / 'checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/model.pdiparams',
]
for p in checks:
    print('OK' if p.is_file() else 'MISSING', p)
"
```

If any file is missing after clone, run `git lfs pull` and verify Git LFS is installed.
