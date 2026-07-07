# Model Checkpoints

The realtime pipeline requires three model bundles. **Weights are not included in this repository** (too large for Git). Place files as follows:

```
checkpoints/
├── Drone-YOLO/
│   └── best.pt                          # Drone-YOLO pedestrian detector
├── gait_model/
│   └── GaitBase_DronGait1-60000.pt      # GaitBase recognizer (DroneGait1)
└── seg_model/
    └── human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/
        ├── deploy.yaml                  # included in repo
        ├── model.pdmodel                # download separately
        └── model.pdiparams              # download separately
```

## Segmentation (PP-HumanSeg v2)

1. Download the mobile inference model from [PaddleSeg model zoo](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9/contrib/PP-HumanSeg/README.md) or your training export.
2. Copy `model.pdmodel` and `model.pdiparams` into the directory above.
3. `deploy.yaml` is already provided; verify `model_dir` paths match.

## Detection (Drone-YOLO)

Train or obtain `best.pt` from your Drone-YOLO / ultralytics training run.  
Path referenced in `realtime_gait/config/default.yaml` → `drone_yolo`.

## Gait recognition (GaitBase)

Obtain `GaitBase_DronGait1-60000.pt` from OpenGait training on DroneGait1.  
Config: `configs/gaitbase/gaitbase_da_dronegait1.yaml`.

## Verify

From the **repository root** (this directory's parent if you cloned into a subfolder — run from the folder that contains `realtime_gait/`):

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

If you migrated from a local backup bundle, copy the three weight files from your old checkpoint tree.
