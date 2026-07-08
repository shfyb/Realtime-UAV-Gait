# Publishing Checklist

Use this checklist before tagging or pushing a public release.

## 1. Local model weights

Model weights are versioned with Git LFS. Install Git LFS before cloning or
pushing a release that includes checkpoint files.

Expected local paths:

```text
checkpoints/
├── Drone-YOLO/best.pt
├── Drone-YOLO/best1.pt
├── gait_model/GaitBase_DronGait1-60000.pt
└── seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/
    ├── deploy.yaml
    ├── model.pdmodel
    └── model.pdiparams
```

## 2. Verify repository contents

```powershell
git status --short
git ls-files checkpoints
git lfs ls-files
```

Expected result:

- `git status --short` only shows intentional release edits.
- `git ls-files checkpoints` shows `.gitkeep`, `README.md`, `deploy.yaml`,
  and lightweight metadata only.
- Model weights appear in `git lfs ls-files`; gallery pickles should still stay
  out of Git.

## 3. Run a lightweight validation

```powershell
python -m compileall realtime_gait opengait
```

GPU inference is not required for this check; it only validates that the
publishable Python modules parse correctly.

## 4. GitHub setup

On GitHub, create an empty repository named `realtime-gait`. Do not add a
README, license, or `.gitignore` there because this repository already has them.

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/realtime-gait.git
git push -u origin main
```

## 5. Public release checklist

- [ ] No private gallery files or personal data under `output/`.
- [ ] No private IP addresses, tokens, or machine-specific paths in committed code.
- [ ] Model weights are tracked by Git LFS, not normal Git blobs.
- [ ] `LICENSE`, `NOTICE.md`, and third-party license obligations reviewed.
- [ ] README clone URL uses the final GitHub owner.
- [ ] `gait_runtime/` contains the runtime compatibility code; `demo/` contains
  only runnable demonstrations.
- [ ] Old experiments belong under `legacy/legacy_scripts/`.
- [ ] Repository topics set: `gait-recognition`, `drone`, `opencv`, `pytorch`,
  `rtsp`, `computer-vision`.

See `docs/RELEASE_AUDIT.md` for the cleanup risks that remain after this
release pass.
