# Publishing to GitHub

## 1. Copy model weights (local only, not pushed)

From your backup `OpenGait/demo/checkpoints/` copy into this repo:

- `Drone-YOLO/best.pt`
- `gait_model/GaitBase_DronGait1-60000.pt`
- `seg_model/.../model.pdiparams` (and `model.pdmodel` if needed)

These paths are in `.gitignore` and will **not** be committed.

Optional: copy gallery pickles to `output/gallery/` for local testing.

## 2. Initialize and first commit

```powershell
cd D:\py\py\realtime-gait
git init
git add .
git status   # verify no .pt / .pkl / output/registry.json
git commit -m "Initial release: realtime drone gait recognition pipeline"
```

## 3. Create GitHub repo and push

On GitHub: **New repository** → name e.g. `realtime-gait` → **do not** add README (already have one).

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/realtime-gait.git
git push -u origin main
```

## 4. Before making public — checklist

- [ ] No personal names in `output/` (gallery is gitignored)
- [ ] No private IP addresses in committed code (defaults use `127.0.0.1`)
- [ ] Model weights not in commit (`git log --stat` / check repo size)
- [ ] LICENSE and README reviewed
- [ ] Update GitHub repo description and topics: `gait-recognition`, `drone`, `opencv`, `pytorch`

## 5. Large files (optional later)

If you ever need to host weights on GitHub, use **Git LFS** or release assets — do not commit multi-hundred-MB `.pt` files to normal Git history.

## Repo locations

| Path | Role |
|------|------|
| `D:\py\py\realtime-gait` | **Publishable** new repo (this folder) |
| `D:\py\py\realtime_gait_windows_needed_bundle\OpenGait` | **Backup** — keep unchanged |
