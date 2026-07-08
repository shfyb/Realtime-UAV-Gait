# Release Audit

This document records the repository cleanup items that matter before a public
GitHub release.

## Current Status

Ready for a source release with required model weights tracked through Git LFS:

- Runtime output is ignored through `output/`.
- Model checkpoints under `checkpoints/` are tracked through Git LFS.
- Runtime-only user data such as gallery files remains ignored.
- Release documentation exists in `README.md`, `docs/PUBLISHING.md`, and
  `docs/WINDOWS_DEPLOY.md`.

## Main Release Risks

1. Third-party source is still bundled under `gait_runtime/`.
   This makes the repository look larger and more experimental than the core
   project. The top-level legacy scripts have been moved to
   `legacy/legacy_scripts/`, and `.gitattributes` marks the largest
   vendored folders so GitHub stats focus on this project. The cleanest future
   release should replace these folders with package dependencies or documented
   submodules.

2. Public weight redistribution must be license-checked.
   The checkpoint files are now intended to ship through Git LFS. Before making
   the repository public, verify the redistribution terms for every model file
   and keep a clear model/license note.

3. Documentation is Windows-first.
   That matches the deployment target, but a public release should explicitly
   label Linux support as community/experimental unless tested.

## Recommended Next Refactor

- Move remaining project-owned runtime adapters from `gait_runtime/` into
  `realtime_gait/modules/` or `realtime_gait/vendor_adapters/`.
- Replace vendored `ultralytics`, `mmseg`, and `paddleseg` trees with pinned
  dependencies where possible.
- Keep only small compatibility patches in this repository, each with a note
  explaining why upstream packages cannot be used directly.
- Add smoke tests that validate imports, config loading, path resolution, and
  gallery serialization without requiring GPU model weights.

## Pre-Release Commands

```powershell
git status --short
git ls-files checkpoints
git lfs ls-files
python -m compileall realtime_gait opengait
```
