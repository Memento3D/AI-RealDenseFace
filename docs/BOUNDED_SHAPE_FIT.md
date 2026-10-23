# v0.5.0 — conservative shared FLAME identity candidate

fit-head is the first implemented step that changes geometry using all three photographs. It consumes the original successful register-views attempt and the corresponding successful MICA head initialization, including identity.npy. No new photos, camera calibration or model download is required. Prior camera experiments stay separate. No originals are overwritten.

## Method and limits

The isolated Python 3.10 worker imports the exact pinned upstream models.flame.FLAME class, loads the pinned FLAME2020 generic model, sets zero expression and zero articulation, and reconstructs the original 300 identity coefficients on CPU. It exports eight local shape directions from coefficient perturbations. No MICA/ArcFace inference or GPU allocation is requested. Decoder face indices must exactly match the OBJ; reconstructed vertices must be within 0.1 mm of the source mesh or the command stops with DECODER_MISMATCH. This guard detects checkpoint/model incompatibility rather than applying an arbitrary deformation.

The main process fits only the first eight identity coefficient offsets, shared across three views. Original register-views camera intrinsics and poses are frozen, with zero distortion as in that baseline. This isolates shape changes; it does not establish that those cameras are correct. Seven existing approximate semantic correspondences per view (nose tip, four eye corners, two mouth corners) give 21 observed 2D points. The same points are used for optimization and in-sample residuals. They are not an identity validation set. Neither contour nor chin is optimized. Expression, jaw articulation, eyes, hair and texture remain unmodeled in this experiment.

A Huber data term (delta=6 pixels) and quadratic prior 25*||delta_beta||^2 penalize departure from MICA. Offsets are bounded at +/-0.5 per coefficient and max vertex displacement is bounded at 3 mm. At most 30 IRLS/Gauss-Newton steps with monotonic backtracking. Candidate triangles must not collapse or reverse relative to the source. These are conservative engineering bounds, not empirically calibrated quality thresholds. Sparse data, camera error, approximate mappings and partial occlusion can still produce a worse face even when loss decreases. Large improvements are not promised.

correspondences.json makes the actual seven FLAME68/MediaPipe pairs and excluded fixed chin explicit. This is an audit record, NOT a claim of a newly validated landmark mapping. Dynamic contour selection and automatic occlusion filtering remain unimplemented; previous fixed chin errors are only an approximate visual reference at side angles. Cameras are deliberately frozen in this release. No need to print or photograph more checkerboards.

## Output

SHAPE_CANDIDATE_EXPORTED: a nonzero bounded geometry candidate was exported. NO_SHAPE_CHANGE: optimization did not change identity coefficients. Neither status means likeness was verified. Failures keep a report after exclusive output creation; existing output directories are rejected.

- head.candidate.obj: actual bounded mesh, source topology and mm units.
- identity.candidate.npy: original 300 coefficients plus eight fitted offsets.
- head.candidate.preview.png: smooth canonical preview of candidate.
- front/left/right.shape_comparison.png: original MICA left, bounded candidate right, same camera.
- front/left/right.candidate.png: candidate overlay per view.
- report.json: objective history, limits, source hashes, decoder consistency error, source/candidate residuals and geometry change flag.
- correspondences.json, shape_basis.npz, shape_worker.log: reproducibility diagnostics containing licensed-model-derived data; do not commit or redistribute runs.

## Windows update

Copy src, tests, scripts, docs, README.md, pyproject.toml into the existing project. Keep configs, both environments, vendor, input and all runs. Install editable with --no-deps. Uses the already configured .venv-mica Python from configs/mica.local.json.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head fit-head `
    --registration-run runs/webcam_registration_001 `
    --head-run runs/head_init_001 `
    --mica-config configs/mica.local.json `
    --output runs/webcam_shape_fit_001
explorer.exe .\runs\webcam_shape_fit_001
```

## Verification

67 local unit tests, architecture boundaries and compilation. Tests verify monotonic reduction on synthetic shared-identity data, unchanged source arrays, no invented change on exact matches, coefficient/displacement bounds under large disagreements, nonfinite rejection, top-level dispatch and rejection of modified identity before worker launch. The pinned upstream FLAME implementation was inspected at af22e7a5810d474bc28a1433db533723d6bd2b07. No licensed weights available locally: actual decoder subprocess and real Windows fitting remain unverified and must run on the user's environment. Do not claim GPU/Windows runtime or real likeness improvement from synthetic tests.
