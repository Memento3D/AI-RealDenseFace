# Single-image experiment and smooth previews — 0.3.0

## Scope

This research command compares MICA geometry initialization from different source
portraits. It is not an alternate production approval path. The three-view
validate-photos/reconstruct-head flow and its checksum/human-review requirements
are unchanged. No texture, hair, fitting or MCP is added.

The MICA model, upstream code, assets config and worker dependencies remain the
same as 0.2.0. No model repin, re-download or Torch installation is needed.

## Upgrade from the working Windows installation

Extract the release to a temporary directory. Copy src/, tests/, docs/, scripts/,
README.md and pyproject.toml into the existing project, replacing corresponding
source files. Preserve .venv/, .venv-mica/, vendor/, input/, runs/ and configs/,
including configs/mica.local.json and your photo threshold config.

From the existing project root:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head --help
```

You should see experiment-head and render-head. The external worker still runs
with its Python 3.10 interpreter specified by the existing MICA config.

## 1. Re-render the webcam baseline without inference

```powershell
.\.venv\Scripts\python.exe -m memento3d_head render-head `
  --mesh runs/head_init_001/head.obj `
  --output runs/head_init_001_smooth
```

PREVIEW_EXPORTED means only a new diagnostic image was generated. Read
runs/head_init_001_smooth/head.preview.smooth.png. The original OBJ/PLY, identity
parameters and baseline report remain untouched. The preview report records
source mesh SHA256 and renderer version; inference_run is false.

Smooth rendering uses area-weighted vertex normals, barycentric interpolation and
a per-pixel depth buffer. Geometry and topology are unchanged. It does not add
subdivision or improve reconstruction accuracy. It shows 0, -45, +45, -90, +90°
canonical yaw with equal scale; these are NOT fitted photograph camera angles.
No textures, face similarity scores or claims of printability are displayed.

## 2. Trial from the original studio portrait

Copy the original clear frontal photograph into input/34586.jpg (not a screenshot
or an AI-generated side view). Keep the old webcam images and runs.

```powershell
$modelPath = 'D:\AI-cache\mediapipe\face_landmarker.task'
$modelHash = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'

.\.venv\Scripts\python.exe -m memento3d_head experiment-head `
  --input input/34586.jpg `
  --source-kind real_capture `
  --model $modelPath `
  --model-sha256 $modelHash `
  --mica-config configs/mica.local.json `
  --config configs/photo_validation.local.json `
  --output runs/studio_head_001
```

This command first runs the existing CPU image inspector with frontal
quality/angle checks, closes MediaPipe, snapshots the normalized pixels and only
then starts the external MICA worker. Thresholds are still experimental. A bad
input gives NEEDS_RETAKE (exit 2), not a fake face or automatic relaxed threshold.

Model/decode/dependency errors give FAILED (exit 1); Ctrl+C gives CANCELLED (130).
Successful inference/export gives GEOMETRY_EXPORTED (0).

The report has pipeline_type=single_image_experiment, pipeline_version=1,
research_only=true, photo_set_validated=false and multi_view_fitting=false.
source-kind is a declaration, not proof of real capture. No prior three-image
report is fabricated or required. Identity/authenticity/likeness are not verified.

Outputs include the standard geometry files, the legacy flat head.preview.png,
and the new **head.preview.smooth.png** with five views. quality records the
single-image measurements and source hash. The worker input is the exact bytes
of front.normalized.png copied as front.input.png. Failure retains diagnostics.

## 3. Compare

Compare head_init_001_smooth/head.preview.smooth.png against
studio_head_001/head.preview.smooth.png, using the same lighting/renderer. Inspect
front.aligned.jpg for crop/alignment differences. Look at face outline, nose,
cheeks, lips and chin; absence of hair/texture is expected. The two source photos
were taken under different conditions and possibly at different times, so this
is exploratory evidence, not a controlled attribution to image sharpness alone.

Do not add smoothing filters or sculpt the baseline before comparison. If changing
thresholds deliberately, save a separate config and use a fresh output folder.
