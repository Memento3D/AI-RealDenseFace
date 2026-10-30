# V1: three photographs to optimized textured GLB

## FLAME2023 update (workflow release 1.1)

The default five-stage workflow now uses the upstream model filename
vendor/RealDenseFace/weights/flame/flame2023.pkl consistently for front fitting,
joint fitting and texture projection. Keep the older Open model and outputs for
comparison, but run again from photographs rather than reusing their parameters.
An explicit --flame-model path is passed across stages. Model filename, SHA256
and flame_assets SHA256 are recorded in pipeline.report.json, with change checks.
The filename declares a variant; a locally computed hash is not independent proof
of download authenticity or that a mislabeled file is the official model.

Some legacy standalone diagnostics remain specific to the Open experiment.
Use scripts/run_v1.py for the updated supported five-stage workflow. An old
Open run is rejected by the texture model hash binding unless its original
model is explicitly supplied.

This release consolidates the existing v0.13.1 reconstruction prototype into a
single Windows command. V1 denotes the workflow version; the core package retains
its existing version. Existing .venv, .venv-rdf, pinned vendor checkout and model
assets must be preserved when overlaying this source archive.

## Setup

Use the existing working environments. No CUDA reinstall is needed.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/observability.txt
```

The existing CPU face parser dependencies and model must already be installed
(see requirements/face-parsing.cpu.txt and scripts/setup_face_parser.py).
This release does not download or include model weights.

## Run

Place one front, left and right photograph in input/03. Supported extensions:
.jpg, .jpeg, .png, with exactly one file per slot. All three must match the
resolution requirements of the existing three-photo fitter.

```powershell
.\.venv\Scripts\python.exe .\scripts\run_v1.py --input-dir input/03
```

For another set, change only --input-dir. No PowerShell script execution policy
change is required. Each run creates a unique directory; old runs are preserved.

## Outputs

- runs/v1_<timestamp>_<id>/head.optimized.glb: final geometry with embedded atlas.
- pipeline.report.json: input hashes, stage timings, resource peaks and status.
- metrics.csv: stage-tagged resource samples.
- 01_front.log through 05_export.log: stage logs.
- Nested intermediate reconstruction, texture and HTML workspace directories.

Import the final GLB in Blender via File > Import > glTF 2.0, then Material
Preview. It is standalone and does not require the original photographs.
The material is unlit to preserve photograph colors. Hidden-region skin fill
is flat, so it differs from the HTML's view-dependent fill lighting. The atlas
uses one tile per triangle; --cell 48 or 64 increases resolution and memory use.

## Boundaries

- The command checks file presence and slot names, not photo pose/quality or
  whether all photographs depict the same person. The existing validation CLI
  remains available separately. Do not label this as an enforced production gate.
- No hair, cap, eyeball or teeth assets are added.
- GLB export does not certify likeness, physical scale, watertightness or printing.
- RAM is summed child-process working set, with possible shared-memory overlap.
- GPU samples are whole-device values; one-second sampling can miss peaks.
- Per-stage wall times include imports, model loading, IO and observer overhead.
- V1 is sequential and does not establish safe concurrent job capacity.
- Source code and synthetic tests can be checked locally; actual Windows/CUDA
  end-to-end results require running on the user's configured machine.

Keep the existing FaceBox no-compile patch in vendor/RealDenseFace. This release
does not reset vendor, alter model pins or overwrite environments/configs.
