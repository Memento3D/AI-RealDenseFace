# Demo from a new three-photo folder — v0.11.0

Keep the working `.venv`, `.venv-mica`, `vendor`, `configs/*.local.json` and model
cache already installed on your Windows machine. Update only this source package.
This demo uses no prior runs, hand-drawn contours, calibration profiles or jaw fits.

## Prepare

Create `input/01` and put exactly one image for each stem inside it:

| Stem | Example |
| --- | --- |
| front | input/01/front.jpg |
| left | input/01/left.png |
| right | input/01/right.png |

JPG/JPEG/PNG/WebP and uppercase extensions work. Do not leave both front.jpg and
front.png in the same folder. Folder names are arbitrary: 01, 02, teacher_demo.
Use real photos of the same person, neutral expression, consistent lighting,
visible face and opposite side views. Existing photo-validation rules still apply.

From the project root in PowerShell, install once after updating:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

## Run

```powershell
.\.venv\Scripts\python.exe -m memento3d_head demo --input-dir input/01
```

The command uses defaults already configured in this project:

- Face landmarker: D:/AI-cache/mediapipe/face_landmarker.task, pinned SHA256.
- MICA configuration: configs/mica.local.json.
- Photo rules: configs/photo_validation.local.json.

Override locations if needed:

```powershell
.\.venv\Scripts\python.exe -m memento3d_head demo `
    --input-dir input/02 `
    --model D:/AI-cache/mediapipe/face_landmarker.task `
    --mica-config configs/mica.local.json `
    --config configs/photo_validation.local.json
```

## One review during the run

After validation passes, the command opens `01_validation`. Inspect
front.landmarks.jpg, left.landmarks.jpg and right.landmarks.jpg. Confirm the
correct person, slots and plausible face landmarks. Return to the terminal and
enter your name at `Reviewer name:` to continue. Blank input cancels; it does not
silently proceed. The name is recorded as a human review declaration, not proof
of identity. If automatic validation rejects an image, retake the indicated view.

After that the remaining stages run sequentially and the final HTML opens.
No PowerShell script or execution-policy change is needed for this command.

## Outputs

Each invocation creates a fresh `runs/demo_01_<timestamp>_<unique>` directory:

| Path | Contents |
| --- | --- |
| 00_inputs | Exact snapshots of the three new files |
| 01_validation | Quality report, normalized images and landmark previews |
| 02_head | MICA mesh, identity coefficients, inference log |
| 03_registration | Three fitted cameras, overlays, normalized images |
| final/multiview_preview.html | Self-contained offline color preview |
| final/head.obj, head.ply | Original untextured geometry for inspection |
| final/identity.npy | MICA identity coefficients |
| final/multiview_projection.npz | Geometry, projective coordinates and coverage |
| final/photos.atlas.png | Packed photographic sources, not a baked head texture |
| final/report.json | Final output hashes, method and limitations |
| report.json | Overall result and stage paths |
| 01_validation.log, 02_head.log, 03_registration.log | Stage diagnostics |

`--output runs/demo_01_001` chooses a destination explicitly. It must not already
exist. `--no-open` disables launching folders/viewer but still requires interactive
review in the terminal. Ctrl+C cancels. Logs and reports remain for diagnosis.

## What to tell the teacher

The prototype checks a real three-photo set, initializes head geometry from the
front photo using MICA/FLAME, estimates cameras for all three views, and projects
photo colors onto the existing surface. The side images contribute camera and
color information; this is not dense three-view geometry reconstruction.

The manual contour/jaw optimization used on the earlier subject is intentionally
excluded. Its annotations do not transfer to a new photo set. New-subject results
may differ from the manually adjusted example. Hair, clothing, separate finished
eyes/teeth, a production UV texture, GLB export and print readiness are outside
this demo. Capture feedback on likeness, visible seams and gray gaps before
adding these steps. No identity or likeness score is claimed.

## Validation performed for this release

Unit/integration tests verify new-input isolation, name resolution, review/failure
handling, report bindings and actual color-viewer generation on synthetic fixtures.
Model inference is mocked only in tests; the real CLI always invokes the installed
MICA worker. End-to-end live Windows/GPU inference has not been run in the packaging
environment. Rehearse once locally before presenting; no timing guarantee is made.
