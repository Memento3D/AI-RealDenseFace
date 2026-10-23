# Frontal photographic color preview — 0.9.0

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head preview-photo-color `
    --candidate-run runs/webcam_jaw_fit_001 `
    --registration-run runs/webcam_registration_001 `
    --output runs/webcam_photo_color_001
Start-Process .\runs\webcam_photo_color_001\photo_preview.html
```

Uses the actual candidate OBJ, frozen frontal registration and normalized photo.
Checks linked report and photo/mesh checksums and original photo dimensions.
Existing outputs are rejected. Does not change mesh, identity or camera and does
not download or execute a neural model. Requires the existing vision dependency
(OpenCV for Rodrigues); rasterization itself uses NumPy/Pillow on CPU.

Method: project the complete mesh into the photo camera and rasterize a
perspective-correct depth buffer. Restrict candidate triangles to the eroded
convex hull of the registered face landmarks, with all three vertices inside.
Keep triangles with view-facing cosine >= 0.4 and four interior depth samples
within 0.75 mm of the full mesh depth buffer. Erode the image-space hull by four
pixels. These engineering settings are experimental, not calibrated quality
thresholds. The mask is NOT skin/hair segmentation. Finite sampling is NOT a
proof that the whole triangle is visible. Fine creases and camera mismatch can
leave holes or produce limited bleed; they must be inspected visually.

Both viewer panels use the SAME candidate geometry. Left is gray, right uses
source-photo color only on passing triangles. Controls: synchronized orbit,
wheel zoom, yaw presets, wireframe, and a photo-color toggle. Unknown surfaces
stay gray. Source image, mesh and projection coordinates are embedded in the
HTML; it opens offline with no external assets. Keep the file private.

Viewer uses homogeneous source-photo coordinates for projective texture lookup,
and shows captured RGB directly without relighting it. Existing shadows remain.
It is NOT intrinsic albedo recovery, a UV atlas, multi-view blending, completed
hair/eyes/teeth, likeness validation or a print-ready model. Do not infer that
photo color corrects geometry. Side/back/head regions are deliberately unknown.

Outputs: photo_preview.html, front.photo.png, front.coverage.png,
photo_projection.npz (homogeneous coordinates + covered face flags), report.json,
head.photo.obj, head.photo.mtl. OBJ stores approximate per-vertex projected UVs;
it is not the exact homogeneous projective lookup used by the HTML. Keep the OBJ,
MTL and front.photo.png together. Uncovered faces use the unknown gray material.

Local actual-data backend trial: 3018 of 9976 entire-head triangles pass. This
30.25% is coverage of the full mesh, NOT a likeness score or percentage of usable
face coverage. Mesh was reproduced from the user's exact reported deltas/basis.
The original uploaded JPEG checksum matches the registered source JPEG. Local
normalized PNG was re-encoded and its bytes differ from the original registered
PNG; the standalone demo records this. The CLI uses the original normalized PNG
from the registration run and rejects checksum mismatches.

115 tests pass, including perspective depth vs analytic ray/plane intersection,
rear-surface rejection, backfacing/mask exclusions, source-bound CLI export,
geometry preservation, changed-photo rejection and output non-overwrite.
JavaScript syntax and image-load/toggle/preset logic were checked with a mock GL
context; browser rendering of the new texture mode still needs user confirmation.
A separate CPU raster preview was inspected at 0 and +/-45 degrees; gray gaps at
creases/eyes/nose/mouth are retained rather than filled with invented colors.
