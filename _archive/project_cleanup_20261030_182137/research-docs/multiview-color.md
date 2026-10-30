# Three-photo color preview (0.10.0)

Run from the project root in PowerShell:

```powershell
& .\scripts\preview_multiview.ps1
```

The helper installs the editable package without changing dependencies, finds the
newest successful fit-jaw result and its exact checksum-linked original camera
registration, creates a fresh result directory, and opens the offline viewer.
It does not choose an unrelated registration when a match is absent.

Explicit CLI:

```powershell
.\.venv\Scripts\python.exe -m memento3d_head preview-multiview-color `
    --candidate-run runs/YOUR_JAW_FIT_RUN `
    --registration-run runs/YOUR_LINKED_REGISTRATION_RUN `
    --output runs/multiview_color_001
```

The viewer provides front, left, right and combined modes. Changing modes resets
the view. In the combined mode each triangle uses the eligible camera with the
highest facing cosine; ties prefer front, left, right. Eligibility retains the
single-photo eroded landmark hull, full-mesh depth buffer and finite interior
samples. Unobserved triangles remain gray. No neural inference is added.

The atlas is a horizontal packing of source photos, not a baked facial UV texture.
The archive stores original mesh arrays, three projective-coordinate arrays,
coverage flags and chosen_source (-1 unknown, 0 front, 1 left, 2 right). Viewer
vertices are duplicated only to separate texture coordinates at seams; physical
triangle positions remain unchanged. No OBJ or GLB export is included in this step.

There is no color balancing, blending, albedo recovery, geometry fitting or
likeness verification. Captured lighting, occluders, camera error and visible
source seams can remain. Coverage counts refer to the entire mesh. The standalone
local example verifies original JPEG hashes but re-encodes normalized PNGs locally;
the production CLI verifies the actual registered PNG hashes instead.

## 0.10.1 correction

The combined mode now retains every eligible frontal triangle. Side images only
supplement uncovered areas outside image-space rectangles around eyes, nose and
mouth. Missing frontal coverage inside those rectangles remains gray. These are
approximate protections derived from registered landmarks, not semantic masks.
This replaces the per-triangle highest-facing selection described above; that
rule remains only for choosing between eligible supplementary side views.

Side RGB offsets use median differences at shared outer-face triangle centers,
require at least 30 samples and are limited to +/-15 per channel. Single-view
modes retain raw source colors. This is a bounded color adjustment, not relighting
or skin albedo recovery. No feathering or seam-free blend is implemented.

The WebGL fragment shader requests high precision where available for projective
texture coordinates, with a medium-precision fallback. Viewer serialization rounds
normals and texture coordinates and removes JSON whitespace; original mesh arrays
and projective coordinates remain in the NPZ archive. Geometry and cameras do not
change. Gray holes and registration mismatch can remain.

## 0.10.2 correction

Depth policy v2 compares the triangle plane and rasterized occluder depth on the
same pixel-center ray. The previous subpixel-vs-pixel comparison could reject
sloped visible surfaces. On the current local photo example this recovers only
one additional triangle; it does not explain or solve the remaining gray holes.

The viewer now fades photographic color to the existing gray shading over one
mesh edge ring outside protected feature regions bordering unpainted triangles. Source eligibility and original
geometry do not change. This is boundary feathering, not source-to-source blending,
texture completion, shape refinement, or alignment correction. Fully interior
photographic regions retain their colors. Existing PNG masks/NPZ coverage remain
geometric eligibility data, not the feathered display opacity.
