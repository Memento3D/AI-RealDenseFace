# Offline 3D comparison — 0.8.0

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head compare-heads `
    --head-run runs/head_init_001 `
    --candidate-run runs/webcam_jaw_fit_001 `
    --output runs/webcam_head_viewer_001
Start-Process .\runs\webcam_head_viewer_001\head_comparison.html
```

Verifies linked original/jaw candidate reports and OBJ checksums, matching face
ordering, finite vertex arrays and unchanged sources. Existing output directories
are rejected. No model inference or network asset is needed. The output contains
one self-contained HTML and a provenance report. Keep HTML private as it embeds
both user meshes. User meshes are never included in source archives.

Download/open the HTML locally in Chrome or Edge. Drag to orbit, wheel to zoom,
choose preset yaw views, reset, show wireframe or per-vertex displacement colors.
Both panels share rotation, distance, projection, light and normalization from the
original bounds. Vertex positions are not independently centered/scaled or
exaggerated. Candidate and baseline normals are calculated from their actual
triangles. A color change represents vertex displacement in mm, not a quality
score or likeness estimate. The original is blue/zero in displacement mode.

WebGL1 draws on interaction/resize rather than in a continuous animation loop.
Geometry is not exported or edited by browser controls. No remote JS, fetch,
photo upload or server is used. A readable error appears when WebGL is unavailable
or its context is lost. The mesh has no texture; canonical viewing angles do not
match the registered photo cameras. Two-finger zoom/pan are not implemented;
one-pointer orbit and wheel zoom are available.

Verification: 108 unit/integration tests pass; source mismatch, topology and
no-overwrite checks are tested. Wheel inspection confirms HTML package data.
JavaScript syntax and event logic (orbit, presets, zoom, wireframe, heat map,
reset and no-WebGL fallback) were checked with a mock GL context. Real browser
GPU rendering was not verified in the build environment: the browser download
failed. This does not establish Windows/browser performance.

The provided standalone HTML uses the actual uploaded original mesh and a
candidate reconstructed from its eight reported coefficients and uploaded basis.
The reconstructed serialized OBJ matches the user's Windows candidate checksum
exactly (including CRLF), and vertices are displayed at that OBJ's saved precision.
No new model inference is involved.
