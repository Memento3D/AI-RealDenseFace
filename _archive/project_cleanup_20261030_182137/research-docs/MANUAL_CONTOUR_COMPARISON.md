# Manual contour comparison (0.6.1)

Run after validate-contours, using a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -m memento3d_head compare-manual-contours `
    --workspace runs/webcam_manual_contours_001 `
    --validation-run runs/webcam_manual_contours_check_002 `
    --registration-run runs/webcam_registration_001 `
    --output runs/webcam_manual_comparison_001
```

The command reads the validated snapshot, verifies its checksum and linked
manifest/registration/photos, and compares the already projected original and
candidate silhouettes. No model download, GPU inference, camera adjustment or
geometry modification occurs. Existing output directories are rejected.

Each image contains baseline (left) and candidate (right). Green is the manual
curve, cyan/purple the proposed silhouette arc, orange sample connections.
Review these connections: the projected whole-head silhouette contains ears
and neck, so automatically selected arcs are NOT verified anatomical matches.

For each closed outline, the method locates nearest sampled endpoints, examines
both connecting arcs, samples 64 equally spaced normalized arclength positions,
and retains the arc with the smaller mean paired distance. Drawing direction
is supported in either order. Sampling the endpoints is approximate. Equal
arclength does not imply anatomical correspondence. Sparse annotation vertices,
rasterization and camera errors affect results. The arc search is diagnostic;
it is not a valid automatic fitting objective or a likeness measurement.

Report distances are per curve in pixels at the original photo resolution.
Candidate-minus-baseline below zero means a smaller distance under this
specific pairing only. No score or automatic acceptance is produced.

Send front.manual_comparison.png, left.manual_comparison.png,
right.manual_comparison.png and report.json for inspection before fitting.
