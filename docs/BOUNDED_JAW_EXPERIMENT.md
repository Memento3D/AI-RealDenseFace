# Bounded jaw experiment — 0.7.0

Run after a v0.6.4 compare-jaw-contours attempt with both baseline frontal curves
supported. Uses the original MICA mesh/identity and the checksum-linked eight
FLAME identity directions already exported by fit-head. No model download,
Torch import, new GPU inference, camera adjustment, or source overwrite.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head fit-jaw `
    --head-run runs/head_init_001 `
    --shape-run runs/webcam_shape_fit_001 `
    --registration-run runs/webcam_registration_001 `
    --workspace runs/webcam_manual_contours_001 `
    --validation-run runs/webcam_manual_contours_check_002 `
    --jaw-comparison-run runs/webcam_jaw_comparison_003 `
    --output runs/webcam_jaw_fit_001
```

Checks report links, original OBJ/identity/basis/photo hashes, basis face order,
decoder/source discrepancy <= 0.1 mm, and a continuous frozen jaw path from the
comparison. Validation annotations are used from their immutable snapshot.
Unsupported frontal curves fail; side contours are never included in fitting.

Freeze baseline jaw edge endpoints and interpolation weights at 64 samples per
curve. Minimize squared frontal jaw residuals plus the 21 approximate semantic
landmark residuals and an identity prior. Camera and expressions stay fixed.
This metric uses FROZEN correspondences and can differ from the v0.6.4 diagnostic
which reselects endpoints; both appear separately in the output report. Neither
measures person likeness. Both frontal training curves must improve or remain
unchanged before a step can be accepted.

Versioned experimental engineering limits (not calibrated quality thresholds):
coefficient delta <= 0.5, vertex displacement <= 3 mm, per-view mean landmark
increase <= 0.5 px and each point increase <= 1 px. Reject collapsed/flipped
triangles. Only eight global FLAME identity dimensions are available; this is
not isolated jaw sculpting. Guards are based on approximate correspondences;
they do not ensure correct anatomy, identity or generalization. All 21 landmark
points participate in fitting. An additional fixed-index chin check is reported
outside the objective, also with unverified semantic pairing.

Outputs: head.candidate.obj, identity.candidate.npy (matching pair), smooth mesh
preview, three jaw comparisons and report.json. Status JAW_EXPERIMENT_EXPORTED
means a candidate was exported, not automatically accepted. No improvement
returns NO_SHAPE_CHANGE. Original head and prior candidates remain untouched.

Local backend experiment on supplied data: max displacement 1.33494 mm;
frontal frozen jaw means 19.5294 -> 18.0825 and 7.2611 -> 6.2304 px.
Landmark means front 6.8018 -> 6.5444, left 5.8248 -> 5.8388, right 5.5781 ->
5.5895 px. Held-out chin: front 9.9629 -> 8.2901, left 32.9859 -> 32.8591,
right 28.8991 -> 29.1309 px. These are mixed diagnostic outcomes, not proof of
improved likeness. The local experiment used 64-point reference samples from
the supplied previous diagnostic report; the CLI uses the original validated
annotation snapshot, so tiny numeric differences are expected.

104 tests pass, including actual numerical fit guards, zero-basis no-change,
unsupported/nonfinite inputs, and a synthetic source-bound CLI export with
no-overwrite and changed-basis rejection. Real Windows CLI/config execution
remains to be confirmed on the user's machine. User assets are excluded from
source archives; a separate result archive contains the local experiment only.
