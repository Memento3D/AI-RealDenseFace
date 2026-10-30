# v0.5.1 — dynamic contour and self-occlusion audit

New audit-contours command inspects original MICA and the v0.5 shape candidate against the same frozen cameras. No geometry, camera, model pin or optimization limits are changed. Existing fit-head behavior remains available unchanged. This release diagnoses correspondence before adding contour terms to optimization; it does not claim a new contour-aware fitted mesh.

The isolated worker exports the pinned FLAME2020 79-row x 17-landmark dynamic jaw embedding and full 68 landmarks. No MICA inference/GPU model is loaded. The camera rotation is converted to FLAME head axes with diag(1,-1,-1) @ R_opencv; the yaw extraction and signed row indexing follow the pinned upstream FLAME lookup convention. Synthetic frontal and +/-20, +/-60 degree rotations test the convention. Frozen camera pose is still an estimate, not independent ground truth.

FLAME dynamic table support is +/-39 degrees. Rounded angles select a row; negative angles map to 39-angle. Outside support, the row is clamped for visualization and angle_clamped=true, within_table_support=false is reported. Do not interpret clamped contour as a validated correspondence or use it to pull the jaw into place. Existing side-view camera estimates may exceed this limit even when MediaPipe yaw differs. This release does not request new photos or camera calibration.

The seventeen approximate MediaPipe face-oval counterparts are [127,234,93,132,58,172,136,150,152,379,365,397,288,361,323,454,356]. These follow a jaw ordering but are not a proven pointwise equivalent to FLAME's iBUG contour. Numbered image diagnostics and explicit semantic_mapping_verified=false permit review. All used_for_fit flags are false in this audit. The seven earlier semantic points are also checked.

Self-occlusion uses ray/triangle intersections from the estimated camera center to each mesh landmark; any mesh intersection more than 1 mm in front of the landmark marks self_occluded=true. Positive camera depth and in-frame status are reported separately. This checks the reconstructed mesh only. It cannot detect occlusion by real hair, hands, glasses or incorrect landmark detections, and is not a photo visibility confidence. Tolerance is an engineering choice, not a calibrated reliability threshold.

Outputs: front.contour_audit.png, left.contour_audit.png, right.contour_audit.png, report.json, contours.npz, contour_worker.log. Panels: original MICA left, v0.5 shape candidate right. Green=observed approximate contour, red=projected mesh, orange=self-occluded point, yellow=residual. Jaw points 0..16 are numbered. Mean contour residual is diagnostic only and is not a likeness score or pass threshold. Source meshes, images and linked reports are hash-bound and never overwritten; every output directory is new.

## Windows

Copy src/tests/scripts/docs/README.md/pyproject.toml to the existing project. Preserve configs, both environments, vendor, input and runs.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head audit-contours `
    --registration-run runs/webcam_registration_001 `
    --head-run runs/head_init_001 `
    --shape-run runs/webcam_shape_fit_001 `
    --mica-config configs/mica.local.json `
    --output runs/webcam_contour_audit_001
explorer.exe .\runs\webcam_contour_audit_001
```

Status CONTOUR_AUDIT_EXPORTED means diagnostics were exported, not that contour correspondences passed. Report preserves all per-point values, row selection and uncertainty flags.

Verification: 73 local tests, architecture boundaries and compilation pass. New tests cover signed yaw/index conversion, clamping, invalid table rejection, ray occlusion, barycentric interpolation, image generation and top-level dispatch. Actual pinned dynamic embedding subprocess and real head audit remain unverified until Windows execution. Upstream lookup code inspected at MICA af22e7a5810d474bc28a1433db533723d6bd2b07; no upstream source or licensed assets are redistributed.
