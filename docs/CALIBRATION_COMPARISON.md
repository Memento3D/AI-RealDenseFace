# v0.4.3 — experimental checkerboard camera comparison

New command: compare-calibration. Consumes the existing successful register-views run, unchanged head.obj, the existing MICA pins and configs/webcam.calibration.experimental.json. The included profile contains derived numeric checkerboard estimates (50/3 mm square, 10x7 inner corners, 1280x720); no photographs, landmarks, weights, credentials or run directories are packaged. It includes hashes linking the original calibration report and ZIP. Neither candidate is certified as the true camera.

Both standard5 and radial2 intrinsics remain fixed. Only per-view pose is refitted using the same seven approximate semantic correspondences and uniform least squares as register-views. This deliberately avoids mixing the depth weighting and Huber changes from refine-cameras with the calibration comparison. Existing and EPNP pose seeds are refined, and the lowest-error positive-depth candidate is selected. This is local pose fitting, not a global-optimality guarantee.

OpenCV distortion order: k1,k2,p1,p2,k3. The same distortion is supplied to solvePnP, pose refinement and ALL projections: fit landmarks, held-out chin and mesh wireframe. Observations stay in original normalized-photo pixel coordinates. Photos are not undistorted separately, avoiding double correction. No implicit resolution scaling is allowed. Same photo dimensions do not establish matching focus/zoom/crop; this remains a declared experimental assumption.

Source registration report, normalized photos, baseline overlays, calibration JSON and mesh are checksum-bound. Input landmarks must agree with baseline observations. Embedding is re-exported by the existing isolated worker after pin verification; no MICA inference or MediaPipe rerun occurs. Original mesh and identity stay unchanged. Every run needs a fresh output folder. Snapshots and provenance are retained.

Outputs:
- front.calibration_comparison.png / left.calibration_comparison.png / right.calibration_comparison.png: labeled columns BASELINE | STANDARD5 | RADIAL2.
- Individual front/left/right.standard5.png and *.radial2.png.
- report.json: raw per-point residuals, delta from baseline (negative means reduced residual), camera matrices, distortion, poses, calibration limitations and provenance.
- calibration.snapshot.json, embedding.npz, embedding_worker.log.

Success status CALIBRATION_COMPARISON_EXPORTED means artifacts were produced. No model is automatically accepted. These are in-sample camera fit residuals; chin is held out but fixed contour correspondence is approximate at side angles. Sparse wireframe still draws hidden surfaces. Lower residual does not establish improved head geometry or identity. Calibration paper flatness/edge coverage were imperfect and are recorded in the profile.

## Windows update and run

Copy src, tests, scripts, docs, README.md and pyproject.toml into the existing project. Copy ONLY the new webcam.calibration.experimental.json into configs. Preserve configs/mica.local.json, all previous configs, both environments, vendor, input and runs. No need to unpack the earlier measurement ZIP into the project.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head compare-calibration `
    --registration-run runs/webcam_registration_001 `
    --head-run runs/head_init_001 `
    --mica-config configs/mica.local.json `
    --calibration configs/webcam.calibration.experimental.json `
    --output runs/webcam_calibration_compare_001
explorer.exe .\runs\webcam_calibration_compare_001
```

Verification: full local regression suite, architecture checks and compilation. Synthetic nonzero-distortion/off-center-camera pose recovery, invalid calibration/dimension rejection, top-level CLI dispatch and comparison command integration (synthetic geometry with mocked external embedding boundary). Test checks both model outputs, unchanged source and refusal to overwrite. No Windows/GPU or real MICA mesh result is claimed. User must run the command on the configured Windows environment to measure real-image results.
