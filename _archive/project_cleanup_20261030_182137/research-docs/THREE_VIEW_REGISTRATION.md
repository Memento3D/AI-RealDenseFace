# Three-view camera registration v1

register-views fits an approximate camera separately for each of three real photographs against the same unchanged head_init mesh. This is not multi-view shape optimization. No MICA inference is requested; MediaPipe and camera fitting use CPU. The existing isolated MICA Python decodes the pinned landmark embedding.

Required: successful reconstruct-head run, original validation report and the exact three input files. Hash binding rejects swapped sides or modified input/mesh/report. It does not prove identity, authenticity or a calibrated capture. Every attempt uses a fresh output folder. Settings and assessment are retained. No originals are overwritten.

Outputs: front.registration.png, left.registration.png, right.registration.png and report.json, plus normalized photographs and embedding diagnostics. Each camera has its own residual report. The seven landmarks used for camera estimation are also used for fit residuals; chin is held out. Residuals are not likeness scores and should not be compared across different pixel resolutions without normalization.

Limitations: assumed focal length and centered principal point; approximate semantic correspondences; no automatic landmark visibility filtering; no hidden-surface removal on sparse wireframe; expressions and perspective may affect residuals. At side angles the far eye may be partly occluded, so manually inspect all three overlays before any shape fitting. side_visibility_verified is false. No three-view real Windows result has been verified in this release.

v0.4.1 also includes the CLI dispatch fix and singleton-batch embedding fix from the preceding Windows diagnosis. Local verification: 51 unit tests, architecture boundary check and compilation. Tests include the top-level register-views dispatch, exact source binding, swapped side rejection, changed mesh/report rejection and batch embedding shapes.

Update existing project: copy src, tests, docs, scripts, README.md and pyproject.toml. Preserve configs, vendor, input, runs and both virtual environments. Install with .venv Python: pip install --no-deps -e . . Use register-views --help for complete flags.
