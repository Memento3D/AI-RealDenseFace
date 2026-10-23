# Shared camera experiment v1 — v0.4.2

refine-cameras consumes a successful register-views report, its normalized images and the exact unchanged head mesh. It does not need original filenames or rerun MediaPipe/MICA inference. The isolated MICA Python decodes the checksum-pinned landmark embedding; no GPU inference is requested. Reports/photos/mesh are checksum-bound. Fresh output directory required.

It tests up to 31 distinct focal candidates from 0.55 to 2.0 times the maximum image dimension, including the baseline factor 1.0. The focal is shared across views; pose is optimized per view. Principal point remains image center and distortion remains zero. All three oriented image dimensions must match. Same dimensions do not prove matching intrinsics: the same camera, zoom and crop are an assumption for this experiment.

A camera-depth heuristic assigns weight 0.35 to the farther eye's two landmarks when abs(detected yaw)>20 degrees. It does not detect actual visibility and can misclassify occlusion. Other fit points retain weight 1. Robust Huber loss has delta=6px and uses analytic pose Jacobians with backtracking. No SciPy dependency. Candidate poses are initialized from the previous cameras, and the finite search/local optimizer does not guarantee a global optimum. The shared objective averages weighted per-view Huber costs. Baseline and candidate are scored using the same weights/loss. All raw landmark residuals remain in the report.

Chin is held out and retains the previous fixed FLAME index only as a diagnostic. Dynamic contour correspondence is NOT implemented; do not use the chin residual as a shape-fitting objective at side angles. No silhouette or shape fitting is performed. Smaller fit loss does not prove improved identity/geometry. Search-boundary optima must be flagged, not treated as calibrated intrinsics. Source mesh is unchanged. Sparse wire still includes hidden surfaces.

Outputs: front/left/right.candidate.png, front/left/right.comparison.png (baseline left, candidate right), report.json and embedding diagnostics. CAMERA_EXPERIMENT_EXPORTED means files were produced, not that the candidate was accepted as better. Report retains per-view baseline/candidate residuals, all focal trials, selected focal, weights, objective and boundary flag.

Windows update: copy src/tests/scripts/docs/README.md/pyproject.toml, preserve environments/configs/vendor/input/runs, install editable with --no-deps. Run refine-cameras --help.

Verification: 56 local tests pass, architecture boundary check passes, source compiles. Includes known synthetic shared-focal recovery, unchanged source points, dimension mismatch rejection, depth-weighting policy, robust outlier cost and top-level CLI dispatch. Actual webcam improvement and Windows execution remain unverified until a user run.
