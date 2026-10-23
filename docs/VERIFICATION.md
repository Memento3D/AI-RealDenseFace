# Verification evidence — 2026-10-21

Environment used to create this archive: Linux, Python 3.12; not the user's
Windows/RTX 4050. Windows installation and GPU benchmarks have NOT been run.

- 16 unittest contract, config, image-ingest and lifecycle tests passed.
- Architecture boundary check passed.
- CLI config-check passed; real MediaPipe 0.10.32 CPU task loaded and ran on
  the three supplied photographs (one front reference, two synthetic side views).
- Official model resource used:
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- Observed SHA256:
  64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
- Estimated yaw: front +1.86°, first side +21.56°, second side -21.27°.
- Status NEEDS_RETAKE because side absolute yaw < experimental minimum 25°.
  The thresholds were not changed to force these samples through.
- Real image quality/model fidelity, identity, authenticity and head reconstruction
  remain unvalidated. This is an adapter execution smoke test, not a benchmark.

No uploaded photos, derived personal images, task weights, environment, cache or
runs are included in this archive. No pretrained model training was performed.


# Update 0.2.0 — 2026-10-21

- 30 CPU tests pass, including original 16 tests.
- Boundary checker and Python compilation pass.
- New tests cover source checksum mismatch, rejected/synthetic input gates,
  review declaration, report retention, cancellation, real child process-group
  termination, OBJ geometry checks asset pin integrity, and raster preview on a synthetic tetrahedron.
- Pinned official MICA source was inspected at
  af22e7a5810d474bc28a1433db533723d6bd2b07. The wrapper uses its MICA class and
  pretrained loading; a local SCRFD ONNX runs via CPU provider rather than loading
  the whole detector/model pack on GPU. No upstream source redistributed.
- mica-status was run on that source and correctly listed absent model files.
- No MICA checkpoint, FLAME generic model or CUDA/Torch is available here.
  Therefore mica-doctor MODEL_LOADED, face mesh generation, 6 GB VRAM fit,
  Windows/Python 3.10 installation and likeness have NOT been verified.
- No inference success, substitute template face or GPU timings are fabricated.
- The optional model requirements are a candidate recipe, not a transitive lock.

# Update 0.3.0 — 2026-10-21

- 41 CPU tests pass (all prior 30 plus 11 research/renderer tests).
- Architecture boundaries and Python compilation pass.
- New tests verify rejected input never reaches the backend, error/cancel reports,
  research flags, exact normalized snapshots, unchanged mesh bytes/vertices,
  depth-buffer ordering, invalid geometry rejection and no output overwrite.
- Real MediaPipe CPU frontal gate was run on the supplied 900x1200 studio portrait:
  yaw about 1.86°, sharpness 243.48, brightness 191.22, no current quality issues.
  Its normalized snapshot was successfully copied and hash-verified. No GPU
  inference was substituted for this gate test.
- Smooth five-view preview was rendered on a synthetic 1106-vertex ellipsoid and
  visually inspected. This synthetic mesh is a renderer test only, not a face.
- Existing MICA initialization has Windows/GPU evidence reported by the user for
  v0.2 (GEOMETRY_EXPORTED, 5023 vertices/9976 faces). The new end-to-end experiment
  has NOT been run here with MICA assets/GPU. Same external worker path is reused.
- No studio-vs-webcam likeness outcome or improvement is claimed in this release.
- No personal images, inference outputs or models are packaged with the source.
