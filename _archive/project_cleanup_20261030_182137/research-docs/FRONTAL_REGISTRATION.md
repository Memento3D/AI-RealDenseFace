# Frontal registration v1 (research only)

register-head fits a camera to an existing FLAME2020 MICA mesh. It does not run MICA again or alter geometry, identity, texture or expression. Uses CPU MediaPipe and OpenCV. The isolated MICA Python only decodes the checksum-pinned upstream landmark embedding (which requires Torch for deserialization); no CUDA inference is requested.

Seven approximate semantic correspondences (eye corners, nose tip, mouth corners) estimate pose with assumed focal length=max(image width,height), centered principal point and zero lens distortion. Uses FLAME full 68-point barycentric embedding. Chin is shown separately, held out of pose estimation. Anatomical mappings are approximate; detectors and expressions introduce additional error. No production threshold or percentage likeness is provided. Fit-point residuals are in-sample and can be small even for incorrect face shape. The held-out chin alone is not independent reconstruction validation. Sparse cyan wireframe has no hidden-surface removal and is not a silhouette segmentation.

Original photograph and mesh must match the SHA256s in the successful reconstruction report. EXIF normalization follows MediaPipeInspector. Near-frontal quality checks remain active. Outputs require a fresh directory, with hashes and camera parameters in report.json. No source files are overwritten.

## Windows

Copy updated src, tests, scripts, docs, pyproject.toml and README.md into the existing project; preserve configs, vendor, environments, input and runs. Install with pip install --no-deps -e . using .venv Python. Run register-head --help for arguments. Use the original image path from experiment-head, not front.aligned.jpg.

Outputs: front.registration.png, report.json, front.normalized.png, embedding.npz and embedding_worker.log. Green=MediaPipe observations, red=projected mesh landmarks, yellow=deviation, cyan=sparse mesh wire. This is camera registration, not multi-view shape fitting.

Verification: synthetic known-camera recovery, invalid point/topology rejection, regression suite and architecture checks run locally. Full pinned embedding subprocess + real MICA head registration must be verified on the user's configured Windows environment; no such end-to-end run is claimed here.
