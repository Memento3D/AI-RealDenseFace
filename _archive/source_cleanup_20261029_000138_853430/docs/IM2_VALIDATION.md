# Validation of Im2Haircut prepared-sample update

2026-10-27: 148 unittest cases passed; architecture boundaries PASS.
New checks cover ambiguous patches failing without mutation, patch idempotence,
invalid/incomplete strand arrays, ordered PLY to NPZ/OBJ roundtrip, refusing to
overwrite exports, and monitoring a real failed subprocess with its exit code
and log preserved even when GPU telemetry is unavailable.

Both patches were also matched against fetched upstream trainer source.
CLI help and parsing passed. Tests used Python 3.12 with isolated NumPy 1.26.4,
Pillow 11.3.0, OpenCV headless 4.11.0.86, SciPy 1.15.3, trimesh 4.7.4,
psutil 7.1.0 and pyhocon 0.3.60. The user runtime is separately Python 3.9/CUDA.

No RTX GPU or Blender runtime was available for validating the new runner and
Blender script here. The previously completed GPU sample with both source fixes
was executed on the user's laptop. No claim of new inference speed, peak VRAM,
Blender rendering correctness, or personalized hair accuracy is made.
