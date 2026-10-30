Extract into the project root; use existing .venv-rdf, project src, pinned vendor and assets.
Run:
.\.venv-rdf\Scripts\python.exe .\scripts\preview_rdf_three_textures.py --run-dir runs/rdf_open_20261023_152718_f3ce74/three_photo_7cf7011b --input-dir input/01

Output: new three_textures_<id>/head.three_textures.html (offline viewer), report.json, OBJ/MTL/source PNGs, texture.sources.npz.
Preserves current geometry, front expression, original source image resolution. Uses saved independent per-photo states for projection. No inference or refitting.
For every visible face prefer front source. Otherwise choose side with better facing angle. Does not blend samples across photos; hard seams can remain.
HTML does perspective-correct photograph lookup without added lighting. OBJ export has standard planar UV coordinates with affine interpolation; keep MTL/PNGs alongside OBJ. This is not a full-head UV atlas.
Unobserved back/scalp regions stay gray. No automatic skin/hair segmentation, color calibration, albedo recovery, hair reconstruction, mouth correction or printability guarantee.
Optional --masks-dir masks_folder requires front.png,left.png,right.png of matching photo resolution: white retains projected color, black excludes. Without masks, hair/accessories/background can be projected.
Verified locally: synthetic projection/visibility and mask checks, source-selection checks, generated viewer JS syntax, project regression suite. CUDA/WebGL appearance must be checked on the target machine.
