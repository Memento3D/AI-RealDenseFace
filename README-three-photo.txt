Three-photo shared identity experiment

Extract into Memento3D-Head-Reconstruction, keeping scripts/.
Requires the existing .venv-rdf environment, weights and Open model.
Requires pinned RDF commit and the previously applied compile_model=False detector fix.

Run in PowerShell from the project root:
.\.venv-rdf\Scripts\python.exe .\scripts\fit_rdf_three_photos.py --params runs/rdf_open_20261023_152718_f3ce74/front.flame.npz --input-dir input/01

A new three_photo_<id> folder is created beside the baseline NPZ.
Open identity.comparison.html and front/left/right.before_after.jpg.
Image panels: BEFORE on left; AFTER on right. Both neutral meshes.
The NPZ stores one shared identity and three independent expression/pose states.
This is a custom experiment using upstream private solver APIs, not an official synchronized multi-camera pipeline.
FLAME2023 Open is experimental here, not equivalent to the original trained FLAME model.
Camera FOV is fixed from the baseline and shared across all three captures.
No texture, hair, automatic acceptance or likeness verification is included.
Residuals measure agreement with neural predictions, not ground truth.
Baseline files are never overwritten.
Local checks: syntax, CLI, project regression tests. GPU execution must be verified on the target machine.
