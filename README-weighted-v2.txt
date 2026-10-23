Extract into the project root. This adds a separate script and preserves the original script/runs.
Run:
.\.venv-rdf\Scripts\python.exe .\scripts\fit_rdf_three_photos_weighted.py --params runs/rdf_open_20261023_152718_f3ce74/front.flame.npz --input-dir input/01 --front-weight 2 --identity-regularization 0.05

Experimental settings: front squared data loss x2; identity anchor regularization 0.05 (previously 0.01). No guarantee of improvement.
One shared identity; independent per-photo pose/expression; fixed shared FOV.
Output: new three_photo_weighted_<id> folder beside baseline.
identity.comparison.html: interactive smooth shading.
smooth.0.png, smooth.-45.png, smooth.45.png: neutral head shaded comparison, BEFORE left / AFTER right. No geometry smoothing.
front/left/right.before_after.jpg: neutral meshes with separately refitted diagnostic cameras, BEFORE left / AFTER right.
report.json: same unweighted diagnostic metric for both meshes (network predictions, not ground truth).
joint.flame.npz: shared identity plus three states.
The neutral diagnostic differs from the fitted per-photo expression and is not a likeness metric. Do not compare directly to earlier fixed-expression side-check reports.
CUDA execution must be verified on the target machine. Existing environment, pinned vendor, no-compile patch, weights, FLAME Open and project src required.
