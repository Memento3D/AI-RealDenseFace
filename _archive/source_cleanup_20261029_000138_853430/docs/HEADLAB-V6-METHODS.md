# HeadLab V6: methods, evidence, limits

## Proposed integration, not a published hair-root algorithm

The V6 integration is the project's own experimental method. It separates
visible hair/skin contacts from silhouettes and ambiguous edges, weakens an
oblique silhouette's interpretation as a scalp root, and uses deterministic
candidate evaluation and rollback. No claim of novel scientific validation.

Semantic skin labels 1/14 supply contact candidates; ear labels 7/8 are separate
negative surface evidence. A two-pixel dilation classifies neighboring contact.
Rear V5 labels were synthetic hair/background: label 1 there is never treated as skin.

For each scalp vertex, project with the recovered camera and test model depth
and surface facing. Depth tolerance is 0.12 reference mm times camera depth-row
norm. Facing must exceed 0.15. Hair signed distance d gives

    p = clip(0.5 + d/(2u), 0, 1), u = 2.5 pixels at 512-pixel image width.

Base positive weights use cosine^4 and a distance confidence margin. Positive
pixels within 2.5 pixels of the projected scalp silhouette receive weight 0.08:
the silhouette remains volume evidence but is weak root evidence. Negative
visible evidence uses cosine^2. Background is a soft silhouette contradiction
weighted 0.45, never an observed anatomical skin or root. Rear weight is 2.

No head-only depth test perfectly models hair self-occlusion: this is explicitly
an approximation, especially around ears and dense overhanging hair.

## Field optimization

Let f0 be the V5 root probability, W the per-view evidence weights, P the hair
probabilities, N the non-hair evidence weights, and L the symmetric graph
Laplacian using inverse edge lengths, normalized by median weighted degree.
For each regularization lambda and veto alpha solve

    [diag(sum(W + alpha*N) + 0.015) + lambda*L] f
        = sum(W*P) + 0.015*f0

with synthetic mouth/neck closure vertices held at zero. Clamp f into [0,1].
Changes outside a 1.5 reference-mm geodesic band around the initial boundary
are prevented. This is a local correction budget, not a person's hair length.
Remove components having no positive evidence, then extract iso-contour 0.5.

Trials: lambda in [0.15,0.5,1.2], alpha in [0,0.5,2]; nine bounded solves.
No left/right symmetry, fixed nape height, fixed sideburn length or hairstyle ID.

## Measurement and acceptance

Render the outer hair footprint and depth-test it against the head in four views.
Measure IoU, false-positive/negative area divided by detected hair area, and
excess in lower/lateral photo-relative monitoring zones. Zones use each mask's
own bounds; they are not anatomical landmarks. Contact-to-predicted-boundary
p95 distance is divided by detected hair width. This is a one-sided contact
consistency proxy, not anatomical 3D hairline accuracy.

The scalar trial objective is mean(excess+missing), plus 0.6 times mean lower
and lateral excess, plus 0.5 mean contact error, plus 0.25 field-edge variation.
Require objective improvement, no per-view IoU loss beyond 0.025 and no lower
excess increase beyond 0.015. These are explicit provisional engineering
thresholds, not validated perceptual or production standards.

After volume optimization, evaluate each repaired candidate against the same
height/algorithm's V5 baseline again. Reject nonconvergence, failed topology or
proxy regression. Export automatic baseline fallback when necessary. Among
the accepted/fallback pool choose the lowest regional score per height.
Generate experimental supports for that selected output, not another candidate.

The same four masks drive fitting and evaluation. Gains are training
consistency; do not treat deterministic repeats as independent people/samples.
The face source vertices are unchanged and export error is recorded separately.

## Upstream sources and implementation distinction

- Laurentini, visual hull / volume intersection: silhouettes constrain outer
  volume; they do not reveal all concavities or hidden roots.
  https://cgvg.polito.it/Articles/cvpr93.pdf
- Laplacian Surface Editing, Sorkine et al. 2004, context for graph/mesh editing.
  https://igl.ethz.ch/projects/Laplacian-mesh-processing/Laplacian-mesh-editing/
- As-Rigid-As-Possible Surface Modeling, Sorkine and Alexa 2007, context for
  the existing uniform-spokes ARAP volume candidate (not a full library port).
  https://igl.ethz.ch/projects/ARAP/
- SciPy bounded least_squares TRF, linear and soft_l1 loss: existing offset fit.
  https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html
- Wendland 1995 compact positive-definite radial functions, context for the
  existing normalized compact kernel basis, not a published hair inference method.
  https://doi.org/10.1007/BF02123482

See V4/V5 methods for the inherited algorithms, exact simplifications and
support-screening limits. No general long/tied-hair inference, FEA, exact
minimum-wall test, calibrated printer slicing or physical print is implemented.
