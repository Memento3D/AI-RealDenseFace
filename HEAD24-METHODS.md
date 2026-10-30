# Head24 implementation notes

Evidence: code supplied from RealDenseFace, upstream commit recorded in run05:
`ebf0063835f520f07fdf5a813b1af9cdc0ae46b5`.
No new upstream implementation is silently patched. `head24-rdf-source.json`
pins 48 supplied Python/config files. This does not cover CUDA binary provenance.

`RealDenseFaceInferencer._init_face_bbox` falls back to a large central default
box when detection returns None. The previous run has no bbox log, so that
branch cannot be proven to have executed in run05. Explicit landmark ROIs avoid
that entire branch in this experimental runner.

`GNFlameOptimizer._register` determines C from camera_view_mat.shape[-3].
Using [3,4,4] would imply three simultaneous views of one pose for observations
[3,V,2]. The worker passes [T=3,C=1,4,4], so x remains [3,118] and K can differ
per frame. Solver ops explicitly accept [T,C,4,4]. This interface has been
inspected, not exercised on CUDA by this release author.

Per frame, decode X_i with root rotation/translation removed and Y_i fully
posed, then estimate a rigid T_i by SVD. Reject max native-coordinate residual
above 2e-5. Export P_i = K_i E_i T_i. q_i comes from Y_i in that view's camera.
Check projected X_i vs q_i. The canonical output remains X_front, preserving
its expression; no forced mouth closure or eye socket carving is performed.

Do not confuse the camera-contract check with reconstruction accuracy. The
front-eye gate uses a different model (MediaPipe) but is only a coarse alignment
test and shares image/occlusion failure modes. Native face resolution, glasses,
lighting, topology accuracy, expression plausibility and physical printability
are not certified by it.

Shared identity in joint fitting is still an estimate. No independent score of
identity improvement or anatomy is provided. The pipeline stops on gate failure;
it does not select a worse fallback and label it successful.

Baseline measured on uploaded run05 workspace, at native 1920x2560:
legacy recovered camera P95: front .000075px, left 11.034354px, right 10.707661px.
These measurements characterize the old data contract, not new inference results.


## Detection patch 24.1

Original-image pass plus square overlapping windows at .65 and .40 of the short
image side, 50% stride. A detection-only power LUT (x**.6) is also scanned. Maps
return to EXIF-normalized native pixels. Duplicates group at IoU>=.35 and center
distance<=.3 of the smaller diagonal; choose the best face/window ratio as the
representative, never average identities or lower size thresholds. This grouping
is not face recognition. Original crop and appearance pixels remain untouched.
No actual FaceLandmarker inference was run by the release author for this patch;
unit tests verify mappings, grouping, thresholds and preservation of source pixels.
