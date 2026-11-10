# V2.3 engineering changes and evidence

## Continuous eye projection

For each source eyeball corner q in the front camera, compute source pixel `(q_x/q_z * W, (1-q_y/q_z)*H)`. Normalize these coordinates into one per-eye crop. Every duplicated triangle corner at the same source vertex must obtain the same UV. Each eye primitive receives its own TEXCOORD_0 accessor and crop texture; positions/normals/index buffers remain source data. This avoids per-face visibility switching and cross-view color discontinuities. Front projection does not reconstruct hidden iris appearance or gaze and is validated here on one frontal capture.

The continuity audit samples 0.25/0.50/0.75 along matching shared edges of visible source-eye triangles, comparing bilinear sRGB texture colors across adjacent faces (sampling in linear light). It reports channel-max RGB8 differences and is not a global perceptual score. Geometry, camera and actual eye aperture are not altered to make this metric better.

## Bounded eyelid detail

Within semantic skin, at distance less than 8 pixels from front eye labels, obtain Lab lightness residual:

`D = L - (0.65*Gaussian(L,2.5) + 0.35*Gaussian(L,8))`.

Apply a smooth distance falloff and strength 0.85, clip D to [-9,+2] Lab L, then add it to the canonical base L while keeping base a/b. Preserve feature cores separately. Outside this eye band and feature mask, reset to the exact base RGB. Sharp shadows/noise can still contribute to residual; this method does not identify intrinsic albedo. The bounds and ROI make that risk explicit, not nonexistent.

## Protected exterior

Reuse the validated source radial correspondence and V2.2 contact shell. For source radius r, shell proposal t, and maximum retraction b=1.0 mm, protect exterior proxy vertices by `p=max(t,r-b)`. Blend to the contact shell over a source wall-offset transition. The exact budget is checked for wall-offset >=0.43 mm before local sideburn editing. It is not an exact semantic outer-surface partition, global volume preservation or ARAP. The cavity/contact, source-ear coverage, central-face and final STL gates are still required.

## Sources and what is actually implemented

- Laplacian Surface Editing, Sorkine et al., SGP 2004: https://igl.ethz.ch/projects/Laplacian-mesh-processing/Laplacian-mesh-editing/ . Local sideburn editing remains the V2.2 normalized-Laplacian displacement variant, not the full rotation-invariant paper implementation.
- ARAP reference considered: https://igl.ethz.ch/projects/ARAP/ . ARAP is not implemented in V2.3.
- Intrinsic Images in the Wild, Bell et al., SIGGRAPH 2014: https://opensurfaces.cs.cornell.edu/publications/intrinsic/ . Reflectance/shading decomposition is ambiguous; frequency filtering is not a complete decomposition.

No new research novelty, optimality, printing success guarantee or benchmark on the user's RTX 4050 follows from this engineering combination.
