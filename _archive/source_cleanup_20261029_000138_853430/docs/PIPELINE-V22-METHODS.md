# Technical notes and sources — V2.2

## Canonical skin with source feature masks

Let C be the robust bright-patch RGB design reference, F(p) the original photo color and alpha(p) an eroded/feathered semantic feature mask. The prepared photo is `(1-alpha)*C + alpha*F(p)`. Alpha is exactly zero on semantic skin, ears, nose and neck. Barycentric projection and stored visibility weights transfer source feature contours into the original UV atlas; pure-base atlas samples are explicitly reset to C. Original eye-component atlas tiles are preserved. No original head feature *tile* is restored wholesale. Positions/UVs remain source geometry.

The principle of separating albedo from illumination is grounded in Smith et al., **A Morphable Face Albedo Model**, CVPR 2020:
https://openaccess.thecvf.com/content_CVPR_2020/html/Smith_A_Morphable_Face_Albedo_Model_CVPR_2020_paper.html

This implementation does **not** run that statistical albedo model, Light Stage acquisition or inverse rendering. It uses a uniform design base for a stylized printable figurine. The report must not call this calibrated intrinsic albedo recovery.

## Scalp shell and independently bounded detail

For each source hair vertex, obtain source radius r, scalp radius s and source cavity entry radius i from a common internal origin. Smooth r on the hair mesh graph to get r_low; detail d is a clipped/scaled `(r-r_low)`. The new radius is:

`r_new = s - inset + (1-exp(-max(r-i,0)/edge_taper)) * (body + inset + d)`.

The inner wall remains inset; the external mean thickness is controlled independently of source high-frequency radial detail. This is a proposed engineering combination, **not a claim of a newly published or proven optimal algorithm**. Only short cavity templates have been exercised. Non-star-shaped styles, hair ties, braids or ponytails need a different representation.

The optional V2.1 cache path inverts the exact recorded monotone radius mapping and audits the inferred source entry radii against fresh source-template ray intersections. It validates hashes, scale, asset, direction/index consistency and a deterministic 128-ray subset. It does not reuse old final STL or old supports as if they belonged to the new edit.

## Local sideburn deformation

Visible semantic-ear corners locate an ear-relative region and handle band. Outside-region vertex displacements are exactly fixed at zero. With normalized graph Laplacian L and handle selection H, solve:

`min_D ||L*D||^2 + w^2 ||H*D - T||^2 + epsilon ||D||^2`.

Targets T extend the sideburn down, reduce its tangential width and round its tip. A subsequent radial reattachment retains the pre-edit scalp clearance of moved vertices. The face mesh never participates in this solve.

This is inspired by **Laplacian Surface Editing**, Sorkine et al., SGP 2004:
https://igl.ethz.ch/projects/Laplacian-mesh-processing/Laplacian-mesh-editing/

It is a small-motion, linear displacement-field variant using a normalized graph operator. It does not implement the paper's full rotation-invariant editing framework. **ARAP is not implemented in this release**; the related reference considered during design is:
https://igl.ethz.ch/projects/ARAP/

## Gates and limits

- Same source face/eye position buffer after GLB export, allowing float32 numerical error only.
- Sampled central face distance after union; eye sockets/lateral temples excluded from that particular test and explicitly reported.
- Visible source-ear corner coverage before/after editing; reject more than 3% newly covered samples per side. This is a sampled source-mesh proxy, not independent anatomy or full collision proof.
- Radial contact screening after repair, not merely before it.
- Full exported STL edge/vertex-link manifold, orientation, positive volume and connected-material check.
- Original face remains a surface boolean input. Hair-only repair may erase fine grooves and is reported when used.

No guarantee of global self-intersection freedom, exact minimum wall thickness, physical print success, source-head accuracy, likeness or aesthetics follows from these gates. Print readiness remains false pending actual machine/resin/slicer validation.
