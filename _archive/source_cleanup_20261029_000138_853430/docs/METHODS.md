# Method and provenance

This is an engineering combination for short source-head-owned hair, not a claim of a newly proven optimal reconstruction method or a new learned model.

## Representation

A closed radial cap is built around the existing personalized head. A periodic angular hairline controls the cap boundary. Inner radius intersects the source head; outer offsets define printable design volume. Source face geometry and photo texture buffers are not solve variables and must remain byte-identical in the assembled GLB.

The visible front/oblique masks constrain meridian hair/skin transitions using source-head z-buffer visibility and facing weights. A bounded linear least-squares system penalizes angular second differences and carries an explicit prior only where no boundary is observed. Rear photo ownership is distinct: front/oblique views are prohibited from setting central posterior roots.

`scipy.optimize.lsq_linear` solves the bounded sparse boundary system:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html

A five-field bounded nonlinear fit adjusts mean volume against source silhouette tangent proxies. Azimuth-dependent basis terms vanish at the polar vertex to avoid a cusp. If no rear photo is accepted, the rear-volume coefficient is fixed to its declared prior rather than driven by an oblique-view tail. Robust loss minimizes consistency error but is not proof of independent reconstruction accuracy.

## Relation to silhouette reconstruction

The general value and limitations of silhouette constraints are discussed by Laurentini, *The Visual Hull Concept for Silhouette-Based Image Understanding* (1994), DOI 10.1109/34.273735:
https://iris.polito.it/handle/11583/1401917

This implementation is **not** a complete visual-hull/space-carving algorithm: it uses an existing head and a restricted radial hair family, with source-internal camera fits and explicitly unknown regions. No unseen shape is labeled as an observed reconstruction.

## Native TripoSR role

The native mesh is not fitted as a complete wig. Its front depth map is aligned to the original crop silhouette, horizontally checked, band-pass filtered and bounded as relief. Large-scale depth/bulges, thin extremities and unsupported posterior shape are discarded. If native/source-mask agreement is insufficient, this relief is disabled and the head-owned cap remains valid.

Source model:
https://github.com/VAST-AI-Research/TripoSR

No GPU inference was run during this postprocessing benchmark. The supplied native has negative signed volume; its face order does not affect the uncullled front-depth-map sampling, and the cap is rebuilt with fresh verified outward topology.

## Posterior photo branch

An optional original back photo is segmented in two passes. Two semantic-ear regions are required. A constrained two-ear similarity fit with fixed posterior yaw and weak perspective provides approximate projection; roll, scale, ear residual and posterior boundary support are gated. It does not estimate full pitch or prove independent calibration. Wrong/non-head input is rejected. The supplied back-photo branch was not validated on the person's actual back photo in this run.

## Validation boundary

- Coordinate/texture preservation of the supplied face buffers.
- Sampled face-distance guard after boolean.
- Head/hair overlap and protected facial sample collision guard.
- Sparse front/side/rear cap coverage screen, not an exact minimum-thickness certificate.
- Edge and vertex-link manifold, orientation, positive volume, one material component after STL roundtrip.
- Source-mask IoU gates, with an explicit rear gate when a posterior view is accepted.
- Experimental support preparation remains uncalibrated and does not constitute machine slicing.

Physical print, exact local wall thickness, global self-intersection, mechanical fit, independent likeness and aesthetics remain unverified.
