# Face surface comparison — 0.6.2

The old compare-manual-contours command measures arcs of a whole-head outer
silhouette. Its neck-crossing arcs are unsuitable for jaw fitting. Preserve
its reports for provenance, but do not interpret those distances as jaw error.

Use the existing editor workspace and corrected validated annotations:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head compare-jaw-contours `
    --workspace runs/webcam_manual_contours_001 `
    --validation-run runs/webcam_manual_contours_check_002 `
    --registration-run runs/webcam_registration_001 `
    --head-run runs/head_init_001 `
    --shape-run runs/webcam_shape_fit_001 `
    --mica-config configs/mica.local.json `
    --output runs/webcam_jaw_comparison_001
```

No new images, annotations, calibration or assets are required. Uses CPU geometry
and the existing isolated worker Python to read the pinned semantic mask;
no model inference or GPU dependency is added. Existing outputs are not overwritten.

Method: compute facing transitions between adjacent triangles on the original
full mesh under the frozen camera. Keep only edges whose endpoints belong to
FLAME face, excluding neck, ears, scalp and eyeballs. Check three positions per
edge against the full mesh for self-occlusion (0.25 mm tolerance). Trace continuous
edge chains, splitting at graph branches. Do not create edges along the semantic
mask cut or connect disconnected chains. No fallback to whole-head silhouettes.

An annotation is compared to a continuous open chain trimmed at its nearest
endpoint samples. A chain is sampled at 64 equal normalized arclength locations.
This remains a proposed correspondence, not verified anatomy. A short chain
may poorly cover an annotation; inspect endpoint distances and overlays. If no
chain yields distinct endpoints, export NO_SUPPORTED_ARC without a distance or
fabricated connection. Green manual traces still appear for unsupported curves.

Artifacts: front/left/right.manual_comparison.png, report.json, face_regions.npz,
face_mask_worker.log. The NPZ contains only semantic indices, not model weights.
Left panel baseline cyan; right panel candidate purple; manual green.

Mesh and camera stay unchanged. Likeness and automatic fitting readiness remain
false. The semantic mask, mesh discretization and ray tolerance can omit genuine
jaw segments; they require inspection on the user's real data. Do not optimize
from these diagnostic distances until curve selection/coverage is reviewed.

Verification: synthetic closed-mesh silhouette/exclusion tests, no artificial
crop-boundary test, full-mesh occlusion test, reversed open-curve comparison,
unsupported-curve rendering, CLI routing, existing source checksum and no-overwrite
integration checks. No Windows/GPU runtime or user FLAME geometry was available
for local end-to-end validation.

## 0.6.3 correction: coverage-gated proposals

Use a new output directory, e.g. runs/webcam_jaw_comparison_002. Existing photo,
annotation workspace, validated snapshot, camera, mesh and asset pins are retained.

The semantic search region now additionally excludes eye_region, forehead, nose
and lips. This produces a lower-face candidate region; FLAME does not provide a
verified jaw-only mask here, so the result is not labeled an anatomical jaw mask.
No synthetic mask cut edges are introduced.

Before selecting a chain, apply explicit experimental engineering guards:
- Endpoint tolerance = max(6 pixels, 0.08 * reference polyline length).
- At least 85% of 64 reference samples within that tolerance of chain samples.
- Trimmed arc/reference length ratio between 0.65 and 1.6.
- Maximum paired arclength distance <= 2 * endpoint tolerance.

These constants are versioned in ARC_POLICY and are NOT calibrated quality
thresholds. Rejected chains retain support diagnostics but no mesh sample pairs,
mean fitting distance or candidate-minus-baseline delta. Disconnected chains are
not joined. NO_SUPPORTED_ARC is a valid diagnostic outcome, not a successful jaw
reconstruction. Report all_curves_supported and supported_curve_counts describe
geometric guard outcomes only; automatic fitting readiness remains false.

surface_chains.json exports ALL extracted projected chains, with source binding,
for follow-up debugging without requiring another blind rerun. Treat this output
as user data; it is never included in source archives. Its coordinate space is
original normalized photo pixels.

Additional verification: all eight previously selected arcs in the user's
0.6.2 report were rejected by the new support guard. This establishes regression
rejection only; full re-extraction on that user's mesh was not available locally.
Synthetic tests cover short chin/mouth stubs, disconnected fragments, detours,
correct full-span curves, reversed order, and no mutation. An integration test
checks the missing-support report/image/artifact path using a mocked extractor.

## 0.6.4: distinguish anatomical jaw proxy from occluding silhouette

Direct inspection of the user's 5023-vertex meshes confirmed that the traced
jaw in oblique photos lies inside the projected head. A connected global
occluding contour therefore cannot stand in for the entire anatomical jaw.
The old face-facing-transition method is retained for research tests, but is
no longer used by compare-jaw-contours.

The new diagnostic uses the documented, checksum-pinned FLAME face vertex mask.
Find closed boundary loops of triangles wholly in the face patch; select the
loop with greatest canonical vertical extent (exclude eye/nostril/mouth holes).
Keep its continuous lower half in neutral FLAME coordinates, using the midpoint
of the face's Y extent. The boundary indices are frozen from the original mesh
and reused on the candidate, so selection does not change with candidate shape.
The pinned mask's face region does not overlap its neck or ear masks. Only the
specific neutral pinned FLAME topology is supported.

This is explicitly a SEMANTIC PATCH BOUNDARY PROXY. Patch-cut edges are used as
an anatomical reference here, not claimed to be a true occluding contour or a
validated anatomical jaw. It is a change of diagnostic definition, not an
improvement in the old silhouette metric. Do not compare scores across these
versions. Far-side parts can project inside the face; visibility is not verified.

Keep the existing experimental endpoint/coverage guards unchanged. For supported
proxy spans report ANATOMICAL_PROXY_SUPPORTED (still unverified correspondence).
For unsupported spans report ANATOMICAL_PROXY_INSUFFICIENT_SUPPORT, omit fitting
metrics, and draw the diagnostic proxy in gray WITHOUT paired connection lines.
All automatic fitting and likeness flags remain false. The output filename is
unchanged for compatibility. surface_chains.json now contains projected anatomical
proxy paths; boundary_source and pipeline_version=4 identify the new semantics.

Local reproduction used the user's exact OBJ checksums and registration report,
and reference samples from their prior contour report. Both frontal spans passed
the existing engineering guards. Both oblique spans failed endpoint guards;
one endpoint differed by about 34 px (right) and 46 px (left). The implementation
has NOT made those viewpoints fit, verified identity likeness, or changed geometry.
The purpose is to expose a continuous anatomical reference for inspection while
preserving the evidence that endpoints remain inconsistent.

Use the same command with --output runs/webcam_jaw_comparison_003.
99 tests pass, including mask-loop extraction, freezing correspondence indices,
unsupported-proxy behavior, reverse drawing order, and source-bound CLI checks.
The isolated Windows worker/config path still requires local user validation.
