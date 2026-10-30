# V3 validation — 2026-10-27

- CPU demo actually run on uploaded case04 `head.workspace(1).html`, workspace SHA256 `0822ea326a0890d895b8a6dcd3caef9ca5fd6cf067fe3531b7d5efc937e0e5f8`.
- Main test suite: **172 tests passed**, including camera recovery, rejection of invalid projection, closed boundary caps, single solid voxel surface, localized smile, continuous teeth band, STL mm/Z-up transform, config constraints, filling enclosed voids and rejection of detached parts.
- Architecture boundaries: PASS.
- Source-expression face positions are compared against source vertices after the declared transform; result is in `validation.actual.json` in the demo.
- Both exported STL variants are loaded back independently to check watertightness, consistent winding, connected components and physical dimensions.
- Measured CPU geometry export time for the delivered run is **18.958 seconds** (`report.json`), excluding static review rendering. It is not a Windows/RTX4050 benchmark or a runtime guarantee. Review rendering and dependency setup add time.
- Upper-head camera reprojection p95 relative to stored projections: front approximately 0, left 2.753 px, right 1.972 px. These are consistency checks against the source workspace, not independent camera calibration.

## Visual review

The generated front/side/back views were inspected. The solid hair silhouette is usable for a design feasibility comparison, while the back still contains an irregular patch from incomplete constraints. It must not be labeled a finished hairstyle. Eyes demonstrate adjustable painted details; expression and gaze need aesthetic review. The synthetic smile demonstrates a continuous shallow colored relief for teeth, but it is not yet a natural personalized smile. The source-expression baseline remains available for comparison.

The current voxel pitch is a coarse **0.23 mm**, intentionally for mass/connection testing. Hair ridges can be lost during hair smoothing. The source face is preserved as a surface and combined with Manifold booleans, avoiding voxel terraces on the face. The source reconstructed head itself may contain expression/likeness errors; V3 does not correct them automatically.

There was no physical resin print, no slicer/support review, no measured resin tolerance, no verified skin color reproduction and no production-readiness validation. A closed mesh is only one geometric precondition.
