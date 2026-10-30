# V3.1 validation — 2026-10-27

- Actual case04 workspace SHA256: `0822ea326a0890d895b8a6dcd3caef9ca5fd6cf067fe3531b7d5efc937e0e5f8`.
- Source-expression face coordinates preserved within 1.60480086e-06 mm after the declared scale/coordinate transform. This does not validate likeness of the source reconstruction.
- Exported STL reloaded independently: watertight, winding consistent, one component; 25878 triangles.
- Separate hair STL reloaded: closed, one component. Front/side/back PNG views inspected: the previous nape notch is absent in this case. The back shape is a design prior, not a recovered back-image shape.
- STL XYZ extents (mm): [23.899799346923828, 26.161248207092285, 35.76995040476322].
- Both eye GLB primitives have UVs and embedded PNG textures; eyes can be welded into positive volumes for boolean union.
- Full suite: 178 tests PASS, including missed-ray rejection, known-ray intersections, missing-view cap continuity, photo-eye fallback, GLB eye textures and groove-depth constraints. Architecture boundaries PASS.
- Measured geometry export in this environment: 22.722 seconds, excluding static review rendering; not a Windows/GPU benchmark.
- PowerShell launcher supplied for the existing Windows project; it was not executed in a Windows session here. Python pipeline/GLB/STL validation ran on Linux CPU with the pinned numerical dependencies.

No slicer, minimum-wall, support, resin-tolerance or physical-print verification. Review geometry and paint separately; colors do not print on the resin printer. Broad lock styling still needs aesthetic approval.
