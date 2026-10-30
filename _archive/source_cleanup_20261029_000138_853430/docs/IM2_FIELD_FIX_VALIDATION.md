# Field-transform update 0.3 validation

Build checks (CPU, Python 3.12):

- Full test suite: **162 tests passed**.
- Architecture boundaries: PASS.
- Added scripts parsed with Python 3.9 syntax grammar: PASS.
- Bash launcher syntax: PASS.
- New tests cover circular interpolation at the 180-degree wrap, vector rotation, rejected affine transforms, preservation of R/outside-mask bytes, rejection of mismatched raw sources, preservation of original datasets, CPU-only zero-worker execution, and failure persistence after one mocked worker without retry.

Synthetic fixtures exercise actual CPU field correction, raster diagnostics, root plots, dataset copies, report JSON and HTML generation. GPU worker behavior is mocked in orchestration tests. Existing diagnostic tests additionally bind NPZ geometry to the recorded PLY and test head occlusion.

The user's raw/aligned map bytes and camera matrices are not present in the build workspace. Therefore the personal-case field transformation and real WSL/GPU worker must still execute on the user's machine. No reconstruction improvement, runtime improvement, calibrated orientation convention or multi-view capability has been validated here.

The source archive contains no user images, runs, weights, vendor repositories, environments or credentials.
