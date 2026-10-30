# V2.1 texture experiment

This release preserves V1 by default. It does not add DECA, MICA, hair, eye
assets or new geometry. It blends existing photograph coverage toward a smooth
approximate skin material over two mesh-edge rings. The unobserved material
extends plausible observed skin colors with a weak base-color prior. No new
source-photo projections are enabled; masked hair and clothes are not restored.
Observed shadows remain, and boundary detail can soften. This is not recovered
hidden texture, albedo estimation or an independently verified likeness result.

## Reuse an existing workspace (CPU, no reconstruction)

```powershell
.\.venv\Scripts\python.exe .\scripts\export_workspace_glb.py `
  --workspace "PATH/TO/head.workspace.html" `
  --output "PATH/TO/head.v2_1.glb" `
  --refine-texture
```

Use a new output filename. Existing files are never overwritten. Import GLB in
Blender with Material Preview. The original HTML remains the V1 comparison;
the refined result exists in the new GLB and its adjacent report.

## New photographs through all five stages

```powershell
.\.venv\Scripts\python.exe .\scripts\run_v2.py --input-dir input/03
```

The run uses the same reconstruction/telemetry machinery as V1, with
texture_refinement_requested=true recorded in pipeline.report.json. The GLB
export report records the actual refinement. V1 remains available through
scripts/run_v1.py without --refine-texture.

## Evaluation performed

The supplied workspace was baked and compared to the supplied GLB using CPU
orthographic rasterization at matching angles. Boundary transition improved in
this inspection; eyes and mouth were not modified. Geometry arrays were checked
for exact preservation. Windows/CUDA end-to-end reconstruction was not rerun.
