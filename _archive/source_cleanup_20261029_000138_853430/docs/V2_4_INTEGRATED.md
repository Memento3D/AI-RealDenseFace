# V2.4 integrated review experiment

This is a combined review prototype, not a claim of completed human likeness.
It preserves V1/V2 reconstruction and adds a separate optional review bundle.

## Existing workspace: no new inference

```powershell
.\.venv\Scripts\python.exe .\scripts\complete_head.py `
  --workspace "PATH/TO/head.workspace.html" `
  --input-dir input/04 `
  --output-dir runs/integrated_04_001
```

Choose a new output directory. Front/left/right RGB pixels must exactly match
the workspace's embedded photographs; mismatched photo sets are rejected.

## All stages from photographs

```powershell
.\.venv\Scripts\python.exe .\scripts\run_v2.py --input-dir input/04 --complete-head
```

The new run's integrated/ directory contains:

- head.skin.glb: color/source-selection experiment, no assets.
- head.hair.glb: short dark hair template.
- head.hair_glasses.glb: hair and rectangular frames.
- head.hair_cap.glb: hair and revised cap.
- head.complete.glb: all assets together, separate named objects.
- head.skin.report.json: color candidate selection and feature policies.
- complete.report.json: input binding, reference usage, limitations and outputs.

Color estimation uses conservative geometric/color skin heuristics and robust
bounded source corrections. One quarter of eligible overlap points selects
between candidates and the old correction; this is not independent test data,
true skin segmentation, color calibration or recovered albedo. Original front
colors are the reference. The method does not establish true skin color.

Existing valid front triangles are preferred around eyes/nose/mouth to reduce
mixed-expression texture. No gaze, iris, eyelid, lip, teeth, jaw or shape fitting
is added. Eyes remain photographic/unlit; no new corneal optics are claimed.

Hair is an original short shell template fitted to canonical scalp geometry,
with an optional heuristic front hairline cue from dark pixels above the eyes.
Glasses are original rounded rectangular frames fitted from existing eye centers,
with no optical lenses. The cap is revised lower and tapered. These assets are
not reconstructed from the additional photos. back.jpg and references/*-glasses
are recorded for human comparison only, not automatically used to fit assets.
For this demo the short dark hairstyle and rectangular frame family were chosen
after visually inspecting the supplied photographs, not by an automated classifier.

Switch parts in Blender's Outliner. Standalone inspection of head.skin.glb is
required so assets do not hide skin or geometry issues. No physical-scale,
collision-free fitting, realism, likeness or printability guarantee is made.
Different assets/head coordinate systems need separate registration work.

Source does not contain user photos, models, weights or generated personal GLBs.
Models and reports from the supplied dataset are delivered separately.
