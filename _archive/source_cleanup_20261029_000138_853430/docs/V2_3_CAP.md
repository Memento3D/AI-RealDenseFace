# Optional original mortarboard prototype

This is an original procedural asset, not a downloaded third-party model. It
contains separate cap body, square board, button, cord and tassel meshes with
independent materials. No AI reconstruction or external model download is used.

Fit an existing canonical Memento3D head GLB (CPU, NumPy only):

```powershell
.\.venv\Scripts\python.exe .\scripts\add_graduation_cap.py `
  --head "PATH/TO/head.optimized.glb" `
  --output-dir "runs/cap_test_001"
```

Choose a new output directory. It creates graduation_cap_01.glb (cap only),
head.with_cap.glb (head and cap), and cap.report.json. All head texture/materials
remain embedded in the combined GLB. It preserves existing head geometry.

The standalone cap is positioned in the fitted head's native coordinate frame;
it is not a standardized centered asset yet. The fit assumes canonical FLAME
axes and uses a heuristic scalp envelope. It does not establish physical units,
collision-free fit on other heads, printability, rigging or textile realism.
This simple first prototype is for inspecting fit and customization workflow.
The cap and head remain overlapping separate surfaces, not a print-ready union.
There is no hair asset. The existing head's own FLAME licensing is unchanged.

Run from photographs with this optional assembly:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_v2.py --input-dir input/03 --graduation-cap
```

The final pipeline report includes head_with_cap_glb. The original uncapped
head.optimized.glb is retained separately. Unflagged V1/V2 runs are unchanged.
