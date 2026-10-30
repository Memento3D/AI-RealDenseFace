# V2.2: existing FLAME eyes as separate objects

run_v2.py now enables texture refinement and --separate-eyes at export.
run_v1.py retains its original export unless flags are explicitly supplied.

```powershell
.\.venv\Scripts\python.exe .\scripts\run_v2.py --input-dir input/03
```

To export an existing workspace without running reconstruction again:

```powershell
.\.venv\Scripts\python.exe .\scripts\export_workspace_glb.py `
  --workspace "PATH/TO/head.workspace.html" `
  --output "PATH/TO/head.v2_2.glb" `
  --refine-texture --separate-eyes
```

Expected GLB objects: Head, Eye_ModelXNegative, Eye_ModelXPositive. Coordinates
are model axes; anatomical left/right are deliberately not inferred from names.
All objects have independent materials and share one embedded atlas. They are
static meshes, not a rig; their pivots are not eye rotation controls.

The exporter requires exactly 3931/546/546 connected vertices after exact welding,
and checks approximate eyeball proportions and separation. Unexpected topology
fails instead of guessing. Existing positions/normals are preserved. The change
does not create eyeballs, replace them with new spheres or estimate gaze.

Where the original workspace already marked a whole eye triangle as valid from
the front photograph, that photograph is preferred. Other eye triangles retain
their existing source weights. No previously excluded photograph projection is
enabled. Skin-boundary fading is disabled on the eyeballs to avoid blending skin
onto eye details. Eye color remains photographic and can retain blur/shadows.

This is an incremental separation/material experiment. It does not add a new
iris/cornea material, gaze fitting, eyelid fitting, animation or printing checks.
Confirm appearance in Blender at matching angles before adopting it as default.
The original workspace HTML remains the previous viewer, not the V2.2 GLB view.

Existing V2_1_TEXTURE.md describes the preceding stage; its run_v2 command now
runs this V2.2 version. To isolate V2.1, export with only --refine-texture.
