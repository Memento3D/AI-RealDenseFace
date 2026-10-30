# Hair library contract v1

The stable interface is metadata and bounded fitting behavior, not a shared
vertex count or a fixed hairstyle. Each asset preserves its own exterior mesh.

## Required source preparation

1. Author verifies source units. STL itself does not prove millimeters.
2. Mesh must be edge-closed, consistently oriented and one intended material
   component. Inspect vertex links too: an edge-watertight mesh can be pinched.
3. Canonical orientation: +Y up, +Z front, approximate cavity axis along Y.
   The importer offers four proper rotations, not arbitrary pose inference.
4. Prepare cavity radius, cavity roof height and lower source anchor. These
   are authored interface descriptors, not automatically recovered true roots.
5. Fit to a reference head and diverse heads before validating an asset's
   declared scale/cage limits. Passing one demo is not a population guarantee.

## Registry structure

`assets/hair/library.json`: schema_version=1, assets=[...]. Per asset:

| Field | Meaning |
|---|---|
| id | Unique [A-Za-z0-9_-], maximum 48 characters; version new revisions |
| name | Human-readable style label |
| path | Project-relative STL path; no parent traversal |
| sha256 | Exact source bytes; verify before fitting |
| source / license_status | Provenance, retained even for supplied demo assets |
| input_units | Author's unit assumption/verification status |
| source_to_y_up | Orthonormal 3×3 rotation, determinant +1 |
| inner_roof_y | Cavity roof in canonical source units |
| inner_radius | Approximate canonical cavity radius |
| source_base_y | Lower canonical anchor |
| limits | scale_min, scale_max, cage_displacement_mm |
| custom | volume, fringe_lift_mm, nape_lift_mm, yaw_deg |

Source scale is currently assumed mm; non-mm assets must be normalized before
registration. Cavity axis should be at X≈0,Z≈0 after rotation. X numerical
centering is automatic; general off-axis cavities need author normalization.
The supplied templates use one declared orientation and per-template roof data.

## Example: inspect a new normalized asset

```powershell
.\.venv\Scripts\python.exe .\scripts\import_pipeline_v2_hair.py `
  --source "D:\your-models\hair.stl" --id "my_style_01" --name "My style" `
  --orientation zup_front_minus_y --inner-roof-y 9.3 --inner-radius 4.7
```

These numerical anchors are examples, not defaults suitable for all models.
The command is dry-run unless `--register` is added. Registration copies the
source to assets/hair/<id>/source.stl and appends a new registry record.
Then run V2 with `-HairIds "my_style_01"`. It must still pass runtime checks.

## Validation states

Imported → author-normalized → fitted to reference → tested across heads →
geometry export validated → print-profile tested. Store evidence separately.
Do not overwrite a source ID to silently change its geometry or validation.

Profile/registry data is excluded from the static code checksum so a library
addition does not break the launcher. Each model's own source checksum remains
mandatory. User custom parameters are run-time overrides with bounded ranges.

## Runtime safeguards

- Scale outside asset limits: reject and allow another registered style.
- Excessive protected facial coverage: reject; do not deform nose/lips to fit hair.
- Vertex pinches or STL precision trouble: at most one bounded hair-only repair.
- Large repair deviation or invalid final surface: reject and retain diagnostics.
- Final printer readiness remains false without the downstream evidence.

Mechanical detachable interfaces and multi-color manufacture are outside this
contract version. Software style replacement does not imply physical snap-fit.
