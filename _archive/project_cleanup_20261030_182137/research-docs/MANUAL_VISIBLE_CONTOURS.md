# v0.6.0 — source-bound offline visible-contour annotation

prepare-contours creates an HTML editor with the three normalized photographs embedded locally, plus optional original/candidate projected outer boundaries. No server, network API, new model, MICA worker or camera tuning is used. The original successful register-views and fit-head runs are required. Meshes, photographs and linked reports are checksum-bound. Sources are never overwritten and each output folder must be new.

The reference boundary is the outer foreground coverage of projected mesh triangles. Internal and hidden triangle edges are not drawn. For an outer silhouette the union of triangle projections suffices, so this is not a depth-buffer implementation or replacement for the previous per-point ray-occlusion test. It covers the whole head, ears and neck, not an anatomical face-only mask. It does not derive photo contours or establish correspondence automatically. In particular, projected skull outline does not imply hair geometry.

## Windows

Copy src, tests, scripts, docs, README.md, pyproject.toml into the existing project, preserving configs/vendor/input/runs and environments. The HTML template is packaged as setuptools package data.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head prepare-contours `
    --registration-run runs/webcam_registration_001 `
    --head-run runs/head_init_001 `
    --shape-run runs/webcam_shape_fit_001 `
    --output runs/webcam_manual_contours_001
Start-Process .\runs\webcam_manual_contours_001\contour_editor.html
```

## Marking

Start with overlay hidden and draw from the actual photograph. Click sequentially along a visible face/background boundary, never through a hidden cheek, hair, ear or neck. Do not infer hidden geometry. Skip uncertain sections. Each segment needs at least three distinct points; multiple separate segments are supported. For the frontal view, split left-image jaw and right-image jaw into separate curves if useful. Side views use visible_profile for clearly visible facial boundary segments. Labels refer to image position, not anatomical left/right.

Buttons: new segment, undo last point, delete selected segment. Click a segment button to select it. Changing type updates selected segment. Zoom maintains original normalized-image pixel coordinates. The optional mesh outline can be toggled for comparison; it is not a tracing target. The app does not automatically snap to it.

Mark at least one segment in each view. Review all three, check the visible-only review declaration, then download visible_contours.json. No browser persistence: data stay in the current tab until exported. Import the JSON later to resume the same workspace. Annotations for another workspace/source are rejected. HTML contains the photographs; do not commit or redistribute generated editor workspaces. Send only the downloaded JSON when requesting annotation review.

## Structural validation after export

Place visible_contours.json in the generated workspace (or supply its actual download path):

```powershell
.\.venv\Scripts\python.exe -m memento3d_head validate-contours `
    --workspace runs/webcam_manual_contours_001 `
    --annotations runs/webcam_manual_contours_001/visible_contours.json `
    --output runs/webcam_manual_contours_check_001
```

This validates schema, source binding, dimensions, curve types, finite in-frame coordinates, distinct adjacent points, segment lengths and explicit human review declaration. ANNOTATIONS_STRUCTURALLY_VALID does NOT verify anatomical accuracy or authorize automatic fitting. Self-intersections/semantic contour correctness are not automatically assessed; visual review remains necessary. The editor is a development reference tool, not a required customer capture step.

## Outputs and verification

Editor workspace: contour_editor.html, manifest.json, report.json. Annotated curves are a separately downloaded file. validate-contours stores a snapshot and report in a fresh output directory. No mesh changes occur. Curve matching and contour fitting are still subsequent work; this release establishes user-reviewed image curves instead of using the failed 17-index pairing.

80 local tests pass, including actual prepare/validate CLI composition on synthetic geometry, original mesh preservation, refusal to overwrite, stale annotation rejection, wrong dimensions/nonfinite/incomplete curves, HTML data escaping and boundary suppression of internal projected triangles. Architecture boundary check and compilation pass; JavaScript syntax checked with Node. Wheel built and HTML package data inspected. No full interactive Windows browser session has been tested locally; first real browser use remains to be verified by the user.
