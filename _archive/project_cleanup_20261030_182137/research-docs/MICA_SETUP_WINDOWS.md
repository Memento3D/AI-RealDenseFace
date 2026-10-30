# MICA head initialization — research integration 0.2.0

## Read this first

This release implements an external adapter and real inference code path, but
**end-to-end MICA inference was not run here**: licensed FLAME assets and MICA
checkpoint are not present, and this environment has no Torch/CUDA. Never treat
CPU contract tests as proof of face fidelity, Windows compatibility or 6 GB fit.

The current command uses the validated FRONT image only. The two side images
remain reference evidence. No multi-view fitting, texture, hair, custom expressions,
MCP, GLB or print-ready export is implemented. MICA predicts canonical head shape,
not the exact input expression. The OBJ/PLY is in mm; identity.npy is retained.

No MICA/FLAME code or weights are redistributed here. Official MICA license is
non-commercial research/education/artistic; it prohibits commercial artifacts and
redistribution without permission. Obtain resources yourself through the authors'
channels. Do not send account passwords or license credentials to chat.

## A. Update the project without deleting runs or input

Unzip the release to a temporary folder and copy its src/, tests/, scripts/, configs/
(existing photo config is unchanged), requirements/, docs/ and pyproject.toml into
this project. Preserve .venv/, input/ and runs/. Reinstall editable package:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
.\.venv\Scripts\python.exe -m memento3d_head --help
```

## B. Clone pinned upstream source

From this project root:

```powershell
New-Item -ItemType Directory -Path vendor -Force | Out-Null
git clone https://github.com/Zielon/MICA.git vendor/MICA
git -C vendor/MICA checkout af22e7a5810d474bc28a1433db533723d6bd2b07
.\.venv\Scripts\python.exe -m memento3d_head mica-status --repo vendor/MICA
```

Do not run the upstream install.sh blindly on Windows: it is a Linux/Conda script,
contains account prompts and installs an old Linux-specific environment. The status
command lists missing assets before any large environment setup.

## C. Obtain model assets

Follow official https://github.com/Zielon/MICA and https://flame.is.tue.mpg.de/ .
Place these files under vendor/MICA/:

- data/pretrained/mica.tar (author's MICA checkpoint linked from README/install.sh)
- data/FLAME2020/generic_model.pkl (FLAME2020, not an unverified 2023 substitution)
- data/FLAME2020/landmark_embedding.npy (already present in pinned source)
- data/FLAME2020/FLAME_masks/FLAME_masks.pkl (FLAME masks from official resources)

For detection, obtain the official antelopev2 model pack via InsightFace resources
referenced by MICA's install.sh. Extract its SCRFD detection ONNX, commonly named
scrfd_10g_bnkps.onnx, to D:\AI-cache\mica\scrfd_10g_bnkps.onnx. Do not rename an
ArcFace recognition ONNX into the detector name; the worker checks task type.
Only this detection model is loaded from the pack, on CPU. No FaceAnalysis auto-download.

This package never downloads these assets or accepts terms on your behalf. Local
pins record the files you obtained; they are not independent authenticity proofs.

## D. Separate model environment — experimental native Windows recipe

Do NOT install these packages into the validation .venv or old Hunyuan environment.
Python 3.10 is deliberate for legacy FLAME/chumpy compatibility. First check:

```powershell
py -0p
```

If Python 3.10 is installed:

```powershell
py -3.10 -m venv .venv-mica
.\.venv-mica\Scripts\python.exe -m pip install --upgrade pip "setuptools<81" wheel
.\.venv-mica\Scripts\python.exe -m pip install "numpy==1.23.5" "scipy==1.10.1"
.\.venv-mica\Scripts\python.exe -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
.\.venv-mica\Scripts\python.exe -m pip install -r requirements/mica-worker.research.txt
.\.venv-mica\Scripts\python.exe -m pip check
```

InsightFace/chumpy may require source builds; InsightFace on Windows may require
Microsoft C++ Build Tools. If installation fails, keep the full error and stop
there instead of changing pins or using random wheels. This recipe is NOT a fully
resolved lock. Torch's CUDA runtime is separate from the validation CPU task.

Check imports/asset presence:

```powershell
.\.venv\Scripts\python.exe -m memento3d_head mica-status `
  --repo vendor/MICA --worker-python .venv-mica/Scripts/python.exe
```

## E. Pin local assets once

Only after the listed assets and environment exist:

```powershell
.\.venv\Scripts\python.exe -m memento3d_head mica-pin `
  --repo vendor/MICA `
  --worker-python .venv-mica/Scripts/python.exe `
  --detector D:/AI-cache/mica/scrfd_10g_bnkps.onnx `
  --output-config configs/mica.local.json
```

The config stores absolute paths and SHA256. It refuses overwrite and rechecks
pins at execution. Do not regenerate config simply to accept unexpected changes.

## F. Actual model-load doctor

```powershell
.\.venv\Scripts\python.exe -m memento3d_head mica-doctor `
  --mica-config configs/mica.local.json `
  --output runs/mica_doctor_001
```

Checks actual CUDA matmul, CPU ONNX detector initialization and MICA model loading.
MODEL_LOADED means model load only; inference_tested remains false. Output/log is
kept on failure. No claim that 6 GB can run full inference follows from doctor.

## G. Run one front-image geometry attempt

After manually checking the validation landmark images, identify the reviewer:

```powershell
.\.venv\Scripts\python.exe -m memento3d_head reconstruct-head `
  --validation-run runs/real_photo_check_001 `
  --front input/WIN_20261021_15_50_54_Pro.jpg `
  --reviewed-by "Don" `
  --mica-config configs/mica.local.json `
  --output runs/head_init_001
```

Front source hash must match the prior report. It rejects NEEDS_RETAKE, synthetic/
unknown declarations unless --allow-synthetic is explicitly given for research,
and a reused output path. Human review and real_capture are declarations, not proof.

Expected only after real success: GEOMETRY_EXPORTED, head.obj, head.ply,
identity.npy, head.preview.png (three canonical orthographic views), front.aligned.jpg,
report.json, worker_result.json, mica_worker.log, validation.snapshot.json.
Report records units, artifact hashes/sizes and measured Torch peak allocated/
reserved bytes; those are not whole-device VRAM peaks or host-RAM measurements.
Mesh validation checks finite coordinates, indices, bounds and degenerate faces;
NOT watertightness or manufacturability. Open OBJ in Blender for further inspection.

Worker is one separate process, batch one, CPU detector + CUDA MICA, no simultaneous
Hunyuan. Timeout/cancel terminates its group; no automatic retries, CPU fallback or
model downgrade. Native-library subprocess failure remains visible in the log.

## Next research gate

Compare grayscale geometry to real photographs before texture/asset work. Determine
whether initialization preserves identity. Only then design three-view fitting and
calibrate data acceptance against reconstruction outcomes.
