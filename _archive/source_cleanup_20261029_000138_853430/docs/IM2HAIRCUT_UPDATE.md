# Memento3D — Im2Haircut prepared-sample runner 0.1

Additive update: does not replace run_v1.py/run_v2.py or the Windows environments.
Uses the WSL environment already installed during this session. No weights,
images, upstream repository or environments are included.

## What is implemented

- One prepared sample per invocation, pinned Im2Haircut commit.
- Idempotent, guarded application of the two trainer fixes already tested on
  the user's laptop: appearance allocation and SDF indexing/coordinate clone.
- Epoch limit, own config per attempt, no overwriting old runs.
- Process-tree RSS and whole-device NVIDIA telemetry CSV; status and hashes.
- Requires the requested last PLY; exports ordered NPZ and OBJ lines.
- Blender import of NPZ as curves, extrusion zero, bevel relative to hair height.
  Material is a default preview color, not reconstructed hair color.

## Current evidence and limits

User ran the official prepared example without upsampling, exported steps
0/100/200/300 and inspected strands in Blender. Approximate elapsed time from
folder/log timestamps was 11 minutes; an observed whole-device sample was
5913 MiB / 6141 MiB. This is not a measured peak or a general 6GB guarantee.
EGL reported AMD Radeon 780M, while CUDA used RTX 4050.

New runner is CPU/unit-tested in the delivered environment; its full GPU run
and the new Blender importer are not verified here. Existing upstream pipeline
and two fixes were run by the user, not on this host.

Raw photographs are NOT accepted by this runner. A prepared dataset also needs
orientation, segmentation, depth, alignment/cameras and auxiliary inputs.
MonoHair, automatic RDF head alignment, recovered hair color, GLB hair mesh,
watertight print output, final checkpoint/resume are not implemented.
Do not simply copy front.jpg into the prepared examples directory.

## Install the update

Extract and merge `scripts/`, `tests/` and this document into your project root.
Run all following commands in Ubuntu, not PowerShell:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate im2haircut
cd /mnt/d/FPTU/07/EXE201/ai-3d-generation/Memento3D-Head-Reconstruction
python -m pip install psutil==7.1.0
```

## Export the existing result without another GPU run

```bash
python scripts/im2_support.py \
  --ply ~/memento-hair/Im2Haircut/experiments/local_smoke_20261027_091808/pointclouds_train/pred_000300.ply \
  --config ~/memento-hair/Im2Haircut/experiments/local_smoke_20261027_091808/recording/config.conf \
  --output runs/im2_existing_300/export
```

Output directory must be new. No GPU inference, and no source PLY modification.
Copy this export directory to Windows if needed for Blender.

In Blender use Scripting > Open `scripts/blender_im2_preview.py` > Run Script.
Choose `hair.strands.npz` in the dialog. Sidebar offers radius ratio and a strand
limit. Default ratio 0.0004 is a preview parameter, not a measured human hair
radius. The script adds an object, preserves the existing scene and coordinates,
and does not auto-save or overwrite a blend file. Save your scene separately.
Full curves may be heavy; use 1500 maximum preview strands when needed.

## A short measured run

Omit --scene to reuse the first sorted prepared image, matching the earlier
manual test. The exact selected filename is printed and recorded:

```bash
python scripts/run_im2_sample.py \
  --epochs 50 --apply-fixes
```

Output: `~/memento-hair/Im2Haircut/experiments/memento_im2_<UTC>_<id>/`.
Use `--scene EXACT_FILENAME.png` to choose another prepared sample explicitly.
Contains `pipeline.report.json`, `metrics.csv`, `inference.log`, `run.conf`,
upstream artifacts and `export/hair.strands.npz`, `hair.strands.obj`.
Follow progress from a second Ubuntu terminal using the printed log path.
Ctrl+C stops the spawned process group and records INTERRUPTED.

The upstream loop stops at epoch > num_epochs. A single prepared example at
`--epochs 50` means 51 updates, with geometry exported at step 50 BEFORE that
step's optimizer update. `--epochs 0` still runs a backward/update step; it is
not feed-forward inference. `--coarse-steps 20` keeps the upstream <= threshold.
Keep it fixed when comparing epoch budgets. No upsampling is enabled.
Checkpoint at step 0 still consumes about 0.85 GB per run; this update does not
change checkpoint logic or claim the checkpoint is the final fitted state.

## Compare settings fairly

Run 50, 100 and 300 SEQUENTIALLY, with the same --scene and --coarse-steps,
each using a new attempt. Compare NPZ geometry and report metrics, then view
with the same material, bevel ratio and camera. Do not attribute changes from
thicker bevel to a better reconstruction. No automatic quality winner is picked.
300 is the reference tested earlier, not an assurance of better quality.

Check physical space with `df -h /mnt/d`; the virtual WSL filesystem's free
space can exceed actual space available on D. Do not run RDF and hair fitting
at the same time on this 6GB GPU.

If failed, open that attempt's inference.log; do not rebuild CUDA by default.
If source guard rejects a modified upstream file, inspect changes rather than
force-resetting the repository or replacing your trained data.

Upstream: https://github.com/Vanessik/Im2Haircut . Preserve its license and
dependency/weight licenses. This update does not grant deployment rights.
