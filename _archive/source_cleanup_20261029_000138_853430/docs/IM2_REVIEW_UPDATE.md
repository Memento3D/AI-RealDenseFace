# Im2Haircut personal review update 0.2

## Install into the existing Windows project

Copy the contents of this source archive into the existing Memento3D-Head-Reconstruction project, allowing code files to be replaced. Preserve the existing input, runs, vendor, weights and environments. This archive contains none of those assets. Do not run from a newly extracted project without your existing runs.

No new model or environment is required. Use the established Ubuntu-22.04 `im2haircut` environment, Python 3.9 / Torch 2.1.1+cu118. This update does not install the main Python package into that environment and does not alter the upstream model pin.

## One command, in Ubuntu

From the Windows project mounted under `/mnt/d`:

```bash
cd /mnt/d/FPTU/07/EXE201/ai-3d-generation/Memento3D-Head-Reconstruction
bash scripts/test_im2_04.sh
```

This command reads `configs/im2.review04.json`, whose paths target the existing personal04 prepared dataset, canonical camera, baseline 50 and baseline 300 runs from this session. The original photo must still be `input/04/front.jpg` with the same SHA256. Paths under `~` are resolved in the WSL user home. If your paths differ, edit the JSON explicitly; the tool never silently selects the newest unrelated run.

The command:

1. Checks upstream commit, original-photo binding, prepared-map bytes, camera consistency and old-run input hashes. Verifies NPZ geometry against the PLY recorded in its source report.
2. Reuses the completed baseline 50 and 300 results. Does not rerun either.
3. Projects the reference head and BFM face into the aligned source image. Exports mask and saved direction-map previews.
4. Projects the saved strands into the source camera, using a CPU triangle depth buffer of the reference head to omit head-occluded samples. Shows support at 1, 2 and 3 px radii.
5. Runs two independent 50-epoch ablations sequentially from the same pinned pretrained weights: `gabor_direction` and `no_depth_loss`. Each worker has logs, config, input hashes, hardware samples and strand exports.
6. Writes one `review.html` and `suite.report.json` in a fresh `runs/im2_review_<UTC time>_<id>/` folder. The page is updated after each result. On worker failure, saves partial results and stops further GPU jobs.

The default suite costs two additional short reconstructions. The 50 epoch limit means 51 updates under the upstream loop semantics. No time guarantee: existing GPU measurements do not predict every new variant.

To inspect existing results without running any GPU ablation:

```bash
bash scripts/test_im2_04.sh --mode review
```

Open the exact `review.html` path printed at the end using Windows Explorer. All images are sibling files; keep the whole review folder when sharing. Send `suite.report.json`, `head_projection.png`, and screenshots of the HTML if further review is needed. Worker logs live under `worker_<variant>/memento_im2_*/inference.log`; inspect those if a GPU worker fails.

## What the two variants actually change

| Variant | Change | What stays the same |
|---|---|---|
| baseline_50, reused | Original configuration | Existing 50-epoch result, used as the controlled reference |
| baseline_300, reused | Longer original run | Context/reference only; not a matched-budget ablation |
| gabor_direction | `dataset_real.convert_gabor_to_strand_map=true` | Prior input still uses the saved Gabor angle; upstream converts the loss-direction features using Gabor with HairStep sign disambiguation |
| no_depth_loss | `loss_config.gaus_depth_loss=0` | Depth remains a network input and is still loaded/computed; this is not a depth-free model or guaranteed memory reduction |

Each new result starts from the same fine/coarse pretrained weights and deterministic seeds already set by upstream. Neither resumes the old optimizer state. No upsampling is enabled.

Only these two audited override keys are accepted by the worker. Source references audited at Im2Haircut commit `4afcc98c303b4108983252b0d4a671ffcee78480`: `configs/static.conf`, `src/datasets/real_dataset.py`, `trainers/trainer_single.py`, `src/gaussian_utils/GaussianTrainer.py`, and `src/model_utils/save_utils.py`. Their actual local hashes are recorded where used. The two earlier guarded trainer fixes must already be present; the suite does not rewrite upstream source.

## Read the review correctly

- Green: the proposed image hair mask. Magenta: projected sparse strand support after reference-head occlusion. These are not a final shaded or Gaussian-rendered hair image.
- Blue head overlay: projected reference head. Orange dots: projected aligned BFM mesh vertices. Both use the same estimated camera; this is not independent camera calibration.
- The support proxy depends strongly on the chosen pixel radius and sparse strand count. It is not a calibrated strand diameter, recovered density, final silhouette IoU, likeness score or automatic quality rank.
- Head depth uses two-sided triangles with perspective interpolation and a small depth tolerance. No hair self-occlusion or sophisticated strand antialiasing is implemented. Segments are sampled eight times each. The mesh is neither moved nor deformed.
- Direction panels compare saved features with the same undirected hue encoding. Their angular disagreement is not ground-truth direction error. The prior alignment's treatment of direction vectors under affine warping remains unverified; this update visualizes it and does not silently rotate the maps.
- Upstream currently loads `orient_conf` from the orientation PNG, not the separate confidence NPY. The default `use_conf=false` is retained. Do not enable confidence weighting assuming the saved NPY will be used.
- No side/back photos participate in optimization. No multi-view fitting, scalp adaptation, hair completion, recovered hair color, upsampling or FLAME attachment is claimed here. Those require separate implementation after the camera/map diagnostics.
- RAM is summed process RSS and can double-count shared pages. VRAM/utilization/temperature/power describe the whole GPU and sampling can miss short peaks. Device VRAM is not exact pipeline allocation.

## Validation

New CPU tests cover known camera coordinates, points behind the camera, reference-head occlusion, sparse-support sensitivity/displacement, OBJ index handling, prohibited overrides, stale original-photo detection, corrupted NPZ/PLY association, and partial failure persistence. A synthetic end-to-end CPU fixture creates actual projection PNGs and HTML while deliberately failing a mocked GPU worker. It does not validate GPU performance or reconstruction quality.

Run from the project root:

```bash
python -m unittest discover -s tests -v
python scripts/check_architecture.py
```

The release's full tests are run in the build environment with the required optional CPU packages. Real WSL/GPU experiments must run on your machine; no new GPU result is bundled or fabricated.
