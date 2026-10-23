"""Experimental shared-identity fit for three sequential captures (pinned RDF)."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

PIN = 'ebf0063835f520f07fdf5a813b1af9cdc0ae46b5'
SLOTS = ('front', 'left', 'right')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path('vendor/RealDenseFace'))
    p.add_argument('--iterations', type=int, default=10)
    args = p.parse_args()
    if not 1 <= args.iterations <= 30:
        p.error('--iterations must be between 1 and 30')
    root = Path(__file__).resolve().parents[1]
    repo = args.repo.resolve()
    sys.path[:0] = [str(root / 'src'), str(repo)]
    from check_rdf_side_views import digest, fit_camera, visible_wire
    import common.fix_chumpy  # noqa: F401
    import cv2
    import numpy as np
    import torch
    from common.config import FLAMEConfig, RealDenseFaceInferenceConfig, FaceBoxConfig, GNFlameOptimizerConfig
    from tracker import RealDenseFaceInferencer, GNFlameOptimizer
    from models.flame.model import FLAME
    from memento3d_head.demo_commands import find_photos
    from memento3d_head.adapters.head_compare import write_comparison
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN:
        raise ValueError('This experiment requires the documented pinned RDF commit')
    params = args.params.resolve()
    photos = find_photos(args.input_dir.resolve())
    cfg = FLAMEConfig(flame_model_path=str(repo / 'weights/flame/flame2023_Open.pkl'),
                      flame_assets_path=str(repo / 'weights/flame/flame_assets.npz'), num_expressions=100)
    weights = repo / 'weights/realdenseface/vits.pth'
    detector = repo / 'weights/facebox/face_box.pth'
    files = [params, Path(cfg.flame_model_path), Path(cfg.flame_assets_path), weights, detector,
             repo / 'configs/model/vits.yaml', repo / 'tracker/fitting/gn_flame_optimizer.py',
             repo / 'tracker/inference/realdenseface_inferencer.py', *photos.values()]
    hashes = {str(f): digest(f) for f in files}
    with np.load(params, allow_pickle=False) as data:
        beta = data['identity'].astype(np.float32).copy()
        fov = float(data['camera_fov_y'])
        width, height = int(data['image_width']), int(data['image_height'])
        n = int(data['num_expressions'])
    if beta.shape != (300,) or not np.isfinite(beta).all() or n != 100 or not 0 < fov < 180:
        raise ValueError('Expected a finite 300-dimensional identity, 100 expressions and valid FOV')
    out = params.parent / ('three_photo_' + uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    report = dict(schema_version=1, status='FAILED', research_only=True, model_variant='FLAME2023_Open',
                  upstream_commit=commit, input_sha256=hashes, shared_identity=True,
                  per_photo_expression_and_pose=True, camera_intrinsics_assumed_shared=True,
                  camera_fov_y=fov, likeness_verified=False, texture_generated=False,
                  baseline_photo_binding_verified=False, iterations=args.iterations,
                  metric_note='Residuals are to network predictions, not ground truth or likeness scores',
                  error=None, views={})
    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        infer = RealDenseFaceInferencer(RealDenseFaceInferenceConfig(
            model_config_path=str(repo / 'configs/model/vits.yaml'), model_weights_path=str(weights),
            flame=cfg, device='cuda', compile_model=False),
            FaceBoxConfig(model_weights_path=str(detector), device='cuda'))
        images, predictions = [], []
        for slot in SLOTS:
            print('Predicting ' + slot, flush=True)
            image = cv2.imdecode(np.fromfile(photos[slot], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise ValueError('All photos must match baseline resolution: ' + slot)
            prediction = infer.infer_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), face_bbox=None)
            images.append(image)
            predictions.append(prediction)
        # Release the neural network before allocating the batched solver.
        del infer
        gc.collect()
        torch.cuda.empty_cache()
        optimizer = GNFlameOptimizer(GNFlameOptimizerConfig(flame=cfg, device='cuda',
            registration_iterations=args.iterations))
        camera = optimizer._build_config_camera(width, height, fov)
        view, projection = optimizer._single_camera_matrices(camera)
        fields = ('vertex_coord', 'vertex_depth', 'vertex_coord_log_var', 'vertex_depth_log_var')
        tensors = []
        for field in fields:
            array = np.stack([getattr(item, field) for item in predictions]).astype(np.float32)
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite prediction: ' + field)
            tensors.append(torch.from_numpy(array).to('cuda'))
        coords, depths, coord_var, depth_var = tensors
        coords[..., 0] /= width
        coords[..., 1] /= height
        coord_var = optimizer._apply_eye_region_boost(coord_var)
        anchor = torch.from_numpy(beta).to('cuda')
        canonical_v, canonical_j = optimizer._calc_canonical_flame(anchor)
        # Initialize each photo independently: no temporal tracking assumption.
        states = []
        for index, slot in enumerate(SLOTS):
            print('Initializing pose ' + slot, flush=True)
            state = optimizer._register_pose(view, projection, coords[index], depths[index],
                coord_var[index], depth_var[index], canonical_v, canonical_j)
            states.append(state)
        print('Joint fit: one identity, three independent expression/pose states', flush=True)
        identity, xs = optimizer._register(view, projection, coords, depths, coord_var, depth_var,
            init_identity=anchor, init_x=torch.stack(states), num_iters=args.iterations)
        candidate = identity.detach().cpu().numpy()
        states_np = xs.detach().cpu().numpy()
        if candidate.shape != (300,) or states_np.shape != (3, 118) or not np.isfinite(candidate).all() or not np.isfinite(states_np).all():
            raise ValueError('Invalid joint solver output')
        np.savez(out / 'joint.flame.npz', identity=candidate, x=states_np, slots=np.array(SLOTS),
            camera_fov_y=fov, image_width=width, image_height=height, num_expressions=n,
            camera_rot=np.asarray(optimizer.config.camera_rotation), camera_pos=np.asarray(optimizer.config.camera_position))
        model = FLAME(cfg, device='cpu')
        faces = model.faces.cpu().numpy()
        with np.load(cfg.flame_assets_path, allow_pickle=False) as assets:
            if not np.array_equal(faces, assets['faces']):
                raise ValueError('Open model and dense-prior face topology differ')
        zero = torch.zeros(118)
        with torch.no_grad():
            before = model.forward(torch.from_numpy(beta), zero).numpy()
            after = model.forward(torch.from_numpy(candidate), zero).numpy()
        for name, vertices in [('before.neutral.obj', before), ('after.neutral.obj', after)]:
            with (out / name).open('w', encoding='ascii') as f:
                for v in vertices:
                    f.write('v %.9g %.9g %.9g\n' % tuple(v))
                for face in faces + 1:
                    f.write('f %d %d %d\n' % tuple(face))
        viewer = out / 'identity.comparison.html'
        write_comparison(viewer, before, after, faces, {'units': 'native_model_units', 'research_only': True})
        html = viewer.read_text(encoding='utf-8').replace('millimeters', 'native_model_units').replace(' mm', ' model units')
        html = html.replace('Mesh gốc', 'Front-only identity').replace('Mesh thử nghiệm', 'Three-photo identity')
        viewer.write_text(html, encoding='utf-8')
        focal = height / (2 * np.tan(np.radians(fov) / 2))
        k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], float)
        # Equal diagnostic: neutral meshes, same confidence subset, independently re-fit camera.
        for i, slot in enumerate(SLOTS):
            points = np.asarray(predictions[i].vertex_coord)
            var = np.asarray(predictions[i].vertex_coord_log_var).ravel()
            valid = np.isfinite(points).all(1) & np.isfinite(var) & (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
            if valid.sum() < 30:
                raise ValueError('Too few valid diagnostic points: ' + slot)
            valid &= var <= np.quantile(var[valid], .5)
            panels, metrics = [], {}
            for name, vertices in [('before', before), ('after', after)]:
                r, t, metric = fit_camera(vertices[valid], points[valid], k)
                panels.append(visible_wire(images[i], vertices, faces, k, r, t))
                metrics[name] = metric
            cv2.imencode('.jpg', np.concatenate(panels, axis=1))[1].tofile(str(out / (slot + '.before_after.jpg')))
            report['views'][slot] = metrics
        if any(digest(Path(path)) != value for path, value in hashes.items()):
            raise ValueError('An input changed during execution')
        report.update(status='JOINT_CANDIDATE_EXPORTED', identity_delta_l2=float(np.linalg.norm(candidate-beta)),
                      max_neutral_displacement_native_units=float(np.linalg.norm(after-before, axis=1).max()),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      automatic_acceptance=False, diagnostic_expression='neutral for both meshes')
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Output: ' + str(out), flush=True)
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
