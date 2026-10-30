"""Review existing personal Im2 results and optionally run two controlled 50-epoch ablations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from im2_support import PIN, sha256
from im2_diagnostics import diagnose, direction_review, overlay, project as project_points, read_obj, surface_depth, write_review

VARIANTS = {
    'gabor_direction': {'dataset_real.convert_gabor_to_strand_map': True},
    'no_depth_loss': {'loss_config.gaus_depth_loss': 0},
}


def resolve_path(value, project):
    path = Path(value).expanduser()
    return (project / path).resolve() if not path.is_absolute() else path.resolve()


def validate_case(case, project, repo):
    """Require original-photo binding and byte-identical dataset/camera artifacts."""
    dataset = repo / 'data' / case['dataset_folder']
    prep = resolve_path(case['prepared_run'], project)
    canonical = resolve_path(case['canonical_run'], project)
    deep = canonical.parent
    original = resolve_path(case['front_photo'], project)
    prep_report = json.loads((prep / 'report.json').read_text())
    if prep_report['input_sha256'] != sha256(original):
        raise ValueError('Original front photo does not match this prepared run')
    names = {
        'resized_img_aligned': 'front.png', 'seg_aligned': 'front.png',
        'body_img_aligned': 'front.png', 'orientation_maps_aligned': 'front.png',
        'strand_map_aligned': 'front.png', 'depth_apple_pro_aligned': 'front.npz',
        'confidence_maps_aligned': 'front.npy',
    }
    hashes = {}
    for folder, filename in names.items():
        relative = folder + '/' + filename
        source, target = prep / relative, dataset / relative
        if sha256(source) != sha256(target):
            raise ValueError('Dataset differs from prepared run: ' + relative)
        hashes[str(target.relative_to(repo / 'data'))] = sha256(target)
    camera_path = dataset / 'proj_matx_inv_aligned/front.txt'
    if sha256(camera_path) != sha256(canonical / 'proj_matx_inv_aligned/front.txt'):
        raise ValueError('Dataset camera differs from canonical run')
    if sha256(dataset / 'resized_img_aligned/front.png') != sha256(deep / 'resized_img_aligned/front.png'):
        raise ValueError('Camera was estimated from a different image')
    hashes[str(camera_path.relative_to(repo / 'data'))] = sha256(camera_path)
    camera = np.loadtxt(camera_path)
    transform = np.loadtxt(canonical / 'proj_matx_aligned/front.txt')
    source_camera = np.loadtxt(canonical / 'bfm_cameras_aligned/front.txt')
    if camera.shape != (3, 4) or transform.shape != (4, 4):
        raise ValueError('Unexpected camera/transform shape')
    if not np.isfinite(camera).all() or np.linalg.matrix_rank(camera) != 3:
        raise ValueError('Invalid canonical camera')
    if not np.allclose(camera @ transform, source_camera, rtol=1e-6, atol=1e-6):
        raise ValueError('Camera transform consistency check failed')
    image = Image.open(dataset / 'resized_img_aligned/front.png').convert('RGB')
    if image.size != (512, 512):
        raise ValueError('Expected 512x512 aligned photo')
    for folder in ('seg_aligned', 'body_img_aligned', 'orientation_maps_aligned', 'strand_map_aligned'):
        with Image.open(dataset / folder / 'front.png') as im:
            if im.size != image.size:
                raise ValueError('Mis-sized map: ' + folder)
    mask = np.asarray(Image.open(dataset / 'seg_aligned/front.png').convert('L')) > 127
    with np.load(dataset / 'depth_apple_pro_aligned/front.npz', allow_pickle=False) as data:
        depth = data['depth']
    confidence = np.load(dataset / 'confidence_maps_aligned/front.npy', allow_pickle=False)
    if not mask.any() or mask.all():
        raise ValueError('Empty or full-frame hair mask')
    for name, array in [('depth', depth), ('confidence', confidence)]:
        if array.shape != mask.shape or not np.isfinite(array).all():
            raise ValueError('Invalid ' + name)
    # Hash source code/assets used by diagnostics or shared across the experiments.
    audit_files = ['src/datasets/real_dataset.py', 'trainers/trainer_single.py',
                   'src/gaussian_utils/GaussianTrainer.py', 'configs/static.conf',
                   'data/head_prior.obj', 'data/scale_preprocess.pickle',
                   'pretrained_models/fine.pth', 'pretrained_models/coarse.pth']
    audit = {f: sha256(repo / f) for f in audit_files}
    return dataset, canonical, camera, image, mask, hashes, audit


def check_run(run, dataset, expected_hashes, repo, required_epochs=None):
    report = json.loads((run / 'pipeline.report.json').read_text())
    if report.get('status') != 'STRANDS_REVIEW_READY' or report.get('repo_commit') != PIN:
        raise ValueError('Existing run is incomplete or uses a different upstream commit')
    command = report['command']
    if command[command.index('--folder_name')+1] != dataset.name:
        raise ValueError('Existing run used a different dataset')
    if report['scene'] != 'front.png':
        raise ValueError('Existing run used a different scene')
    if required_epochs is not None and report['epoch_limit'] != required_epochs:
        raise ValueError('Existing baseline has a different epoch limit')
    if report.get('experiment_overrides'):
        raise ValueError('Existing reference must be an unmodified baseline')
    for name, digest in expected_hashes.items():
        if report['data_sha256'].get(name) != digest:
            raise ValueError('Existing run used different map/camera bytes: ' + name)
    if report['trainer_sha256'] != sha256(repo / 'trainers/trainer_single.py'):
        raise ValueError('Trainer changed since the reference run')
    for name in ('fine', 'coarse'):
        if report[name + '_weights_sha256'] != sha256(repo / 'pretrained_models' / (name + '.pth')):
            raise ValueError('Weights changed since the reference run')
    if report['config_sha256'] != sha256(run / 'run.conf'):
        raise ValueError('Reference run.conf has changed')
    from pyhocon import ConfigFactory
    reference_config = ConfigFactory.parse_file(str(run / 'run.conf')).as_plain_ordered_dict()
    baseline_config = ConfigFactory.parse_file(str(repo / 'configs/static.conf')).as_plain_ordered_dict()
    reference_config.pop('visuals_config', None)
    baseline_config.pop('visuals_config', None)
    if reference_config != baseline_config:
        raise ValueError('Reference used a different non-logging model/loss configuration')
    validate_npz(run, report)
    return report


def validate_npz(run, report):
    ply = run / 'pointclouds_train' / ('pred_%06d.ply' % report['epoch_limit'])
    if sha256(ply) != report['export']['source_sha256']:
        raise ValueError('Source PLY changed')
    with np.load(run / 'export/hair.strands.npz', allow_pickle=False) as data:
        strands = data['strands']
    expected = (report['export']['strand_count'], report['export']['points_per_strand'], 3)
    if strands.shape != expected or not np.isfinite(strands).all():
        raise ValueError('Invalid strand export')
    # Old reports did not bind NPZ bytes: verify geometry against their bound PLY.
    import trimesh
    points = np.asarray(trimesh.load(ply, process=False).vertices)
    if not np.array_equal(strands.reshape(-1, 3), points.astype(strands.dtype)):
        raise ValueError('NPZ does not reproduce the recorded PLY geometry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/im2.review04.json')
    parser.add_argument('--mode', choices=['all', 'review'], default='all')
    parser.add_argument('--epochs', type=int, default=50, choices=range(1, 301), metavar='1..300')
    parser.add_argument('--repo', type=Path, default=Path.home() / 'memento-hair/Im2Haircut')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Use Ubuntu/WSL with the im2haircut conda environment')
    project = Path(__file__).resolve().parents[1]
    repo = args.repo.expanduser().resolve()
    case = json.loads(args.case.read_text(encoding='utf-8'))
    if Path(case['dataset_folder']).name != case['dataset_folder'] or case['dataset_folder'] in ('.', '..'):
        raise ValueError('Dataset folder must be a single name')
    out = project / 'runs' / ('im2_review_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'RUNNING', 'case': case, 'mode': args.mode,
              'epoch_limit_for_new_runs': args.epochs, 'results': [],
              'automatic_quality_ranking': False, 'multi_view_optimization': False,
              'hair_color_recovered': False, 'flame_attachment_implemented': False,
              'notes': ['Single-view only; side/back inputs are not used.',
                        'Sparse pixel support changes with display radius and is not a final render.',
                        'Existing 300-epoch result is a longer-run reference, not a controlled 50-epoch ablation.',
                        'No-depth-loss still loads depth as a network input; this is not a depth-free model.',
                        'Gabor mode changes loss-direction features; the prior network still consumes the Gabor angle input.',
                        'No upsampling, camera fitting, synthetic hair completion or automatic winner selection.']}
    def save():
        (out / 'suite.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        write_review(out, report)
    print('Review output:', out, flush=True)
    save()
    try:
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != PIN:
            raise ValueError('Unexpected Im2Haircut commit')
        dataset, canonical, camera, image, mask, hashes, audit = validate_case(case, project, repo)
        report['input_sha256'] = hashes
        report['audit_sha256'] = audit
        report['case_config_sha256'] = sha256(args.case)
        report['diagnostic_code_sha256'] = sha256(Path(__file__).with_name('im2_diagnostics.py'))
        # Confirm allowed overrides are available before launching any GPU subprocess.
        from pyhocon import ConfigFactory
        config = ConfigFactory.parse_file(str(repo / 'configs/static.conf'))
        if (repo / config['dataset']['data_path']).resolve() != (repo / 'data').resolve():
            raise ValueError('Unexpected dataset root in upstream config')
        if config['dataset_real']['convert_gabor_to_strand_map'] is not False or config['loss_config']['gaus_depth_loss'] != .01:
            raise ValueError('Upstream baseline configuration differs from the audited defaults')
        source = (repo / 'src/datasets/real_dataset.py').read_text()
        if 'if self.convert_gabor_to_strand_map:' not in source:
            raise ValueError('Gabor loss-direction switch missing from dataset implementation')
        overlay(image, mask, np.zeros_like(mask)).save(out / 'input_review.png')
        report['direction_diagnostic'] = direction_review(dataset, mask, out)
        vertices, faces = read_obj(repo / 'data/head_prior.obj')
        head_depth = surface_depth(vertices, faces, camera, image.size)
        if np.isfinite(head_depth).sum() == 0:
            raise ValueError('Reference head projects entirely outside the camera image')
        head_image = np.asarray(image, dtype=float).copy()
        visible = np.isfinite(head_depth)
        head_image[visible] = .55*head_image[visible]+.45*np.array([50, 100, 255])
        preview = Image.fromarray(np.uint8(head_image))
        draw = ImageDraw.Draw(preview)
        bfm, _ = read_obj(canonical / 'bfm_meshes_our_space_aligned/front.obj')
        uv, _, positive = project_points(bfm[::20], camera)
        for (x, y), valid in zip(uv, positive):
            if valid and 0 <= x < image.width and 0 <= y < image.height:
                draw.ellipse((x-1, y-1, x+1, y+1), fill=(255, 140, 20))
        preview.save(out / 'head_projection.png')
        report['reference_head_projected_pixels'] = int(visible.sum())
        references = []
        for label in ('baseline_50', 'baseline_300'):
            run = resolve_path(case[label], project)
            epoch = args.epochs if label == 'baseline_50' else 300
            old = check_run(run, dataset, hashes, repo, epoch)
            references.append((label, run, old))
        def review(label, run, old, reused):
            diagnostic = diagnose(run / 'export/hair.strands.npz', camera, image, mask, head_depth, out / label)
            item = {'variant': label, 'status': 'DIAGNOSTIC_EXPORTED', 'run': str(run),
                    'reused': reused, 'epoch_limit': old['epoch_limit'],
                    'metrics': old.get('metrics'), 'overrides': old.get('experiment_overrides', {}),
                    'npz_sha256': sha256(run / 'export/hair.strands.npz'), 'diagnostic': diagnostic}
            report['results'].append(item)
            save()
        for label, run, old in references:
            review(label, run, old, True)
        print('Existing results reviewed. Open:', out / 'review.html', flush=True)
        failed = False
        if args.mode == 'all':
            for label, overrides in VARIANTS.items():
                print('Running ablation:', label, flush=True)
                if {f: sha256(repo / f) for f in audit} != audit:
                    raise ValueError('Shared source/weights changed during the suite')
                if any(sha256(repo / 'data' / f) != digest for f, digest in hashes.items()):
                    raise ValueError('Dataset changed during the suite')
                private = out / ('worker_' + label)
                private.mkdir()
                override_file = private / 'overrides.json'
                override_file.write_text(json.dumps(overrides, indent=2), encoding='utf-8')
                command = [sys.executable, '-u', str(project / 'scripts/run_im2_sample.py'),
                           '--repo', str(repo), '--folder', dataset.name, '--scene', 'front.png',
                           '--epochs', str(args.epochs), '--output-root', str(private),
                           '--overrides', str(override_file)]
                # Worker owns GPU monitoring and signal cleanup. Keep its live OBS output visible.
                result = subprocess.run(command, cwd=str(project))
                candidates = list(private.glob('memento_im2_*'))
                if result.returncode or len(candidates) != 1:
                    failed = True
                    report['results'].append({'variant': label, 'status': 'FAILED',
                                              'exit_code': result.returncode, 'worker_root': str(private)})
                    save()
                    # Stop after worker failure; do not spend more GPU time on cascading errors.
                    break
                run = candidates[0]
                old = json.loads((run / 'pipeline.report.json').read_text())
                if old['status'] != 'STRANDS_REVIEW_READY' or old.get('experiment_overrides') != overrides:
                    raise ValueError('Worker did not export the requested variant')
                if any(old['data_sha256'].get(f) != digest for f, digest in hashes.items()):
                    raise ValueError('Worker used different dataset bytes')
                validate_npz(run, old)
                review(label, run, old, False)
        report['status'] = 'PARTIAL_FAILURE' if failed else 'REVIEW_REQUIRED'
        return 1 if failed else 0
    except KeyboardInterrupt:
        report['status'] = 'INTERRUPTED'
        return 130
    except Exception as error:
        report['status'] = 'FAILED'
        report['error'] = str(error)
        raise
    finally:
        save()
        print(report['status'] + ': ' + str(out / 'review.html'), flush=True)
        print('Report:', out / 'suite.report.json', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
