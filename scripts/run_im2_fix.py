"""One consolidated field-transform experiment: CPU checks, reused references, at most one 50-epoch worker."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from im2_support import PIN, sha256
from im2_diagnostics import diagnose, read_obj, surface_depth
from im2_field_fix import check_affine, correct_fields, field_comparison, root_review
from run_im2_review import resolve_path, validate_case, check_run, validate_npz


def prepare_copy(dataset, prep, matrix, target, output):
    """Create a separate dataset. Original pixel/camera/depth bytes are never changed."""
    raw_rgb_path = prep / 'resized_img/front.png'
    raw_rgb = np.asarray(Image.open(raw_rgb_path).convert('RGB'))
    rgb = np.asarray(Image.open(dataset / 'resized_img_aligned/front.png').convert('RGB'))
    size = (rgb.shape[1], rgb.shape[0])
    reconstructed = cv2.warpAffine(raw_rgb, matrix, size)
    if not np.array_equal(rgb, reconstructed):
        raise ValueError('Configured affine does not reproduce aligned RGB exactly; stop before GPU')
    raw_gabor_path = prep / 'orientation_maps/front.png'
    raw_strand_path = prep / 'strand_map/front.png'
    raw_mask_path = prep / 'seg/front.png'
    raw_gabor = np.asarray(Image.open(raw_gabor_path).convert('L'))
    raw_strand = np.asarray(Image.open(raw_strand_path).convert('RGB'))
    raw_mask = np.asarray(Image.open(raw_mask_path).convert('L')) > 127
    old_gabor = np.asarray(Image.open(dataset / 'orientation_maps_aligned/front.png').convert('L'))
    old_strand = np.asarray(Image.open(dataset / 'strand_map_aligned/front.png').convert('RGB'))
    mask = np.asarray(Image.open(dataset / 'seg_aligned/front.png').convert('L')) > 127
    # Bind the raw scalar feature to the existing aligned map as well as the RGB.
    legacy_gabor = cv2.warpAffine(raw_gabor, matrix, size)
    if not np.array_equal(old_gabor, legacy_gabor):
        raise ValueError('Raw Gabor does not reproduce existing aligned Gabor; stop instead of combining unrelated maps')
    gabor, strand, note = correct_fields(raw_gabor, raw_strand, raw_mask, old_strand, mask, matrix)
    note['change_diagnostic'] = field_comparison(old_gabor, gabor, old_strand, strand, mask, output)
    note['affine_2d'] = matrix.tolist()
    note['aligned_rgb_reproduced_exactly'] = True
    note['legacy_gabor_reproduced_exactly'] = True
    note['rotation_image_degrees'] = float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0])))
    sources = [raw_rgb_path, raw_gabor_path, raw_strand_path, raw_mask_path]
    note['raw_sha256'] = {str(path): sha256(path) for path in sources}
    # Save raw vs legacy Gabor previews too; noise is not assumed to come only from alignment.
    Image.open(raw_gabor_path).convert('L').save(output / 'gabor_raw_encoded.png')
    Image.fromarray(old_gabor).save(output / 'gabor_legacy_encoded.png')
    target.mkdir(parents=True, exist_ok=False)
    copied = {}
    for folder, filename in (
        ('resized_img_aligned', 'front.png'), ('seg_aligned', 'front.png'),
        ('body_img_aligned', 'front.png'), ('orientation_maps_aligned', 'front.png'),
        ('strand_map_aligned', 'front.png'), ('confidence_maps_aligned', 'front.npy'),
        ('depth_apple_pro_aligned', 'front.npz'), ('proj_matx_inv_aligned', 'front.txt'),
    ):
        relative = folder + '/' + filename
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset / relative, dst)
        copied[relative] = sha256(dataset / relative)
    Image.fromarray(gabor).save(target / 'orientation_maps_aligned/front.png')
    Image.fromarray(strand).save(target / 'strand_map_aligned/front.png')
    changed = ('orientation_maps_aligned/front.png', 'strand_map_aligned/front.png')
    for relative, digest in copied.items():
        if relative not in changed and sha256(target / relative) != digest:
            raise ValueError('Unexpected non-direction modification: ' + relative)
    note['dataset'] = str(target)
    note['original_dataset'] = str(dataset)
    note['target_sha256'] = {relative: sha256(target / relative) for relative in copied}
    note['changed_direction_files'] = [relative for relative in changed if sha256(target / relative) != copied[relative]]
    (target / 'field_fix.report.json').write_text(json.dumps(note, indent=2), encoding='utf-8')
    return note


def render_report(out, report):
    esc = lambda value: html.escape(str(value))
    text = '''<!doctype html><meta charset="utf-8"><title>Im2 field transform experiment</title>
<style>body{font:16px system-ui;background:#151b24;color:#eee;margin:auto;max-width:1700px;padding:25px}section{background:#222c37;padding:18px;margin:18px 0}.row{display:flex;flex-wrap:wrap}figure{flex:1;min-width:220px;margin:5px}img{max-width:100%;width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #637185}</style>
<h1>One field-transform experiment</h1>
<p>Original camera, masks, depth, roots, model weights and loss weights retained. Two direction encodings transformed on a private dataset copy. No additional models or downloads.</p>
<p>This is a combined preprocessing hypothesis, not two isolated ablations. HairStep XY convention is explicit but not independently calibrated. Native Gabor noise and unseen side/back shape are not recovered.</p>
<p>Roots: visible roots green inside photo hair mask, red outside. Roots can lie under hair: red alone does not prove a scalp error. Early segments: first 5% of strand points. Full support: sparse proxy at 1 px, not a final render.</p>'''
    text += '<h2>Status: '+esc(report['status'])+'</h2>'
    if report.get('error'):
        text += '<pre>'+esc(report['error'])+'</pre>'
    if (out / 'fields_before_after.png').is_file():
        text += '<h2>Field transform before / after</h2><img style="max-width:1024px" src="fields_before_after.png">'
    rows = []
    for item in report['results']:
        label = item['variant']
        metrics = item.get('metrics') or {}
        diag = item.get('diagnostic') or {}
        sensitivity = diag.get('radius_sensitivity', [{}])[0]
        rows.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in (
            label, item.get('epoch_limit'), metrics.get('wall_seconds'),
            metrics.get('peak_process_tree_rss_mib'), metrics.get('peak_device_vram_mib'),
            sensitivity.get('sparse_support_iou_proxy')))+'</tr>')
        images = [('roots.png', 'Visible roots'), ('early_segments.png', 'Early segments'), ('overlay_r1.png', 'Full strands, radius 1 px')]
        text += '<section><h2>'+esc(label)+'</h2><div class="row">'
        for filename, title in images:
            if (out / label / filename).is_file():
                text += '<figure><img src="'+esc(label+'/'+filename)+'"><figcaption>'+esc(title)+'</figcaption></figure>'
        text += '</div><details><summary>Measurements / provenance</summary><pre>'+esc(json.dumps(item, indent=2))+'</pre></details></section>'
    text += '<h2>Resource measurements and sparse proxy</h2><table><tr><th>Variant</th><th>Epoch limit</th><th>Seconds</th><th>Tree RSS MiB</th><th>Whole GPU VRAM MiB</th><th>Sparse IoU proxy r1</th></tr>'+''.join(rows)+'</table>'
    text += '<p>Existing 300 epochs is a longer-run reference, not matched-budget evidence. GPU baselines and thermal conditions differ. No automatic quality winner.</p><details><summary>Full experiment report</summary><pre>'+esc(json.dumps(report, indent=2))+'</pre></details>'
    (out / 'review.html').write_text(text, encoding='utf-8')


def check_corrected_sample(repo, target):
    """Load once on CPU with the actual upstream Dataset, before launching the sole GPU worker."""
    import torch
    from pyhocon import ConfigFactory
    sys.path.insert(0, str(repo))
    from src.datasets.real_dataset import HairstyleRealDataset
    config = ConfigFactory.parse_file(str(repo / 'configs/static.conf'))
    if (repo / config['dataset']['data_path']).resolve() != (repo / 'data').resolve():
        raise ValueError('Unexpected upstream data root')
    options = dict(config['dataset_real'])
    options.update(device='cpu', infer_path=str(target), scene='front.png')
    # Resolve shared asset paths without changing the process working directory.
    for name in ('path_to_meshgrid_data', 'gs_scale_path'):
        options[name] = str(repo / options[name])
    actual = HairstyleRealDataset(**options)
    if len(actual) != 1:
        raise ValueError('Corrected dataset must contain exactly one scene')
    for value in actual[0]:
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError('Corrected sample contains a non-finite tensor')
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError('Corrected sample contains non-finite camera/features')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument('--case', type=Path, default=project / 'configs/im2.review04.json')
    parser.add_argument('--repo', type=Path, default=Path.home() / 'memento-hair/Im2Haircut')
    parser.add_argument('--cpu-only', action='store_true', help='Prepare and inspect fields/roots without any GPU worker')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run in Ubuntu/WSL using im2haircut')
    repo = args.repo.expanduser().resolve()
    case = json.loads(args.case.read_text(encoding='utf-8'))
    start = time.perf_counter()
    token = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]
    out = project / 'runs' / ('im2_fix_' + token)
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'RUNNING', 'results': [], 'case': case,
              'new_gpu_runs_started': 0, 'automatic_quality_ranking': False,
              'multi_view_optimization': False, 'hair_color_recovered': False,
              'geometry_modified_by_postprocessing': False, 'scalp_modified': False,
              'note': 'At most one 50-epoch reconstruction; no automatic retry or baseline reruns.'}
    def save():
        report['elapsed_wall_seconds'] = round(time.perf_counter()-start, 3)
        (out / 'experiment.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        render_report(out, report)
    print('Output:', out, flush=True)
    save()
    try:
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != PIN:
            raise ValueError('Unexpected upstream commit')
        if Path(case['dataset_folder']).name != case['dataset_folder'] or case['dataset_folder'] in ('.', '..'):
            raise ValueError('Dataset folder must be a single name')
        dataset, canonical, camera, image, mask, hashes, audit = validate_case(case, project, repo)
        report['audit_sha256'] = audit
        report['case_config_sha256'] = sha256(args.case)
        report['code_sha256'] = {name: sha256(Path(__file__).with_name(name)) for name in ('run_im2_fix.py', 'im2_field_fix.py', 'im2_diagnostics.py')}
        matrix = check_affine(case['affine_2d'])
        vertices, faces = read_obj(repo / 'data/head_prior.obj')
        head_depth = surface_depth(vertices, faces, camera, image.size)
        if not np.isfinite(head_depth).any():
            raise ValueError('Reference head is outside camera image')
        observed_roots = {}
        def review(label, run, old, reused):
            diag = diagnose(run / 'export/hair.strands.npz', camera, image, mask, head_depth, out / label)
            with np.load(run / 'export/hair.strands.npz', allow_pickle=False) as data:
                strands = data['strands']
            observed_roots[label] = strands[:, 0].copy()
            if label != 'baseline_50' and observed_roots[label].shape == observed_roots['baseline_50'].shape:
                report.setdefault('root_position_comparisons', {})[label] = {
                    'max_indexed_displacement_native_units': float(np.linalg.norm(observed_roots[label]-observed_roots['baseline_50'], axis=-1).max()),
                    'note': 'Same export ordering; numerical agreement does not prove anatomical placement.',
                }
            roots = root_review(strands, camera, image, mask, head_depth, out / label)
            report['results'].append({'variant': label, 'run': str(run), 'epoch_limit': old['epoch_limit'],
                'reused': reused, 'npz_sha256': sha256(run / 'export/hair.strands.npz'),
                'metrics': old.get('metrics'), 'diagnostic': diag, 'roots': roots})
            save()
        for label, epochs in [('baseline_50', 50), ('baseline_300', 300)]:
            run = resolve_path(case[label], project)
            old = check_run(run, dataset, hashes, repo, epochs)
            review(label, run, old, True)
        prep = resolve_path(case['prepared_run'], project)
        target = repo / 'data' / ('fieldfix04_' + token)
        report['field_fix'] = prepare_copy(dataset, prep, matrix, target, out)
        report['status'] = 'CPU_CHECKS_PASSED'
        save()
        print('CPU checks passed. Private dataset:', target, flush=True)
        if args.cpu_only:
            report['status'] = 'CPU_REVIEW_READY'
            return 0
        check_corrected_sample(repo, target)
        if any(sha256(repo / f) != digest for f, digest in audit.items()):
            raise ValueError('Shared source/weights changed after CPU checks')
        worker = out / 'worker_field_fix'; worker.mkdir()
        command = [sys.executable, '-u', str(project / 'scripts/run_im2_sample.py'),
                   '--repo', str(repo), '--folder', target.name, '--scene', 'front.png',
                   '--epochs', '50', '--output-root', str(worker)]
        report['worker_command'] = command
        report['new_gpu_runs_started'] = 1
        report['status'] = 'GPU_RUNNING'
        save()
        print('Starting ONE 50-epoch worker; previous variants will not rerun.', flush=True)
        result = subprocess.run(command, cwd=str(project))
        report['worker_exit_code'] = result.returncode
        runs = list(worker.glob('memento_im2_*'))
        if result.returncode or len(runs) != 1:
            report['status'] = 'WORKER_FAILED'
            report['error'] = 'Worker failed; inspect worker_field_fix/*/inference.log. No retry was started.'
            return 1
        run = runs[0]
        old = json.loads((run / 'pipeline.report.json').read_text())
        if old['status'] != 'STRANDS_REVIEW_READY':
            raise ValueError('Worker did not export complete strands')
        if old.get('experiment_overrides'):
            raise ValueError('Unexpected model/loss overrides in field-transform worker')
        for relative, digest in report['field_fix']['target_sha256'].items():
            if old['data_sha256'].get(target.name+'/'+relative) != digest:
                raise ValueError('Worker input does not match corrected dataset: ' + relative)
        validate_npz(run, old)
        review('field_fix_50', run, old, False)
        report['status'] = 'REVIEW_REQUIRED'
        return 0
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
        print('Report:', out / 'experiment.report.json', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
