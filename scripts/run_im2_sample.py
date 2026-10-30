"""Benchmark ONE prepared Im2Haircut example in WSL; raw-photo preprocessing is not implemented."""
import argparse
import csv
from datetime import datetime, timezone
import json
import importlib.metadata
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

from im2_support import PIN, export_strands, prepare_fixes, sha256


def gpu_sample():
    try:
        result = subprocess.run(['nvidia-smi', '--id=0',
            '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        if result.returncode:
            return [None] * 4
        fields = result.stdout.strip().split(',')
        values = []
        for field in fields:
            try:
                values.append(float(field.strip()))
            except ValueError:
                values.append(None)
        return values if len(values) == 4 else [None] * 4
    except (OSError, subprocess.TimeoutExpired):
        return [None] * 4


def monitored(command, cwd, env, output, interval):
    import psutil
    rows = []
    started = time.perf_counter()
    baseline = gpu_sample()[0]
    interrupted = False
    with (output / 'inference.log').open('w', encoding='utf-8') as log, \
            (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as metrics:
        writer = csv.writer(metrics)
        writer.writerow(['elapsed_s', 'tree_rss_mib', 'device_vram_mib', 'gpu_util_pct',
                         'gpu_temp_c', 'gpu_power_w'])
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                rss = 0
                try:
                    process = psutil.Process(child.pid)
                    processes = [process] + process.children(recursive=True)
                except psutil.Error:
                    processes = []
                for process in processes:
                    try:
                        rss += process.memory_info().rss
                    except psutil.Error:
                        pass
                row = [round(time.perf_counter() - started, 3), rss / 2**20, *gpu_sample()]
                rows.append(row)
                writer.writerow(row)
                metrics.flush()
                if len(rows) == 1 or len(rows) % 5 == 0:
                    print(f'[OBS] {row[0]:.1f}s | RAM {row[1]:.0f} MiB | device VRAM {row[2]} MiB', flush=True)
                # Poll rather than waiting uninterruptibly for the entire interval.
                deadline = time.perf_counter() + interval
                while child.poll() is None and time.perf_counter() < deadline:
                    time.sleep(min(0.1, max(0, deadline - time.perf_counter())))
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
    def peak(column):
        values = [r[column] for r in rows if r[column] is not None]
        return max(values) if values else None
    return {
        'exit_code': code, 'interrupted': interrupted,
        'wall_seconds': round(time.perf_counter() - started, 3),
        'sample_count': len(rows), 'sampling_interval_seconds': interval,
        'gpu_memory_baseline_mib': baseline,
        'peak_process_tree_rss_mib': peak(1), 'peak_device_vram_mib': peak(2),
        'peak_gpu_utilization_pct': peak(3), 'peak_gpu_temperature_c': peak(4),
        'peak_gpu_power_w': peak(5),
        'notes': ['RSS sums may double-count shared pages.',
                  'GPU metrics are for the entire device, not this process.',
                  'Sampling may miss brief peaks; wall time includes monitoring overhead.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.home() / 'memento-hair/Im2Haircut')
    parser.add_argument('--scene', help='Exact prepared filename; omitted = first sorted image, recorded in report')
    parser.add_argument('--folder', default='examples', help='Already preprocessed dataset folder')
    parser.add_argument('--epochs', type=int, default=50, help='Upstream epoch limit; one sample yields epochs+1 updates')
    parser.add_argument('--coarse-steps', type=int, default=20)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--apply-fixes', action='store_true', help='Apply only the two exact audited trainer fixes')
    parser.add_argument('--overrides', type=Path, help='JSON with audited experiment options only')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run in Ubuntu/WSL using the im2haircut environment')
    if not 0 <= args.epochs <= 300 or args.coarse_steps < 0 or args.interval < 0.2:
        parser.error('epochs: 0..300; coarse steps >=0; interval >=0.2 seconds')
    if any(Path(value).name != value or value in ('.', '..') or '\\' in value
           for value in (args.folder,) + ((args.scene,) if args.scene else ())):
        parser.error('scene/folder must be a single filename/directory name')
    repo = args.repo.expanduser().resolve()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN:
        raise ValueError('Unexpected Im2Haircut commit; refusing to run')
    import psutil  # fail before starting the GPU worker if observability is unavailable
    from pyhocon import ConfigFactory
    overrides = load_overrides(args.overrides)
    trainer_hash = prepare_fixes(repo, args.apply_fixes)
    if args.scene is None:
        prepared = repo / 'data' / args.folder / 'resized_img_aligned'
        images = sorted(path.name for path in prepared.iterdir()
            if path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg'))
        if not images:
            raise ValueError('No prepared images')
        args.scene = images[0]
    photo = repo / 'data' / args.folder / 'resized_img_aligned' / args.scene
    assets = [repo / 'configs/static.conf', repo / 'pretrained_models/fine.pth',
              repo / 'pretrained_models/coarse.pth', repo / 'data/pointcloud.ply', photo]
    for path in assets:
        if not path.is_file():
            raise FileNotFoundError(path)
    root = (args.output_root or repo / 'experiments').expanduser().resolve()
    out = root / ('memento_im2_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    # Keep the upstream loop semantics explicit. It stops when epoch > num_epochs.
    base = assets[0].read_text(encoding='utf-8')
    base += ''.join(f'\n{key} = {json.dumps(value)}\n' for key, value in overrides.items())
    config = out / 'run.conf'
    config.write_text(base + f'\nvisuals_config.num_epochs = {args.epochs}\n'
        f'visuals_config.pc_freq = {max(1, args.epochs)}\nvisuals_config.logging_freq = 10\n', encoding='utf-8')
    parsed = ConfigFactory.parse_file(str(config))
    command = [sys.executable, '-u', str(repo / 'run_image_reconstruction.py'),
        '--conf_path', str(config), '--savedir', str(out), '--ckpt_path', str(assets[1]),
        '--pointcloud_path_head', str(assets[3]), '--folder_name', args.folder,
        '--scene', args.scene, '--num_workers', '0', '--num_steps_coarse', str(args.coarse_steps),
        '--unfreeze_time_for_pca', '-1', '--render_direction', '--binarize_masks',
        '-r', '1', '--ip', '127.0.0.1', '--port', '6019']
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['CUDA_LAUNCH_BLOCKING'] = '1'  # keep the verified upstream execution mode
    env['PYTHONPATH'] = os.pathsep.join([str(repo / 'submodules/external/VOODOO3D-official'),
        str(repo / 'submodules/external/GaussianHaircut'), env.get('PYTHONPATH', '')])
    report = {'schema_version': 1, 'status': 'RUNNING', 'repo_commit': commit,
        'trainer_sha256': trainer_hash, 'command': command, 'python': sys.executable,
        'scene': args.scene, 'prepared_photo_sha256': sha256(photo),
        'fine_weights_sha256': sha256(assets[1]), 'coarse_weights_sha256': sha256(assets[2]),
        'config_sha256': sha256(config), 'epoch_limit': args.epochs,
        'expected_single_sample_updates': args.epochs + 1,
        'upsampling_enabled': False, 'raw_photo_preprocessing': False,
        'head_alignment_verified': False, 'automatic_quality_ranking': False}
    report['experiment_overrides'] = overrides
    if args.overrides:
        report['overrides_sha256'] = sha256(args.overrides)
    report['packages'] = {}
    for package in ('torch', 'torchvision', 'numpy', 'scipy', 'scikit-image',
                    'timm', 'moderngl', 'pytorch3d', 'psutil'):
        try:
            report['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][package] = None
    # Bind auxiliary masks/depth/cameras as well as the visible input photo.
    data_root = repo / 'data'
    report['data_sha256'] = {str(path.relative_to(data_root)): sha256(path)
        for path in sorted(data_root.rglob('*')) if path.is_file()}
    report['upstream_worktree_status'] = subprocess.check_output(
        ['git', '-C', str(repo), 'status', '--short'], text=True)
    def save():
        (out / 'pipeline.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    print(f'Sample: {args.scene}\nOutput: {out}\nLive log: {out / "inference.log"}', flush=True)
    try:
        report['metrics'] = monitored(command, repo, env, out, args.interval)
        if report['metrics']['interrupted']:
            report['status'] = 'INTERRUPTED'
            return 130
        if report['metrics']['exit_code'] != 0:
            report['status'] = 'FAILED'
            return 1
        # Require the requested last geometry, not merely a successful process exit.
        ply = out / 'pointclouds_train' / f'pred_{args.epochs:06d}.ply'
        report['export'] = export_strands(ply, out / 'export', int(parsed['dataset']['num_points']))
        report['status'] = 'STRANDS_REVIEW_READY'
        report['geometry_note'] = 'PLY is evaluated before the optimizer update at its named step; not a saved final optimizer state.'
        return 0
    except Exception as error:
        report['status'] = 'FAILED'
        report['error'] = str(error)
        raise
    finally:
        save()
        print(f'{report["status"]}: {out / "pipeline.report.json"}', flush=True)


def load_overrides(path):
    """Restrict ablations to options audited in the pinned upstream source."""
    values = json.loads(path.read_text(encoding='utf-8')) if path else {}
    allowed = {'dataset_real.convert_gabor_to_strand_map': bool,
               'loss_config.gaus_depth_loss': (int, float)}
    if not isinstance(values, dict) or set(values) - set(allowed):
        raise ValueError('Unsupported experiment override')
    for key, value in values.items():
        if key == 'dataset_real.convert_gabor_to_strand_map':
            if type(value) is not bool:
                raise ValueError('Gabor switch must be boolean')
        elif type(value) not in (int, float) or not 0 <= value <= 0.01:
            raise ValueError('Depth loss must be a finite number from 0 to 0.01')
    return values


if __name__ == '__main__':
    raise SystemExit(main())
