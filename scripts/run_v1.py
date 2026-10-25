"""V1: three photographs -> optimized textured GLB, with per-stage telemetry."""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from uuid import uuid4

import psutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu():
    names = ('gpu_memory_used_mib', 'gpu_utilization_pct', 'gpu_temperature_c', 'gpu_power_w')
    try:
        r = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        values = r.stdout.strip().splitlines()[0].split(',')
        result = {}
        for name, value in zip(names, values):
            try: result[name] = float(value.strip())
            except ValueError: result[name] = None
        return {name: result.get(name) for name in names}
    except (OSError, subprocess.SubprocessError, IndexError):
        return dict.fromkeys(names)


def ram(pid):
    try:
        parent = psutil.Process(pid)
        processes = [parent] + parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.
    total = 0
    for p in processes:
        try: total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    return total / 1024**2


def latest(parent, pattern, status):
    candidates = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        report = p / 'report.json'
        if report.is_file() and json.loads(report.read_text(encoding='utf-8-sig')).get('status') == status:
            return p
    raise RuntimeError(f'Missing successful output: {pattern}, {status}')


def peak(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return max(values) if values else None


def run_stage(name, command, cwd, output, all_rows):
    print(f'\n[{name}]', flush=True)
    started = time.perf_counter()
    proc = subprocess.Popen([str(c) for c in command], cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
        env={**__import__('os').environ, 'PYTHONIOENCODING':'utf-8', 'PYTHONUNBUFFERED':'1'})
    lines = queue.Queue()
    def read():
        for line in proc.stdout: lines.put(line)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    rows = []
    try:
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            next_sample = 0.
            while True:
                elapsed = time.perf_counter() - started
                if elapsed >= next_sample:
                    memory = psutil.virtual_memory()
                    row = dict(stage=name, stage_elapsed_seconds=round(elapsed,3), ram_tree_mib=ram(proc.pid),
                        system_ram_used_pct=memory.percent, system_ram_available_mib=memory.available/1024**2, **gpu())
                    rows.append(row); all_rows.append(row)
                    next_sample = elapsed + 1.
                while True:
                    try: line = lines.get_nowait()
                    except queue.Empty: break
                    print(line, end='', flush=True); log.write(line); log.flush()
                if proc.poll() is not None and not reader.is_alive() and lines.empty(): break
                time.sleep(.05)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            if sys.platform == 'win32':
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'], capture_output=True, check=False)
            else: proc.terminate()
            proc.wait()
        proc.stdout.close()
    return dict(stage=name, exit_code=code, wall_seconds=round(time.perf_counter()-started,3),
        peak_ram_tree_mib=peak(rows,'ram_tree_mib'), peak_gpu_memory_used_mib=peak(rows,'gpu_memory_used_mib'),
        peak_gpu_utilization_pct=peak(rows,'gpu_utilization_pct'),
        peak_gpu_temperature_c=peak(rows,'gpu_temperature_c'),peak_gpu_power_w=peak(rows,'gpu_power_w'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--face-parser-model', type=Path, default=Path('D:/AI-cache/face-parsing/resnet18.onnx'))
    p.add_argument('--cell', type=int, choices=(16,32,48,64), default=32)
    p.add_argument('--flame-model', type=Path, default=ROOT/'vendor/RealDenseFace/weights/flame/flame2023.pkl')
    p.add_argument('--refine-texture', action='store_true',help='Enable experimental V2.1 texture boundary refinement')
    args = p.parse_args()
    folder = args.input_dir.resolve()
    repo = ROOT/'vendor/RealDenseFace'
    rdf = ROOT/'.venv-rdf/Scripts/python.exe'
    python = Path(sys.executable)
    parser_model = args.face_parser_model.resolve()
    flame_model = args.flame_model.resolve()
    photos = {}
    if not folder.is_dir(): p.error(f'Missing input directory: {folder}')
    for slot in ('front','left','right'):
        matches = [x for x in folder.iterdir() if x.is_file() and x.stem.lower()==slot and x.suffix.lower() in ('.jpg','.jpeg','.png')]
        if len(matches)!=1: p.error(f'Expected exactly one {slot}.jpg/.jpeg/.png in {folder}')
        photos[slot] = matches[0]
    required = [rdf, parser_model, repo/'fit_single_image.py', repo/'configs/model/vitb.yaml',
        repo/'weights/realdenseface/vitb.pth', repo/'weights/realdenseface/vits.pth', flame_model,
        repo/'weights/flame/flame_assets.npz', repo/'weights/facebox/face_box.pth']
    for f in required:
        if not f.is_file(): p.error(f'Missing file: {f}')
    for script in ('fit_rdf_three_photos.py','preview_rdf_three_textures.py','optimize_head.py','export_workspace_glb.py'):
        if not (ROOT/'scripts'/script).is_file(): p.error(f'Missing script: {script}')
    out = ROOT/'runs'/('v1_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    rows, stages = [], []
    report = dict(schema_version=1,pipeline_type='three_photo_to_glb',pipeline_version=1,
        status='RUNNING',started_at_utc=datetime.now(timezone.utc).isoformat(),
        input_dir=str(folder),input_sha256={s:sha(f) for s,f in photos.items()},stages=stages,
        workflow_release='v1-flame2023-1.1',model_variant=flame_model.stem,
        texture_refinement_requested=args.refine_texture,
        flame_model_path=str(flame_model),flame_model_sha256=sha(flame_model),
        flame_assets_sha256=sha(repo/'weights/flame/flame_assets.npz'),
        photo_quality_gate_executed=False,likeness_verified=False,printability_verified=False,
        measurement_notes=['RAM is summed child process working sets; shared pages can overlap.',
            'GPU metrics are device-wide, not process-specific. One-second sampling can miss peaks.',
            'Wall times include imports, model loading, IO and monitoring overhead.',
            'Monitoring parent RAM is excluded. No concurrent-load capability is established.'])
    started = time.perf_counter()
    report['gpu_baseline'] = gpu()
    def execute(name, command, cwd=ROOT):
        stage = run_stage(name,command,cwd,out,rows)
        stages.append(stage)
        (out/'pipeline.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if stage['exit_code']: raise RuntimeError(f'{name} failed; see {name}.log')
    try:
        execute('01_front',[rdf,repo/'fit_single_image.py','--input',photos['front'],
            '--output_npz',out/'front.flame.npz','--output_image',out/'front.reconstruction.jpg',
            '--flame_model',flame_model,'--flame_assets',repo/'weights/flame/flame_assets.npz',
            '--facebox_weights',repo/'weights/facebox/face_box.pth','--num_expressions','100','--device','cuda','--no_compile'],repo)
        if sha(flame_model)!=report['flame_model_sha256']: raise RuntimeError('FLAME model changed during front fitting')
        execute('02_joint',[rdf,ROOT/'scripts/fit_rdf_three_photos.py','--params',out/'front.flame.npz','--input-dir',folder,'--repo',repo,'--flame-model',flame_model])
        head = latest(out,'three_photo_*','JOINT_CANDIDATE_EXPORTED')
        execute('03_texture',[rdf,ROOT/'scripts/preview_rdf_three_textures.py','--run-dir',head,'--input-dir',folder,'--repo',repo,'--flame-model',flame_model])
        texture = latest(head,'three_textures_*','THREE_TEXTURES_PREVIEW_EXPORTED')
        execute('04_optimize',[python,ROOT/'scripts/optimize_head.py','--head-run',head,'--texture-run',texture,'--auto-mask','--face-parser-model',parser_model])
        workspace = latest(head,'head_workspace_*','TEXTURE_REVIEW_READY')
        execute('05_export',[python,ROOT/'scripts/export_workspace_glb.py','--workspace',workspace/'head.workspace.html','--output',out/'head.optimized.glb','--cell',args.cell]+(['--refine-texture'] if args.refine_texture else []))
        for slot, f in photos.items():
            if sha(f)!=report['input_sha256'][slot]: raise RuntimeError('Input photograph changed during run')
        if sha(flame_model)!=report['flame_model_sha256']: raise RuntimeError('FLAME model changed during run')
        if sha(repo/'weights/flame/flame_assets.npz')!=report['flame_assets_sha256']: raise RuntimeError('FLAME assets changed during run')
        report.update(status='GLB_EXPORTED',glb=str(out/'head.optimized.glb'),glb_sha256=sha(out/'head.optimized.glb'),workspace=str(workspace/'head.workspace.html'))
    except (Exception, KeyboardInterrupt) as e:
        report.update(status='FAILED',error=str(e) or 'Interrupted')
    finally:
        report.update(wall_seconds=round(time.perf_counter()-started,3),
            peak_ram_tree_mib=peak(rows,'ram_tree_mib'),peak_gpu_memory_used_mib=peak(rows,'gpu_memory_used_mib'),
            peak_gpu_utilization_pct=peak(rows,'gpu_utilization_pct'),
            peak_gpu_temperature_c=peak(rows,'gpu_temperature_c'),peak_gpu_power_w=peak(rows,'gpu_power_w'))
        if rows:
            with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (out/'pipeline.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2));print('Output: '+str(out))
    return 0 if report['status']=='GLB_EXPORTED' else 1


if __name__=='__main__':
    raise SystemExit(main())
