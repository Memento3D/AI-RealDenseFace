"""Local, sequential RDF ablation; immutable source crops and fresh attempts only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def run(source, output, python):
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=False)
    crops = source / 'rdf_crops'
    files = [source / 'job.json', source / 'config.snapshot.json', crops / 'report.json']
    files += [crops / (s + '.png') for s in ('front', 'left', 'right')]
    source_hashes = {str(p): sha(p) for p in files}
    report = {'schema_version': 1, 'status': 'RUNNING', 'source': str(source),
              'source_sha256': source_hashes, 'python': str(python),
              'created_at': datetime.now(timezone.utc).isoformat(),
              'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                             (Path(__file__), ROOT/'tools/headapi/rdf_worker.py',
                              ROOT/'tools/headapi/fitting_contract.py', ROOT/'tools/headapi/face_export.py')},
              'variants': {}, 'likeness_verified': False, 'ready_for_machine': False,
              'scope': 'A/B fitting experiment only; no new silhouette loss or hair processing'}
    write(output/'experiment.json', report)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}
    try:
        for name, views in [('A_front', ['front']), ('B_three', ['front', 'left', 'right'])]:
            target = output / name
            command = [str(python), '-m', 'tools.headapi.rdf_worker', '--repo',
                       str(ROOT/'vendor/RealDenseFace'), '--crops', str(crops),
                       '--output', str(target), '--views', *views]
            print('Starting '+name, flush=True)
            started = time.monotonic()
            with (output/(name+'.log')).open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=1800)
            row = {'command': command, 'exit_code': result.returncode,
                   'process_elapsed_seconds': time.monotonic()-started}
            report['variants'][name] = row
            write(output/'experiment.json', report)
            if result.returncode:
                raise RuntimeError(name+' failed; see its log')
            with (output/(name+'.export.log')).open('w', encoding='utf-8') as log:
                subprocess.run([str(python), '-m', 'tools.headapi.face_export', '--fit',
                                str(target), '--crops', str(crops)], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
            row['fit_report'] = json.loads((target/'report.json').read_text())
            row['artifacts'] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
        import numpy as np
        from PIL import Image, ImageDraw
        with np.load(output/'A_front/projection_bundle.npz', allow_pickle=False) as a, np.load(output/'B_three/projection_bundle.npz', allow_pickle=False) as b:
            if not np.array_equal(a['faces'], b['faces']):
                raise ValueError('Topology mismatch')
            distances = np.linalg.norm(a['vertices']-b['vertices'], axis=1)
            report['shape_change'] = {'mean_native': float(distances.mean()),
                'p95_native': float(np.quantile(distances, .95)), 'max_native': float(distances.max()),
                'scope': 'Corresponding vertex displacement; expression may differ; not reconstruction accuracy or mm'}
        images = [Image.open(output/n/'rdf_face_gray.png').convert('RGB') for n in ('A_front','B_three')]
        sheet = Image.new('RGB', (max(i.width for i in images), sum(i.height+30 for i in images)), '#171c24')
        draw = ImageDraw.Draw(sheet)
        y = 0
        for label, im in zip(('A: front only', 'B: three views'), images):
            draw.text((8,y+6), label, fill='white'); sheet.paste(im, (0,y+30)); y += im.height+30
        sheet.save(output/'comparison_gray.png')
        report['comparison_note'] = 'Existing preview renderer auto-centers/scales each mesh; not a common calibrated camera or quantitative image comparison.'
        report['source_unchanged'] = all(sha(Path(p)) == h for p,h in source_hashes.items())
        if not report['source_unchanged']:
            raise ValueError('Source changed during experiment')
        report['status'] = 'EXPERIMENT_COMPLETED'
    except BaseException as error:
        report['status'] = 'FAILED'
        report['error'] = str(error)
        raise
    finally:
        write(output/'experiment.json', report)
        print(str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or ROOT/'runs/experiments'/('rdf-ab-'+uuid4().hex)
    run(args.source, destination.resolve(), Path(sys.executable).resolve())
