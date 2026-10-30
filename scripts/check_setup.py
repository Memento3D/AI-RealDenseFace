"""Read-only imports, assets and CUDA matmul checks; no face inference."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PIN = 'ebf0063835f520f07fdf5a813b1af9cdc0ae46b5'
EXPECTED = {
    'flame/flame2023.pkl': '8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0',
    'flame/flame_assets.npz': '922cb76f3afef0c2e9801a63991de269a98e6b67b1728232493f851589151dd3',
}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)
    return message


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment', choices=('main', 'rdf'), required=True)
    p.add_argument('--repo', type=Path, default=ROOT / 'vendor/RealDenseFace')
    p.add_argument('--parser-model', type=Path, default=Path('D:/AI-cache/face-parsing/resnet18.onnx'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.output and args.output.exists():
        p.error('output exists; choose a new path')
    checks = []

    def check(name, fn):
        try:
            checks.append({'name': name, 'ok': True, 'detail': fn()})
        except Exception as exc:
            checks.append({'name': name, 'ok': False, 'detail': f'{type(exc).__name__}: {exc}'})

    check('python', lambda: require(sys.version_info[:2] == (3, 11), 'documented baseline: Python 3.11'))
    modules = ['numpy', 'PIL', 'cv2']
    if args.environment == 'main':
        sys.path.insert(0, str(ROOT / 'src'))
        modules += ['memento3d_head', 'mediapipe', 'onnxruntime', 'psutil']
    else:
        sys.path.insert(0, str(args.repo.resolve()))
        modules += ['torch', 'torchvision', 'scipy', 'timm', 'roma', 'yaml', 'tqdm', 'trimesh',
                    'common.fix_chumpy', 'chumpy', 'nvdiffrast.torch', 'flame_solver.cuda_ext']
    for name in modules:
        check('import:' + name, lambda name=name: str(getattr(importlib.import_module(name), '__version__', 'import OK')))
    check('pip_check', lambda: subprocess.check_output([sys.executable, '-m', 'pip', 'check'], text=True, stderr=subprocess.STDOUT).strip())
    if args.environment == 'rdf':
        check('upstream_commit', lambda: require(subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip() == PIN, 'pinned upstream'))
        check('windows_patch', lambda: require('compile_model=self.config.compile_model' in (args.repo / 'tracker/inference/realdenseface_inferencer.py').read_text(encoding='utf-8-sig'), 'compile forwarding present'))

        def cuda():
            import torch
            require(torch.cuda.is_available(), 'CUDA unavailable')
            a = torch.ones((32, 32), device='cuda')
            require((a @ a)[0, 0].item() == 32, 'CUDA matmul failed')
            torch.cuda.synchronize()
            return {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
        check('cuda_matmul', cuda)

        def dino():
            import timm
            return require(bool(timm.list_models('*dinov3*')), 'DINOv3 definitions available')
        check('dinov3', dino)
        for rel in ['facebox/face_box.pth', 'realdenseface/vitb.pth', 'realdenseface/vits.pth', *EXPECTED]:
            def asset(rel=rel):
                path = args.repo / 'weights' / rel
                require(path.stat().st_size > 0, 'empty asset')
                sha = digest(path)
                if rel in EXPECTED:
                    require(sha == EXPECTED[rel], 'asset differs from documented model/assets hash')
                return {'sha256': sha, 'reference_hash_checked': rel in EXPECTED}
            check('asset:' + rel, asset)
    else:
        def parser_asset():
            from memento3d_head.adapters.automatic_face_masks import MODEL_SHA256
            return require(digest(args.parser_model) == MODEL_SHA256, 'face parser checksum matches')
        check('face_parser_model', parser_asset)
    report = {'status': 'SETUP_CHECK_PASSED' if all(c['ok'] for c in checks) else 'SETUP_REQUIRED',
              'environment': args.environment, 'python': sys.executable, 'checks': checks,
              'inference_tested': False, 'fresh_windows_install_verified': False}
    body = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            f.write(body + '\n')
    print(body)
    return 0 if report['status'] == 'SETUP_CHECK_PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
