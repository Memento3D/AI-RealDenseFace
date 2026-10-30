"""Forward --no_compile to FaceDetector in the pinned upstream checkout."""
import argparse
from pathlib import Path
import subprocess

PIN = 'ebf0063835f520f07fdf5a813b1af9cdc0ae46b5'
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=ROOT / 'vendor/RealDenseFace')
    args = p.parse_args()
    commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN:
        raise SystemExit('REFUSED: unexpected commit; no file changed')
    path = args.repo / 'tracker/inference/realdenseface_inferencer.py'
    source = path.read_bytes()
    newline = b'\r\n' if b'\r\n' in source else b'\n'
    old = b'            device=config.device,\n        )'.replace(b'\n', newline)
    new = b'            device=config.device,\n            compile_model=self.config.compile_model,\n        )'.replace(b'\n', newline)
    if source.count(new) == 1:
        print('ALREADY_PATCHED')
        return
    if source.count(old) != 1:
        raise SystemExit('REFUSED: source block differs; no file changed')
    path.write_bytes(source.replace(old, new, 1))
    print('PATCHED: FaceDetector respects --no_compile')


if __name__ == '__main__':
    main()
