"""Archive only hash-verified root files; leave source/model/data paths untouched."""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def verify_runtime(root):
    manifest = json.loads((root / 'pipeline-v23-code.json').read_text(encoding='utf-8-sig'))
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file() or sha(path) != entry['sha256']:
            raise ValueError('V2.3 release mismatch: ' + entry['path'])


def load_plan(root, plan_path):
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    for entry in plan['files']:
        name = entry['path']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Only root file names are allowed')
        if Path(name).suffix.lower() not in ('.ps1', '.md', '.txt'):
            raise ValueError('Unsupported archive file type')
        if entry['category'] not in ('docs', 'launchers'):
            raise ValueError('Invalid category')
        p = root / name
        if not p.is_file() or sha(p) != entry['sha256']:
            raise ValueError('File changed/missing; nothing moved: ' + name)
    return plan


def restore(root, archive):
    archive = archive.resolve()
    if archive.parent != (root / '_archive').resolve():
        raise ValueError('Rollback directory must be directly inside project/_archive')
    journal = json.loads((archive / 'journal.json').read_text(encoding='utf-8'))
    for entry in journal['files']:
        src = archive / entry['category'] / entry['path']
        dst = root / entry['path']
        if src.exists():
            if sha(src) != entry['sha256'] or (dst.exists() and (not dst.is_file() or sha(dst) != entry['sha256'])):
                raise ValueError('Rollback conflict: ' + entry['path'])
        elif not dst.is_file() or sha(dst) != entry['sha256']:
            raise ValueError('Missing/changed rollback file: ' + entry['path'])
    for entry in journal['files']:
        src = archive / entry['category'] / entry['path']
        if src.exists() and not (root / entry['path']).exists():
            src.rename(root / entry['path'])
    journal['status'] = 'RESTORED'
    write_json(archive / 'journal.json', journal)
    print('RESTORED:', archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--rollback', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.rollback:
        restore(root, args.rollback)
        return
    verify_runtime(root)
    plan = load_plan(root, Path(__file__).with_name('cleanup-plan.json'))
    for entry in plan['files']:
        print(entry['category'] + ': ' + entry['path'])
    print('FILES:', len(plan['files']))
    if not args.apply:
        print('PREVIEW ONLY. Use -Apply to archive these files.')
        return
    archive = root / '_archive' / ('source_cleanup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    archive.mkdir(parents=True, exist_ok=False)
    journal = {'status': 'MOVING', 'files': plan['files']}
    write_json(archive / 'journal.json', journal)
    try:
        for entry in plan['files']:
            target = archive / entry['category'] / entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            (root / entry['path']).rename(target)
        verify_runtime(root)
        journal['status'] = 'ARCHIVED'
        write_json(archive / 'journal.json', journal)
    except BaseException:
        restore(root, archive)
        raise
    print('ARCHIVED:', archive)
    print('Legacy launchers are inactive while archived; restore before running them.')
    print('No Python, configs, models, environments, inputs or runs were removed.')


if __name__ == '__main__':
    main()
