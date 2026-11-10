"""Offline integrity/release checks. No network calls or storage credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('Unsafe relative asset path')
    path = root/value
    cursor = path
    while cursor != root:
        if cursor.is_symlink() or (cursor.exists() and getattr(cursor.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Linked asset path is not allowed')
        cursor = cursor.parent
    path = path.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or external asset file: '+str(relative))
    return path


def audit(root):
    root = Path(root).resolve()
    inventory = json.loads(contained(root,'inventory.json').read_text(encoding='utf-8-sig'))
    files = inventory.get('Files', inventory.get('files', []))
    if not files:
        raise ValueError('Empty asset inventory')
    for entry in files:
        file = contained(root,entry.get('Path',entry.get('path')))
        if sha(file) != entry.get('SHA256',entry.get('sha256')):
            raise ValueError('Checksum mismatch: '+str(file.relative_to(root)))
    catalog = json.loads(contained(root,'catalog.json').read_text(encoding='utf-8-sig'))
    if catalog.get('schema_version') != 1 or not catalog.get('assets'):
        raise ValueError('Unsupported or empty catalog')
    results = []
    seen = set()
    for asset in catalog['assets']:
        key = (asset['id'], asset['version'])
        if key in seen:
            raise ValueError('Duplicate asset version')
        seen.add(key)
        for part in key:
            if not part or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in part):
                raise ValueError('Invalid asset identifier')
        relative = f'library/{key[0]}/{key[1]}/fitted-reference.glb'
        file = contained(root,relative)
        if sha(file) != asset['reference']['sha256']:
            raise ValueError('Reference asset checksum mismatch')
        with file.open('rb') as stream:
            magic,version,total = struct.unpack('<III',stream.read(12))
            if magic != 0x46546C67 or version != 2 or total != file.stat().st_size:
                raise ValueError('Invalid GLB header')
            length,kind = struct.unpack('<II',stream.read(8))
            if kind != 0x4E4F534A or length > total-20:
                raise ValueError('Invalid GLB JSON')
            glb = json.loads(stream.read(length))
        names = [n.get('name','') for n in glb.get('nodes',[]) if 'mesh' in n]
        if not names or any('Personalized' in name or 'Body' in name for name in names):
            raise ValueError('Shared accessory unexpectedly contains head/body')
        blockers = []
        if asset.get('status') != 'APPROVED_PREVIEW' or asset.get('published'):
            blockers.append('Candidate has not completed the explicit preview release workflow')
        if asset.get('reference',{}).get('placement_scope','').startswith('Existing single-head'):
            blockers.append('Placement is specific to the local reference head')
        if asset.get('license_status'):
            blockers.append(asset['license_status'])
        blockers.extend(asset.get('limitations',[]))
        results.append({'id':key[0], 'version':key[1], 'bytes':file.stat().st_size,
                        'preview_release_blockers':blockers,
                        'print_ready':bool(asset.get('print_ready',False))})
    return {'integrity':'PASS','verified_files':len(files),'assets':results,
            'publishable':all(not item['preview_release_blockers'] for item in results),
            'scope':'Local integrity and explicit metadata gates only; no inference, collision or print certification'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--require-publishable',action='store_true')
    args = parser.parse_args()
    report = audit(args.library)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if args.require_publishable and not report['publishable']:
        raise SystemExit(2)


if __name__ == '__main__':main()
