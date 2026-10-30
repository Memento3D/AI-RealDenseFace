import hashlib,json
from pathlib import Path

CODE_REVISION='107cefdc244c39106fa830359024f6a2f1c78871'
MODEL_REVISION='5b521936b01fbe1890f6f9baed0254ab6351c04a'
DINO_REVISION='f205d5d8e640a89a2b8ef0369670dfc37cc07fc2'
ROOT=Path(__file__).resolve().parents[2]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def check_repo(repo):
 manifest=json.loads((Path(repo)/'probe.source.json').read_text())
 if not isinstance(manifest.get('files'),dict) or not manifest['files']:raise ValueError('Empty or invalid upstream source manifest')
 if manifest['revision']!=CODE_REVISION:raise ValueError('Unexpected upstream revision')
 for name,digest in manifest['files'].items():
  p=Path(repo)/name
  if not p.resolve().is_relative_to(Path(repo).resolve()) or sha(p)!=digest:raise ValueError('Upstream source changed: '+name)
 return manifest


def source_repo(root=ROOT):
 """Resolve a verified source tree without moving Windows-locked directories."""
 vendor=Path(root)/'vendor'
 for name in ['TripoSR-probe','TripoSR-probe.download','TripoSR-probe-source']:
  candidate=vendor/name
  if not candidate.exists():continue
  if not candidate.is_dir() or candidate.is_symlink():raise ValueError('Unexpected source path: '+str(candidate))
  if not any(candidate.iterdir()):continue
  if not (candidate/'probe.source.json').is_file():raise ValueError('Existing source tree is incomplete or unverified; retained unchanged: '+str(candidate))
  check_repo(candidate)
  return candidate
 return None
