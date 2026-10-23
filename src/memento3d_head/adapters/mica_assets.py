"""Pin external MICA code and assets without copying restricted upstream files."""
import json
import subprocess
from pathlib import Path
from memento3d_head.adapters.head_store import sha256
from memento3d_head.domain.contracts import ValidationError

MICA_COMMIT = 'af22e7a5810d474bc28a1433db533723d6bd2b07'
REQUIRED = ('data/pretrained/mica.tar', 'data/FLAME2020/generic_model.pkl',
            'data/FLAME2020/landmark_embedding.npy', 'data/FLAME2020/FLAME_masks/FLAME_masks.pkl')


def inspect_repo(repo: Path) -> dict:
    """Check exact upstream revision and reject tracked source changes."""
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.PIPE).strip()
    commit = git('rev-parse','HEAD')
    dirty = git('status','--porcelain','--untracked-files=no')
    if commit != MICA_COMMIT or dirty:
        raise ValidationError('UPSTREAM_MISMATCH','MICA must be at the documented clean pinned commit')
    return {'commit':commit,'missing':[name for name in REQUIRED if not (repo/name).is_file()]}


def pin_assets(repo: Path, python: Path, detector: Path, destination: Path) -> dict:
    """Register hashes of user-obtained assets; never download or overwrite config."""
    state = inspect_repo(repo)
    if state['missing']:
        raise ValidationError('MISSING_ASSETS','Missing: '+', '.join(state['missing']))
    if not python.is_file() or not detector.is_file() or detector.suffix.lower() != '.onnx':
        raise ValidationError('MISSING_ASSETS','Worker Python and local detector ONNX are required')
    config = {'schema_version':1,'upstream_commit':MICA_COMMIT,'repo':str(repo.resolve()),
              'python':str(python.resolve()), 'timeout_seconds':600,
              'assets':{name:{'path':str((repo/name).resolve()),'sha256':sha256(repo/name)} for name in REQUIRED}}
    config['assets']['detector']={'path':str(detector.resolve()),'sha256':sha256(detector)}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(config,stream,indent=2)
    return config


def verify_config(path: Path) -> dict:
    """Verify local pins before starting the external model process."""
    config=json.loads(path.read_text(encoding='utf-8'))
    if config.get('schema_version') != 1 or config.get('upstream_commit') != MICA_COMMIT:
        raise ValidationError('INVALID_MICA_CONFIG','Unsupported config/upstream version')
    repo=Path(config['repo'])
    inspect_repo(repo)
    if set(config['assets']) != set(REQUIRED)|{'detector'}:
        raise ValidationError('INVALID_MICA_CONFIG','Missing or unknown model pins')
    for name,item in config['assets'].items():
        file=Path(item['path'])
        if name!='detector' and file.resolve() != (repo/name).resolve():
            raise ValidationError('INVALID_MICA_CONFIG','Asset path differs from fixed upstream layout')
        if not file.is_file() or sha256(file)!=item['sha256']:
            raise ValidationError('MODEL_CHECKSUM_MISMATCH',f'Asset changed or missing: {name}')
    if not Path(config['python']).is_file():
        raise ValidationError('WORKER_PYTHON_NOT_FOUND','External Python path does not exist')
    if type(config.get('timeout_seconds')) is not int or not 1<=config['timeout_seconds']<=3600:
        raise ValidationError('INVALID_MICA_CONFIG','Timeout must be 1..3600 seconds')
    return config
