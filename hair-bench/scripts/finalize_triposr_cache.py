"""Complete an interrupted TripoSR setup from existing cached weights.
No package installs, model-weight downloads, cache deletion or inference here.
"""
import argparse,hashlib,json,os,re,subprocess,sys,tempfile,urllib.request
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify_cache_blob(path):
    path=Path(path);resolved=path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size==0:raise RuntimeError('Missing/empty cached file: '+str(path))
    fingerprint=digest(resolved);name=resolved.name
    if re.fullmatch('[0-9a-f]{64}',name) and name!=fingerprint:
        raise RuntimeError('Cached LFS blob SHA256 mismatch; file retained: '+str(resolved))
    return fingerprint

def link_existing(source,destination):
    source=Path(source).resolve(strict=True);destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        if not destination.is_file():raise RuntimeError('Existing target is not a readable file; retained: '+str(destination))
        if not destination.samefile(source) and digest(destination)!=digest(source):
            raise RuntimeError('Existing target differs; refusing overwrite: '+str(destination))
        return 'reused'
    destination.parent.mkdir(parents=True,exist_ok=True);destination.symlink_to(source)
    return 'linked'

def cached_files(state,entry):
    from huggingface_hub import hf_hub_download
    result={}
    for name in ['config.yaml','model.ckpt']:
        result[name]=Path(hf_hub_download(repo_id=entry['weights_repo'],filename=name,
            revision=entry['weights_revision'],cache_dir=str(state/'cache/huggingface/hub'),local_files_only=True))
    return result

def validate_repo(repo,entry):
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if actual!=entry['commit']:raise RuntimeError('Repository commit differs from HairBench lock; no checkout performed')
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    if dirty.strip():raise RuntimeError('Isolated repository has local changes; retained without modification')

def fetch_dino(state,entry):
    # Only a small pinned JSON config may be fetched; never model weights.
    from huggingface_hub import hf_hub_download
    try:
        source=Path(hf_hub_download(repo_id='facebook/dino-vitb16',filename='config.json',
            revision=entry['dino_config_revision'],cache_dir=str(state/'cache/huggingface/hub'),local_files_only=True))
        raw=source.read_bytes()
    except Exception as exc:
        from huggingface_hub.utils import LocalEntryNotFoundError
        if not isinstance(exc,LocalEntryNotFoundError):raise
        url='https://huggingface.co/facebook/dino-vitb16/resolve/'+entry['dino_config_revision']+'/config.json'
        print('Fetching small pinned DINO config only (not DINO weights)...',flush=True)
        with urllib.request.urlopen(url,timeout=180) as r:raw=r.read(1024*1024+1)
    if len(raw)>1024*1024:raise RuntimeError('DINO config exceeds 1 MiB safety limit')
    data=json.loads(raw)
    if not isinstance(data,dict) or not data:raise RuntimeError('Invalid DINO config JSON')
    target=state/'weights/triposr/dino-config/config.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_file() or json.loads(target.read_text())!=data:
            raise RuntimeError('Existing DINO config differs; refusing overwrite: '+str(target))
    else:
        with target.open('xb') as f:f.write(raw)
    return target

def smoke_check(repo,scripts,state):
    env=os.environ.copy();env.update(HF_HOME=str(state/'cache/huggingface'),TORCH_HOME=str(state/'cache/torch'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2')
    code="import sys,torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('GPU:',torch.cuda.get_device_name(0),flush=True); sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2]);from run_model import install_cpu_mc;install_cpu_mc();from tsr.system import TSR;print('TRIPOSR_IMPORT_CHECKED',flush=True)"
    subprocess.run([sys.executable,'-u','-c',code,str(repo),str(scripts)],env=env,check=True)

def complete(kit,state):
    kit=Path(kit).resolve();state=Path(state).resolve()
    entry=json.loads((kit/'models.lock.json').read_text())['models']['triposr']
    if entry['weights_repo']!='stabilityai/TripoSR':raise RuntimeError('Unexpected weights repository')
    python=state/'envs/triposr/bin/python';repo=state/'repos/triposr'
    if Path(sys.executable).absolute()!=python.absolute():raise RuntimeError('Run this helper with the existing TripoSR Python, not the project CPU venv')
    validate_repo(repo,entry)
    record=state/'triposr.setup.json'
    if record.exists():
        existing=json.loads(record.read_text())
        for key,value in [('python',str(python)),('repo',str(repo)),('weights',str(state/'weights/triposr')),('source_commit',entry['commit']),('weights_revision',entry['weights_revision'])]:
            if existing.get(key)!=value:raise RuntimeError('Existing setup record differs; retained: '+key)
    sources=cached_files(state,entry);fingerprints={}
    # Verify both sources before creating any links.
    for name,source in sources.items():
        print('Verifying cached',name,flush=True);fingerprints[name]=verify_cache_blob(source)
    for name,source in sources.items():print(name,link_existing(source,state/'weights/triposr'/name),flush=True)
    fetch_dino(state,entry)
    smoke_check(repo,kit/'scripts',state)
    freeze=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
    freeze_path=state/'envs/triposr/pip-freeze.txt'
    if not freeze_path.exists():freeze_path.write_text(freeze)
    result={'model':'triposr','status':'SETUP_IMPORT_CHECKED','python':str(python),'repo':str(repo),
        'weights':str(state/'weights/triposr'),'source_commit':entry['commit'],'weights_revision':entry['weights_revision'],
        'inference_verified':False,'completion_method':'existing cached checkpoint linked; existing environment import/GPU checked',
        'cached_file_sha256':fingerprints,'dino_config_revision':entry['dino_config_revision']}
    fd,tmp=tempfile.mkstemp(prefix='.triposr.setup.',suffix='.json',dir=state)
    try:
        with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2)
        os.replace(tmp,record)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print('TRIPOSR_CACHED_SETUP_READY',flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kit',type=Path,required=True)
    p.add_argument('--state',type=Path,default=Path('/home/indows/.local/share/memento-hairbench-v1'))
    a=p.parse_args()
    if a.state.resolve()!=(Path.home()/'.local/share/memento-hairbench-v1').resolve():
        raise RuntimeError('Selected WSL default user differs from the existing HairBench state owner')
    complete(a.kit,a.state)
if __name__=='__main__':main()
