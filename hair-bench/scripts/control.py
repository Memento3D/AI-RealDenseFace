"""Isolated Linux/WSL benchmark controller. No privileged installs, no credentials."""
import argparse,csv,hashlib,html,json,os,platform,shutil,subprocess,sys,time,traceback,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def host_disks():
    paths=json.loads(os.environ.get('MEMENTO_HAIRBENCH_HOST_DISKS','[]'))
    return [{'path':str(p),'free_gib':shutil.disk_usage(p).free/1024**3} for p in paths]

def check_live_disk(env=None):
    config=env if env is not None else os.environ
    paths=json.loads(config.get('MEMENTO_HAIRBENCH_HOST_DISKS','[]'))
    for path in paths:
        free=shutil.disk_usage(path).free/1024**3
        if free<3.:raise RuntimeError('BLOCKED_HOST_DISK: '+str(path)+' has '+str(round(free,2))+' GiB free; 3 GiB emergency reserve. Existing files retained.')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def probe(state):
    executable=shutil.which('nvidia-smi') or ('/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').is_file() else None)
    gpus=[];error=None
    if executable:
        p=subprocess.run([executable,'--query-gpu=index,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
        if p.returncode==0:
            for row in csv.reader(p.stdout.splitlines()):
                try:gpus.append(dict(index=int(row[0]),name=row[1].strip(),total_mib=float(row[2]),free_mib=float(row[3]),driver=row[4].strip()))
                except (ValueError,IndexError):error='nvidia-smi did not supply numeric VRAM data: '+repr(row)
        else:error=p.stderr.strip()
    available=None
    try:
        info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());available=int(info['MemAvailable'].split()[0])/1024
    except (OSError,KeyError):pass
    state=Path(state);state.mkdir(parents=True,exist_ok=True)
    physical=host_disks();virtual_free=shutil.disk_usage(state).free/1024**3
    return {'platform':platform.platform(),'is_wsl':'microsoft' in platform.release().lower(),'gpus':gpus,'gpu_error':error,
        'available_ram_mib':available,'free_disk_gib':min([virtual_free]+[d['free_gib'] for d in physical]),'virtual_disk_free_gib':virtual_free,'physical_host_disks':physical,'physical_host_disk_checked':bool(physical),'state_directory':str(state)}

def plan(hw,models):
    result=[]
    for name in models:
        if name=='difflocks':status='DEFERRED_GATED_UNVERIFIED_6GB';reason='Registration-required checkpoints, custom CUDA kernels and actual VRAM use need separate verification.'
        elif not hw['gpus']:status='BLOCKED_GPU_NOT_VISIBLE';reason='No NVIDIA GPU visible in this Linux/WSL process.'
        elif hw['gpus'][0]['total_mib']<5500:status='BLOCKED_LOW_VRAM';reason='Runner conservative policy, not an official model minimum.'
        else:status='READY_TO_ATTEMPT';reason='6GB-class trial, not a claim of verified fit or successful inference.'
        result.append({'model':name,'status':status,'reason':reason})
    return result

def terminate_group(p):
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait()

def command(args,log,cwd=None,env=None,timeout=3600):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    check_live_disk(env)
    with log.open('a') as f:
        f.write('\nCOMMAND: '+json.dumps([str(x) for x in args])+'\n');f.flush()
        print('['+log.stem+'] '+str(args[0])+'; progress log: '+str(log),flush=True)
        p=subprocess.Popen([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,cwd=cwd,env=env,start_new_session=True)
        started=time.monotonic()
        try:
            while True:
                remaining=timeout-(time.monotonic()-started)
                if remaining<=0:raise RuntimeError('Command timed out; model process group terminated. See '+str(log))
                try:
                    code=p.wait(timeout=min(15.,remaining));break
                except subprocess.TimeoutExpired:
                    check_live_disk(env)
                    print('['+log.stem+'] running '+str(int(time.monotonic()-started))+'s; log bytes='+str(log.stat().st_size),flush=True)
        except BaseException:
            terminate_group(p)
            f.write('\nINTERRUPTED_OR_BLOCKED: subprocess group terminated; cached files retained.\n');f.flush()
            raise
    if code:raise RuntimeError('Command failed ('+str(code)+'); see '+str(log))

def setup(name,entry,state,log):
    if name=='difflocks':return {'model':name,'status':'DEFERRED_GATED_UNVERIFIED_6GB','registration_url':'https://difflocks.is.tue.mpg.de/','repo':entry['repo'],'commit':entry['commit']}
    env=os.environ.copy();env.update(HF_HOME=str(state/'cache/huggingface'),TORCH_HOME=str(state/'cache/torch'),MAX_JOBS='1',OMP_NUM_THREADS='2',UV_PYTHON_INSTALL_DIR=str(state/'python'))
    uvroot=state/'bootstrap';uvpython=state/'bootstrap/bin/uv'
    if not uvpython.is_file():
        command([sys.executable,'-m','pip','install','--target',str(uvroot),'uv==0.6.14'],log,env=env)
    env['PYTHONPATH']=str(uvroot);uv=[str(uvpython)]
    venv=state/'envs'/name;python=venv/'bin/python'
    if not python.is_file():command(uv+['venv','--seed','--python','3.10',str(venv)],log,env=env)
    repo=state/'repos'/name
    if not repo.exists():command(['git','clone',entry['repo'],str(repo)],log,env=env)
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    if dirty:raise RuntimeError('Isolated repo has local changes; refusing to overwrite '+str(repo))
    if actual!=entry['commit']:command(['git','-C',str(repo),'checkout','--detach',entry['commit']],log,env=env)
    command([python,'-m','pip','install','torch==2.5.0','torchvision==0.20.0','--index-url','https://download.pytorch.org/whl/cu121'],log,env=env)
    command([python,'-m','pip','install','-r',ROOT/'requirements'/(name+'.txt')],log,env=env)
    weights=state/'weights'/name;weights.mkdir(parents=True,exist_ok=True)
    if name=='triposr':
        code="from huggingface_hub import hf_hub_download; import sys; [hf_hub_download(sys.argv[1],f,revision=sys.argv[2],local_dir=sys.argv[3],local_dir_use_symlinks=False) for f in ['config.yaml','model.ckpt']]"
    else:
        code="from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],allow_patterns=['hunyuan3d-dit-v2-mini/config.yaml','hunyuan3d-dit-v2-mini/model.fp16.safetensors'])"
    command([python,'-c',code,entry['weights_repo'],entry['weights_revision'],str(weights)],log,env=env)
    if name=='triposr':
        command([python,'-c',"from huggingface_hub import hf_hub_download;import sys;hf_hub_download('facebook/dino-vitb16','config.json',revision=sys.argv[1],local_dir=sys.argv[2],local_dir_use_symlinks=False)",entry['dino_config_revision'],str(weights/'dino-config')],log,env=env)
    command([python,'-c','import torch; assert torch.cuda.is_available(), "CUDA not visible"; print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))'],log,env=env)
    smoke=("import sys;sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2]);from run_model import install_cpu_mc;install_cpu_mc();from tsr.system import TSR" if name=='triposr' else "import sys;sys.path.insert(0,sys.argv[1]);from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline")
    command([python,'-c',smoke,str(repo),str(ROOT/'scripts')],log,env=env)
    with (state/'envs'/name/'pip-freeze.txt').open('w') as f:subprocess.run([python,'-m','pip','freeze'],stdout=f,check=True)
    result={'model':name,'status':'SETUP_IMPORT_CHECKED','python':str(python),'repo':str(repo),'weights':str(weights),'source_commit':entry['commit'],'weights_revision':entry['weights_revision'],'inference_verified':False}
    (state/(name+'.setup.json')).write_text(json.dumps(result,indent=2));return result

def inference_environment(mode,state):
    env=os.environ.copy();env.update(HF_HOME=str(state/'cache/huggingface'),TORCH_HOME=str(state/'cache/torch'),OMP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if mode=='run':env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return env


def render_report(out,data):
    data['any_inference_succeeded']=any(x.get('status')=='INFERRED_NATIVE_MESH' for x in data.get('runs',[]))
    esc=html.escape;rows=''
    for r in (data.get('runs') or data['plan']):rows+='<tr><td>'+esc(r['model'])+'</td><td>'+esc(r['status'])+'</td><td>'+esc(str(r.get('error',r.get('reason',''))))+'</td><td>'+esc(str(r.get('seconds','')))+'</td></tr>'
    text='<meta charset="utf-8"><title>Hair benchmark</title><style>body{background:#17232d;color:#eef;font:16px system-ui;margin:30px}td,th{padding:12px;border:1px solid #567}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>Benchmark tóc — mặt nguồn cố định</h1><p>READY_TO_ATTEMPT / SETUP_IMPORT_CHECKED không phải kết quả inference. Native mesh chưa có nghĩa đã fit hoặc in được.</p><table><tr><th>Model</th><th>Trạng thái thực</th><th>Chi tiết</th><th>Giây</th></tr>'+rows+'</table><pre>'+esc(json.dumps(data,indent=2,ensure_ascii=False))+'</pre>'
    (out/'report.html').write_text(text,encoding='utf-8');(out/'report.json').write_text(json.dumps(data,indent=2))

def normalize_subject(manifest):
    """Convert only path fields; never rewrite the Windows-side source manifest."""
    result=dict(manifest)
    for key in ['source_run','portrait','hair_rgb','hair_rgba','face_reference','viewer_project','viewer_run']:
        value=result.get(key)
        if isinstance(value,str) and len(value)>2 and value[1]==':':
            result[key]=subprocess.check_output(['wslpath','-a','-u',value],text=True).strip()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args();r=json.loads(a.request.read_text(encoding='utf-8-sig'));out=Path(r['output_linux']);out.mkdir(parents=True,exist_ok=True)
    state=Path(r.get('state_linux') or Path.home()/'.local/share/memento-hairbench-v1');os.environ['MEMENTO_HAIRBENCH_HOST_DISKS']=json.dumps(r.get('host_disks_linux',[]));hw=probe(state);lock=json.loads((ROOT/'models.lock.json').read_text())['models'];models=r.get('models',['triposr','hunyuan-mini','difflocks'])
    if any(x not in lock for x in models):raise ValueError('Unknown model ID')
    data={'hardware':hw,'windows_hardware':r.get('windows_hardware'),'plan':plan(hw,models),'runs':[],'mode':r['mode'],'development_host_gpu_inference_verified':False,'ready_for_machine':False}
    data['existing_setup']={name:{'environment_exists':(state/'envs'/name/'bin/python').is_file(),'setup_checked':(state/(name+'.setup.json')).is_file()} for name in models}
    (out/'hardware.json').write_text(json.dumps(hw,indent=2));render_report(out,data)
    if r['mode']=='inspect':print(json.dumps(data,indent=2));return 0
    if not hw['gpus']:print('BLOCKED: NVIDIA GPU not visible in WSL; no GPU packages installed.');return 2
    minimum_disk=10 if r['mode'] in ['setup','all'] else 5
    data['disk_policy']={'start_free_gib':minimum_disk,'emergency_reserve_gib':3,'scope':'conservative runner policy, not a guarantee of model storage needs'}
    if not hw['physical_host_disk_checked']:
        data['error']='Cannot establish physical host-disk capacity. Use RUN-HAIR-BENCH.ps1 from patch 1.0.1.';render_report(out,data);return 2
    if hw['free_disk_gib']<minimum_disk or (hw['available_ram_mib'] is not None and hw['available_ram_mib']<5000):
        data['error']='Insufficient free RAM/disk under runner policy: 5 GiB RAM / '+str(minimum_disk)+' GiB effective disk. Cached files retained.';render_report(out,data);return 2
    subject=Path(r['subject_linux']) if r.get('subject_linux') else None
    manifest=normalize_subject(json.loads(subject.read_text(encoding='utf-8-sig'))) if subject and subject.is_file() else None
    if manifest:
        subject=out/'subject.linux.json';subject.write_text(json.dumps(manifest,indent=2))
        data['subject_manifest']=str(subject)
    if r['mode'] in ['run','all']:
        if not manifest:raise ValueError('Prepared subject.json is required')
        face=Path(manifest['face_reference']);gold=manifest['face_sha256']
        if sha(face)!=gold:raise ValueError('Original face changed before benchmark')
    for item in data['plan']:
        name=item['model'];log=out/(name+'.log')
        if item['status']=='READY_TO_ATTEMPT' and probe(state)['free_disk_gib']<minimum_disk:
            data['runs'].append({'model':name,'status':'BLOCKED_HOST_DISK','error':'Not enough effective disk for the next model; queue stopped.','ready_for_machine':False});render_report(out,data);return 2
        if item['status']!='READY_TO_ATTEMPT':data['runs'].append(item);continue
        try:
            if r['mode'] in ['setup','all']:setup_info=setup(name,lock[name],state,log)
            else:setup_info=json.loads((state/(name+'.setup.json')).read_text())
            if r['mode']=='setup':data['runs'].append(setup_info);continue
            free=probe(state)['gpus']
            if not free or free[0]['free_mib']<4500:raise RuntimeError('Less than 4500 MiB free GPU memory under benchmark policy; close other GPU applications')
            folder=out/name
            env=inference_environment(r['mode'],state)
            command([setup_info['python'],ROOT/'scripts/run_model.py','--model',name,'--repo',setup_info['repo'],'--weights',setup_info['weights'],'--subject',subject,'--output',folder],log,env=env,timeout=1800)
            result=json.loads((folder/'result.json').read_text());data['runs'].append(result)
            if result['status']=='INFERRED_NATIVE_MESH':
                target=Path(manifest['viewer_project'])/'assets/hair'/(name+'.glb');shutil.copy2(folder/'native.glb',target)
                target.with_suffix('.hair.json').write_text(json.dumps({'name':name+' — generated','up_axis':'Z' if name=='triposr' else 'Y'}))
        except KeyboardInterrupt:
            data['runs'].append({'model':name,'status':'INTERRUPTED','error':'Cancelled by user; current subprocess group terminated, cached files retained.','ready_for_machine':False})
            data['interrupted']=True;render_report(out,data);print('Cancelled; cached files retained. Report: '+str(out/'report.html'),flush=True);return 130
        except Exception as exc:
            actual=out/name/'result.json'
            data['runs'].append(json.loads(actual.read_text()) if actual.is_file() else {'model':name,'status':'FAILED_OR_BLOCKED','error':str(exc),'ready_for_machine':False})
        finally:
            if manifest and sha(Path(manifest['face_reference']))!=manifest['face_sha256']:raise RuntimeError('Original face invariant failed')
            render_report(out,data)
    render_report(out,data);print(json.dumps({'report':str(out/'report.html'),'runs':data['runs']},indent=2));return 0 if any(x['status'] in ['SETUP_IMPORT_CHECKED','INFERRED_NATIVE_MESH'] for x in data['runs']) else 2
if __name__=='__main__':raise SystemExit(main())
