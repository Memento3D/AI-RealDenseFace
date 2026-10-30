"""Durable local job orchestration. One executor, subprocess environments kept separate."""
import copy
import hashlib
import html
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[2]
SLOTS=('front','left','right')
TERMINAL={'MVP_ASSEMBLED','COMPONENTS_EXPORTED','RAW_EXPORTED','PARTIAL','FAILED','INTERRUPTED'}


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path,root=ROOT):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data.get('schema_version')!=1:raise ValueError('Unsupported API config')
    data=copy.deepcopy(data)
    for name,value in data['paths'].items():
        p=Path(value);data['paths'][name]=str(p if p.is_absolute() else (root/p).resolve())
    data['root']=str(Path(root).resolve())
    return data


class ProcessLock:
    """OS advisory lock releases on process death; never kill another process."""
    def __init__(self,path):self.path=Path(path);self.file=None
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open('a+b')
        self.file.seek(0);self.file.write(b'0');self.file.flush();self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt;msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();self.file=None;raise RuntimeError('Another Head API owns this jobs directory')
    def release(self):
        if self.file:self.file.close();self.file=None


class JobService:
    def __init__(self,config,executor=None):
        self.config=copy.deepcopy(config);self.root=Path(config['root'])
        self.jobs=Path(config['paths']['jobs']);self.jobs.mkdir(parents=True,exist_ok=True)
        self.executor=executor or ThreadPoolExecutor(max_workers=1,thread_name_prefix='headapi')
        self.mutex=threading.RLock();self.lock=ProcessLock(self.jobs/'api.lock')
    def start(self):
        self.lock.acquire()
        for path in self.jobs.glob('*/job.json'):
            row=json.loads(path.read_text())
            if row.get('status') not in TERMINAL:
                row.update(status='INTERRUPTED',error='API process stopped; submit a new attempt; old files retained')
                write(path,row)
    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=False);self.lock.release()
    def preflight(self,mode):
        names=['cpu_python','probe_python','landmarker']
        if mode in ('components','mvp'):names+=['rdf_python','rdf_repo']
        if mode in ('mvp','assembly'):
            if not self.config['paths'].get('parser'):raise ValueError('Missing parser config; update configs/headapi.json')
            names+=['parser']
        missing=[name for name in names if not Path(self.config['paths'][name]).exists()]
        if mode in ('components','mvp'):
            repo=Path(self.config['paths']['rdf_repo'])
            missing += ['RDF/'+rel for rel in ('weights/flame/flame2023.pkl','weights/flame/flame_assets.npz','weights/realdenseface/vitb.pth','weights/facebox/face_box.pth') if not (repo/rel).is_file()]
        if missing:raise ValueError('Missing prerequisites: '+', '.join(missing))
    def get(self,job_id):
        if not re.fullmatch(r'[0-9a-f]{32}',job_id):raise KeyError(job_id)
        p=self.jobs/job_id/'job.json'
        if not p.is_file():raise KeyError(job_id)
        with self.mutex:return json.loads(p.read_text(encoding='utf-8'))
    def update(self,job_id,**changes):
        with self.mutex:
            row=self.get(job_id);row.update(changes);write(self.jobs/job_id/'job.json',row)
        return row
    def submit(self,images,mode,resolution):
        if mode not in ('raw','components','mvp'):raise ValueError('Invalid mode')
        if resolution not in (128,192,256):raise ValueError('Invalid resolution')
        if 'front' not in images or not set(images)<=set(SLOTS):raise ValueError('FRONT_REQUIRED / invalid slots')
        self.preflight(mode)
        with self.mutex:
            active=sum(json.loads(p.read_text()).get('status') not in TERMINAL for p in self.jobs.glob('*/job.json'))
            if active>=self.config.get('max_pending_jobs',3):raise OverflowError('QUEUE_FULL')
            job_id=uuid4().hex;folder=self.jobs/job_id;folder.mkdir()
            inputs=folder/'inputs';inputs.mkdir();observations=[]
            for slot in SLOTS:
                if slot not in images:continue
                entry=images[slot];path=inputs/(slot+'.png');path.write_bytes(entry['png'])
                original=folder/'uploads';original.mkdir(exist_ok=True)
                (original/(slot+'.'+entry['format'].lower())).write_bytes(entry['original'])
                observations.append({'slot':slot,'original_sha256':hashlib.sha256(entry['original']).hexdigest(),
                    'normalized_sha256':sha(path),'normalized_size':entry['size'],
                    'normalization':'EXIF transpose and lossless RGB PNG; source upload retained',
                    'used_by_triposr':slot=='front','requested_for_rdf':mode in ('components','mvp')})
            row={'job_id':job_id,'status':'QUEUED','mode':mode,'resolution':resolution,
                 'created_at':datetime.now(timezone.utc).isoformat(),'observations':observations,
                 'stages':[],'artifacts':[],'errors':[],'ready_for_machine':False,
                 'hair_isolated':False,'hybrid_assembly_performed':False,'face_identity_verified':False,
                 'profile_images_condition_triposr':False,'rdf_views_actually_used':[],
                 'code_sha256':self.code_hashes()}
            write(folder/'config.snapshot.json',self.config);write(folder/'job.json',row)
            self.executor.submit(self.run,job_id)
        return row
    def submit_assembly(self,parent_id):
        self.preflight('assembly')
        with self.mutex:
            parent=self.get(parent_id)
            if parent['status'] not in TERMINAL:raise ValueError('Source job must finish before assembly')
            folder=self.jobs/parent_id
            for name in ('rdf/projection_bundle.npz','rdf/report.json','rdf_crops/report.json','raw/raw_native.glb','raw/report.json','observation/report.json'):
                if not (folder/name).is_file():raise ValueError('Source job missing '+name)
            if json.loads((folder/'rdf/report.json').read_text()).get('status')!='FIT_GATES_PASSED':raise ValueError('RDF source job failed')
            if json.loads((folder/'raw/report.json').read_text()).get('status')!='RAW_PROPOSAL_EXPORTED':raise ValueError('TripoSR source job failed')
            active=sum(json.loads(p.read_text()).get('status') not in TERMINAL for p in self.jobs.glob('*/job.json'))
            if active>=self.config.get('max_pending_jobs',3):raise OverflowError('QUEUE_FULL')
            job_id=uuid4().hex;out=self.jobs/job_id;out.mkdir()
            row={'job_id':job_id,'parent_job_id':parent_id,'job_type':'assembly_only','mode':'mvp','status':'QUEUED',
                'resolution':parent['resolution'],'created_at':datetime.now(timezone.utc).isoformat(),
                'observations':copy.deepcopy(parent['observations']),'rdf_views_actually_used':parent['rdf_views_actually_used'],
                'stages':[],'errors':[],'artifacts':[],'code_sha256':self.code_hashes(),
                'ready_for_machine':False,'hair_isolated':False,'hybrid_assembly_performed':False,'face_identity_verified':False,
                'profile_images_condition_triposr':False}
            write(out/'job.json',row);write(out/'config.snapshot.json',self.config)
            self.executor.submit(self.run_assembly,job_id)
        return row
    def assemble_stages(self,job_id,source):
        c=self.config['paths'];out=self.jobs/job_id/'fusion'
        steps=[('06_mvp_render',[c['probe_python'],'-m','tools.headapi.fusion','--job',source,'--output',out,'--phase','prepare']),
               ('07_mvp_landmarks',[c['cpu_python'],'-m','tools.headapi.fusion_landmarks','--image',out/'registration_front.png','--task',c['landmarker'],'--output',out/'raw_landmarks.json']),
               ('08_mvp_assembly',[c['probe_python'],'-m','tools.headapi.fusion','--job',source,'--output',out,'--phase','assemble','--parser',c['parser']])]
        for name,args in steps:
            if not self.stage(job_id,name,args):return False
        result=json.loads((out/'assembly.report.json').read_text())
        if result.get('status')!='MVP_ASSEMBLED' or not result.get('rdf_geometry_unchanged'):raise ValueError('Assembly output contract failed')
        self.update(job_id,hybrid_assembly_performed=True,hair_isolated=True,hair_segmentation_verified=False,rdf_geometry_unchanged=True)
        return True
    def run_assembly(self,job_id):
        self.update(job_id,status='RUNNING')
        try:
            job=self.get(job_id)
            if self.code_hashes()!=job['code_sha256']:raise ValueError('Pipeline code changed after submission')
            success=self.assemble_stages(job_id,self.jobs/job['parent_job_id'])
            if self.code_hashes()!=job['code_sha256']:raise ValueError('Pipeline code changed during assembly')
            self.update(job_id,status='MVP_ASSEMBLED' if success else 'FAILED',current_stage=None)
        except Exception as exc:
            row=self.get(job_id);row['errors'].append({'stage':row.get('current_stage'),'error':str(exc)})
            self.update(job_id,status='FAILED',errors=row['errors'],current_stage=None)
        finally:
            try:self.review_page(job_id)
            finally:self.update(job_id,artifacts=self.collect(job_id),finished_at=datetime.now(timezone.utc).isoformat())
    def code_hashes(self):
        files=[]
        for sub in ('tools/headapi','tools/triposr_probe'):
            files.extend((self.root/sub).glob('*.py'))
        for name in ('head24_policy.py','head24_detection.py','detect_head24_profile.py','eye_components.py'):
            path=self.root/'scripts'/name
            if path.is_file():files.append(path)
        return {p.relative_to(self.root).as_posix():sha(p) for p in sorted(files)}
    def environment(self):
        env=os.environ.copy();cache=Path(self.config['paths']['cache_root'])
        for key,sub in {'HF_HOME':'hf','HF_HUB_CACHE':'hub','HUGGINGFACE_HUB_CACHE':'hub',
                         'TRANSFORMERS_CACHE':'transformers','TORCH_HOME':'torch','U2NET_HOME':'rembg',
                         'TEMP':'temp','TMP':'temp'}.items():
            target=cache/sub;target.mkdir(parents=True,exist_ok=True);env[key]=str(target)
        env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
        return env
    def stage(self,job_id,name,args):
        folder=self.jobs/job_id;log=folder/(name+'.log');row=self.get(job_id)
        row['stages'].append({'name':name,'status':'RUNNING'});self.update(job_id,current_stage=name,stages=row['stages'])
        try:
            with log.open('w',encoding='utf-8') as stream:
                done=subprocess.run([str(v) for v in args],cwd=self.root,env=self.environment(),
                    stdout=stream,stderr=subprocess.STDOUT,timeout=self.config.get('stage_timeout_seconds',1800),check=False)
            code=done.returncode;error=None if code==0 else 'Stage exited '+str(code)
        except subprocess.TimeoutExpired:code=-1;error='STAGE_TIMEOUT'
        except Exception as exc:code=-1;error=str(exc)
        row=self.get(job_id);row['stages'][-1].update(status='EXPORTED' if code==0 else 'FAILED',exit_code=code,log=log.name)
        if error:row['errors'].append({'stage':name,'error':error,'log':log.name})
        self.update(job_id,stages=row['stages'],errors=row['errors'])
        return code==0
    def collect(self,job_id):
        folder=self.jobs/job_id;allowed={'.glb','.obj','.png','.jpg','.json','.html','.log','.npz'}
        rows=[]
        for path in sorted(folder.rglob('*')):
            rel=path.relative_to(folder)
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in allowed:continue
            if rel.parts[0] in ('inputs','uploads') or rel.as_posix() in ('job.json','config.snapshot.json'):continue
            rows.append({'path':rel.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
        return rows
    def run(self,job_id):
        folder=self.jobs/job_id;c=self.config['paths'];job=self.get(job_id);rdf_ok=False;raw_ok=False
        self.update(job_id,status='RUNNING')
        try:
            for entry in job['observations']:
                if sha(folder/'inputs'/(entry['slot']+'.png'))!=entry['normalized_sha256']:
                    raise ValueError('Input changed after upload: '+entry['slot'])
            if self.code_hashes()!=job['code_sha256']:raise ValueError('Pipeline code changed after submission; submit a new attempt')
            if job['mode'] in ('components','mvp'):
                rdf_ok=self.stage(job_id,'01_rdf_observations',[c['cpu_python'],'-m','tools.headapi.prepare_faces',
                    '--input',folder/'inputs','--task',c['landmarker'],'--output',folder/'rdf_crops','--rdf-python',c['rdf_python'],'--repo',c['rdf_repo']])
                if rdf_ok:
                    rdf_ok=self.stage(job_id,'02_rdf_fit',[c['rdf_python'],'-m','tools.headapi.rdf_worker',
                        '--repo',c['rdf_repo'],'--crops',folder/'rdf_crops','--output',folder/'rdf'])
                if rdf_ok:
                    report=json.loads((folder/'rdf/report.json').read_text())
                    if report['status']!='FIT_GATES_PASSED':raise ValueError('RDF did not pass its gates')
                    self.update(job_id,rdf_views_actually_used=report['views_used'])
                    rdf_ok=self.stage(job_id,'02b_rdf_export',[c['probe_python'],'-m','tools.headapi.face_export',
                        '--fit',folder/'rdf','--crops',folder/'rdf_crops'])
            crop_ok=self.stage(job_id,'03_head_observation',[c['cpu_python'],'-m','tools.triposr_probe.prepare',
                '--source',folder/'inputs/front.png','--task',c['landmarker'],'--output',folder/'observation'])
            if crop_ok:
                raw_ok=self.stage(job_id,'04_triposr',[c['probe_python'],'-m','tools.triposr_probe.worker',
                    '--image',folder/'observation/head_crop.png','--output',folder/'raw','--resolution',job['resolution']])
            if raw_ok:
                r=json.loads((folder/'raw/report.json').read_text())
                if r['status']!='RAW_PROPOSAL_EXPORTED':raise ValueError('TripoSR did not export')
                raw_ok=self.stage(job_id,'05_audit',[c['probe_python'],'-m','tools.triposr_probe.audit',
                    '--mesh',folder/'raw/raw_native.glb','--report',folder/'raw/report.json','--output',folder/'audit'])
            status=('COMPONENTS_EXPORTED' if rdf_ok and raw_ok else 'PARTIAL' if rdf_ok or raw_ok else 'FAILED') if job['mode'] in ('components','mvp') else ('RAW_EXPORTED' if raw_ok else 'FAILED')
            if job['mode']=='mvp' and rdf_ok and raw_ok:
                status='MVP_ASSEMBLED' if self.assemble_stages(job_id,folder) else 'PARTIAL'
            if self.code_hashes()!=job['code_sha256']:raise ValueError('Pipeline code changed during job')
            self.update(job_id,status=status,current_stage=None)
        except Exception as exc:
            row=self.get(job_id);row['errors'].append({'stage':row.get('current_stage'),'error':str(exc)})
            self.update(job_id,status='FAILED',errors=row['errors'],current_stage=None)
        finally:
            try:self.review_page(job_id)
            finally:self.update(job_id,artifacts=self.collect(job_id),finished_at=datetime.now(timezone.utc).isoformat())
    def review_page(self,job_id):
        folder=self.jobs/job_id;row=self.get(job_id)
        text='<meta charset="utf-8"><title>Head API result</title><style>body{background:#171c24;color:white;font:16px system-ui;margin:24px}img{max-width:100%}a{color:#acf}</style>'
        text+='<h1>'+html.escape(row['status'])+'</h1><p>'+('MVP: RDF head + aligned TripoSR hair surface. Not a manufacturing solid or identity guarantee.' if row.get('hybrid_assembly_performed') else 'TripoSR head and RDF face are separate components; assembly not completed.')+'</p>'
        if (folder/'fusion/mvp_head.glb').is_file():text+='<p><a href="fusion/mvp_head.glb">MVP GLB</a> · <a href="fusion/review.html">Assembly review</a></p>'
        for label,relative in [('MVP assembled head','fusion/mvp_color.png'),('MVP gray geometry','fusion/mvp_gray.png'),('TripoSR front-conditioned head','raw/raw_preview.png'),('RDF geometry','rdf/rdf_face_gray.png'),('RDF photographic color approximation','rdf/rdf_face_photo.png')]:
            if (folder/relative).is_file():text+='<h2>'+label+'</h2><img src="'+relative+'">'
        text+='<p>'+('MVP uses solved landmark alignment. Source diagnostic previews retain their own cameras; export coordinates are RDF native.' if row.get('hybrid_assembly_performed') else 'RDF previews use a display-only axis permutation; exports retain RDF coordinates. Cameras are not jointly registered.')+'</p>'
        text+='<pre>'+html.escape(json.dumps({'observations':row['observations'],'rdf_views_actually_used':row['rdf_views_actually_used'],'errors':row['errors']},indent=2))+'</pre>'
        (folder/'review.html').write_text(text,encoding='utf-8')
    def logs(self,job_id,stage=None):
        row=self.get(job_id);stage=stage or row.get('current_stage')
        if stage not in {s['name'] for s in row['stages']}:raise KeyError('Unknown stage')
        path=self.jobs/job_id/(stage+'.log')
        if not path.is_file():return ''
        with path.open('rb') as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-65536))
            return stream.read().decode('utf-8',errors='replace')
    def artifact(self,job_id,relative):
        row=self.get(job_id)
        if relative not in {a['path'] for a in row['artifacts']}:raise KeyError(relative)
        root=(self.jobs/job_id).resolve();path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise KeyError(relative)
        return path
