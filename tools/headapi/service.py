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
from uuid import UUID, uuid4

ROOT=Path(__file__).resolve().parents[2]
SLOTS=('front','left','right','back')
RDF_SLOTS=('front','left','right')
TERMINAL={'MVP_EXPORTED','MVP_ASSEMBLED','COMPONENTS_EXPORTED','RAW_EXPORTED','PARTIAL','FAILED','INTERRUPTED'}


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
        if mode!='mvp':raise ValueError('Only the RDF + hair-cap MVP is supported')
        names=['cpu_python','rdf_python','haircap_python','rdf_repo','landmarker','parser']
        missing=[name for name in names if not Path(self.config['paths'][name]).exists()]
        repo=Path(self.config['paths']['rdf_repo'])
        missing += ['RDF/'+rel for rel in ('weights/flame/flame2023.pkl','weights/flame/flame_assets.npz','weights/realdenseface/vitb.pth','weights/facebox/face_box.pth') if not (repo/rel).is_file()]
        if missing:raise ValueError('Missing prerequisites: '+', '.join(missing))
    def get(self,job_id):
        if not isinstance(job_id,str) or not re.fullmatch(r'(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',job_id):raise KeyError(job_id)
        p=self.jobs/job_id/'job.json'
        if not p.is_file():raise KeyError(job_id)
        with self.mutex:return json.loads(p.read_text(encoding='utf-8'))
    def update(self,job_id,**changes):
        with self.mutex:
            row=self.get(job_id);row.update(changes);write(self.jobs/job_id/'job.json',row)
        return row
    def submit(self,images,mode='mvp',job_id=None,attempt_id=None,image_metadata=None,profile='default'):
        if mode!='mvp':raise ValueError('Only the RDF + hair-cap MVP is supported')
        if not set(images)<=set(SLOTS) or not set(RDF_SLOTS)<=set(images):
            raise ValueError('MVP requires front, left and right views; back is optional')
        self.preflight(mode)
        if profile!='default':raise ValueError("Only profile 'default' is supported")
        job_id=job_id or uuid4().hex
        if not re.fullmatch(r'(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',job_id):
            raise ValueError('Invalid job id')
        if attempt_id is not None:
            try:attempt_id=str(UUID(str(attempt_id)))
            except (ValueError,TypeError,AttributeError) as exc:raise ValueError('Invalid attempt id') from exc
        metadata_by_view={item['view_hint']:item for item in (image_metadata or [])}
        with self.mutex:
            existing_path=self.jobs/job_id/'job.json'
            if existing_path.is_file():
                existing=json.loads(existing_path.read_text(encoding='utf-8'))
                same_images=all(
                    next((item for item in existing.get('observations',[]) if item.get('slot')==slot),{}).get('normalized_sha256')
                    ==hashlib.sha256(entry['png']).hexdigest()
                    for slot,entry in images.items()
                )
                if (existing.get('attempt_id')==attempt_id and existing.get('profile')==profile
                        and len(existing.get('observations',[]))==len(images) and same_images):
                    return existing
                raise ValueError('Job id already exists for a different attempt or input')
            active=sum(json.loads(p.read_text()).get('status') not in TERMINAL for p in self.jobs.glob('*/job.json'))
            if active>=self.config.get('max_pending_jobs',3):raise OverflowError('QUEUE_FULL')
            folder=self.jobs/job_id;folder.mkdir()
            inputs=folder/'inputs';inputs.mkdir();observations=[]
            for slot in SLOTS:
                if slot not in images:continue
                entry=images[slot];path=inputs/(slot+'.png');path.write_bytes(entry['png'])
                original=folder/'uploads';original.mkdir(exist_ok=True)
                (original/(slot+'.'+entry['format'].lower())).write_bytes(entry['original'])
                image_meta=metadata_by_view.get(slot,{})
                observations.append({'slot':slot,'image_id':image_meta.get('image_id'),
                    'view_hint':image_meta.get('view_hint',slot),'original_sha256':hashlib.sha256(entry['original']).hexdigest(),
                    'normalized_sha256':sha(path),'normalized_size':entry['size'],
                    'normalization':'EXIF transpose and lossless RGB PNG; source upload retained',
                    'requested_for_rdf':slot in RDF_SLOTS,'used_for_hair_cap':True})
            row={'job_id':job_id,'status':'QUEUED','mode':mode,
                 'attempt_id':attempt_id,'profile':profile,'created_at':datetime.now(timezone.utc).isoformat(),
                 'observations':observations,
                 'stages':[],'artifacts':[],'errors':[],'ready_for_machine':False,
                 'hair_cap_exported':False,'hair_flow_or_part_inferred':False,
                 'face_identity_verified':False,'rdf_views_actually_used':[],
                 'code_sha256':self.code_hashes()}
            write(folder/'config.snapshot.json',self.config);write(folder/'job.json',row)
            self.executor.submit(self.run,job_id)
        return row
    def code_hashes(self):
        files=list((self.root/'tools/headapi').glob('*.py'))
        files.extend(path for path in
            (self.root/'tools/triposr_probe'/name for name in ('audit.py','common.py'))
            if path.is_file())
        for name in ('head24_policy.py','head24_detection.py','detect_head24_profile.py','eye_components.py',
                     'scalp_cap.py','back_view.py','figurine22_hair.py','headlab_raster.py',
                     'add_graduation_cap.py','figurine_preview.py'):
            path=self.root/'scripts'/name
            if path.is_file():files.append(path)
        path=self.root/'src/memento3d_head/adapters/automatic_face_masks.py'
        if path.is_file():files.append(path)
        path=self.root/'configs/hair-cap.json'
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
        folder=self.jobs/job_id;allowed={'.glb','.obj','.png','.jpg','.json','.html','.log','.npz','.stl'}
        rows=[]
        for path in sorted(folder.rglob('*')):
            rel=path.relative_to(folder)
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in allowed:continue
            if rel.parts[0] in ('inputs','uploads') or rel.as_posix() in ('job.json','config.snapshot.json'):continue
            rows.append({'path':rel.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
        return rows
    def run(self,job_id):
        folder=self.jobs/job_id;c=self.config['paths'];job=self.get(job_id);rdf_ok=False;cap_ok=False
        self.update(job_id,status='RUNNING',started_at=datetime.now(timezone.utc).isoformat())
        try:
            for entry in job['observations']:
                if sha(folder/'inputs'/(entry['slot']+'.png'))!=entry['normalized_sha256']:
                    raise ValueError('Input changed after upload: '+entry['slot'])
            if self.code_hashes()!=job['code_sha256']:raise ValueError('Pipeline code changed after submission; submit a new attempt')
            rdf_ok=self.stage(job_id,'01_rdf_observations',[c['cpu_python'],'-m','tools.headapi.prepare_faces',
                '--input',folder/'inputs','--task',c['landmarker'],'--output',folder/'rdf_crops','--rdf-python',c['rdf_python'],'--repo',c['rdf_repo']])
            if rdf_ok:
                rdf_ok=self.stage(job_id,'02_rdf_fit',[c['rdf_python'],'-m','tools.headapi.rdf_worker',
                    '--repo',c['rdf_repo'],'--crops',folder/'rdf_crops','--output',folder/'rdf'])
            if rdf_ok:
                report=json.loads((folder/'rdf/report.json').read_text())
                if report['status']!='FIT_GATES_PASSED':raise ValueError('RDF did not pass its gates')
                used_views=report.get('views_used',[])
                if (not used_views or 'front' not in used_views
                        or any(view not in RDF_SLOTS for view in used_views)):
                    raise ValueError('RDF did not use a supported front-containing view set')
                self.update(job_id,rdf_views_actually_used=used_views)
                rdf_ok=self.stage(job_id,'02b_rdf_export',[c['haircap_python'],'-m','tools.headapi.face_export',
                    '--fit',folder/'rdf','--crops',folder/'rdf_crops'])
            if rdf_ok:
                cap_ok=self.stage(job_id,'03_hair_cap',[c['haircap_python'],'-m','tools.headapi.hair_cap_worker',
                    '--job',folder,'--output',folder/'hair','--parser',c['parser'],
                    '--profile',self.root/'configs/hair-cap.json'])
            if cap_ok:
                scene_path=folder/'hair/scene.glb'
                if not scene_path.is_file() or scene_path.stat().st_size<12:
                    raise ValueError('Scene GLB output is missing or invalid')
                cap_report=json.loads((folder/'hair/report.json').read_text())
                if cap_report.get('status')!='HAIR_CAP_EXPORTED' or not cap_report.get('geometry',{}).get('watertight'):
                    raise ValueError('Hair-cap output contract failed')
                self.update(job_id,hair_cap_exported=True,
                    hair_flow_or_part_inferred=cap_report.get('hair_flow_or_part_inferred',False))
            status='MVP_EXPORTED' if rdf_ok and cap_ok else 'PARTIAL' if rdf_ok or cap_ok else 'FAILED'
            if self.code_hashes()!=job['code_sha256']:raise ValueError('Pipeline code changed during job')
            self.update(job_id,status=status,current_stage=None)
        except Exception as exc:
            row=self.get(job_id);row['errors'].append({'stage':row.get('current_stage'),'error':str(exc)})
            self.update(job_id,status='FAILED',errors=row['errors'],current_stage=None)
        finally:
            try:self.review_page(job_id)
            finally:self.update(job_id,artifacts=self.collect(job_id),finished_at=datetime.now(timezone.utc).isoformat(),
                                completed_at=datetime.now(timezone.utc).isoformat())
    def review_page(self,job_id):
        folder=self.jobs/job_id;row=self.get(job_id)
        text='<meta charset="utf-8"><title>Head API result</title><style>body{background:#171c24;color:white;font:16px system-ui;margin:24px}img{max-width:100%}a{color:#acf}</style>'
        text+='<h1>'+html.escape(row['status'])+'</h1><p>RealDenseFace head and a separate experimental solid hair-cap proposal. The cap uses hair masks and multi-view silhouette/volume fitting; hair flow is not inferred. Neither output is certified for printing.</p>'
        if (folder/'hair/hair_cap_native.glb').is_file():text+='<p><a href="hair/hair_cap_native.glb">Hair cap (RDF native units)</a> · <a href="hair/hair_cap_design_mm.stl">Hair cap design-scale STL</a> · <a href="hair/report.json">Hair-cap report</a></p>'
        for label,relative in [('RDF geometry','rdf/rdf_face_gray.png'),('RDF photographic color approximation','rdf/rdf_face_photo.png')]:
            if (folder/relative).is_file():text+='<h2>'+label+'</h2><img src="'+relative+'">'
        for slot in ('front','left','right','back'):
            relative='hair/'+slot+'.fit.png'
            if (folder/relative).is_file():text+='<h2>'+slot+' · tóc dự đoán / mask ảnh</h2><img src="'+relative+'">'
        text+='<p>Scale 22 mm là giả định theo chiều cao mesh, không phải số đo vật lý. Ba ảnh dùng prior giới hạn cho tóc khuất phía sau; ảnh back thêm ràng buộc silhouette sau đầu nhưng không quan sát đỉnh từ trên xuống.</p>'
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
