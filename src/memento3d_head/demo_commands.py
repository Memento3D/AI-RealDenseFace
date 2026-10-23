"""One folder, one isolated demo attempt using existing measured stages."""
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
from memento3d_head import __version__
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError

SLOTS=('front','left','right')
MODEL_HASH='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'


def add_commands(commands):
    p=commands.add_parser('demo',help='Fresh three-photo folder to offline colored-head preview')
    p.add_argument('--input-dir',required=True,type=Path)
    p.add_argument('--output',type=Path,help='Must not exist; default runs/demo_<folder>_<unique>')
    p.add_argument('--model',type=Path,default=Path('D:/AI-cache/mediapipe/face_landmarker.task'))
    p.add_argument('--model-sha256',default=MODEL_HASH)
    p.add_argument('--mica-config',type=Path,default=Path('configs/mica.local.json'))
    p.add_argument('--config',type=Path,default=Path('configs/photo_validation.local.json'))
    p.add_argument('--no-open',action='store_true',help='Do not launch local folders/viewer')


def find_photos(directory):
    if not directory.is_dir():raise ValidationError('INPUT_NOT_FOUND','Input folder does not exist: '+str(directory))
    files=list(directory.iterdir());result={}
    for slot in SLOTS:
        matches=[p for p in files if p.is_file() and p.stem.lower()==slot and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp')]
        if len(matches)!=1:raise ValidationError('PHOTO_NAMES','Expected exactly one '+slot+'.jpg/.jpeg/.png/.webp; found '+str(len(matches)))
        result[slot]=matches[0].resolve()
    return result


def run_stage(name,arguments,output):
    """Isolate CPU inspectors in separate processes; retain logs and reject failures."""
    command=[sys.executable,'-m','memento3d_head',*map(str,arguments),'--output',str(output)]
    log=output.parent/(name+'.log')
    print('\n['+name+'] '+str(output),flush=True)
    with log.open('w',encoding='utf-8') as stream:
        completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    report_path=output/'report.json'
    report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else {}
    expected={'01_validation':'NEEDS_HUMAN_REVIEW','02_head':'GEOMETRY_EXPORTED','03_registration':'REGISTRATIONS_EXPORTED'}[name]
    if completed.returncode or report.get('status')!=expected:
        error=report.get('error') or {'message':str(report.get('assessment',{}).get('issues',[]))}
        raise ValidationError('STAGE_FAILED',name+': '+str(error)+'; log: '+str(log))
    print('  '+report['status'],flush=True)
    return report


def review_photos(validation,open_files=True):
    if open_files:
        try:
            if os.name=='nt':os.startfile(str(validation.resolve()))
            else:webbrowser.open(validation.resolve().as_uri())
        except OSError:print('Open the validation folder manually to inspect the images.',flush=True)
    print('\nReview normalized/landmark images in '+str(validation),flush=True)
    print('Check same person, correct three views, visible face and landmarks. Enter reviewer name to continue; blank cancels.',flush=True)
    if not sys.stdin.isatty():raise ValidationError('REVIEW_REQUIRED','Run demo in an interactive terminal to review this photo set')
    name=input('Reviewer name: ').strip()
    if not name:raise ValidationError('REVIEW_CANCELLED','No review declaration; reconstruction was not started')
    return name


def execute(args):
    output=None;started=time.monotonic();report={'schema_version':1,'pipeline_type':'three_photo_head_demo','pipeline_version':1,'package_version':__version__,'status':'FAILED','stages':{},'error':None}
    try:
        from memento3d_head.adapters.mica_assets import verify_config
        from memento3d_head.adapters.smooth_preview import read_geometry
        from memento3d_head.adapters.multiview_projection import build_preview
        from memento3d_head.settings import load_settings
        sources=find_photos(args.input_dir.resolve())
        verify_config(args.mica_config.resolve());load_settings(args.config)
        if sha256(args.model)!=args.model_sha256.lower():raise ValidationError('MODEL_HASH_MISMATCH','Face landmarker does not match requested pin')
        hashes={s:sha256(p) for s,p in sources.items()}
        destination=args.output or Path('runs')/('demo_'+args.input_dir.resolve().name+'_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        try:destination.resolve().mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a fresh output directory') from error
        output=destination.resolve();snapshots=output/'00_inputs';snapshots.mkdir();photos={}
        for slot,path in sources.items():
            photos[slot]=snapshots/(slot+path.suffix.lower());shutil.copyfile(path,photos[slot])
            if sha256(photos[slot])!=hashes[slot]:raise ValidationError('SOURCE_CHANGED','Input changed during snapshot: '+slot)
        report['input']={s:{'path':str(sources[s]),'sha256':hashes[s]} for s in SLOTS}
        common=['--model',args.model.resolve(),'--model-sha256',args.model_sha256,'--config',args.config.resolve()]
        photo_args=[item for slot in SLOTS for item in ('--'+slot,photos[slot])]
        validation=output/'01_validation';head=output/'02_head';registration=output/'03_registration';final=output/'final'
        def record(name,value):
            report['stages'][name]={'status':value['status'],'report':str({'validation':validation,'head':head,'registration':registration}[name]/'report.json')}
            LocalStore().write_json(output/'report.json',report)
        record('validation',run_stage('01_validation',['validate-photos',*photo_args,'--source-kind','real_capture',*common],validation))
        reviewer=review_photos(validation,not args.no_open)
        report['review']={'reviewed_by':reviewer,'human_review_is_declaration':True}
        record('head',run_stage('02_head',['reconstruct-head','--validation-run',validation,'--front',photos['front'],'--reviewed-by',reviewer,'--mica-config',args.mica_config.resolve()],head))
        record('registration',run_stage('03_registration',['register-views','--head-run',head,'--validation-run',validation,*photo_args,*common,'--mica-config',args.mica_config.resolve()],registration))
        r=json.loads((registration/'report.json').read_text(encoding='utf-8'));h=json.loads((head/'report.json').read_text(encoding='utf-8'))
        mesh=head/'head.obj';mesh_hash=sha256(mesh)
        if r['input']['mesh']!=mesh_hash or h['artifacts']['head.obj']['sha256']!=mesh_hash or r['input']['head_report']!=sha256(head/'report.json'):
            raise ValidationError('SOURCE_MISMATCH','New registration is not bound to this new head')
        normalized={s:registration/(s+'.normalized.png') for s in SLOTS}
        for slot,path in normalized.items():
            if sha256(path)!=r['artifacts'][path.name]['sha256']:raise ValidationError('SOURCE_MISMATCH','Normalized photo changed: '+slot)
        final.mkdir();print('\n[04_color] '+str(final),flush=True)
        vertices,faces=read_geometry(mesh)
        provenance={'head_obj_sha256':mesh_hash,'head_report_sha256':sha256(head/'report.json'),'registration_report_sha256':sha256(registration/'report.json'),'manual_jaw_fit_performed':False}
        info=build_preview(final,vertices,faces,r,normalized,provenance)
        for name in ('head.obj','head.ply','identity.npy'):shutil.copyfile(head/name,final/name)
        if sha256(mesh)!=mesh_hash or any(sha256(photos[s])!=hashes[s] or sha256(sources[s])!=hashes[s] for s in SLOTS):
            raise ValidationError('SOURCE_CHANGED','Source changed during demo')
        color_report={'schema_version':1,'pipeline_type':'demo_photo_color_projection','pipeline_version':1,'status':'MULTIVIEW_COLOR_EXPORTED','input':provenance,'projection':info,'research_only':True,'manual_jaw_fit_performed':False,'geometry_views':['front'],'color_views':list(SLOTS),'likeness_verified':False,'printability_verified':False,'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in final.iterdir() if p.is_file()}}
        LocalStore().write_json(final/'report.json',color_report)
        report.update(status='DEMO_PREVIEW_EXPORTED',final_viewer=str(final/'multiview_preview.html'),manual_jaw_fit_performed=False,geometry_views=['front'],color_views=list(SLOTS),likeness_verified=False,printability_verified=False)
        if not args.no_open:webbrowser.open((final/'multiview_preview.html').as_uri())
    except (KeyboardInterrupt,EOFError):
        report['status']='CANCELLED';report['error']={'code':'INTERRUPTED','message':'Demo cancelled; no automatic continuation'}
    except Exception as error:
        report['error']={'code':error.code if isinstance(error,ValidationError) else 'DEMO_FAILED','message':str(error)}
    report['elapsed_seconds']=round(time.monotonic()-started,3)
    if output is not None:LocalStore().write_json(output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if report['status']=='DEMO_PREVIEW_EXPORTED' else (130 if report['status']=='CANCELLED' else 1)
