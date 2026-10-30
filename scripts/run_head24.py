"""One-shot photo -> crop -> gated RDF -> explicit-camera workspace -> V2.4 demo."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback,webbrowser
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_manifest(path,root):
    for e in json.loads(path.read_text(encoding='utf-8-sig'))['files']:
        p=root/e['path']
        if not p.is_file() or sha(p)!=e['sha256']:raise ValueError('Code/config changed: '+str(p))

def resume_artifacts(source):
    source=source.resolve()
    report=json.loads((source/'pipeline.report.json').read_text(encoding='utf-8-sig'))
    completed={row['stage'] for row in report.get('stages',[]) if row.get('exit_code')==0}
    if not {'01_prepare','02_fit'}<=completed:raise ValueError('Resume requires successful prepare and fit stages')
    crops=source/'01_crops';fit=source/'02_fit'
    crop_report=json.loads((crops/'report.json').read_text(encoding='utf-8-sig'))
    fit_report=json.loads((fit/'report.json').read_text(encoding='utf-8-sig'))
    if crop_report.get('status')!='CROPS_PREPARED' or fit_report.get('status')!='FIT_GATES_PASSED':
        raise ValueError('Resume artifacts have not passed their gates')
    if sha(crops/'report.json')!=fit_report.get('crop_report_sha256'):
        raise ValueError('Resume crop metadata checksum mismatch')
    if sha(fit/'projection_bundle.npz')!=fit_report.get('projection_bundle_sha256'):
        raise ValueError('Resume projection bundle checksum mismatch')
    for slot in ('front','left','right'):
        if sha(crops/(slot+'.png'))!=crop_report['views'][slot]['crop_sha256']:
            raise ValueError('Resume crop image checksum mismatch: '+slot)
    return crops,fit,{'resumed_from':str(source),'input':report.get('input'),
        'reused_verified_artifacts':['01_crops','02_fit'],
        'source_fit_report_sha256':sha(fit/'report.json')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'input/05');p.add_argument('--task',type=Path,default=Path('D:/AI-cache/mediapipe/face_landmarker.task'));p.add_argument('--output',type=Path);p.add_argument('--check-only',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
    a.input=a.input.resolve();a.task=a.task.resolve()
    out=(a.output or ROOT/'runs'/(('head24_resume_' if a.resume else 'head24_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if out.exists():raise ValueError('Output exists: use a fresh directory')
    out.mkdir(parents=True);report={'status':'RUNNING','version':'Head24.4 cranial frame and observed feature fallback','input':str(a.input.resolve()),'stages':[],'ready_for_machine':False,'physical_print_verified':False,'windows_gpu_inference_verified_by_release_author':False};start=time.perf_counter()
    repo=ROOT/'vendor/RealDenseFace';rdf=ROOT/'.venv-rdf/Scripts/python.exe'
    try:
        check_manifest(ROOT/'pipeline-v23-code.json',ROOT)
        check_manifest(ROOT/'head24-code.json',ROOT)
        check_manifest(ROOT/'head24-rdf-source.json',repo)
        if a.resume:
            crops,fit,reused=resume_artifacts(a.resume)
            report.update(reused)
        else:
            if not a.input.is_dir():raise ValueError('Missing input: '+str(a.input))
            for path in [a.task,rdf,repo/'weights/flame/flame2023.pkl',repo/'weights/flame/flame_assets.npz',repo/'weights/realdenseface/vitb.pth',repo/'weights/facebox/face_box.pth']:
                if not path.is_file():raise ValueError('Missing prerequisite: '+str(path))
        if a.check_only:
            report['status']='PREREQUISITE_FILES_VERIFIED';return 0
        def execute(name,command,cwd=ROOT):
            print('\n['+name+']',flush=True)
            with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                proc=subprocess.Popen([str(x) for x in command],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'})
                for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                code=proc.wait()
            report['stages'].append({'stage':name,'exit_code':code})
            if code:raise RuntimeError(name+' failed; inspect '+str(out/(name+'.log')))
        workspace=out/'03_workspace';result=out/'04_figurine'
        if not a.resume:
            crops=out/'01_crops';fit=out/'02_fit'
            execute('01_prepare',[sys.executable,ROOT/'scripts/prepare_head24.py','--input',a.input.resolve(),'--task',a.task,'--output',crops])
            execute('02_fit',[rdf,ROOT/'scripts/fit_head24.py','--repo',repo,'--crops',crops,'--output',fit],repo)
        execute('03_workspace',[sys.executable,ROOT/'scripts/build_head24_workspace.py','--fit',fit,'--crops',crops,'--output',workspace])
        execute('04_figurine',[sys.executable,ROOT/'scripts/run_pipeline_v24.py','--profile',ROOT/'configs/pipeline-v24.json','--workspace',workspace/'head.workspace.html','--parser-model','D:/AI-cache/face-parsing/resnet18.onnx','--output',result])
        final=json.loads((result/'report.json').read_text())
        if final.get('status')!='DEMO_EXPORTED':raise ValueError('Figurine did not select a valid demo candidate')
        report.update(status='DEMO_EXPORTED',review=str(result/'review.html'),workspace=str(workspace/'head.workspace.html'))
        print('Open:',result/'review.html',flush=True)
        if os.name=='nt':os.startfile(result/'review.html')
        return 0
    except Exception as e:
        report['error']=str(e);report['status']='FAILED'
        for sub in ('01_crops','02_fit'):
            file=out/sub/'report.json'
            if file.is_file():
                stage_report=json.loads(file.read_text())
                if stage_report.get('status') in ('NEEDS_RETAKE','DETECTION_UNRESOLVED'):
                    report['status']=stage_report['status'];report['stage_error']=stage_report.get('error');report['stage_report']=str(file)
        rejection=out/'04_figurine/rejection.json'
        if rejection.is_file():
            report['status']='INPUT_GATE_REJECTED';report['rejection']=json.loads(rejection.read_text())
        print(traceback.format_exc(),flush=True);return 2 if report['status'] in ('NEEDS_RETAKE','DETECTION_UNRESOLVED','INPUT_GATE_REJECTED') else 1
    finally:
        report['seconds']=time.perf_counter()-start
        (out/'pipeline.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':raise SystemExit(main())
