"""Sequential local fixed-crop-FOV sweep; no training or production policy change."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[2]


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    variants={}
    report={'schema_version':1,'status':'RUNNING','source':str(source),'variants':variants,
            'likeness_verified':False,'ready_for_machine':False,
            'policy':'fixed crop FOV shared across views as an experimental constraint, not measured intrinsics',
            'budget':'20 single-frame registration iterations, then 10 joint iterations; no FOV-search warm-start',
            'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__),ROOT/'tools/headapi/rdf_worker.py',ROOT/'tools/headapi/camera_diagnostics.py']}}
    def save():
        (output/'experiment.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'}
    for fov in [15.,21.803398874989483,30.,40.,50.]:
        name='fov_'+str(round(fov,4)).replace('.','_')
        command=[sys.executable,'-m','tools.headapi.rdf_worker','--repo',str(ROOT/'vendor/RealDenseFace'),
                 '--crops',str(source/'rdf_crops'),'--output',str(output/name),'--fixed-fov',str(fov)]
        print('Starting '+name,flush=True)
        row={'fov':fov,'command':command};variants[name]=row
        try:
            with (output/(name+'.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            row['exit_code']=result.returncode
            if result.returncode:
                row['status']='FAILED';save();continue
            for stage,args in [('export',['-m','tools.headapi.face_export','--fit',str(output/name),'--crops',str(source/'rdf_crops')]),
                               ('diagnostics',['-m','tools.headapi.camera_diagnostics','--experiment',str(output),'--source',str(source),'--output',str(output/(name+'_diagnostics')),'--variants',name])]:
                with (output/(name+'.'+stage+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
            row['status']='COMPLETED'
            row['fit']=json.loads((output/name/'report.json').read_text())
            row['diagnostics']=json.loads((output/(name+'_diagnostics')/'diagnostics.json').read_text())['variants'][name]['views']
        except Exception as error:
            row['status']='FAILED';row['error']=str(error)
        save()
    report['status']='SWEEP_FINISHED'
    save()
    print(str(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.source.resolve(),(a.output or ROOT/'runs/experiments'/('rdf-fov-'+uuid4().hex)).resolve())
