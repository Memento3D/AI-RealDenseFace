"""Controlled solver initialization experiment with checksum-bound cached predictions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[2]


def run():
    source=ROOT/'runs/headapi/9bf00af2-3a4b-40e1-bd08-5706c66ef19d'
    output=ROOT/'runs/experiments'/('rdf-warm-'+uuid4().hex);output.mkdir(parents=True)
    report={'schema_version':1,'source':str(source),'status':'RUNNING','variants':{},
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'tools/headapi/rdf_worker.py',ROOT/'tools/headapi/camera_diagnostics.py']},
        'likeness_verified':False,'ready_for_machine':False,
        'scope':'same neural predictions, cold vs baseline-state initialization; warm prior has extra prior optimization history'}
    def save(): (output/'experiment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'}
    seed=output/'seed_21'
    for name,fov,warm in [('seed_21',21.803398874989483,False),('cold_40',40,False),('warm_40',40,True),('cold_50',50,False),('warm_50',50,True)]:
        row={'fov':fov,'warm_start':warm};report['variants'][name]=row;save()
        command=[sys.executable,'-m','tools.headapi.rdf_worker','--repo',str(ROOT/'vendor/RealDenseFace'),
                 '--crops',str(source/'rdf_crops'),'--output',str(output/name),'--fixed-fov',str(fov)]
        if name!='seed_21':command+=['--prediction-cache',str(seed/'predictions')]
        if warm:command+=['--warm-fit',str(seed)]
        row['command']=command
        print('Starting '+name,flush=True)
        try:
            with (output/(name+'.log')).open('w',encoding='utf-8') as log:
                process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            row['exit_code']=process.returncode
            if (output/name/'report.json').exists():row['fit']=json.loads((output/name/'report.json').read_text())
            if process.returncode:
                row['status']='FAILED';save()
                if name=='seed_21':raise RuntimeError('Seed failed')
                continue
            for stage,args in [('export',['-m','tools.headapi.face_export','--fit',str(output/name),'--crops',str(source/'rdf_crops')]),
                ('diagnostics',['-m','tools.headapi.camera_diagnostics','--experiment',str(output),'--source',str(source),'--output',str(output/(name+'_diagnostics')),'--variants',name])]:
                with (output/(name+'.'+stage+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
            row['diagnostics']=json.loads((output/(name+'_diagnostics')/'diagnostics.json').read_text())['variants'][name]['views']
            row['status']='COMPLETED'
        except Exception as error:
            row['status']='FAILED';row['error']=str(error)
            if name=='seed_21':report['status']='FAILED';save();raise
        save()
    report['status']='EXPERIMENT_FINISHED';save();print(output,flush=True)


if __name__=='__main__':run()
