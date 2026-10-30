"""Run baseline and refined texture sequentially on the same immutable photo set."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]

def photos(folder):
    result = {}
    for slot in ('front', 'left', 'right'):
        matches = [p for p in folder.iterdir() if p.is_file() and p.stem.lower() == slot and p.suffix.lower() in ('.jpg','.jpeg','.png')]
        if len(matches) != 1:
            raise ValueError('Expected exactly one image for '+slot)
        p = matches[0]
        result[slot] = {'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    args = p.parse_args()
    folder = args.input_dir.resolve()
    original = photos(folder)
    out = ROOT/'runs'/('comparison_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True)
    report = {'status':'RUNNING','input':original,'results':[], 'hair_models':{'MonoHair':'NOT_INTEGRATED','Im2Haircut':'NOT_INTEGRATED'},'automatic_quality_ranking':False}
    def save():
        (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    for label, script in [('baseline','run_v1.py'),('texture_refined','run_v2.py')]:
        if photos(folder) != original:
            report['status']='INPUT_CHANGED';save();return 1
        before = set((ROOT/'runs').glob('v1_*'))
        started = time.perf_counter()
        log = out/(label+'.log')
        cmd = [sys.executable,str(ROOT/'scripts'/script),'--input-dir',str(folder)]
        print('Running '+label+'; log: '+str(log),flush=True)
        with log.open('w',encoding='utf-8') as f:
            child = subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                if sys.platform=='win32':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                else:child.terminate()
                child.wait();report['status']='INTERRUPTED';save();return 130
        candidates = sorted(set((ROOT/'runs').glob('v1_*'))-before)
        report['results'].append({'variant':label,'command':cmd,'exit_code':code,'wall_seconds':round(time.perf_counter()-started,3),'log':str(log),'new_run_candidates':[str(x) for x in candidates], 'run_association_unambiguous':len(candidates)==1})
        save()
        if code != 0:
            report['status']='FAILED';save();print('Stopped: inspect '+str(log));return code
    report['status']='REVIEW_REQUIRED' if photos(folder)==original else 'INPUT_CHANGED'
    save();print('Report: '+str(out/'comparison.json'))
    return 0 if report['status']=='REVIEW_REQUIRED' else 1

if __name__=='__main__':raise SystemExit(main())
