"""Bounded subprocess execution, per-stage logs and actual measured wall time."""
import os,subprocess,time
from pathlib import Path
from .io import write

def execute(name,command,cwd,folder,timeout=1800,env=None):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter();log=folder/(name+'.log')
 print('['+name+']',flush=True)
 with log.open('w',encoding='utf-8') as stream:
  # Avoid PIPE deadlock and preserve partial logs on timeout. Poll without sleeping
  # inside tool orchestration; this runs on the user's machine as a normal CLI.
  try:
   result=subprocess.run([str(c) for c in command],cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1',**(env or {})})
   code=result.returncode;status='STAGE_COMPLETED' if code==0 else 'STAGE_FAILED'
  except subprocess.TimeoutExpired:code=None;status='STAGE_TIMEOUT'
 record={'stage':name,'status':status,'exit_code':code,'seconds':time.perf_counter()-start,'log':str(log),'command':[str(x) for x in command]};write(folder/(name+'.stage.json'),record)
 return record
