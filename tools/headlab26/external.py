"""Explicit optional research backends; no fallback success or shared output folders."""
from pathlib import Path
import os,shutil
from .io import write,read,sha,SLOTS
from .config import resolve
from .process import execute

def pixel3dmm(root,crops,out,profile):
 spec=profile['pixel3dmm'];repo=resolve(root,spec['repo']);python=resolve(root,spec['python']);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 source=out/'observations';source.mkdir()
 for i,slot in enumerate(SLOTS):shutil.copyfile(Path(crops)/(slot+'.png'),source/(str(i).zfill(5)+'.png'))
 env={'PIXEL3DMM_CODE_BASE':repo.as_posix(),'PIXEL3DMM_PREPROCESSED_DATA':(out/'preprocessed').resolve().as_posix(),'PIXEL3DMM_TRACKING_OUTPUT':(out/'tracking').resolve().as_posix()}
 name=source.name;steps=[('crop',[python,repo/'scripts/run_cropping.py','--video_or_images_path',source.resolve().as_posix()]),('segment',[python,repo/'scripts/run_facer_segmentation.py','--video_name',name]),('normals',[python,repo/'scripts/network_inference.py','model.prediction_type=normals','video_name='+name]),('uv',[python,repo/'scripts/network_inference.py','model.prediction_type=uv_map','video_name='+name]),('track',[python,repo/'scripts/track.py','video_name='+name,'is_discontinuous=True','global_camera=False','include_neck=False','use_flame2023=True','ignore_mica=True','batch_size='+str(min(3,spec['batch_size'])),'iters='+str(spec['iterations']),'global_iters='+str(spec['global_iterations']),'save_meshes=True'])]
 # Do not invoke the upstream os.system-based umbrella script. Each action has
 # its own exit code and fresh per-dataset output paths.
 timings=[]
 for stage,command in steps:
  row=execute('pixel_'+stage,command,repo,out,profile['execution']['stage_timeout_seconds'],env);timings.append(row)
  if row['exit_code']!=0:raise RuntimeError('Pixel3DMM '+stage+' failed; '+row['log'])
 command=[python,'-m','tools.headlab26.pixel_export','--tracking',out/'tracking','--preprocessed',out/'preprocessed'/name,'--output',out/'common']
 row=execute('pixel_contract',command,root,out,profile['execution']['stage_timeout_seconds'],env)
 if row['exit_code']!=0:raise RuntimeError('Pixel camera/mesh contract export failed; '+row['log'])
 write(out/'native.execution.json',{'status':'UPSTREAM_EXECUTION_COMPLETED','stages':timings,'coordinate_export':'explicit OpenGL to pixel camera conversion; requires independent overlays','source_code_sha256':{p.relative_to(repo).as_posix():sha(p) for p in [repo/'scripts/track.py',repo/'src/pixel3dmm/tracking/tracker.py']}})
 return out/'common'
