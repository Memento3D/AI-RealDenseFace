"""Measured appearance states and source-bound reuse, without refitting identity."""
from pathlib import Path
import shutil
import numpy as np
from .io import sha,key,write,read

def appearance_key(workspace,profile):
 return key({'workspace':sha(workspace),'semantic':sha(Path(profile['semantic26'])/'report.json') if profile.get('semantic26') else None,'profile':{k:profile[k] for k in ['head_neck_mm','appearance23','skin245','glasses245','bright_skin']},'appearance_code':sha(Path(__file__).with_name('appearance.py')),'face_code':sha(Path(__file__).with_name('face.py'))})

def appearance_cached(workspace,face,profile,out,cache=None):
 out=Path(out)
 if cache is None:
  from .appearance import canonical_appearance
  native,unlit,skin=canonical_appearance(workspace,face,profile,out)
  write(out/'face.json',face['report']);write(out/'bright_skin.json',skin)
  files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.glb','.json','.png'] and p.name!='appearance.binding.json'}
  write(out/'appearance.binding.json',{'key':appearance_key(workspace,profile),'files':files,'skin':skin})
 else:
  cache=Path(cache);binding=read(cache/'appearance.binding.json')
  if not binding.get('files'):raise ValueError('Empty appearance cache')
  if binding['key']!=appearance_key(workspace,profile):raise ValueError('Appearance cache does not belong to this source/policy/code')
  for name,digest in binding['files'].items():
   source=cache/name
   if not source.resolve().is_relative_to(cache.resolve()) or sha(source)!=digest:raise ValueError('Appearance cache changed')
   if (out/name).exists():
    if sha(out/name)!=digest:raise ValueError('Output already contains different appearance file '+name)
   else:shutil.copyfile(source,out/name)
  skin=binding['skin'];face['skin']=np.asarray(skin['skin_base_rgb8'],float)/255;face['report']['skin']=skin
  native=out/'canonical_native.glb';unlit=out/'canonical_unlit_native.glb'
 return native,unlit,skin

def export(workspace,model,profile,out):
 from .face import load_face
 from run_pipeline_v24 import preview
 out=Path(out);out.mkdir(parents=True,exist_ok=False);face=load_face(workspace,model,profile,out);native,unlit,skin=appearance_cached(workspace,face,profile,out)
 frame=([0,0,0],profile['head_neck_mm']*1.4)
 for name in ['source_reference','canonical_reference','canonical_unlit']:preview(out/(name+'.glb'),out/(name+'.png'),frame,profile['preview_size'])
 # Reseal after adding previews. All appearance inputs are still immutable.
 binding=read(out/'appearance.binding.json');binding['files']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='appearance.binding.json'};write(out/'appearance.binding.json',binding)
 return skin
