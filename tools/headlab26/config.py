"""Portable defaults resolved from the project, never a particular subject/run."""
import sys
from pathlib import Path
from .io import read

def python_in(root,name):
 paths=[Path(root)/name/'Scripts/python.exe',Path(root)/name/'bin/python']
 return next((p for p in paths if p.is_file()),paths[0])

def resolve(root,path):return (Path(root)/path).resolve() if not Path(path).is_absolute() else Path(path).resolve()

def load(root,path=None):
 root=Path(root).resolve();profile=read(path or root/'configs/headlab26.json')
 cache=Path(profile['paths'].get('cache_root','D:/AI-cache'))
 # AI-cache root is a model-storage preference, not a dataset or subject binding.
 profile['resolved']={'project':str(root),'rdf_python':str(python_in(root,'.venv-rdf')),'rdf_repo':str(root/'vendor/RealDenseFace'),'parser_model':str(cache/'face-parsing/resnet18.onnx'),'landmarker_model':str(cache/'mediapipe/face_landmarker.task')}
 for name,value in profile['paths'].items():
  if name!='cache_root' and value is not None:profile['resolved'][name]=str(resolve(root,value))
 f=profile['fitting']
 for name in ['glasses_weight','nonface_weight','outside_weight','minimum_robust_weight']:
  if not isinstance(f[name],(int,float)) or not 0<f[name]<=1:raise ValueError('Invalid '+name)
 if not 1<=f['joint_refits']<=5 or not 1<=f['joint_iterations']<=100 or not 1<=f['single_iterations']<=100 or not 0<f['huber_delta_px']<=100:raise ValueError('Invalid fitting budget')
 hair=profile['figurine']
 if any(x not in ['radial','arap'] for x in hair['hair_methods']) or not hair['hair_methods']:raise ValueError('Invalid hair methods')
 if not .1<=hair.get('arap_cell_mm',.55)<=2 or not 32<=hair.get('arap_max_nodes',6000)<=12000 or not 0<hair.get('arap_min_fraction',.125)<=1:raise ValueError('Invalid ARAP control budget')
 if profile['render']['preview_size']>600 or not 8<=profile['render']['atlas_cell']<=32:raise ValueError('Render budget exceeded')
 return profile

def prerequisites(profile,backend):
 r=profile['resolved'];required={'prepare':[r['landmarker_model']],'rdf_baseline':[r['rdf_python'],Path(r['rdf_repo'])/'weights/flame/flame2023.pkl',Path(r['rdf_repo'])/'weights/flame/flame_assets.npz',Path(r['rdf_repo'])/'weights/realdenseface/vitb.pth',Path(r['rdf_repo'])/'weights/facebox/face_box.pth',r['parser_model']],'rdf_occlusion':[],'pixel3dmm':[],'trust':[]}
 if backend=='rdf_occlusion':return prerequisites(profile,'rdf_baseline')
 if backend in ('pixel3dmm','trust'):
  spec=profile.get(backend,{})
  if not spec.get('python') or not spec.get('repo'):return {'status':'BACKEND_NOT_CONFIGURED','missing':['configure '+backend+'.python and .repo'],'backend':backend}
  root=Path(r['project']);repo=resolve(root,spec['repo']);paths=[resolve(root,spec['python'])]
  if backend=='pixel3dmm':paths +=[repo/'scripts/track.py',repo/'scripts/run_cropping.py',repo/'pretrained_weights/normals.ckpt',repo/'pretrained_weights/uv.ckpt']
  else:paths +=[repo/'lib/model.py',repo/'test.py']
  paths +=[repo/x for x in spec.get('required_files',[])]
 else:paths=required[backend]
 missing=[str(p) for p in paths if not Path(p).is_file()]
 return {'status':'PREREQUISITES_MISSING' if missing else 'PREREQUISITE_FILES_PRESENT','missing':missing,'backend':backend,'execution_verified':False}
