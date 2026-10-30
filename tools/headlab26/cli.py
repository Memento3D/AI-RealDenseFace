"""Portable batch entry point: isolate datasets, fit variants, measure and compare."""
import argparse,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
if not __package__:__package__='tools.headlab26'
from .io import read,write,sha,key,stamp,discover,inspect_dataset,seal_stage,reuse_stage,verify_code,SLOTS
from .config import load,prerequisites,resolve
from .process import execute
from .report import summarize
from .cache import stage_code


def execute_python(stage,args,out,profile,python=None):
 row=execute(stage,[python or sys.executable,*args],ROOT,out,profile['execution']['stage_timeout_seconds'])
 if row['exit_code']!=0:raise RuntimeError(stage+' '+row['status']+'; read '+row['log'])
 return row


def prepare_one(info,out,profile):
 from .prepare import run
 source=out/'input_snapshot';source.mkdir()
 for slot,row in info['views'].items():shutil.copyfile(row['path'],source/(slot+Path(row['path']).suffix.lower()))
 r=profile['resolved'];run(source,Path(r['landmarker_model']),out/'crops',Path(r['rdf_python']),Path(r['rdf_repo']))
 metadata=read(out/'crops/report.json')
 for slot in SLOTS:
  metadata['views'][slot]['original_path']=info['views'][slot]['path']
  metadata['views'][slot]['original_sha256']=info['views'][slot]['sha256']
 write(out/'crops/report.json',metadata)


def pixel_observations(common,task,original_crops):
 import mediapipe as mp
 import numpy as np
 from PIL import Image
 from .prepare import TASK_SHA
 if sha(task)!=TASK_SHA:raise ValueError('Landmarker checksum mismatch')
 cropmeta=read(Path(original_crops)/'report.json');rows={}
 opt=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=1)
 with mp.tasks.vision.FaceLandmarker.create_from_options(opt) as detector:
  for slot in SLOTS:
   image=Image.open(common/(slot+'.png')).convert('RGB');result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(image)))
   if not result.face_landmarks:raise ValueError('Independent landmark observation unavailable for Pixel crop '+slot)
   points=np.array([[p.x*image.width,p.y*image.height] for p in result.face_landmarks[0]])
   row=dict(cropmeta['views'][slot]);row.update(crop_sha256=sha(common/(slot+'.png')),landmarks_crop=points.tolist(),face_box_crop=np.r_[points.min(0),points.max(0)].tolist(),normalized_to_crop=np.eye(3).tolist(),native_scale_verified=False)
   rows[slot]=row
 write(common/'report.json',{'status':'CROPS_PREPARED','views':rows,'source_crop_metadata_sha256':sha(Path(original_crops)/'report.json'),'native_scale_verified':False,'scope':'independent landmarks on Pixel native crops; native image scale not inferred from resized images'})
 from eye_components import identify
 from head24_policy import project,eye_gate
 with np.load(common/'projection_bundle.npz',allow_pickle=False) as bundle:
  _,eye_info=identify(bundle['vertices'][bundle['faces']].reshape(-1,3));xy,_=project(np.asarray(eye_info['eye_centers_native']),bundle['cameras'][0])
 lm=np.asarray(rows['front']['landmarks_crop']);observed=np.asarray([(lm[33]+lm[133])/2,(lm[362]+lm[263])/2]);gate=eye_gate(xy,observed)
 native=read(common/'native.report.json');fit={'status':'FIT_GATES_PASSED','variant':'pixel3dmm','projection_bundle_sha256':sha(common/'projection_bundle.npz'),'crop_report_sha256':sha(common/'report.json'),'native_report':native,'front_eye_center_gate':gate,'gate_scope':'numeric cameras and finite mesh only; independent face diagnostics not a calibrated pass'}
 write(common/'fit.report.json',fit)
 # Common consumers use separate fit and crop dirs; do not overwrite crop report.
 fitdir=common/'fit';fitdir.mkdir();shutil.copyfile(common/'projection_bundle.npz',fitdir/'projection_bundle.npz');write(fitdir/'report.json',fit)
 return fitdir,common


def run_variant(backend,dataset,crops,semantic,out,profile,code_key,resume=None):
 started=time.perf_counter();row={'backend':backend,'status':'RUNNING','stages':[],'ready_for_machine':False,'timing_scope':'fit and face diagnostics/appearance; figurine and manufacturing have separate stage timings'};folder=out/backend;folder.mkdir()
 try:
  pre=prerequisites(profile,backend);row['prerequisites']=pre
  if pre['status']!='PREREQUISITE_FILES_PRESENT':row['status']=pre['status'];return row
  resources=[Path(profile['resolved']['rdf_repo'])/rel for rel in ['weights/flame/flame2023.pkl','weights/flame/flame_assets.npz','weights/realdenseface/vitb.pth','weights/facebox/face_box.pth']]
  resource_hashes={str(p):sha(p) for p in resources if p.is_file()}
  stage_key=key({'models':resource_hashes,'source':dataset['input_fingerprint'],'input_dir':dataset['input_dir'],'variant':backend,'fitting':profile['fitting'],'code':stage_code(ROOT,'fit'),'crop_metadata':sha(crops/'report.json'),'semantic':sha(semantic/'report.json')})
  reused=False
  if backend.startswith('rdf_'):
   fit=folder/'fit'
   if resume is not None:reused=reuse_stage(resume/backend/'fit',fit,stage_key)
   if not reused:
    settings=folder/'fit.settings.json';write(settings,profile['fitting'])
    stage=execute_python('fit_'+backend,['-m','tools.headlab26.rdf_worker','--repo',profile['resolved']['rdf_repo'],'--crops',crops,'--output',fit,'--settings',settings,'--variant',backend,'--semantic',semantic,'--prediction-cache',out/'neural_predictions'],folder,profile,profile['resolved']['rdf_python']);row['stages'].append(stage);seal_stage(fit,stage_key)
   row['fit_reused_verified']=reused
  elif backend=='pixel3dmm':
   from .external import pixel3dmm
   native=pixel3dmm(ROOT,crops,folder/'native',profile);fit,crops=pixel_observations(native,Path(profile['resolved']['landmarker_model']),crops)
   from .semantics import run as parse
   semantic=folder/'semantic';parse(crops,Path(profile['resolved']['parser_model']),semantic)
  else:raise ValueError('Unknown backend '+backend)
  from .diagnostics import run as diagnostic
  row['diagnostics']=diagnostic(fit,crops,semantic,folder/'diagnostics',{'head_neck_mm':22.,'preview_size':profile['render']['preview_size']})
  row['fit_dir']=str(fit);row['crops_dir']=str(crops);row['semantic_dir']=str(semantic)
  workspace=folder/'workspace';stage=execute_python('workspace',['scripts/build_head24_workspace.py','--fit',fit,'--crops',crops,'--output',workspace],folder,profile);row['stages'].append(stage)
  row['workspace']=str(workspace/'head.workspace.html')
  from .face_states import export as states
  appearance_settings=read(ROOT/'configs/pipeline-v245.json');appearance_settings['semantic26']=str(semantic);appearance_settings['appearance23']['atlas_cell']=profile['render']['atlas_cell'];appearance_settings['preview_size']=profile['render']['preview_size']
  try:
   tick=time.perf_counter();row['skin']=states(workspace/'head.workspace.html',Path(profile['resolved']['parser_model']),appearance_settings,folder/'appearance');row['stages'].append({'stage':'face_appearance','seconds':time.perf_counter()-tick})
   for src,dst in [('source_reference.png','face_source.png'),('canonical_reference.png','face_canonical.png')]:shutil.copyfile(folder/'appearance'/src,folder/'diagnostics'/dst)
   row['appearance_cache']=str(folder/'appearance')
  except Exception as exc:row['appearance_error']=str(exc)

  row['status']='FACE_DIAGNOSTICS_EXPORTED';row['face_quality']='UNVERIFIED_WITH_OBSERVATION_WARNINGS'
 except Exception as exc:
  row['status']='FAILED';row['error']=str(exc);(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
 finally:
  row['seconds']=time.perf_counter()-started;write(folder/'report.json',row)
 return row


def choose_variants(valid,best_only):
 # An eye contour detector under glasses is not evidence of recovered eyes.
 supported=[v for v in valid if v['diagnostics']['eyes'][0].get('eye_semantic_pixels',0)>=20 and v['diagnostics']['eyes'][0].get('glasses_pixels',0)<20 and v['diagnostics'].get('eye_contour_error_proxy_median') is not None]
 if supported:
  ordered=sorted(supported,key=lambda v:v['diagnostics']['eye_contour_error_proxy_median'])
  ordered += [v for v in valid if v not in ordered]
  reason='visible front-eye semantic support; lowest detector-contour proxy, not likeness or beauty'
 else:
  ordered=sorted(valid,key=lambda v:v['backend']!='rdf_baseline')
  reason='eye evidence occluded/unavailable: retain baseline ordering; no claim that eye recovery improved'
 return (ordered[:1] if best_only else ordered),reason


def export_figurine(variant,dataset,out,profile):
 folder=out/variant['backend'];fit=Path(variant['fit_dir']);crops=Path(variant['crops_dir']);workspace=Path(variant['workspace']).parent
 # Face-only source-color and common-color diagnostics remain inspectable even
 # if all hair candidates fail. No shape/eye anatomy is reconstructed here.
 settings=read(ROOT/'configs/pipeline-v245.json');settings['semantic26']=variant['semantic_dir'];settings['reuse_face26']=variant.get('appearance_cache');settings['appearance23']['atlas_cell']=profile['render']['atlas_cell'];settings['preview_size']=profile['render']['preview_size'];settings['hair26']={'methods':profile['figurine']['hair_methods'],'iterations':profile['figurine']['arap_iterations'],'max_displacement_mm':profile['figurine']['arap_max_displacement_mm'],'ear_coverage_limit':profile['figurine']['ear_coverage_limit'],'cell_mm':profile['figurine'].get('arap_cell_mm',.55),'max_nodes':profile['figurine'].get('arap_max_nodes',6000),'min_fraction':profile['figurine'].get('arap_min_fraction',.125)}
 write(folder/'figurine.settings.json',settings)
 stage=execute('figurine',['%s'%sys.executable,'-m','tools.headlab26.figurine','--workspace',workspace/'head.workspace.html','--parser-model',profile['resolved']['parser_model'],'--profile',folder/'figurine.settings.json','--output',folder/'figurine'],ROOT,folder,profile['execution']['stage_timeout_seconds']);variant['stages'].append(stage)
 result_path=folder/'figurine/report.json'
 if result_path.is_file():
  result=read(result_path);variant['figurine_status']=result['status'];variant['selected']=result.get('selected');variant['skin']=result.get('face',{}).get('skin');variant['manufacturing']='UNVERIFIED';variant['hair_candidate_counts']={'exported':sum(c['status']=='EXPORTED' for c in result.get('candidates',[])),'rejected':sum(c['status']=='FAILED' for c in result.get('candidates',[]))}
 for src,dst in [('shared/source_reference.png','face_source.png'),('shared/canonical_reference.png','face_canonical.png')]:
  if (folder/'figurine'/src).is_file():shutil.copyfile(folder/'figurine'/src,folder/'diagnostics'/dst)
 if stage['exit_code']!=0:variant['figurine_error']='No valid candidate or failed stage; face diagnostics retained'
 if (folder/'figurine/selected/head_mm.stl').is_file():
  from .manufacturing import run as manufacturing
  variant['manufacturing_report']=manufacturing(folder/'figurine/selected/head_mm.stl',folder/'manufacturing',profile['manufacturing'],ROOT)
 write(folder/'report.json',variant)


def run_dataset(info,batch,profile,backends,code_key,check_only,resume=None):
 dataset=dict(info,variants=[],stages=[]);out=batch/'datasets'/info['id'];out.mkdir(parents=True)
 try:
  write(out/'input.report.json',info)
  if info['errors']:return dataset
  required=prerequisites(profile,'prepare');dataset['prepare_prerequisites']=required
  if check_only:
   dataset['backend_prerequisites']=[prerequisites(profile,b) for b in backends];dataset['status']='PREFLIGHT_ONLY';return dataset
  if required['status']!='PREREQUISITE_FILES_PRESENT':dataset['status']=required['status'];dataset['errors']=required['missing'];return dataset
  cpu=prerequisites(profile,'rdf_baseline');missing=[p for p in cpu.get('missing',[]) if p in (profile['resolved']['parser_model'],)]
  if missing:dataset['status']='PREREQUISITES_MISSING';dataset['errors']=missing;return dataset
  roi_weight=Path(profile['resolved']['rdf_repo'])/'weights/facebox/face_box.pth'
  source_key=key({'input_fingerprint':info['input_fingerprint'],'input_dir':info['input_dir'],'code':stage_code(ROOT,'prepare'),'landmarker_sha':sha(profile['resolved']['landmarker_model']),'roi_weight_sha256':sha(roi_weight) if roi_weight.is_file() else None})
  reused=False
  if resume is not None:reused=reuse_stage(resume/'crops',out/'crops',source_key)
  tick=time.perf_counter()
  if not reused:prepare_one(info,out,profile);seal_stage(out/'crops',source_key)
  dataset['stages'].append({'stage':'prepare','seconds':time.perf_counter()-tick,'reused_verified':reused})
  semantic_key=key({'crops':sha(out/'crops/report.json'),'parser':sha(profile['resolved']['parser_model']),'code':stage_code(ROOT,'semantic')});reused=False
  if resume is not None:reused=reuse_stage(resume/'semantic',out/'semantic',semantic_key)
  tick=time.perf_counter()
  if not reused:
   from .semantics import run as parse
   parse(out/'crops',Path(profile['resolved']['parser_model']),out/'semantic');seal_stage(out/'semantic',semantic_key)
  dataset['stages'].append({'stage':'semantic','seconds':time.perf_counter()-tick,'reused_verified':reused})
  for backend in backends:
   print('Dataset',info['id'],'backend',backend,flush=True);row=run_variant(backend,info,out/'crops',out/'semantic',out,profile,code_key,resume);dataset['variants'].append(row);write(out/'dataset.report.json',dataset)
  valid=[r for r in dataset['variants'] if r['status']=='FACE_DIAGNOSTICS_EXPORTED']
  if valid and profile['figurine']['enabled']:
   selected,dataset['selection_scope']=choose_variants(valid,profile['figurine']['best_only'])
   for variant in selected:
    try:export_figurine(variant,dataset,out,profile)
    except Exception as exc:variant['figurine_error']=str(exc);write(out/variant['backend']/'report.json',variant)
  if profile.get('trust',{}).get('enabled'):
   pre=prerequisites(profile,'trust');dataset['trust']=pre
   if pre['status']=='PREREQUISITE_FILES_PRESENT':
    spec=profile['trust'];row=execute('trust',[resolve(ROOT,spec['python']),'-m','tools.headlab26.trust_worker','--repo',resolve(ROOT,spec['repo']),'--crops',out/'crops','--output',out/'trust'],ROOT,out,profile['execution']['stage_timeout_seconds']);dataset['trust']={'status':'MODEL_ALBEDO_ESTIMATES_EXPORTED' if row['exit_code']==0 and (out/'trust/report.json').is_file() else 'FAILED','execution':row}
  dataset['status']='DIAGNOSTICS_EXPORTED' if valid else 'NO_SUCCESSFUL_BACKEND'
 except Exception as exc:dataset['status']='FAILED';dataset['errors'].append(str(exc));(out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
 finally:write(out/'dataset.report.json',dataset)
 return dataset


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=Path,default=ROOT/'input');p.add_argument('--input-dir',type=Path);p.add_argument('--datasets');p.add_argument('--backends');p.add_argument('--profile',type=Path);p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--check-only',action='store_true');p.add_argument('--face-only',action='store_true');p.add_argument('--all-variants',action='store_true');a=p.parse_args(argv)
 profile=load(ROOT,a.profile)
 if a.face_only:profile['figurine']['enabled']=False
 if a.all_variants:profile['figurine']['best_only']=False
 backends=(a.backends.split(',') if a.backends else profile['backends'])
 if not backends or any(b not in ['rdf_baseline','rdf_occlusion','pixel3dmm'] for b in backends):raise ValueError('Unsupported/empty backend list')
 backends=list(dict.fromkeys(backends));output=(a.output or ROOT/'runs'/('headlab26_'+stamp())).resolve()
 if output.exists():raise ValueError('Output already exists; never overwrite a run')
 verify_code(ROOT,ROOT/'head26-code.json');verify_code(ROOT,ROOT/'pipeline-v23-code.json');verify_code(ROOT,ROOT/'head245-code.json')
 if Path(profile['resolved']['rdf_repo']).is_dir():verify_code(Path(profile['resolved']['rdf_repo']),ROOT/'head24-rdf-source.json')
 output.mkdir(parents=True);write(output/'profile.used.json',profile)
 code_key=key({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/headlab26').glob('*.py')})
 report={'version':'HeadLab26.1','status':'RUNNING','input_root':str(a.input_root.resolve()),'code_key':code_key,'datasets':[],'ready_for_machine':False,'physical_print_verified':False,'backends':backends};started=time.perf_counter();old={}
 if a.resume:
  previous=read(a.resume/'batch.report.json');old={r['id']:r for r in previous['datasets']}
 try:
  for name,folder in discover(a.input_root,a.datasets,a.input_dir):
   print('\nDataset:',name,flush=True)
   try:info=inspect_dataset(name,folder)
   except Exception as exc:info={'id':name,'input_dir':str(folder),'status':'INPUT_INVALID','errors':[str(exc)],'views':{}}
   source=None
   if a.resume and name in old and info.get('input_fingerprint')==old[name].get('input_fingerprint'):source=a.resume/'datasets'/name
   row=run_dataset(info,output,profile,backends,code_key,a.check_only,source);report['datasets'].append(row);summarize(output,report)
  if not report['datasets']:report['status']='NO_DATASETS'
  elif a.check_only:report['status']='PREFLIGHT_ISSUES' if any(d.get('errors') or d.get('prepare_prerequisites',{}).get('status')!='PREREQUISITE_FILES_PRESENT' or any(p['status']!='PREREQUISITE_FILES_PRESENT' for p in d.get('backend_prerequisites',[])) for d in report['datasets']) else 'PREFLIGHT_ONLY'
  else:report['status']='BATCH_COMPLETED_WITH_WARNINGS' if any(d['status']!='DIAGNOSTICS_EXPORTED' or any(v['status']!='FACE_DIAGNOSTICS_EXPORTED' or v.get('appearance_error') or v.get('figurine_error') for v in d.get('variants',[])) for d in report['datasets']) else 'BATCH_DIAGNOSTICS_EXPORTED'
 except KeyboardInterrupt:report['status']='INTERRUPTED'
 except Exception as exc:report['status']='FAILED';report['error']=str(exc)
 finally:
  report['seconds']=time.perf_counter()-started;summarize(output,report);print('Review:',output/'index.html',flush=True)
 return 0 if report['status'] in ('PREFLIGHT_ONLY','BATCH_DIAGNOSTICS_EXPORTED') else 2

if __name__=='__main__':raise SystemExit(main())
