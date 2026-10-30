"""Real TripoSR inference, with raw exports and explicit CPU isosurface extraction."""
import argparse,sys,time,traceback,json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .common import ROOT,MODEL_REVISION,DINO_REVISION,check_repo,sha,write,source_repo


def preview(mesh,path,size=320):
 # Diagnostic orthographic painter preview, Z up, not registered input cameras.
 p=np.asarray(mesh.vertices);center=(p.min(0)+p.max(0))/2;p=p-center;scale=.8*size/max(np.ptp(p,axis=0).max(),1e-8);sheet=Image.new('RGB',(size*4,size+24),(28,32,38));draw=ImageDraw.Draw(sheet)
 colors=np.asarray(mesh.visual.vertex_colors)[:,:3] if mesh.visual.kind=='vertex' else np.full((len(p),3),180)
 for index,degrees in enumerate([0,90,180,270]):
  angle=np.deg2rad(degrees);depth=p[:,0]*np.cos(angle)+p[:,1]*np.sin(angle);horizontal=-p[:,0]*np.sin(angle)+p[:,1]*np.cos(angle);xy=np.column_stack([horizontal*scale+size*.5,-p[:,2]*scale+size*.5]);faces=np.asarray(mesh.faces);order=np.argsort(depth[faces].mean(1));panel=Image.new('RGB',(size,size),(35,40,47));d=ImageDraw.Draw(panel)
  # Display decimation is explicit; exported mesh retains every extracted face.
  order=order[::max(1,len(order)//100000)]
  for f in faces[order]:d.polygon([tuple(point) for point in xy[f]],fill=tuple(np.asarray(colors[f].mean(0),dtype=np.uint8)))
  sheet.paste(panel,(index*size,24));draw.text((index*size+8,5),'native '+str(degrees)+' deg',fill='white')
 sheet.save(path)


def run(a):
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();report={'status':'RUNNING','model':'TripoSR','model_revision':MODEL_REVISION,'source_image_sha256':sha(a.image),'ready_for_machine':False,'hair_isolated':False,'face_identity_verified':False,'source_face_preserved':False,'units':'native neural field units, not mm','stages':{}}
 try:
  import torch,rembg,trimesh
  from huggingface_hub import hf_hub_download
  from .surface import register_backend
  report['surface_backend']=register_backend();repo=source_repo()
  if repo is None:raise ValueError('Pinned TripoSR source is missing; run setup first')
  report['source_repo']=str(repo);check_repo(repo);sys.path.insert(0,str(repo))
  from tsr.system import TSR
  from tsr.utils import remove_background,resize_foreground
  import tsr.models.tokenizers.image as tokenizer_module
  original_download=tokenizer_module.hf_hub_download
  def pinned_tokenizer_download(*args,**kwargs):
   if kwargs.get('repo_id')!='facebook/dino-vitb16':raise ValueError('Unexpected image tokenizer repository')
   kwargs['revision']=DINO_REVISION;path=original_download(*args,**kwargs);report['dino_config']={'revision':DINO_REVISION,'sha256':sha(path)};return path
  tokenizer_module.hf_hub_download=pinned_tokenizer_download
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: stop instead of silently attempting slow CPU neural inference')
  report['gpu']={'name':torch.cuda.get_device_name(0),'total_vram_bytes':torch.cuda.get_device_properties(0).total_memory,'torch_version':torch.__version__,'torch_cuda':torch.version.cuda}
  report['active_stage']='download_or_cache';tick=time.perf_counter();files={name:Path(hf_hub_download('stabilityai/TripoSR',name,revision=MODEL_REVISION)) for name in ['config.yaml','model.ckpt']};report['model_files_sha256']={name:sha(path) for name,path in files.items()};report['stages']['download_or_cache_seconds']=time.perf_counter()-tick
  report['active_stage']='load_model';tick=time.perf_counter();model=TSR.from_pretrained(str(files['config.yaml'].parent),config_name='config.yaml',weight_name='model.ckpt');model.renderer.set_chunk_size(a.chunk_size);model.eval().to('cuda:0');torch.cuda.synchronize();report['stages']['load_seconds']=time.perf_counter()-tick
  if getattr(a,'prepared_image',False):
   image=Image.open(a.image).convert('RGB');image.save(out/'model_input.png')
   report['prepared_image_reused']=True
   report['foreground']={'scope':'existing model input reused; no new background removal or crop'}
  else:
   report['active_stage']='foreground';tick=time.perf_counter();session=rembg.new_session('u2net');rgba=remove_background(Image.open(a.image).convert('RGB'),session);rgba.save(out/'foreground_rgba.png');alpha=np.asarray(rgba.getchannel('A'));report['foreground']={'pixels_above_half_alpha':int((alpha>127).sum()),'top_alpha_contact':bool((alpha[0]>127).any()),'side_alpha_contact':bool((alpha[:,0]>127).any() or (alpha[:,-1]>127).any()),'scope':'person foreground, not a validated hair mask'}
   if (alpha>127).sum()<100:raise ValueError('Foreground removal returned insufficient support')
   observed=resize_foreground(rgba,.85);rgb=np.asarray(observed,float)/255.;rgb=rgb[:,:,:3]*rgb[:,:,3:]+.5*(1-rgb[:,:,3:]);image=Image.fromarray(np.rint(rgb*255).astype(np.uint8));image.save(out/'model_input.png');report['stages']['foreground_seconds']=time.perf_counter()-tick
   bg_weight=Path(os.environ['U2NET_HOME'])/'u2net.onnx'
   if bg_weight.is_file():report['background_model']={'name':'u2net','local_sha256':sha(bg_weight),'independently_pinned':False}
  report['active_stage']='neural';torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
  with torch.inference_mode():codes=model([image],device='cuda:0')
  torch.cuda.synchronize();report['stages']['neural_seconds']=time.perf_counter()-tick
  resolutions=sorted(set(getattr(a,'resolutions',None) or [a.resolution]));report['variants']=[]
  # One neural forward pass shared by every extraction; no separate AI guesses.
  from .audit import metrics,render
  for resolution in resolutions:
   report['active_stage']='extract_surface_'+str(resolution);tick=time.perf_counter()
   target=out if len(resolutions)==1 else out/('resolution_'+str(resolution))
   if target!=out:target.mkdir()
   item={'resolution':resolution,'status':'RUNNING','directory':target.name if target!=out else '.'}
   try:
    with torch.inference_mode():mesh=model.extract_mesh(codes,True,resolution=resolution,threshold=25.0)[0]
    torch.cuda.synchronize()
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():raise ValueError('Empty or nonfinite extracted mesh')
    mesh.export(target/'raw_native.glb');mesh.export(target/'raw_native.obj')
    preview(mesh,target/'raw_preview.png');render(mesh,target/'geometry_gray.png',gray=True)
    item.update(status='EXPORTED',geometry=metrics(mesh),seconds=time.perf_counter()-tick)
    item['output_sha256']={p.name:sha(p) for p in target.iterdir() if p.is_file()}
    write(target/'geometry.json',item['geometry'])
    del mesh
   except torch.cuda.OutOfMemoryError as exc:
    item.update(status='CUDA_OUT_OF_MEMORY',error=str(exc),seconds=time.perf_counter()-tick)
    torch.cuda.empty_cache()
   report['variants'].append(item)
   write(out/'report.json',report)
  report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
  if len(resolutions)==1 and report['variants'][0]['status']=='EXPORTED':report['geometry']=report['variants'][0]['geometry']
  report['status']='RAW_PROPOSAL_EXPORTED' if all(v['status']=='EXPORTED' for v in report['variants']) else 'RAW_PROPOSALS_PARTIAL'
  report['comparison']='same prepared pixels, same latent field, same threshold 25; extraction resolution only'
  report['source_image_unchanged']=sha(a.image)==report['source_image_sha256']
  report['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('report.json','review.html')}
 except Exception as exc:
  report['status']='FAILED';report['error']=str(exc);(out/'traceback.txt').write_text(traceback.format_exc());raise
 finally:
  report['seconds']=time.perf_counter()-started;write(out/'report.json',report)
  text='<!doctype html><meta charset="utf-8"><title>Raw TripoSR probe</title><style>body{background:#171c24;color:white;font:16px system-ui;margin:24px}img{max-width:100%}a{color:#aaddff}</style><h1>TripoSR — raw head proposal</h1><p>Single-image generated head/bust. Hair is not isolated. Hidden shape is inferred. Units are native; not ready for printing. Preview cameras are unregistered.</p>'
  for name in ['model_input.png','raw_preview.png']:
   if (out/name).is_file():text+='<p>'+name+'</p><img src="'+name+'">'
  for variant in report.get('variants',[]):
   directory=variant['directory'];prefix='' if directory=='.' else directory+'/'
   text+='<h2>Resolution '+str(variant['resolution'])+' — '+variant['status']+'</h2>'
   for name in ['raw_preview.png','geometry_gray.png']:
    if (out/prefix/name).is_file():text+='<p>'+name+'</p><img src="'+prefix+name+'">'
   if (out/prefix/'raw_native.glb').is_file():text+='<p><a href="'+prefix+'raw_native.glb">GLB</a></p>'
  for name in ['raw_native.glb','raw_native.obj','report.json']:
   if (out/name).is_file():text+='<p><a href="'+name+'">'+name+'</a></p>'
  (out/'review.html').write_text(text,encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',type=int,choices=[128,192,256],default=128);p.add_argument('--chunk-size',type=int,default=2048);p.add_argument('--resolutions',type=int,nargs='+',choices=[128,192,256]);p.add_argument('--prepared-image',action='store_true');a=p.parse_args()
 if not 128<=a.chunk_size<=8192:p.error('chunk-size must be between128 and8192')
 run(a)
 if json.loads((a.output/'report.json').read_text())['status']=='RAW_PROPOSALS_PARTIAL':raise SystemExit(2)
