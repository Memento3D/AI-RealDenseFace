"""TRUST inference on real face+scene observations with strict weight existence checks."""
import argparse,sys,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

def run(repo,crops,output):
 import torch
 sys.path.insert(0,str(repo))
 from test import fixed_dict,config_test
 from lib.model import TRUST
 from lib import util
 from tools.headlab26.io import read,write,sha,SLOTS
 # Upstream _create_model silently keeps random encoders if optional weights are
 # absent. Refuse this before model construction, and record every loaded hash.
 config=util.dict2obj({**fixed_dict,**config_test,'batch_size':1,'n_scenelight':3,'n_facelight':27,'tex_type':'BalanceAlb','resume_training':True,'pretrained_modelpath_scene':'./data/E_scene_light_BalanceAlb.tar','pretrained_modelpath_facel':'./data/E_face_light_BalanceAlb.tar','pretrained_modelpath_albedo':'./data/E_albedo_BalanceAlb.tar'})
 paths=[repo/'data/deca_model.tar',repo/'data/E_scene_light_BalanceAlb.tar',repo/'data/E_face_light_BalanceAlb.tar',repo/'data/E_albedo_BalanceAlb.tar']
 for name in ['topology_path','flame_model_path','flame_lmk_embedding_path','BalanceAlb_tex_path','lightprobe_normal_path','lightprobe_albedo_path']:paths.append(repo/getattr(config,name))
 for path in paths:
  if not path.is_file():raise ValueError('Missing mandatory TRUST resource: '+str(path))
 original=sys.path[:];import os
 os.chdir(repo);model=TRUST(config)
 for encoder in [model.E_flame,model.E_albedo,model.E_scene_light,model.E_face_light]:encoder.eval()
 metadata=read(crops/'report.json');output.mkdir(parents=True,exist_ok=False);rows=[]
 for slot in SLOTS:
  image=Image.open(crops/(slot+'.png')).convert('RGB');box=np.asarray(metadata['views'][slot]['face_box_crop']);center=(box[:2]+box[2:])/2;half=max(box[2:]-box[:2])*.8;roi=tuple(np.r_[center-half,center+half]);face=image.crop(roi).resize((224,224),Image.Resampling.BILINEAR)
  scene_path=Path(metadata['views'][slot]['original_path']);scene=ImageOps.exif_transpose(Image.open(scene_path)).convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
  tensor=lambda im:torch.from_numpy(np.asarray(im,dtype='float32').transpose(2,0,1)/255)[None].to('cuda')
  with torch.no_grad():scale,sh,tex,light,albedo,ops,shape=model.encoding(tensor(face),tensor(scene))
  alb=albedo[0].detach().cpu().numpy().transpose(1,2,0);rgb=np.rint(np.clip(alb,0,1)*255).astype('uint8');Image.fromarray(rgb).save(output/(slot+'.albedo.png'))
  np.savez_compressed(output/(slot+'.params.npz'),scene_scale=scale.detach().cpu().numpy(),texcode=tex.detach().cpu().numpy(),light=light.detach().cpu().numpy())
  pixels=rgb.reshape(-1,3);valid=(pixels.min(1)>5)&(pixels.max(1)<250);estimate=np.median(pixels[valid],axis=0) if valid.sum()>100 else None
  if estimate is None:raise ValueError('TRUST albedo texture has insufficient finite usable pixels')
  rows.append({'view':slot,'rgb8_texture_median_proxy':estimate.astype(int).tolist(),'crop_sha256':sha(crops/(slot+'.png')),'scene_sha256':sha(scene_path),'scope':'unmasked UV albedo median proxy, NOT a skin-only color truth or validated material swatch'})
 write(output/'report.json',{'status':'MODEL_ALBEDO_ESTIMATES_EXPORTED','model':'TRUST ECCV2022 BalanceAlb','views':rows,'weights_sha256':{p.name:sha(p) for p in paths},'true_skin_color_verified':False,'automatically_applied_to_face':False,'warning':'Albedo models may bias skin tone; compare UV textures and calibrated references before applying a base color'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--crops',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.repo.resolve(),a.crops.resolve(),a.output.resolve())
