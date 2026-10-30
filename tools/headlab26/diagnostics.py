"""Face-only geometry and image-bound eyelid proxies, distinct from mesh export success."""
import json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.distance import cdist
from .io import write,read,sha,SLOTS
from .policy import eye_aperture

# FLAME2023 eyelid vertex groups from the pinned RDF optimizer, not eyeball centres.
LIDS=([807,814,822,824,827,991,994,995,1093,1096,1113,1170,1175,1195,1200,1201,1202,1216,1218,1329,1331,1336,1340,1343,1344,1354,1355,1357,3833,3855],
[2264,2267,2271,2273,2276,2355,2358,2359,2402,2403,2406,2436,2437,2447,2448,2449,2450,2451,2453,2485,2486,2487,2491,2494,2495,2505,2506,2508,3632,3689])
MP_EYES=([33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466])

def eye_measurements(vertices,camera,landmarks,labels,native_scale):
 from head24_policy import project
 projected=[project(vertices[np.asarray(group)],camera)[0] for group in LIDS]
 observed=[np.asarray(landmarks)[group] for group in MP_EYES]
 centers=np.asarray([p.mean(0) for p in observed]);distance=float(np.linalg.norm(centers[1]-centers[0]))
 if distance<1:raise ValueError('Degenerate independent eye observation')
 if sum(np.linalg.norm(projected[i].mean(0)-centers[i]) for i in range(2))>sum(np.linalg.norm(projected[i].mean(0)-centers[1-i]) for i in range(2)):observed=observed[::-1]
 rows=[]
 for i,(a,b) in enumerate(zip(projected,observed)):
  d=cdist(a,b);error=float((d.min(0).mean()+d.min(1).mean())/2/distance)
  xy=np.rint(b).astype(int);valid=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<labels.shape[1])&(xy[:,1]<labels.shape[0]);classes=labels[xy[valid,1],xy[valid,0]]
  rows.append({'group':i,'contour_distance_interocular_proxy':error,'model_lid_band_height_width_ratio':eye_aperture(a),'observed_contour_height_width_ratio':eye_aperture(b),'native_interocular_px':distance*native_scale if native_scale is not None else None,'observed_glasses_sample_fraction':float((classes==6).mean()) if len(classes) else None,'confidence':'LOW_OCCLUSION_OR_UNKNOWN','correspondence_scope':'unordered lid-band to independent detector contour; not exact anatomical vertex correspondence or gaze'})
 return rows,projected,observed

def run(fit,crops,semantic,out,profile):
 from head24_policy import project
 from eye_components import identify
 from run_pipeline_v2 import glb
 import trimesh
 from run_pipeline_v24 import preview
 out=Path(out);out.mkdir(parents=True,exist_ok=False);crop=read(Path(crops)/'report.json');fitreport=read(Path(fit)/'report.json')
 if sha(Path(crops)/'report.json')!=fitreport['crop_report_sha256']:raise ValueError('Fit/crop metadata belongs to a different observation')
 for slot in SLOTS:
  if sha(Path(crops)/(slot+'.png'))!=crop['views'][slot]['crop_sha256']:raise ValueError('Diagnostic crop bytes changed: '+slot)
 with np.load(Path(fit)/'projection_bundle.npz',allow_pickle=False) as data:
  source=data['vertices'];faces=data['faces'];views=data['view_vertices'];cameras=data['cameras']
 if sha(Path(fit)/'projection_bundle.npz')!=fitreport['projection_bundle_sha256']:raise ValueError('Fit bundle changed')
 corners=source[faces].reshape(-1,3);component,_=identify(corners);head=corners[np.repeat(component==0,3)];lo,hi=head.min(0),head.max(0);scale=profile['head_neck_mm']/(hi[1]-lo[1]);center=(lo+hi)/2;model=trimesh.Trimesh((source-center)*scale,faces,process=False)
 glb(out/'face_gray.glb',[('SourceFace',model,[.68,.68,.68,1.])]);preview(out/'face_gray.glb',out/'face_gray.png',([0,0,0],profile['head_neck_mm']*1.4),profile['preview_size'])
 rows=[];all_metrics=[]
 for i,slot in enumerate(SLOTS):
  image=Image.open(Path(crops)/(slot+'.png')).convert('RGB');labels=np.asarray(Image.open(Path(semantic)/(slot+'.labels.png')));lm=crop['views'][slot].get('landmarks_crop');overlay=image.copy();draw=ImageDraw.Draw(overlay);eye_rows=[]
  if lm is not None:
   native_scale=None if crop['views'][slot].get('native_scale_verified') is False else 1/float(np.asarray(crop['views'][slot]['normalized_to_crop'])[0,0]);eye_rows,pred,obs=eye_measurements(views[i],cameras[i],lm,labels,native_scale)
   for group in pred:
    for p in group:draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=(255,80,200))
   for group in obs:
    for p in group:draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=(40,230,80))
   all_metrics.extend(r['contour_distance_interocular_proxy'] for r in eye_rows)
  overlay.save(out/(slot+'.eye_overlay.png'))
  rows.append({'view':slot,'eyes':eye_rows,'independent_landmarks_available':lm is not None,'eye_semantic_pixels':int(np.isin(labels,[4,5]).sum()),'glasses_pixels':int((labels==6).sum()),'native_face_size_px':crop['views'][slot].get('native_face_size_px'),'face_luma_median':crop['views'][slot].get('face_luma_median'),'warnings':crop['views'][slot].get('warnings',[])})
 result={'status':'FACE_DIAGNOSTICS_EXPORTED','variant':fitreport.get('variant','legacy_import'),'geometry_changed':False,'face_accuracy_verified':False,'eyes':rows,'eye_contour_error_proxy_median':float(np.median(all_metrics)) if all_metrics else None,'mouth_anatomical_error':'NOT_MEASURED: no independently validated 3D lip correspondence','scope':'source geometry and per-view image-bound detector comparison; not identity proof, ground-truth calibration, or print validation'}
 write(out/'report.json',result);return result
