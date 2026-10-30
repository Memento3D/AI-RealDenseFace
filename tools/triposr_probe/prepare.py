"""Crop a head observation without projecting it onto a scalp or painting over hair."""
import argparse,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from .common import ROOT,sha,write
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]

def crop_box(box):
 x0,y0,x1,y1=map(float,box);w=x1-x0;h=y1-y0
 if w<=0 or h<=0:raise ValueError('Invalid observed face box')
 side=max(w*1.8,h*2.05);cx=(x0+x1)/2;cy=(y0+y1)/2-.20*h
 return tuple(map(int,np.rint([cx-side/2,cy-side/2,cx+side/2,cy+side/2])))

def run(source,task,out,whole_image=False):
 source=Path(source).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=False)
 with Image.open(source) as im:
  if im.width*im.height>24_000_000:raise ValueError('Image pixel budget exceeded')
  image=ImageOps.exif_transpose(im).convert('RGB')
 report={'source':str(source),'source_sha256':sha(source),'native_size':list(image.size),'hair_mask_used':False,'source_pixels_projected_onto_scalp':False,'missing_hair_recovered':False}
 if whole_image:box=(0,0,image.width,image.height);report['scope']='whole image explicitly requested'
 else:
  import mediapipe as mp
  from head24_detection import scan_faces
  from head24_policy import choose_face
  if sha(task)!='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff':raise ValueError('Unexpected MediaPipe task checksum')
  options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=5)
  with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
   def detect(rgb):
    r=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)));return [np.array([[p.x,p.y] for p in points]) for points in r.face_landmarks]
   found,trace=scan_faces(image,detect)
  boxes=[row['box'] for row in found];idx=choose_face(boxes,image.size);box=crop_box(boxes[idx]);report.update(face_box_native=np.asarray(boxes[idx]).tolist(),detection=trace,scope='geometric face-derived head crop, not independently validated complete hair mask')
 left,top,right,bottom=box;crop=Image.new('RGB',(right-left,bottom-top),'white');ix0=max(0,left);iy0=max(0,top);ix1=min(image.width,right);iy1=min(image.height,bottom)
 crop.paste(image.crop((ix0,iy0,ix1,iy1)),(ix0-left,iy0-top));crop.thumbnail((1024,1024),Image.Resampling.LANCZOS);crop.save(out/'head_crop.png')
 overlay=image.copy();ImageDraw.Draw(overlay).rectangle(box,outline=(255,0,200),width=max(2,image.width//400));overlay.thumbnail((1200,1200));overlay.save(out/'crop_overlay.jpg')
 report.update(crop_box_native=list(box),white_padding_applied=bool(left<0 or top<0 or right>image.width or bottom>image.height),padding_scope='canvas only; no evidence that hair outside original image was observed',crop_sha256=sha(out/'head_crop.png'),status='OBSERVATION_CROP_EXPORTED');write(out/'report.json',report)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--whole-image',action='store_true');a=p.parse_args()
 try:run(a.source,a.task,a.output,a.whole_image)
 except Exception as exc:
  a.output.mkdir(parents=True,exist_ok=True);write(a.output/'failure.json',{'status':'OBSERVATION_PREPARATION_FAILED','error':str(exc),'neural_inference_executed':False});raise
