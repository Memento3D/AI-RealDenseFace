"""Scored FaceBox bbox fallback for side photos; never invents landmark coordinates."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from head24_detection import scan_windows,merge_detections
PIN='3fde8561f059dd60d876021cbd2f67d955d5e02f24c25b297913a4115e433b4d'

def map_box(box,score,window,padded_side,check_edges):
    if not np.isfinite(score) or score<.6:return None
    b=np.asarray(box,float)/512*padded_side
    x0,y0,x1,y1=window;w=x1-x0;h=y1-y0
    if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=b[:2]).any():return None
    # Reject padding/outside/partial tile detections instead of clipping them into plausible boxes.
    if (b[:2]<0).any() or b[2]>w or b[3]>h:return None
    if check_edges and (b[0]<.015*w or b[1]<.015*h or b[2]>.985*w or b[3]>.985*h):return None
    native=b+np.array([x0,y0,x0,y0])
    return {'box':native,'score':float(score),'window':list(window),'effective_face_fraction':float(min((b[2]-b[0])/w,(b[3]-b[1])/h))}

def run(repo,source,out):
    if out.exists():raise ValueError('Profile detection output already exists')
    sys.path.insert(0,str(repo))
    import torch
    from models.facebox.detector import FaceDetector
    torch.set_num_threads(4)
    weights=repo/'weights/facebox/face_box.pth'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=PIN:raise ValueError('FaceBox weights checksum mismatch')
    detector=FaceDetector(str(weights),(512,512),threshold=.6,half=False,device='cpu',compile_model=False)
    with Image.open(source) as im:image=ImageOps.exif_transpose(im).convert('RGB')
    windows=scan_windows(image.size);candidates=[];report={'backend':'facebox','weights_sha256':PIN,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'native_size':image.size,'calls':0,'above_score_threshold':0,'rejected_padding_or_tile_edge':0,'threshold':.6,'landmarks_available':False,'fallback_full_image_box_used':False,'max_score_seen':0.0}
    lut=[round(255*(i/255.)**.6) for i in range(256)]
    for variant in ('original','shadow_lift_detection_only'):
        for index,window in enumerate(windows):
            tile=image.crop(window)
            if variant!='original':tile=tile.point(lut*3)
            side=max(tile.size);canvas=Image.new('RGB',(side,side));canvas.paste(tile,(0,0));canvas=canvas.resize((512,512),Image.Resampling.BILINEAR)
            tensor=torch.from_numpy(np.array(canvas)).permute(2,0,1).unsqueeze(0).float()/255.
            box,score=detector.detect_one(tensor);report['calls']+=1
            report['max_score_seen']=max(report['max_score_seen'],float(score))
            if score>=.6:report['above_score_threshold']+=1
            record=map_box(box,score,window,side,index!=0)
            if record is not None:record['variant']=variant;candidates.append(record)
            elif score>=.6:report['rejected_padding_or_tile_edge']+=1
            if report['calls']%12==0:print('Profile FaceBox windows:',report['calls'],'/',2*len(windows),flush=True)
    clusters=merge_detections(candidates)
    report['candidates']=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in clusters]
    report['status']='BBOX_CANDIDATES_FOUND' if clusters else 'DETECTION_UNRESOLVED'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Profile FaceBox candidates:',len(clusters),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.repo.resolve(),a.source.resolve(),a.output.resolve())
