"""Independent MediaPipe landmark detection, ambiguity checks and traceable head crops."""
import argparse,json,hashlib,sys,subprocess,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from head24_policy import choose_face,crop_transform,apply_pixels,InputRejected
from head24_detection import scan_faces
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
TASK_SHA='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'

def available_slots(folder):
    slots=[]
    for slot in ('front','left','right'):
        paths=[p for p in Path(folder).iterdir() if p.is_file() and p.stem.lower()==slot and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')]
        if len(paths)>1:raise InputRejected('DUPLICATE_VIEW: '+slot)
        if paths:slots.append(slot)
    if 'front' not in slots:raise InputRejected('FRONT_REQUIRED')
    return slots


def face_crop(box,size):
    # This crop is for RDF face fitting. Hair observation uses its separate crop.
    x0,y0,x1,y1=np.asarray(box,float)
    if not np.isfinite([x0,y0,x1,y1]).all() or x1<=x0 or y1<=y0:raise InputRejected('INVALID_FACE_BOX')
    side=int(np.ceil(max(x1-x0,y1-y0)*1.8))
    left=int(round((x0+x1-side)/2));top=int(round((y0+y1-side)/2-.10*(y1-y0)))
    a=1024/side
    return [left,top,left+side,top+side],np.array([[a,0,-a*left],[0,a,-a*top],[0,0,1.]])


def run(folder,task,out,rdf_python=None,repo=None):
    if hashlib.sha256(task.read_bytes()).hexdigest()!=TASK_SHA:raise ValueError('MediaPipe model checksum mismatch')
    import mediapipe as mp
    options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(task),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=5,output_facial_transformation_matrixes=True)
    out.mkdir(parents=True,exist_ok=False);report={'status':'RUNNING','views':{},'task_sha256':TASK_SHA,'thresholds_calibrated':False,'subject_identity_verified':False}
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            for slot in available_slots(folder):
                files=[p for p in folder.iterdir() if p.is_file() and p.stem.lower()==slot and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')]
                if len(files)!=1:raise InputRejected('Expected exactly one image for '+slot)
                source=files[0];raw=source.read_bytes()
                with Image.open(source) as original:
                    if original.width*original.height>24_000_000:raise InputRejected('IMAGE_PIXEL_BUDGET_EXCEEDED')
                    image=ImageOps.exif_transpose(original).convert('RGB')
                report['current_view']=slot
                print('Detecting subject:',slot,flush=True)
                def detect(array):
                    result=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=array))
                    return [np.array([[p.x,p.y] for p in face]) for face in result.face_landmarks]
                candidates,detection=scan_faces(image,detect)
                report['views'][slot]={'detection':detection}
                (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                points=[r['points'] for r in candidates];boxes=[r['box'] for r in candidates]
                roi_backend='mediapipe';roi_score=None
                if not boxes and slot!='front':
                    bbox_file=out/(slot+'.facebox.json')
                    command=[str(rdf_python or ROOT/'.venv-rdf/Scripts/python.exe'),str(ROOT/'scripts/detect_head24_profile.py'),'--repo',str(repo or ROOT/'vendor/RealDenseFace'),'--source',str(source.resolve()),'--output',str(bbox_file.resolve())]
                    print('No profile landmarks; trying scored FaceBox ROI:',slot,flush=True)
                    with (out/(slot+'.facebox.log')).open('w',encoding='utf-8') as log:
                        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'})
                        for line in process.stdout:print(line,end='',flush=True);log.write(line)
                        code=process.wait()
                    if code:raise RuntimeError('FaceBox profile worker failed: '+str(bbox_file.with_suffix('.log')))
                    fallback=json.loads(bbox_file.read_text(encoding='utf-8'))
                    if fallback['source_sha256']!=hashlib.sha256(raw).hexdigest() or tuple(fallback['native_size'])!=image.size:raise ValueError('Profile detection source mismatch')
                    detection['facebox_fallback']=fallback
                    candidates=fallback['candidates'];boxes=[np.asarray(c['box'],float) for c in candidates];points=[None]*len(boxes)
                    roi_backend='facebox'
                    report['views'][slot]={'detection':detection}
                    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                overview=image.copy();draw=ImageDraw.Draw(overview)
                for index,box in enumerate(boxes):
                    draw.rectangle(tuple(box),outline=(255,180,0),width=4)
                    draw.text((box[0],max(0,box[1]-20)),str(index),fill=(255,180,0))
                overview.thumbnail((1280,1280));overview.save(out/(slot+'.detections.jpg'))
                selected=choose_face(boxes,image.size);landmarks=points[selected]
                if roi_backend=='facebox':roi_score=candidates[selected]['score']
                crop,matrix=face_crop(boxes[selected],image.size)
                head=image.crop(crop).resize((1024,1024),Image.Resampling.LANCZOS);head.save(out/(slot+'.png'))
                xy=None if landmarks is None else apply_pixels(landmarks,matrix)
                face_box_crop=apply_pixels(np.asarray(boxes[selected]).reshape(2,2),matrix).reshape(-1).tolist()
                roi=image.crop(tuple(np.asarray(boxes[selected],int)))
                rgb=np.asarray(roi,float)/255;light=rgb@np.array([.2126,.7152,.0722])
                row={'original_path':str(source.resolve()),'original_sha256':hashlib.sha256(raw).hexdigest(),'crop_sha256':hashlib.sha256((out/(slot+'.png')).read_bytes()).hexdigest(),'normalized_size':image.size,'detection':detection,'detected_faces':len(points),'selection':'largest face area with weak horizontal-center prior; reject comparable candidates','selected_index':selected,'coordinate_frame':'EXIF-normalized source pixels','face_box_normalized':boxes[selected].tolist(),'crop_normalized':crop,'canvas_padding_only':bool(crop[0]<0 or crop[1]<0 or crop[2]>image.width or crop[3]>image.height),'normalized_to_crop':matrix.tolist(),'crop_to_normalized':np.linalg.inv(matrix).tolist(),'landmarks_crop':None if xy is None else xy.tolist(),'roi_backend':roi_backend,'roi_score':roi_score,'face_box_crop':face_box_crop,'face_luma_median':float(np.median(light)),'face_luma_p95':float(np.quantile(light,.95)),'warnings':[]}
                row['native_face_size_px']=(np.asarray(boxes[selected])[2:]-np.asarray(boxes[selected])[:2]).tolist()
                if landmarks is not None:
                    left=(landmarks[33]+landmarks[133])/2;right=(landmarks[362]+landmarks[263])/2
                    row['native_interocular_px']=float(np.linalg.norm(right-left))
                if xy is None:row['warnings'].append('PROFILE_LANDMARKS_UNAVAILABLE: scored bbox only; no fabricated facial points')
                if np.median(light)<.25:row['warnings'].append('LOW_FACE_LIGHT: no albedo/illumination recovery attempted')
                report['views'][slot]=row
                overlay=head.copy();draw=ImageDraw.Draw(overlay)
                if xy is None:
                    draw.rectangle(tuple(face_box_crop),outline=(255,180,0),width=3)
                    draw.text((8,8),'FaceBox ROI only - no profile landmarks',fill=(255,180,0))
                for p in ([] if xy is None else xy):draw.ellipse((p[0]-1,p[1]-1,p[0]+1,p[1]+1),fill=(0,255,0))
                overlay.save(out/(slot+'.landmarks.png'))
                print('Head24 crop:',slot,'faces:',len(points),flush=True)
        report['status']='CROPS_PREPARED'
    except Exception as e:
        report['status']=('DETECTION_UNRESOLVED' if str(e).startswith('NO_FACE_DETECTED') else 'NEEDS_RETAKE') if isinstance(e,InputRejected) else 'FAILED';report['error']=str(e);raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rdf-python',type=Path);p.add_argument('--repo',type=Path);a=p.parse_args();run(a.input,a.task,a.output,a.rdf_python,a.repo)
