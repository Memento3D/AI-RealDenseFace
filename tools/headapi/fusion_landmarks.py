"""Run the existing CPU MediaPipe task on the generated-head review rendering."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
TASK_SHA='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'


def run(image,task,out):
    if hashlib.sha256(Path(task).read_bytes()).hexdigest()!=TASK_SHA:raise ValueError('Landmarker pin mismatch')
    import mediapipe as mp
    rgb=np.asarray(Image.open(image).convert('RGB'))
    options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=2)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as model:
        found=model.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
    if len(found.face_landmarks)!=1:raise ValueError('MVP_ALIGNMENT_FACE_UNRESOLVED: expected exactly one face on generated-head render')
    points=[[float(p.x*rgb.shape[1]),float(p.y*rgb.shape[0])] for p in found.face_landmarks[0]]
    Path(out).write_text(json.dumps({'landmarks':points,'image_sha256':hashlib.sha256(Path(image).read_bytes()).hexdigest(),'task_sha256':TASK_SHA,'scope':'detected on synthetic render, not ground truth identity'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.image,a.task,a.output)
