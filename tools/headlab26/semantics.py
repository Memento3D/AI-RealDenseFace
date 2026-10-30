"""Parse each cropped observation once; bind labels to the exact image/model bytes."""
from pathlib import Path
import numpy as np
from PIL import Image
from .io import sha,read,write,SLOTS

def run(crops,model,out):
 from memento3d_head.adapters.automatic_face_masks import MODEL_SHA256,preprocess,restore_crop
 if sha(model)!=MODEL_SHA256:raise ValueError('Parser checksum mismatch')
 import onnxruntime as ort
 settings=ort.SessionOptions();settings.intra_op_num_threads=4
 session=ort.InferenceSession(str(model),sess_options=settings,providers=['CPUExecutionProvider']);metadata=read(Path(crops)/'report.json');out=Path(out);out.mkdir(exist_ok=False,parents=True);rows={}
 for slot in SLOTS:
  path=Path(crops)/(slot+'.png');image=Image.open(path).convert('RGB');box=np.asarray(metadata['views'][slot]['face_box_crop'],float);center=(box[:2]+box[2:])/2;side=max(box[2:]-box[:2])*1.6;lo=np.maximum(np.floor(center-side/2),0).astype(int);hi=np.minimum(np.ceil(center+side/2),image.size).astype(int);box=tuple(np.r_[lo,hi])
  logits=session.run(None,{session.get_inputs()[0].name:preprocess(image.crop(box))})[0]
  if logits.shape!=(1,19,512,512) or not np.isfinite(logits).all():raise ValueError('Invalid semantic model output')
  label=restore_crop(logits[0].argmax(0).astype('uint8'),box,image.size);Image.fromarray(label).save(out/(slot+'.labels.png'));counts=np.bincount(label.ravel(),minlength=19)
  rows[slot]={'crop_sha256':sha(path),'crop_pixel_sha256':__import__('hashlib').sha256(np.asarray(image,dtype=np.uint8).tobytes()).hexdigest(),'labels_sha256':sha(out/(slot+'.labels.png')),'skin_pixels':int(counts[1]),'eye_pixels':int(counts[4]+counts[5]),'glasses_pixels':int(counts[6]),'hair_pixels':int(counts[17]),'eye_observation':'SEMANTIC_SUPPORT_PRESENT' if counts[4]+counts[5]>=20 else 'NO_EYE_SEMANTIC_SUPPORT','mask_ground_truth_verified':False}
 write(out/'report.json',{'status':'SEMANTIC_OBSERVATIONS_EXPORTED','parser_sha256':sha(model),'views':rows})
