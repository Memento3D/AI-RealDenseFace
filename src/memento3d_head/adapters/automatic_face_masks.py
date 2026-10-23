"""Pinned BiSeNet ONNX parsing; mask policies are provisional and not identity checks."""
import base64
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageFilter
from memento3d_head.adapters.head_texture_workspace import digest,read_source,SLOTS

MODEL_SHA256='0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'
MODEL_URL='https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx'
SOURCE_COMMIT='8a4729d95118d0e97c44185f9bdef3d6bfeaaf99'
LABELS=('background','skin','left_brow','right_brow','left_eye','right_eye','glasses','left_ear','right_ear','earring','nose','mouth','upper_lip','lower_lip','neck','necklace','cloth','hair','hat')
KEEP=(1,2,3,4,5,7,8,10,11,12,13,14)


def crop_box(coordinates,size):
    width,height=size;q=np.asarray(coordinates,float)
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or np.any(q[:,2]<=0):raise ValueError('Invalid crop coordinates')
    xy=np.column_stack((q[:,0]/q[:,2]*width,(1-q[:,1]/q[:,2])*height))
    in_frame=(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
    if in_frame.sum()<100:raise ValueError('Too few projected vertices to locate head')
    low=np.quantile(xy[in_frame],.01,axis=0);high=np.quantile(xy[in_frame],.99,axis=0)
    side=int(np.ceil(float(np.max(high-low))*1.15));center=(low+high)*.5
    if side<64 or side>max(size)*2:raise ValueError('Unreasonable head crop')
    x=int(np.floor(center[0]-side/2));y=int(np.floor(center[1]-side/2))
    return (x,y,x+side,y+side)


def preprocess(image):
    rgb=np.asarray(image.resize((512,512),Image.Resampling.BILINEAR).convert('RGB'),np.float32)/255
    return np.ascontiguousarray(((rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None])


def decode_logits(output):
    logits=np.asarray(output,np.float32)
    if logits.shape!=(1,19,512,512) or not np.isfinite(logits).all():raise ValueError('Expected finite [1,19,512,512] parsing logits')
    logits=logits[0];shift=logits-logits.max(axis=0,keepdims=True)
    probabilities=np.exp(shift);probabilities/=probabilities.sum(axis=0,keepdims=True)
    confidence=probabilities.max(axis=0)
    retained_probability=probabilities[list(KEEP)].sum(axis=0)
    labels=logits.argmax(axis=0).astype(np.uint8)
    return labels,confidence,retained_probability


def restore_crop(array,box,size,is_float=False):
    canvas=Image.new('F' if is_float else 'L',size,0)
    cropped=Image.fromarray(np.asarray(array,dtype=np.float32 if is_float else np.uint8))
    side=box[2]-box[0]
    cropped=cropped.resize((side,side),Image.Resampling.BILINEAR if is_float else Image.Resampling.NEAREST)
    canvas.paste(cropped,(box[0],box[1]))
    return np.asarray(canvas).copy()


def generate_masks(texture_run,model_path,output,threshold=.55,session=None):
    texture_run=Path(texture_run);model_path=Path(model_path);output=Path(output)
    if not 0<threshold<1:raise ValueError('Invalid confidence threshold')
    if digest(model_path)!=MODEL_SHA256:raise ValueError('Face parser checksum mismatch; use the pinned setup script')
    import onnxruntime as ort
    if session is None:
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    model_input=session.get_inputs()[0]
    if model_input.type!='tensor(float)' or model_input.shape[1:]!=[3,512,512]:raise ValueError('Unexpected face parser input')
    viewer,data,_,_=read_source(texture_run);source_hash=digest(viewer)
    output.mkdir(parents=True,exist_ok=False);annotations=dict(schema_version=1,source_viewer_sha256=source_hash,skin_rgb=None,masks={},origin='automatic_face_parsing',human_reviewed=False)
    report=dict(status='FAILED',model_sha256=MODEL_SHA256,model_url=MODEL_URL,reference_code_commit=SOURCE_COMMIT,
        backend='onnxruntime',onnxruntime_version=ort.__version__,providers=session.get_providers(),
        classes=list(LABELS),retained_class_ids=list(KEEP),confidence_threshold=threshold,
        mask_policy_version=2,threshold_applies_to='sum_probability_of_retained_classes',confidence_calibrated=False,earbud_detection_available=False,views={},warnings=[],inference_tested=False)
    started=time.perf_counter();skin_pixels=[]
    try:
        for i,slot in enumerate(SLOTS):
            print('Automatic face parsing: '+slot,flush=True)
            with Image.open(io.BytesIO(base64.b64decode(data['layers'][i]['image'].split(',',1)[1],validate=True))) as src:image=src.convert('RGB')
            size=image.size;box=crop_box(data['layers'][i]['coordinates'],size)
            outputs=session.run([session.get_outputs()[0].name],{model_input.name:preprocess(image.crop(box))})
            labels_small,confidence_small,group_small=decode_logits(outputs[0])
            labels=restore_crop(labels_small,box,size);confidence=restore_crop(confidence_small,box,size,True)
            retained_probability=restore_crop(group_small,box,size,True)
            keep=np.isin(labels,KEEP)&(retained_probability>=threshold)
            mask=Image.fromarray(keep.astype(np.uint8)*255).filter(ImageFilter.MinFilter(3));keep=np.asarray(mask)>0
            candidate=(labels==1)&(confidence>=max(.75,threshold))&keep
            rgb=np.asarray(image);pixels=rgb[candidate]
            if len(pixels)>100:
                light=pixels.mean(1);low,high=np.quantile(light,[.3,.8]);skin_pixels.append(pixels[(light>=low)&(light<=high)][::max(1,len(pixels)//10000)])
            raw=io.BytesIO();mask.save(raw,format='PNG');annotations['masks'][slot]='data:image/png;base64,'+base64.b64encode(raw.getvalue()).decode('ascii')
            mask.save(output/(slot+'.mask.png'));Image.fromarray(labels).save(output/(slot+'.labels.png'))
            overlay=np.asarray(image).copy();overlay[~keep]=(overlay[~keep]*.4+np.array([230,50,70])*.6).astype(np.uint8)
            Image.fromarray(overlay).save(output/(slot+'.mask_review.jpg'))
            counts=np.bincount(labels.ravel(),minlength=19)
            warning=[]
            if counts[1]<size[0]*size[1]*.01:warning.append('little_skin_detected')
            if counts[6]>50:warning.append('glasses_detected')
            if np.count_nonzero(candidate)<100:warning.append('insufficient_high_confidence_skin')
            if any(counts[j]<10 for j in (4,5)) and slot=='front':warning.append('front_eye_regions_missing')
            report['views'][slot]=dict(crop_xyxy=list(box),image_size=list(size),retained_pixels=int(keep.sum()),skin_sample_pixels=int(candidate.sum()),class_pixels={LABELS[j]:int(counts[j]) for j in range(19)},warnings=warning)
            report['warnings'] += [slot+':'+w for w in warning]
        if skin_pixels:annotations['skin_rgb']=np.rint(np.median(np.concatenate(skin_pixels),axis=0)).astype(int).tolist()
        else:report['warnings'].append('no_automatic_skin_color; workspace uses fallback requiring review')
        if digest(viewer)!=source_hash or digest(model_path)!=MODEL_SHA256:raise ValueError('Source viewer/model changed during parsing')
        (output/'auto_masks.json').write_text(json.dumps(annotations),encoding='utf-8')
        report.update(status='AUTO_MASKS_EXPORTED',inference_tested=True,skin_rgb=annotations['skin_rgb'],semantic_accuracy_verified=False)
    except Exception as error:report['error']=str(error);raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    del session
    return output/'auto_masks.json',report
