"""Fixed-mask sensitivity study on cached correspondences; no camera fitting."""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tools.headapi.face_texture_trial import ROOT, sha, robust_skin_gain, apply_gain
from tools.headapi.texture_overlap_trial import metrics


def image_fields(rgb, labels):
    skin=(labels==1).astype(np.uint8)
    # Explicit outside background ensures distance is bounded at the image edge.
    distance=cv2.distanceTransform(np.pad(skin,1),cv2.DIST_L2,5)[1:-1,1:-1]
    light=cv2.cvtColor((rgb/255.).astype(np.float32),cv2.COLOR_RGB2LAB)[...,0]
    blurred=cv2.GaussianBlur(light,(0,0),1)
    gx=cv2.Sobel(blurred,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(blurred,cv2.CV_32F,0,1,ksize=3)/8
    return distance,np.hypot(gx,gy)


def sample(field,xy):
    return cv2.remap(field,xy[:,0].astype(np.float32)[None],xy[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]


def run():
    source=ROOT/'runs/experiments/input04-texture-skin-01'
    parent=ROOT/'runs/experiments/input04-texture-residual-01/report.json'
    prior=json.loads(parent.read_text());provenance=json.loads((source/'report.json').read_text())
    output=ROOT/'runs/experiments/input04-texture-boundary-01';output.mkdir(parents=True,exist_ok=False)
    hashes={str(parent):sha(parent),str(source/'report.json'):sha(source/'report.json')}
    bundle=ROOT/'runs/experiments/rdf-flame-parameters-40ab0c23fed043eb875537da057027e8/shape_expression_0_05/projection_bundle.npz'
    if sha(bundle)!=provenance['input_sha256'][str(bundle)]:raise ValueError('Geometry checksum mismatch')
    hashes[str(bundle)]=sha(bundle)
    with np.load(bundle,allow_pickle=False) as b:views=b['view_vertices'];cameras=b['cameras']
    fields=[]
    for slot in ['front','left','right']:
        photo=ROOT/'runs/experiments/input04-validation-01/rdf_crops'/(slot+'.png');labels=source/(slot+'.labels.png')
        for path in [photo,labels]:
            if sha(path)!=provenance['input_sha256'][str(path)]:raise ValueError('Source pin mismatch')
            hashes[str(path)]=sha(path)
        rgb=np.asarray(Image.open(photo).convert('RGB'));distance,gradient=image_fields(rgb,np.asarray(Image.open(labels)))
        fields.append((rgb,distance,gradient))
    policies=[('control',0,None),('boundary_8',8,None),('boundary_12',12,None),('boundary_8_grad_2',8,2.),('boundary_12_grad_1',12,1.)]
    report={'status':'SENSITIVITY_STUDY_ONLY','geometry_changed':False,'texture_exported':False,'thresholds_tuned_on_holdout':False,
            'input_sha256':hashes,'code_sha256':sha(Path(__file__)),
            'dependency_sha256':{str(p):sha(p) for p in [ROOT/'tools/headapi/face_texture_trial.py',ROOT/'tools/headapi/texture_overlap_trial.py']},
            'gradient':'Lab L units/pixel; Gaussian sigma=1; Sobel /8',
            'limitations':['Same capture and approximate correspondences','Fixed thresholds are uncalibrated and resolution dependent','Filtered holdout has changed population; use full fixed holdout for candidate comparisons','One-pixel offsets are photometric sensitivity probes, not camera/registration corrections'],'views':[]}
    for vi,slot in [(1,'left'),(2,'right')]:
        path=source/(slot+'.samples.npz')
        if sha(path)!=prior['input_sha256'][str(path)]:raise ValueError('Cached samples changed since residual audit')
        hashes[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as s:
            ids=s['vertex_ids'];a=s['reference'];b=s['sample'];hold=s['holdout'];old=s['global_color']
        xy=[];distances=[];gradients=[]
        for index in [0,vi]:
            h=np.c_[views[index][ids],np.ones(len(ids))]@cameras[index].T;points=h[:,:2]/h[:,2,None]
            xy.append(points);distances.append(sample(fields[index][1],points));gradients.append(sample(fields[index][2],points))
        distance=np.minimum(*distances);gradient=np.maximum(*gradients)
        row={'view':slot,'fixed_skin_holdout_count':int(hold.sum()),'previous_skin_global':metrics(a[hold],old[hold]),'policies':[]}
        for name,min_distance,max_gradient in policies:
            keep=distance>=min_distance
            if max_gradient is not None:keep&=gradient<=max_gradient
            train=~hold&keep;selected_hold=hold&keep
            item={'name':name,'minimum_boundary_distance_px':min_distance,'maximum_gradient':max_gradient,'train_count':int(train.sum()),'filtered_holdout_count':int(selected_hold.sum())}
            if train.sum()<32:
                item['status']='INSUFFICIENT_TRAIN_SAMPLES';row['policies'].append(item);continue
            gain,used=robust_skin_gain(a[train],b[train]);corrected=apply_gain(b,gain)
            item.update({'status':'MEASURED' if used>=32 else 'FALLBACK_IDENTITY_INSUFFICIENT_ROBUST_SAMPLES','robust_samples':used,'gain':gain.tolist(),'fixed_skin_holdout':metrics(a[hold],corrected[hold])})
            if selected_hold.sum()>=8:item['filtered_holdout_diagnostic']={'raw':metrics(a[selected_hold],b[selected_hold]),'corrected':metrics(a[selected_hold],corrected[selected_hold])}
            row['policies'].append(item)
        # Perturb each source separately: do not search for or apply the best offset.
        sensitivity=[]
        for index,points in zip([0,vi],xy):
            rgb=fields[index][0];base=sample(rgb,points).astype(float)
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                shifted=sample(rgb,points+[dx,dy]).astype(float)
                sensitivity.append({'image':'front' if index==0 else slot,'offset_px':[dx,dy],'holdout_color_change':metrics(base[hold],shifted[hold])})
        row['one_pixel_sensitivity']=sensitivity
        np.savez_compressed(output/(slot+'.diagnostics.npz'),vertex_ids=ids,holdout=hold,boundary_distance=distance,gradient=gradient)
        report['views'].append(row)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['views'],indent=2),flush=True)


if __name__=='__main__':run()
