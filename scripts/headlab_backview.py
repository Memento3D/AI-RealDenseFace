"""Automatic rear-view observation with explicit domain and registration limits.

Dark-hair/light-background color-assisted segmentation is used only when its
contrast tests pass. A face parser's 'hat' label is never blindly remapped.
Camera registration is approximate, not a calibration ground truth.
"""
import json
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from figurine_hairline32 import project


def largest(mask):
    labels,count=ndi.label(mask)
    if not count:raise ValueError('No connected rear-hair proposal')
    sizes=np.bincount(labels.ravel());sizes[0]=0
    return labels==sizes.argmax()


def auto_crop(image):
    image=ImageOps.exif_transpose(image).convert('RGB');rgb=np.asarray(image,float);h,w=rgb.shape[:2]
    dark=rgb.mean(2)<75
    dark[:int(h*.12)]=False;dark[int(h*.96):]=False
    dark[:,:int(w*.18)]=False;dark[:,int(w*.82):]=False
    labels,n=ndi.label(dark);objects=ndi.find_objects(labels);candidates=[]
    for i,slices in enumerate(objects,1):
        if slices is None:continue
        yy,xx=slices;area=int((labels[slices]==i).sum());ww=xx.stop-xx.start;hh=yy.stop-yy.start
        if area<h*w*.006 or not (.07*w<ww<.65*w) or not (.1*h<hh<.85*h):continue
        distance=abs((xx.start+xx.stop)/2-w*.5)/w
        candidates.append((area/(1+distance*2),slices))
    if not candidates:raise ValueError('Rear image has no reliable central dark-hair crop; image type unsupported')
    _,(yy,xx)=max(candidates,key=lambda x:x[0]);side=int(max(xx.stop-xx.start,yy.stop-yy.start)*1.4)
    cx=(xx.start+xx.stop)/2;cy=(yy.start+yy.stop)/2
    return image,(int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2))


def segment_rear(image):
    image,box=auto_crop(image);rgb=np.asarray(image,float);h,w=rgb.shape[:2]
    lum=rgb.mean(2);opponent=(rgb[:,:,0]-rgb[:,:,2])/(rgb[:,:,0]+rgb[:,:,2]+1)
    sigmoid=lambda x:1/(1+np.exp(-np.clip(x,-40,40)))
    probability=sigmoid((90-lum)/12)*sigmoid((.08-opponent)/.04)
    roi=np.zeros((h,w),bool);x0,y0,x1,y1=box;roi[max(y0,0):min(y1,h),max(x0,0):min(x1,w)]=True
    raw=(probability>.65)&roi
    hair=largest(ndi.binary_fill_holes(ndi.binary_closing(raw,iterations=2)))
    y,x=np.where(hair);bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
    if not .015<hair.mean()<.35:raise ValueError('Rear hair area outside supported range')
    ring=ndi.binary_dilation(hair,iterations=max(3,int(w*.012)))&~ndi.binary_dilation(hair,iterations=2)
    hair_lum=float(np.median(lum[hair]));outside_lum=float(np.median(lum[ring]))
    if outside_lum-hair_lum<30:raise ValueError('Rear hair/background contrast too weak for color-assisted segmentation')
    x0,y0,x1,y1=bbox;ww=x1-x0;hh=y1-y0
    skin=(opponent>.10)&(lum>30)&(lum<190)&roi&~hair
    anchors=[]
    for side in ['image_left','image_right']:
        selection=skin.copy()
        selection[:int(y0+.5*hh)]=False;selection[int(y0+.94*hh):]=False
        if side=='image_left':selection[:,int(x0+.14*ww):]=False;selection[:,:max(int(x0-.14*ww),0)]=False
        else:selection[:,:int(x1-.14*ww)]=False;selection[:,min(int(x1+.14*ww),w):]=False
        yy,xx=np.where(selection)
        if len(xx)<30:raise ValueError('Rear image does not expose two usable ear-region anchors')
        anchors.append([float(np.median(xx)),float(np.median(yy))])
    label=np.zeros((h,w),np.uint8);label[skin]=1;label[hair]=17
    # Known skin evidence below the hairline only where color supports skin.
    report={'method':'guarded dark-hair/color-opponent connected segmentation','domain':'dark hair, lighter background, two visible ear regions; not a general rear segmentation model',
        'crop':box,'hair_bbox':bbox,'hair_pixels':int(hair.sum()),'median_hair_luminance':hair_lum,'median_outer_ring_luminance':outside_lum,
        'ear_region_anchors_image_px':anchors,'semantic_ear_landmarks_verified':False,'mask_human_reviewed':False}
    return image,hair,label,report


def camera_matrix(parameters,size,focal):
    yaw,pitch,roll,logscale,cx,cy=parameters[:6]
    base=np.diag([-1.,-1.,1.])
    rotation=Rotation.from_euler('yxz',[yaw,pitch,roll],degrees=True).as_matrix()@base
    distance=focal/np.exp(logscale)
    matrix=np.empty((3,4));matrix[0,:3]=focal*rotation[0]+cx*rotation[2];matrix[1,:3]=focal*rotation[1]+cy*rotation[2];matrix[2,:3]=rotation[2]
    matrix[:,3]=[cx*distance,cy*distance,distance]
    return matrix


def source_anchors(mesh,base_offset):
    p=np.asarray(mesh.vertices);height=np.ptp(p[:,1]);center=(p.min(0)+p.max(0))/2
    band=(p[:,1]>center[1]-.12*height)&(p[:,1]<center[1]+.27*height)
    result=[]
    for sign in [1,-1]:
        ids=np.flatnonzero(band);threshold=np.quantile(sign*p[ids,0],.985)
        selected=p[ids][sign*p[ids,0]>=threshold]
        result.append(np.median(selected,axis=0))
    top=p[np.argmax(p[:,1])].copy();top[0]=center[0];top[1]+=base_offset
    return np.vstack((result,top))


def fit_rear_camera(mesh,mask,segmentation,size,base_offset=.45):
    # Only observed ear regions and top-of-hair anchor are fitted. Pose priors
    # resolve ambiguity, they are not image observations or calibration evidence.
    h,w=mask.shape;ears=np.asarray(segmentation['ear_region_anchors_image_px'],float)
    bbox=segmentation['hair_bbox'];top=[float((bbox[0]+bbox[2])/2),float(bbox[1])]
    target=np.vstack((ears,top));source=source_anchors(mesh,base_offset)
    initial_scale=np.linalg.norm(ears[1]-ears[0])/np.linalg.norm(source[1]-source[0])
    focal=1.5*max(size)
    center=np.mean(target[:2],axis=0)
    init=np.array([0.,0.,0.,np.log(initial_scale),center[0],center[1]+initial_scale*np.mean(source[:2,1]),base_offset])
    lower=np.array([-25,-30,-15,np.log(initial_scale*.6),center[0]-w*.2,center[1]-h*.35,0.])
    upper=np.array([25,30,15,np.log(initial_scale*1.5),center[0]+w*.2,center[1]+h*.35,3.])
    initializations=[]
    for yaw in [-12,0,12]:
        for pitch in [-12,8]:
            start=init.copy();start[:2]=[yaw,pitch];initializations.append(start)
    def residual(params):
        adjusted=source.copy();adjusted[-1,1]+=params[6]-base_offset
        xy,z=project(adjusted,camera_matrix(params,size,focal))
        if (z<=0).any():return np.full(10,1e4)
        # Priors explicitly regularize underdetermined rear pose.
        return np.r_[(xy-target).ravel()/2,params[0]/5,params[1]/20,params[2]/15,(params[6]-base_offset)/1.5]
    results=[least_squares(residual,x,bounds=(lower,upper),loss='soft_l1',max_nfev=100) for x in initializations]
    best=min(results,key=lambda r:r.cost);camera=camera_matrix(best.x,size,focal)
    source=source.copy();source[-1,1]+=best.x[6]-base_offset
    projected,_=project(source,camera);errors=np.linalg.norm(projected-target,axis=1)
    if not best.success or errors.max()>8:raise ValueError('Rear camera registration failed anchor consistency gate')
    return camera,{'method':'multi-start bounded pinhole pose fit with approximate ear-region and hair-top anchors; pose priors',
        'parameters':dict(zip(['yaw_delta_deg','pitch_delta_deg','roll_delta_deg','log_pixels_per_mm','image_cx','image_cy','hair_top_nuisance_offset_mm'],best.x.tolist())),
        'yaw_is_prior_regularized':True,'hair_top_is_nuisance_not_skull_landmark':True,'hair_top_nuisance_at_bound':bool(best.x[6]>2.999 or best.x[6]<.001),'focal_assumed_px':focal,'focal_calibrated':False,'anchor_errors_px':errors.tolist(),'anchor_max_error_px':float(errors.max()),
        'target_anchors_px':target.tolist(),'projected_anchors_px':projected.tolist(),'source_anchors_mm':source.tolist(),
        'starts':len(results),'gate':'approximate anchor reprojection <=8 px; not an independent camera calibration',
        'absolute_rear_pose_ground_truth_verified':False,'accepted_for_experiment':True,'camera_matrix':camera.tolist()}


def prepare_back(path,closed,output):
    image,hair,label,seg=segment_rear(Image.open(path))
    size=(512,round(512*image.height/image.width));factor=np.asarray(size)/image.size
    mask=np.asarray(Image.fromarray(np.uint8(hair)*255).resize(size,Image.Resampling.NEAREST))>0
    small_label=np.asarray(Image.fromarray(label).resize(size,Image.Resampling.NEAREST))
    scaled=dict(seg);scaled['ear_region_anchors_image_px']=(np.asarray(seg['ear_region_anchors_image_px'])*factor).tolist()
    scaled['hair_bbox']=(np.asarray(seg['hair_bbox'])*np.r_[factor,factor]).tolist()
    camera,registration=fit_rear_camera(closed,mask,scaled,size)
    rgb=np.asarray(image.resize(size)).copy();rgb[mask]=np.uint8(rgb[mask]*.45+np.array([30,210,70])*.55)
    canvas=Image.fromarray(rgb);draw=ImageDraw.Draw(canvas)
    for target,pred in zip(registration['target_anchors_px'],registration['projected_anchors_px']):
        x,y=target;draw.ellipse((x-3,y-3,x+3,y+3),outline='cyan',width=1)
        x,y=pred;draw.ellipse((x-2,y-2,x+2,y+2),outline='red',width=1)
    canvas.save(output/'back.registration.png');Image.fromarray(np.uint8(mask)*255).save(output/'back.mask.png')
    report={'segmentation':seg,'registration':registration}
    (output/'back.observation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return image,mask,small_label,camera,report
