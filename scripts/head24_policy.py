"""Deterministic, provisional preflight gates; no claim of identity or printability."""
import numpy as np

class InputRejected(ValueError):
    pass


def choose_face(boxes, size):
    w,h=size
    if np.asarray(boxes).size==0:raise InputRejected('NO_FACE_DETECTED: detector returned no valid face landmarks')
    candidates=[]
    for i,b in enumerate(np.asarray(boxes,float).reshape(-1,4)):
        if not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:continue
        if min(b[2]-b[0],b[3]-b[1])<64:continue
        area=(b[2]-b[0])*(b[3]-b[1])
        dx=((b[0]+b[2])/2/w-.5)
        candidates.append((area*np.exp(-4*dx*dx),i,b))
    if not candidates:raise InputRejected('NO_FACE_MEETS_NATIVE_SIZE: candidates exist, but none has 64 native pixels per side')
    candidates.sort(key=lambda x:x[0],reverse=True)
    if len(candidates)>1 and candidates[1][0]>.6*candidates[0][0]:
        raise InputRejected('AMBIGUOUS_SUBJECT: multiple similarly prominent faces')
    return candidates[0][1]


def crop_transform(box,size,target=1024):
    w,h=size;x0,y0,x1,y1=np.asarray(box,float)
    side=int(np.ceil(max(x1-x0,y1-y0)*2.4))
    if side>min(w,h):raise InputRejected('HEAD_CROP_CLIPPED: insufficient frame margin')
    cx=(x0+x1)/2;cy=(y0+y1)/2-.16*(y1-y0)
    left=int(np.clip(round(cx-side/2),0,w-side));top=int(np.clip(round(cy-side/2),0,h-side))
    a=target/side
    matrix=np.array([[a,0,-a*left],[0,a,-a*top],[0,0,1.]])
    return [left,top,left+side,top+side],matrix


def apply_pixels(points,matrix):
    p=np.asarray(points,float);q=np.column_stack((p,np.ones(len(p))))@np.asarray(matrix).T
    return q[:,:2]/q[:,2,None]


def eye_gate(projected, observed, limit=.25):
    p=np.asarray(projected,float);o=np.asarray(observed,float)
    if p.shape!=(2,2) or o.shape!=(2,2) or not np.isfinite(p).all() or not np.isfinite(o).all():
        raise InputRejected('INVALID_EYE_MEASUREMENTS')
    distance=float(np.linalg.norm(o[1]-o[0]))
    if distance<24:raise InputRejected('EYES_TOO_SMALL_OR_PROFILE: independent front check unavailable')
    # Component labels are model-coordinate based, not anatomical left/right.
    a=np.linalg.norm(p-o,axis=1);b=np.linalg.norm(p[::-1]-o,axis=1)
    errors=min((a,b),key=lambda x:x.sum())/distance
    ratio=float(np.linalg.norm(p[1]-p[0])/distance)
    report={'max_eye_center_error_in_interocular_units':float(errors.max()),'projected_to_observed_eye_distance':ratio,'threshold':limit,'independent_of_rdf_predictions':True,'calibrated':False}
    if errors.max()>limit or not .65<ratio<1.4:
        raise InputRejected('FRONT_ALIGNMENT_REJECTED: '+str(report))
    return report


def project(points,camera):
    p=np.column_stack((points,np.ones(len(points))))@np.asarray(camera).T
    if not np.isfinite(p).all() or (p[:,2]<=0).any():raise InputRejected('INVALID_CAMERA_DEPTH')
    return p[:,:2]/p[:,2,None],p[:,2]


def rigid_map(source,target):
    a=np.asarray(source,float);b=np.asarray(target,float)
    ca=a.mean(0);cb=b.mean(0);u,_,vt=np.linalg.svd((a-ca).T@(b-cb))
    correction=np.eye(3);correction[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@correction@u.T
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=cb-rotation@ca
    residual=np.linalg.norm(a@rotation.T+transform[:3,3]-b,axis=1)
    if residual.max()>2e-5:raise InputRejected('ROOT_TRANSFORM_NOT_RIGID')
    return transform,float(residual.max())


def validate_camera_contract(vertices,coordinates,camera,size,tolerance=.05):
    xy,_=project(vertices,camera);q=np.asarray(coordinates,float)
    target=np.column_stack((q[:,0]/q[:,2]*size[0],(1-q[:,1]/q[:,2])*size[1]))
    errors=np.linalg.norm(xy-target,axis=1)
    if not np.isfinite(errors).all() or errors.max()>tolerance:
        raise InputRejected('EXPLICIT_CAMERA_COORDINATE_MISMATCH')
    return {'p95_px':float(np.quantile(errors,.95)),'max_px':float(errors.max()),'scope':'stored per-view geometry and explicit camera, not independent calibration'}


def inference_box(row,slot):
    """Front requires independent landmarks; profile can use an explicit scored detector ROI."""
    landmarks=row.get('landmarks_crop')
    if landmarks is not None:
        p=np.asarray(landmarks,float)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<400 or not np.isfinite(p).all():raise InputRejected('INVALID_LANDMARK_ROI')
        box=np.r_[p.min(0),p.max(0)]
    else:
        if slot=='front':raise InputRejected('FRONT_INDEPENDENT_LANDMARKS_REQUIRED')
        if row.get('roi_backend')!='facebox' or row.get('roi_score',0)<.6:raise InputRejected('UNSUPPORTED_PROFILE_ROI')
        box=np.asarray(row.get('face_box_crop'),float)
    if box.shape!=(4,) or not np.isfinite(box).all() or (box[2:]<=box[:2]).any() or (box[:2]<0).any() or (box[2:]>1024).any():
        raise InputRejected('INVALID_EXPLICIT_ROI')
    return box.astype('float32')


def bbox_gate(projected,box):
    """Coarse independent ROI consistency only, not anatomical/profile landmark accuracy."""
    p=np.asarray(projected,float);b=np.asarray(box,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<20 or not np.isfinite(p).all():raise InputRejected('INVALID_PROFILE_PROJECTION')
    lo,hi=np.quantile(p,[.05,.95],axis=0);size=b[2:]-b[:2]
    ratios=(hi-lo)/size
    center_error=float(np.linalg.norm(((lo+hi)-(b[:2]+b[2:]))/2)/np.linalg.norm(size))
    result={'projected_bbox_q05_q95':np.r_[lo,hi].tolist(),'roi':b.tolist(),'extent_ratios':ratios.tolist(),'center_error_in_roi_diagonals':center_error,'calibrated':False,'scope':'coarse fit-to-detector ROI; not independent facial anatomy'}
    if not np.isfinite(ratios).all() or (ratios<.25).any() or (ratios>2.5).any() or center_error>.6:
        raise InputRejected('PROFILE_ALIGNMENT_REJECTED: '+str(result))
    return result
