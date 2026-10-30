"""Optional posterior view, admitted only with a segmented hair region and two ears.

Ear similarity calibration is an approximation, not independent camera ground truth.
A front/oblique view never owns the central posterior hairline.
"""
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from memento3d_head.adapters.automatic_face_masks import preprocess,restore_crop
from figurine22_hair import ear_anchors

def prepare_back(path,model,face):
    import onnxruntime as ort
    image=Image.open(path).convert('RGB');opt=ort.SessionOptions();opt.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),sess_options=opt,providers=['CPUExecutionProvider'])
    def parse(box):
        logits=session.run(None,{session.get_inputs()[0].name:preprocess(image.crop(box))})[0]
        return restore_crop(logits[0].argmax(0).astype('uint8'),box,image.size)
    labels=parse((0,0,*image.size));yy,xx=np.nonzero(labels==17)
    if len(xx)<200:raise ValueError('Back image has no supported hair/head region; refuse guessed posterior camera')
    low=np.array([xx.min(),yy.min()]);high=np.array([xx.max(),yy.max()]);center=(low+high)/2
    side=int(max(high-low)*1.6);center[1]+=.12*side
    box=[int(center[0]-side/2),int(center[1]-side/2),int(center[0]+side/2),int(center[1]+side/2)]
    labels=parse(box);groups,n=ndi.label(np.isin(labels,[7,8]));ears=[]
    for k in range(1,n+1):
        y,x=np.nonzero(groups==k)
        if len(x)>=20:ears.append((len(x),np.array([np.median(x),np.median(y)])))
    ears=sorted(ears,key=lambda x:x[0],reverse=True)[:2]
    if len(ears)!=2:raise ValueError('Back image needs two supported ear regions for automatic alignment; refusing silhouette-only scale guessing')
    targets=np.array(sorted([e[1] for e in ears],key=lambda p:p[0]))
    anchors=ear_anchors(face);world=np.array([next(a['center'] for a in anchors if a['side']==s) for s in [1,-1]])
    base=np.array([[-1.,0,0],[0,-1.,0]]);uv=world@base.T
    a=uv[1]-uv[0];b=targets[1]-targets[0];roll=np.arctan2(b[1],b[0])-np.arctan2(a[1],a[0]);roll=(roll+np.pi)%(2*np.pi)-np.pi
    if abs(roll)>np.deg2rad(25):raise ValueError('Back ear alignment exceeds supported roll')
    scale=np.linalg.norm(b)/np.linalg.norm(a)
    if not 4<scale<150:raise ValueError('Back view scale outside supported head crop')
    rot=np.array([[np.cos(roll),-np.sin(roll)],[np.sin(roll),np.cos(roll)]])
    A=scale*rot@base;offset=targets.mean(0)-A@world.mean(0)
    epsilon=.001;camera=np.zeros((3,4));camera[:2,:3]=A+offset[:,None]*np.array([0,0,epsilon])[None,:];camera[:2,3]=offset;camera[2]=[0,0,epsilon,1]
    q=np.c_[world,np.ones(2)]@camera.T;error=np.linalg.norm(q[:,:2]/q[:,2,None]-targets,axis=1)
    if error.max()>5:raise ValueError('Back ear reprojection failed')
    return image,labels,camera,{'method':'two-ear similarity alignment, fixed posterior yaw, weak perspective',
        'crop_xyxy':box,'ear_targets_px':targets.tolist(),'ear_fit_max_px':float(error.max()),'scale_px_per_mm':float(scale),
        'roll_deg':float(np.rad2deg(roll)),'independent_camera_calibration_verified':False,'pitch_estimated':False}
