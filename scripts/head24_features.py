"""Image-bound front landmark regions; no recovery of hidden anatomy or iris details."""
import hashlib
import numpy as np
from PIL import Image,ImageDraw


def front_observation(data,image):
    observation=data.get('front_observation')
    if observation is None:return None
    pixels=np.asarray(image.convert('RGB'),dtype=np.uint8)
    if hashlib.sha256(pixels.tobytes()).hexdigest()!=observation.get('crop_pixel_sha256'):
        raise ValueError('Front observation belongs to another crop')
    p=np.asarray(observation.get('landmarks_crop'),float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<400 or not np.isfinite(p).all():raise ValueError('Invalid front observation landmarks')
    if (p<0).any() or (p[:,0]>=image.width).any() or (p[:,1]>=image.height).any():raise ValueError('Front landmarks outside crop')
    return p


def mouth_region(points,size):
    # Four MediaPipe landmarks delimit a bounded region, not a reconstructed lip contour.
    image=Image.new('L',size);draw=ImageDraw.Draw(image)
    draw.polygon([tuple(points[i]) for i in (61,0,291,17)],fill=255)
    return np.asarray(image)>0


def observed_mouth_anchor(vertices,q,mask,head_corners,front_weight,scale,center):
    uv=q[:,:2]/q[:,2,None];xy=np.column_stack((uv[:,0]*mask.shape[1],(1-uv[:,1])*mask.shape[0])).astype(int)
    valid=(xy.min(1)>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]<mask.shape[0])&head_corners&(front_weight>1e-8)
    support=np.zeros(len(vertices),bool);support[valid]=mask[xy[valid,1],xy[valid,0]]
    points=(vertices[support]-center)*scale
    if len(np.unique(points,axis=0))<12:raise ValueError('Insufficient source surface support in observed mouth ROI')
    lo,hi=np.quantile(points,[.03,.97],axis=0);anchor=np.median(points,axis=0);width=hi[0]-lo[0]
    if width<=.5:raise ValueError('Observed mouth ROI has invalid source width')
    return anchor,width,{'method':'front landmark mouth ROI on visible source vertices; not a recovered lip contour','support_corners':int(support.sum()),'support_unique_vertices':len(np.unique(points,axis=0)),'center_mm':anchor.tolist(),'width_mm':float(width),'source_geometry_modified':False}
