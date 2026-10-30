"""Use a cropped native mesh only for bounded front-surface relief, never its volume."""
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import trimesh
from headlab_raster import fragments

def relief(native_path,subject,front_mask,points,camera,origin,ds,fraction,amplitude=.14):
    mesh=trimesh.load(native_path,force='mesh',process=True);v=mesh.vertices;size=256
    y0,z0=v[:,1:].min(0);y1,z1=v[:,1:].max(0)
    xy=np.c_[(v[:,1]-y0)/(y1-y0)*(size-1),(z1-v[:,2])/(z1-z0)*(size-1)]
    depth=np.full(size*size,np.inf)
    for ids,pixels,w in fragments(xy,mesh.faces,size,size):
        values=np.sum(w*v[mesh.faces[ids],0],axis=1);np.minimum.at(depth,pixels,values)
    depth=depth.reshape(size,size);valid=np.isfinite(depth)
    box=subject['hair_crop_xyxy'];target=np.asarray(Image.fromarray(front_mask[box[1]:box[3],box[0]:box[2]]).resize((size,size),Image.Resampling.NEAREST))>0
    def iou(a,b):return float((a&b).sum()/max((a|b).sum(),1))
    scores=[iou(valid,target),iou(valid[:,::-1],target)];flip=bool(scores[1]>scores[0])
    if flip:depth=depth[:,::-1];valid=valid[:,::-1]
    info={'method':'native front depth band-pass mapped through the original portrait crop',
        'raw_native_used_as_scalp_volume':False,'native_input_watertight':bool(mesh.is_watertight),'native_input_signed_volume':float(mesh.volume),
        'native_input_is_volume':bool(mesh.is_volume),'native_front_axis_assumption':'-X, image width Y, up Z; horizontal flip tested against source mask',
        'best_native_mask_iou':max(scores),'horizontal_flip':flip,'amplitude_limit_mm':amplitude}
    if max(scores)<.60:
        return np.zeros(len(points)),dict(info,used=False,reason='native silhouette disagrees with source crop')
    nearest=ndi.distance_transform_edt(~valid,return_distances=False,return_indices=True);filled=depth[tuple(nearest)]
    band=ndi.gaussian_filter(filled,1.5)-ndi.gaussian_filter(filled,10.)
    core=valid&(ndi.distance_transform_edt(valid)>4);scale=max(float(np.quantile(abs(band[core]),.95)),1e-6)
    band=np.clip(band/scale,-1,1)*amplitude
    q=np.c_[points,np.ones(len(points))]@camera.T;image_xy=q[:,:2]/q[:,2,None]
    uv=np.c_[(image_xy[:,0]-box[0])/(box[2]-box[0])*(size-1),(image_xy[:,1]-box[1])/(box[3]-box[1])*(size-1)]
    theta=np.arctan2(ds[:,0],ds[:,2]);visible=np.maximum(np.cos(theta),0)**6
    taper=np.clip((1-fraction)/.18,0,1);taper=taper*taper*(3-2*taper)
    inside=(uv[:,0]>=0)&(uv[:,0]<size)&(uv[:,1]>=0)&(uv[:,1]<size)
    values=ndi.map_coordinates(band,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=0)
    safe=ndi.map_coordinates(core.astype(float),[uv[:,1],uv[:,0]],order=1,mode='constant',cval=0)
    result=values*safe*visible*taper*inside*(1-ds[:,1]**2)
    return result,dict(info,used=True,affected_vertices=int((abs(result)>1e-5).sum()),actual_max_abs_mm=float(abs(result).max()))
