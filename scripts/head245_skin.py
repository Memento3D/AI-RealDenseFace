"""Multiview observed-skin selection. No albedo recovery or skin-colour ground truth."""
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab,deltaE_ciede2000

def skin_reference(images,labels,settings):
    rows=[];masks=[];colors=[]
    for i,(im,lab) in enumerate(zip(images,labels)):
        rgb=np.asarray(im.convert('RGB'),float)/255
        safe=ndi.binary_erosion(lab==1,iterations=5)
        safe&=~ndi.binary_dilation(np.isin(lab,[2,3,4,5,6,11,12,13,17]),iterations=8)
        safe&=(rgb.min(2)>.025)&(rgb.max(2)<.96)
        # Reject textured/edge pixels, not an assumed ethnicity or skin hue.
        lum=rgb@np.array([.2126,.7152,.0722]);variation=ndi.maximum_filter(lum,5)-ndi.minimum_filter(lum,5)
        safe&=variation<.10
        if safe.sum()<settings['minimum_pixels']:
            rows.append({'view':i,'accepted':False,'reason':'insufficient safe skin pixels'});masks.append(safe);colors.append(None);continue
        low,high=np.quantile(lum[safe],[.78,.95]);patch=safe&(lum>=low)&(lum<=high)
        pix=rgb[patch];color=np.median(pix,axis=0);mad=float(np.median(np.linalg.norm(pix-color,axis=1)))
        # Prefer better-exposed, stable skin in *any* view. Not the brightest single pixel.
        score=float(color@np.array([.2126,.7152,.0722])-.5*mad)
        rows.append({'view':i,'accepted':True,'safe_pixels':int(safe.sum()),'patch_pixels':int(patch.sum()),'rgb8':np.rint(color*255).astype(int).tolist(),'rgb_mad':mad,'score':score})
        masks.append(patch);colors.append(color)
    accepted=[i for i,c in enumerate(colors) if c is not None]
    if not accepted:raise ValueError('No trustworthy semantic skin patch in any view')
    selected=max(accepted,key=lambda i:rows[i]['score']);base=colors[selected]
    labs=rgb2lab(np.asarray([colors[i] for i in accepted]).reshape(1,-1,3))[0]
    spread=float(deltaE_ciede2000(labs,np.tile(rgb2lab(base.reshape(1,1,3))[0,0],(len(labs),1))).max())
    override=settings.get('override_rgb')
    if override is not None:
        a=np.asarray(override,float)
        if a.shape!=(3,) or not np.isfinite(a).all() or (a<0).any() or (a>255).any():raise ValueError('Invalid skin override RGB')
        base=a/255
    return base,masks,{'method':'eroded multiview semantic skin; stable bright quantiles; exposure-ranked observation',
        'selected_view':['front','left','right'][selected],'selected_index':selected,'views':rows,'skin_base_rgb8':np.rint(base*255).astype(int).tolist(),
        'cross_view_deltaE00_max':spread,'confidence':'LOW' if spread>settings['disagreement_deltaE00'] else 'UNCALIBRATED',
        'color_status':'USER_DESIGN_OVERRIDE' if override is not None else 'OBSERVED_UNCALIBRATED',
        'warnings':['cross-view lighting/color disagreement'] if spread>settings['disagreement_deltaE00'] else [],
        'illumination_calibrated':False,'true_skin_albedo_verified':False,'skin_color_truth_verified':False,'source_geometry_modified':False}
