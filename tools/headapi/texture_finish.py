"""Conservative atlas finishing; inferred pixels have separate provenance."""
import numpy as np
import cv2


def boundary_pairs(owner, eligible):
    pairs=[]
    for axis in [0,1]:
        a=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
        b=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
        keep=eligible[a]&eligible[b]&(owner[a]!=owner[b])&(owner[a]>0)&(owner[b]>0)
        pairs.append((a,b,keep))
    return pairs


def seam_metric(atlas,pairs):
    values=[np.linalg.norm(atlas[a].astype(float)-atlas[b].astype(float),axis=-1)[keep] for a,b,keep in pairs]
    values=np.concatenate(values)
    return {'count':len(values),'median_rgb_jump':float(np.median(values)) if len(values) else None,'p90_rgb_jump':float(np.percentile(values,90)) if len(values) else None}


def finish_atlas(atlas,owner,interior,skin,protected,confidence):
    observed=owner>0;skin=skin.astype(bool)&observed;interior=interior.astype(bool)
    protected=cv2.dilate(protected.astype(np.uint8),np.ones((17,17),np.uint8))>0
    border_distance=cv2.distanceTransform(np.pad(interior.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
    def normalized_blur(mask):
        weight=cv2.GaussianBlur(mask.astype(np.float32),(0,0),1)
        color=cv2.GaussianBlur(atlas.astype(np.float32)*mask[...,None],(0,0),1)
        return color/np.maximum(weight[...,None],1e-6)
    # Bounded local diffusion at source-owner boundaries; UV chart edges stay untouched.
    eligible=skin&~protected&(border_distance>8)&(confidence>=.35)
    pairs=boundary_pairs(owner,eligible)
    seams=np.zeros(owner.shape,np.uint8)
    for a,b,keep in pairs:
        seams[a][keep]=1;seams[b][keep]=1
    distance=cv2.distanceTransform(1-seams,cv2.DIST_L2,5) if seams.any() else np.full(owner.shape,1e6,np.float32)
    active=eligible&(distance<12)
    delta=np.zeros_like(atlas,dtype=np.float32)
    correction=np.clip(normalized_blur(eligible)-atlas.astype(np.float32),-6,6)
    delta[active]=correction[active]*(.5*(1-distance[active]/12))[:,None]
    adjusted=np.clip(np.rint(atlas.astype(float)+delta),0,255).astype(np.uint8)
    # Fill only enclosed tiny components with a complete, smooth observed-skin ring.
    missing=interior&~observed
    count,labels,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),connectivity=8)
    filled=np.zeros(owner.shape,np.uint8);skipped={'large':0,'unsafe_ring':0,'high_variation':0}
    for label in range(1,count):
        x,y,w,h,area=stats[label]
        if area>64:skipped['large']+=1;continue
        x0=max(0,x-5);x1=min(owner.shape[1],x+w+5);y0=max(0,y-5);y1=min(owner.shape[0],y+h+5)
        component=(labels[y0:y1,x0:x1]==label)
        ring=(cv2.dilate(component.astype(np.uint8),np.ones((9,9),np.uint8))>0)&~component
        safe=skin[y0:y1,x0:x1]&~protected[y0:y1,x0:x1]&(confidence[y0:y1,x0:x1]>=.35)
        touches_image=x==0 or y==0 or x+w==owner.shape[1] or y+h==owner.shape[0]
        if touches_image or not ring.any() or not safe[ring].all():skipped['unsafe_ring']+=1;continue
        colors=adjusted[y0:y1,x0:x1][ring]
        if np.max(np.std(colors.astype(float),axis=0))>12:skipped['high_variation']+=1;continue
        patch=cv2.inpaint(adjusted[y0:y1,x0:x1],component.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
        adjusted[y0:y1,x0:x1][component]=patch[component];filled[y0:y1,x0:x1][component]=1
    provenance=np.zeros(owner.shape,np.uint8);provenance[observed]=1;provenance[filled>0]=2
    output_conf=confidence.copy();output_conf[filled>0]=.15
    if not np.array_equal(adjusted[~observed&(filled==0)],atlas[~observed&(filled==0)]):raise ValueError('Changed unsupported unknown texels')
    if not np.array_equal(adjusted[protected],atlas[protected]):raise ValueError('Changed protected features')
    return adjusted,output_conf,provenance,{
        'seam_before':seam_metric(atlas,pairs),'seam_after':seam_metric(adjusted,pairs),
        'observed_texels':int(observed.sum()),'filled_texels':int(filled.sum()),'remaining_missing_interior_texels':int((missing&(filled==0)).sum()),
        'changed_observed_texels':int((np.any(adjusted!=atlas,axis=-1)&observed).sum()),
        'maximum_observed_channel_change':int(np.max(np.abs(adjusted.astype(np.int16)-atlas.astype(np.int16))[observed])) if observed.any() else 0,
        'skipped_hole_components':skipped,'protected_unchanged':True,'unsupported_unknown_unchanged':True,
        'confidence':'heuristic only; inferred fills capped at 0.15; no calibrated probability',
        'policy':{'version':2,'method':'bounded normalized local diffusion','seam_radius':12,'blur_sigma':1,'correction_channel_clip':6,'strength':.5,'protection_radius':8,'minimum_confidence':.35,'maximum_hole_area':64,'ring_radius':4,'maximum_ring_std':12}}
