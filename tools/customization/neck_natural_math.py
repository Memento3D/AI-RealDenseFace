"""Local neck-only geometry and color reconstruction helpers.

These functions never alter facial expression or infer missing facial detail.
"""
import numpy as np


def smoothstep(t):
    t=np.clip(t,0,1)
    return t*t*(3-2*t)


def neck_base_mask(vertices,neck_indices,boundary_indices,ceiling):
    v=np.asarray(vertices,float)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or not np.isfinite(ceiling):raise ValueError('Invalid base geometry')
    neck=np.asarray(neck_indices,int);boundary=np.asarray(boundary_indices,int)
    for indices in [neck,boundary]:
        if indices.ndim!=1 or (indices<0).any() or (indices>=len(v)).any():raise ValueError('Invalid semantic vertex indices')
    mask=np.zeros(len(v),bool);mask[neck]=True
    # FLAME's outer neck/shoulder boundary includes vertices omitted by its
    # neck mask. Leaving them untouched creates a visible horizontal shelf.
    base=boundary[v[boundary,2]<ceiling];mask[base]=True
    return mask


def soften_neck_base(vertices,neck_mask,cut_height,transition=.04,target_center=None,target_radii=None):
    v=np.asarray(vertices,float);mask=np.asarray(neck_mask,bool)
    if v.ndim!=2 or v.shape[1]!=3 or mask.shape!=(len(v),) or not np.isfinite(v).all():raise ValueError('Invalid neck geometry')
    if transition<=0:raise ValueError('Invalid transition height')
    band=v[mask&(v[:,2]>cut_height+transition*.6)&(v[:,2]<cut_height+transition*1.8)]
    if len(band)<8:raise ValueError('Insufficient neck support vertices')
    lower,upper=np.quantile(band[:,:2],[.03,.97],axis=0)
    center=(lower+upper)/2;radii=np.maximum((upper-lower)/2,[.015,.02])
    if target_center is not None:center=np.asarray(target_center,float)
    if target_radii is not None:radii=np.asarray(target_radii,float)
    if center.shape!=(2,) or radii.shape!=(2,) or not np.isfinite([center,radii]).all() or (radii<=0).any():raise ValueError('Invalid collar envelope')
    offset=v[:,:2]-center;radius=np.linalg.norm(offset/radii,axis=1)
    target=center+offset/np.maximum(radius[:,None],1)
    weight=smoothstep((cut_height+transition-v[:,2])/transition)*mask
    result=v.copy();result[:,:2]+=(target-v[:,:2])*weight[:,None]
    if not np.array_equal(result[~mask],v[~mask]):raise AssertionError('Non-neck geometry changed')
    return result,{'center_xy':center.tolist(),'radii_xy':radii.tolist(),'transition_height_m':transition,
                   'max_neck_displacement_m':float(np.linalg.norm(result-v,axis=1).max())}


def quadratic_features(points):
    x,y,z=np.asarray(points,float).T
    return np.column_stack([np.ones(len(x)),x,y,z,x*x,y*y,z*z,x*y,x*z,y*z])


def fit_skin_field(points,colors):
    p=np.asarray(points,float);c=np.asarray(colors,float)
    if p.ndim!=2 or p.shape[1]!=3 or c.shape!=p.shape or len(p)<32 or not np.isfinite(p).all() or not np.isfinite(c).all():raise ValueError('Insufficient finite skin samples')
    center=np.median(p,axis=0);scale=np.maximum(np.ptp(p,axis=0),.02)
    X=quadratic_features((p-center)/scale);weights=np.ones(len(p))
    ridge=np.diag([1e-8]+[.5]*9)
    for _ in range(5):
        coef=np.linalg.solve(X.T@(X*weights[:,None])+ridge,X.T@(c*weights[:,None]))
        residual=np.linalg.norm(X@coef-c,axis=1)
        cutoff=max(6,float(np.median(residual))*1.5)
        weights=np.minimum(1,cutoff/np.maximum(residual,1e-6))
    low=np.quantile(c,.08,axis=0)-8;high=np.quantile(c,.92,axis=0)+8
    return {'center':center,'scale':scale,'coefficients':coef,'minimum':np.maximum(low,0),'maximum':np.minimum(high,255)}


def skin_field(points,model):
    X=quadratic_features((np.asarray(points)-model['center'])/model['scale'])
    return np.clip(X@model['coefficients'],model['minimum'],model['maximum'])


def crown_similarity(source_vertices,target_top,pivot):
    v=np.asarray(source_vertices,float);pivot=np.asarray(pivot,float)
    height=float(v[:,2].max()-pivot[2])
    if height<=.05:raise ValueError('Invalid head extent')
    ratio=float((target_top-pivot[2])/height)
    if not .9<=ratio<=1.2:raise ValueError('Headwear fit exceeds reviewed similarity range')
    matrix=np.eye(4);matrix[:3,:3]*=ratio;matrix[:3,3]=pivot*(1-ratio)
    return matrix
