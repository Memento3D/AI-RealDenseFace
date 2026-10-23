"""Bounded shared-focal camera experiment; robust loss never modifies geometry."""
import numpy as np
from memento3d_head.adapters.photo_registration import PAIRS
from memento3d_head.domain.contracts import ValidationError


def project(points, camera, rotation, translation):
    import cv2
    return cv2.projectPoints(points,rotation,translation,camera,None)[0].reshape(-1,2)


def huber_cost(residual, weights, delta=6.):
    lengths=np.linalg.norm(residual,axis=1)
    loss=np.where(lengths<=delta,.5*lengths**2,delta*(lengths-.5*delta))
    return float(np.sum(weights*loss)/np.sum(weights))


def heuristic_weights(points, baseline, yaw):
    """Camera depth chooses farther eye; this is not an occlusion detector."""
    import cv2
    weights=np.ones(7)
    if abs(yaw)>20:
        depth=(points@cv2.Rodrigues(np.asarray(baseline['rotation_vector']))[0].T
               +np.asarray(baseline['translation_mm']))[:,2]
        far=[1,2] if depth[[1,2]].mean()>depth[[3,4]].mean() else [3,4]
        weights[far]=.35
    return weights


def refine_pose(points, targets, size, focal, weights, initial):
    """IRLS with analytic OpenCV pose Jacobian and monotonic backtracking."""
    import cv2
    w,h=size
    camera=np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]],dtype=float)
    rotation=np.asarray(initial['rotation_vector'],dtype=float).reshape(3,1).copy()
    translation=np.asarray(initial['translation_mm'],dtype=float).reshape(3,1).copy()
    # Keep approximately constant projected scale when changing focal length.
    translation[2]*=focal/initial['camera_matrix'][0][0]
    def valid(r,t):
        depth=(points@cv2.Rodrigues(r)[0].T+t.reshape(3))[:,2]
        return np.isfinite(depth).all() and np.all(depth>1.)
    if not valid(rotation,translation):
        raise ValidationError('CAMERA_FIT_FAILED','Invalid initial depth')
    for _ in range(35):
        pixels,jac=cv2.projectPoints(points,rotation,translation,camera,None)
        residual=pixels.reshape(-1,2)-targets
        norm=np.linalg.norm(residual,axis=1)
        robust=weights*np.minimum(1.,6./np.maximum(norm,1e-12))
        factor=np.repeat(np.sqrt(robust),2)
        j=jac[:,:6]*factor[:,None]; b=residual.reshape(-1)*factor
        step=np.linalg.lstsq(np.vstack((j,np.eye(6)*1e-5)),np.r_[-b,np.zeros(6)],rcond=None)[0]
        cost=huber_cost(residual,weights)
        accepted=False
        for rate in (1.,.5,.25,.125,.0625):
            r=rotation+rate*step[:3,None];t=translation+rate*step[3:,None]
            if valid(r,t) and huber_cost(project(points,camera,r,t)-targets,weights)<cost-1e-10:
                rotation,translation=r,t;accepted=True;break
        if not accepted or np.linalg.norm(step)<1e-7:break
    pixels=project(points,camera,rotation,translation)
    return (camera,rotation,translation,pixels),huber_cost(pixels-targets,weights)


def search_shared_focal(points, views, sizes):
    if len(set(sizes.values()))!=1:
        raise ValidationError('INCOMPATIBLE_IMAGES','Shared focal v1 requires identical oriented image dimensions')
    selected=np.asarray(points)[[i for i,_,_ in PAIRS]]
    if selected.shape!=(7,3) or not np.isfinite(selected).all():
        raise ValidationError('INVALID_CORRESPONDENCES','Invalid landmark geometry')
    weights={slot:heuristic_weights(selected,v,v['detected_yaw_degrees']) for slot,v in views.items()}
    targets={slot:np.array([row['observed_px'] for row in view['landmarks'][:7]]) for slot,view in views.items()}
    base=max(next(iter(sizes.values())))
    factors=np.unique(np.r_[np.linspace(.55,2.,30),1.])
    trials=[];best=None
    for factor in factors:
        solutions={};costs={}
        try:
            for slot in views:
                solution,cost=refine_pose(selected,targets[slot],sizes[slot],float(base*factor),weights[slot],views[slot])
                solutions[slot]=solution;costs[slot]=cost
        except ValidationError:
            continue
        objective=float(np.mean(list(costs.values())))
        trials.append({'focal_px':float(base*factor),'objective':objective,'per_view':costs})
        if best is None or objective<best[0]:best=(objective,float(base*factor),solutions)
    if best is None:raise ValidationError('CAMERA_FIT_FAILED','No valid focal candidates')
    baseline=float(np.mean([huber_cost(np.array([r['projected_px'] for r in views[s]['landmarks'][:7]])-targets[s],weights[s]) for s in views]))
    return best[2],weights,{'focal_px':best[1],'search_bounds_px':[float(base*.55),float(base*2.)],
        'at_search_boundary':bool(np.isclose(best[1],base*.55) or np.isclose(best[1],base*2.)),
        'baseline_weighted_huber':baseline,'candidate_weighted_huber':best[0],
        'candidate_reduces_fit_objective':best[0]<baseline,'trials':trials,
        'intrinsics_calibrated':False,'visibility_policy':'depth_based_far_eye_weight_0.35_above_20deg_heuristic',
        'huber_delta_px':6.,'weights':{s:w.tolist() for s,w in weights.items()},
        'geometry_modified':False,'likeness_verified':False}
