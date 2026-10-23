"""Bounded frontal anatomical-proxy experiment with three-view landmark guards."""
import numpy as np
from memento3d_head.adapters.shape_fitting import barycentric_points
from memento3d_head.adapters.photo_registration import PAIRS
from memento3d_head.adapters.manual_contour_comparison import resample
from memento3d_head.adapters.jaw_surface import arc_support
from memento3d_head.domain.contracts import ValidationError

POLICY={'version':1,'max_vertex_displacement_mm':3.,'max_coefficient_delta':.5,
        'max_view_mean_landmark_increase_px':.5,'max_individual_landmark_increase_px':1.,
        'prior_strength':25.,'jaw_weight':10.,'max_steps':30,'calibrated':False}


def project(points,view):
    import cv2
    r=np.asarray(view['rotation_vector'],float);t=np.asarray(view['translation_mm'],float)
    if np.any((points@cv2.Rodrigues(r)[0].T+t)[:,2]<=0):
        raise ValidationError('INVALID_DEPTH','Geometry crossed a frozen camera plane')
    return cv2.projectPoints(points,r,t,np.asarray(view['camera_matrix'],float),None)[0].reshape(-1,2)


def frozen_samples(base,boundary_ids,front_view,curves):
    """Freeze baseline mesh edge indices/weights; never reselect during optimization."""
    path=base[boundary_ids];xy=project(path,front_view)
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    samples=[]
    for curve in curves:
        target=resample(curve['points_px'],64)
        dense=resample(xy,1024)
        a=int(np.argmin(np.linalg.norm(dense-target[0],axis=1)))
        b=int(np.argmin(np.linalg.norm(dense-target[-1],axis=1)))
        if a==b:raise ValidationError('UNSUPPORTED_FRONT_JAW','Collapsed frontal endpoints')
        positions=np.linspace(a,b,64)/1023*lengths[-1]
        edge=np.clip(np.searchsorted(lengths,positions,side='right')-1,0,len(path)-2)
        weight=(positions-lengths[edge])/np.maximum(lengths[edge+1]-lengths[edge],1e-12)
        ids0=np.asarray(boundary_ids)[edge];ids1=np.asarray(boundary_ids)[edge+1]
        points=base[ids0]*(1-weight[:,None])+base[ids1]*weight[:,None]
        support=arc_support(target,project(points,front_view))
        if not support['supported']:raise ValidationError('UNSUPPORTED_FRONT_JAW','Frontal jaw does not pass coverage guards')
        samples.append((ids0,ids1,weight,target,curve['kind']))
    return samples


def fit_jaw(base,basis,faces,indices,weights,views,boundary_ids,curves):
    base=np.asarray(base,float);basis=np.asarray(basis,float)
    if basis.shape!=(len(base),3,8) or not np.isfinite(basis).all():
        raise ValidationError('INVALID_BASIS','Expected eight finite FLAME directions')
    samples=frozen_samples(base,boundary_ids,views['front'],curves)
    landmark_indices=[i for i,_,_ in PAIRS]
    targets={slot:np.array([r['observed_px'] for r in views[slot]['landmarks'][:7]]) for slot in ('front','left','right')}
    def measure(delta):
        mesh=base+np.einsum('ijk,k->ij',basis,delta)
        landmarks=barycentric_points(mesh,faces,indices,weights)[landmark_indices]
        errors={slot:project(landmarks,views[slot])-targets[slot] for slot in targets}
        jaw=[]
        for a,b,w,target,kind in samples:
            points=mesh[a]*(1-w[:,None])+mesh[b]*w[:,None]
            jaw.append(project(points,views['front'])-target)
        return mesh,errors,jaw
    _,initial_errors,initial_jaw=measure(np.zeros(8))
    initial_norms={slot:np.linalg.norm(e,axis=1) for slot,e in initial_errors.items()}
    def residual(delta):
        _,errors,jaw=measure(delta)
        return np.r_[np.concatenate([e.reshape(-1) for e in errors.values()]),
                     np.concatenate([e.reshape(-1) for e in jaw])*np.sqrt(POLICY['jaw_weight']/64),
                     delta*np.sqrt(2*POLICY['prior_strength'])]
    def permitted(delta,mesh,errors):
        if np.max(np.abs(delta))>POLICY['max_coefficient_delta']+1e-12:return False
        if np.max(np.linalg.norm(mesh-base,axis=1))>POLICY['max_vertex_displacement_mm']+1e-10:return False
        for slot,e in errors.items():
            norm=np.linalg.norm(e,axis=1);before=initial_norms[slot]
            if norm.mean()>before.mean()+POLICY['max_view_mean_landmark_increase_px']+1e-10:return False
            if np.any(norm>before+POLICY['max_individual_landmark_increase_px']+1e-10):return False
        t0=base[faces];t1=mesh[faces]
        n0=np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0]);n1=np.cross(t1[:,1]-t1[:,0],t1[:,2]-t1[:,0])
        return bool(np.all(np.einsum('ij,ij->i',n0,n1)>0))
    delta=np.zeros(8);history=[float(np.dot(residual(delta),residual(delta)))];rejected_steps=0
    for _ in range(POLICY['max_steps']):
        error=residual(delta)
        jac=np.column_stack([(residual(delta+np.eye(8)[i]*1e-4)-error)/1e-4 for i in range(8)])
        step=np.linalg.lstsq(jac,-error,rcond=None)[0];accepted=False
        for scale in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625):
            trial=delta+step*scale;mesh,errors,jaw=measure(trial)
            objective=float(np.dot(residual(trial),residual(trial)))
            # Do not accept a tradeoff that worsens either frontal training curve.
            improved=all(np.linalg.norm(e,axis=1).mean()<=np.linalg.norm(old,axis=1).mean()+1e-10
                         for e,old in zip(jaw,initial_jaw))
            if objective<history[-1]-1e-8 and improved and permitted(trial,mesh,errors):
                delta=trial;history.append(objective);accepted=True;break
            rejected_steps+=1
        if not accepted or np.linalg.norm(step)<1e-6:break
    mesh,errors,jaw=measure(delta)
    # Retain a chin check outside the fitting objective; semantic pairing is approximate.
    held_out={}
    old_chin=barycentric_points(base,faces,indices,weights)[[8]]
    new_chin=barycentric_points(mesh,faces,indices,weights)[[8]]
    for slot,view in views.items():
        if len(view.get('landmarks',[]))>7:
            target=np.asarray(view['landmarks'][7]['observed_px'],float)
            held_out[slot]={'baseline_residual_px':float(np.linalg.norm(project(old_chin,view)[0]-target)),
                            'candidate_residual_px':float(np.linalg.norm(project(new_chin,view)[0]-target)),
                            'used_for_fit':False,'semantic_correspondence_verified':False}
    return mesh,delta,{'policy':POLICY,'objective_history':history,'delta_coefficients':delta.tolist(),
        'max_vertex_displacement_mm':float(np.linalg.norm(mesh-base,axis=1).max()),'rejected_steps':rejected_steps,
        'held_out_chin':held_out,'landmarks':{slot:{'baseline_mean_px':float(initial_norms[slot].mean()),
             'candidate_mean_px':float(np.linalg.norm(e,axis=1).mean()),
             'baseline_per_point_px':initial_norms[slot].tolist(),'candidate_per_point_px':np.linalg.norm(e,axis=1).tolist()} for slot,e in errors.items()},
        'frontal_training_curves':[{'kind':sample[-1],'baseline_mean_px':float(np.linalg.norm(before,axis=1).mean()),
             'candidate_mean_px':float(np.linalg.norm(after,axis=1).mean())} for sample,before,after in zip(samples,initial_jaw,jaw)],
        'correspondences_frozen':True,'cameras_modified':False,'side_contours_used_for_fit':False,
        'likeness_verified':False,'automatic_acceptance':False}
