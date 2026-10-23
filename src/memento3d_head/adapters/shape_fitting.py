"""Conservative eight-parameter FLAME identity fitting with frozen cameras."""
import numpy as np
from memento3d_head.domain.contracts import ValidationError
from memento3d_head.adapters.photo_registration import PAIRS


def barycentric_points(vertices,faces,indices,weights):
    indices=np.asarray(indices,dtype=int).reshape(-1);weights=np.asarray(weights).reshape(68,3)
    return np.einsum('lij,li->lj',vertices[faces[indices]],weights)


def fit_shape(base,basis,faces,indices,weights,views,max_steps=30):
    import cv2
    base=np.asarray(base,dtype=float);basis=np.asarray(basis,dtype=float)
    if (basis.shape!=(len(base),3,8) or not np.isfinite(base).all() or not np.isfinite(basis).all()):
        raise ValidationError('INVALID_BASIS','Expected eight finite FLAME shape directions')
    p0=barycentric_points(base,faces,indices,weights)[[i for i,_,_ in PAIRS]]
    point_basis=np.stack([barycentric_points(basis[:,:,i],faces,indices,weights)[[j for j,_,_ in PAIRS]] for i in range(8)],axis=-1)
    cameras=[]
    for slot in ('front','left','right'):
        view=views[slot]
        camera=np.asarray(view['camera_matrix'],dtype=float)
        rotation=np.asarray(view['rotation_vector'],dtype=float)
        translation=np.asarray(view['translation_mm'],dtype=float)
        target=np.array([row['observed_px'] for row in view['landmarks'][:7]],dtype=float)
        if (camera.shape!=(3,3) or rotation.shape!=(3,) or translation.shape!=(3,) or target.shape!=(7,2)
                or not all(np.isfinite(a).all() for a in (camera,rotation,translation,target))):
            raise ValidationError('INVALID_CORRESPONDENCES','Invalid frozen camera or observed points')
        cameras.append((camera,rotation,translation,target))
    def residual(delta):
        points=p0+np.einsum('ijk,k->ij',point_basis,delta)
        result=[]
        for k,r,t,target in cameras:
            if np.any((points@cv2.Rodrigues(r)[0].T+t)[:,2]<=0):raise ValidationError('INVALID_DEPTH','Shape crossed camera plane')
            result.append(cv2.projectPoints(points,r,t,k,None)[0].reshape(-1,2)-target)
        return np.concatenate(result).reshape(-1)
    def objective(delta):
        norms=np.linalg.norm(residual(delta).reshape(-1,2),axis=1)
        robust=np.where(norms<=6.,.5*norms**2,6.*(norms-3.))
        return float(robust.sum()+25.*np.dot(delta,delta))
    def valid(delta):
        displacement=np.einsum('ijk,k->ij',basis,delta)
        return np.max(np.abs(delta))<=.5+1e-12 and np.max(np.linalg.norm(displacement,axis=1))<=3.+1e-10
    delta=np.zeros(8);history=[objective(delta)]
    for _ in range(max_steps):
        errors=residual(delta);norm=np.linalg.norm(errors.reshape(-1,2),axis=1)
        factor=np.repeat(np.sqrt(np.minimum(1.,6./np.maximum(norm,1e-12))),2)
        jac=np.column_stack([(residual(delta+np.eye(8)[i]*1e-4)-errors)/1e-4 for i in range(8)])
        a=np.vstack((jac*factor[:,None],np.sqrt(50.)*np.eye(8)))
        b=np.r_[-errors*factor,-np.sqrt(50.)*delta]
        step=np.linalg.lstsq(a,b,rcond=None)[0]
        accepted=False
        for rate in (1.,.5,.25,.125,.0625,.03125,.015625):
            candidate=delta+rate*step
            if valid(candidate) and objective(candidate)<history[-1]-1e-8:
                delta=candidate;history.append(objective(delta));accepted=True;break
        if not accepted or np.linalg.norm(step)<1e-6:break
    candidate=base+np.einsum('ijk,k->ij',basis,delta)
    triangles0=base[faces];triangles=candidate[faces]
    normals0=np.cross(triangles0[:,1]-triangles0[:,0],triangles0[:,2]-triangles0[:,0])
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    if np.any(np.sum(normals*normals0,axis=1)<=0):
        raise ValidationError('GEOMETRY_REJECTED','Candidate has collapsed or flipped triangles')
    info={'active_identity_dimensions':8,'delta_coefficients':delta.tolist(),'max_abs_coefficient_delta':.5,
          'max_allowed_vertex_displacement_mm':3.,'max_vertex_displacement_mm':float(np.linalg.norm(candidate-base,axis=1).max()),
          'prior_strength':25.,'huber_delta_px':6.,'objective_history':history,'cameras_modified':False,
          'contours_used_for_fit':False,'expressions_modified':False,
          'candidate_reduces_regularized_objective':history[-1]<history[0]-1e-8,
          'correspondence_policy':'seven_existing_approximate_semantic_points_per_view_no_occlusion_filter',
          'likeness_verified':False}
    return candidate,delta,info
