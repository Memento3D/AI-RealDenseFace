"""Fixed experimental checkerboard intrinsics, distortion-aware pose fitting."""
import json
import numpy as np
from memento3d_head.domain.contracts import ValidationError


def load_calibration(path, image_size):
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version')!=1 or data.get('status')!='EXPERIMENTAL_CALIBRATION_NEEDS_REVIEW':
        raise ValidationError('INVALID_CALIBRATION','Unsupported experimental calibration schema')
    if list(image_size)!=data.get('image_size'):
        raise ValidationError('CALIBRATION_SIZE_MISMATCH','Calibration resolution must match photos; no implicit scaling')
    if set(data.get('models',{}))!={'standard5','radial2'}:
        raise ValidationError('INVALID_CALIBRATION','Both standard5 and radial2 models are required')
    for item in data['models'].values():
        k=np.asarray(item['camera_matrix'],dtype=float);d=np.asarray(item['distortion'],dtype=float)
        if (k.shape!=(3,3) or d.shape!=(5,) or not np.isfinite(k).all() or not np.isfinite(d).all()
                or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1])
                or abs(k[0,1])>1e-9 or abs(k[1,0])>1e-9):
            raise ValidationError('INVALID_CALIBRATION','Invalid camera matrix or five-term distortion')
    return data


def fit_fixed_camera(points, targets, camera, distortion, initial, mesh_vertices):
    import cv2
    points=np.asarray(points,dtype=float);targets=np.asarray(targets,dtype=float)
    camera=np.asarray(camera,dtype=float);distortion=np.asarray(distortion,dtype=float)
    if (points.shape!=(7,3) or targets.shape!=(7,2) or not np.isfinite(points).all()
            or not np.isfinite(targets).all() or np.linalg.matrix_rank(points-points.mean(axis=0))<3):
        raise ValidationError('INVALID_CORRESPONDENCES','Need seven valid noncoplanar correspondences')
    seeds=[(np.asarray(initial['rotation_vector'],dtype=float).reshape(3,1),
            np.asarray(initial['translation_mm'],dtype=float).reshape(3,1))]
    ok,r,t=cv2.solvePnP(points,targets,camera,distortion,flags=cv2.SOLVEPNP_EPNP)
    if ok:seeds.append((r,t))
    candidates=[]
    for r,t in seeds:
        try:
            r,t=cv2.solvePnPRefineLM(points,targets,camera,distortion,r.copy(),t.copy())
            depth=(np.asarray(mesh_vertices)@cv2.Rodrigues(r)[0].T+t.reshape(3))[:,2]
            projected=cv2.projectPoints(points,r,t,camera,distortion)[0].reshape(-1,2)
            if not np.isfinite(depth).all() or np.any(depth<=0) or not np.isfinite(projected).all():continue
            cost=float(np.mean(np.sum((projected-targets)**2,axis=1)))
            candidates.append((cost,(camera,r,t,projected)))
        except cv2.error:
            continue
    if not candidates:raise ValidationError('CAMERA_FIT_FAILED','No positive-depth fixed-intrinsics pose')
    return min(candidates,key=lambda item:item[0])[1]
