"""Approximate frontal camera fit; no changes to identity, vertices or expressions."""
import numpy as np
from PIL import Image, ImageDraw
from memento3d_head.domain.contracts import ValidationError

# Zero-based 68-point FLAME to MediaPipe semantic correspondences, approximate.
# No contour landmarks: their visibility depends on the camera.
PAIRS=((30,1,'nose_tip'),(36,33,'eye_outer_a'),(39,133,'eye_inner_a'),
       (42,362,'eye_inner_b'),(45,263,'eye_outer_b'),
       (48,61,'mouth_corner_a'),(54,291,'mouth_corner_b'))


def embedded_points(vertices, faces, face_indices, barycentric):
    indices=np.asarray(face_indices,dtype=int).reshape(-1)
    weights=np.asarray(barycentric,dtype=float)
    if weights.shape == (1, 68, 3):
        weights = weights[0]
    if (len(vertices)!=5023 or len(faces)!=9976 or indices.shape!=(68,) or weights.shape!=(68,3)
            or indices.min()<0 or indices.max()>=len(faces) or not np.isfinite(weights).all()
            or not np.allclose(weights.sum(axis=1),1,atol=1e-5)):
        raise ValidationError('INVALID_EMBEDDING','Expected MICA FLAME2020 topology and 68 landmarks')
    return (vertices[faces[indices]]*weights[:,:,None]).sum(axis=1)


def fit_camera(points, targets, size):
    import cv2
    points=np.asarray(points,dtype=np.float64)
    targets=np.asarray(targets,dtype=np.float64)
    if (points.ndim!=2 or points.shape[1]!=3 or targets.shape!=(len(points),2)
            or len(points)<6 or not np.isfinite(points).all() or not np.isfinite(targets).all()
            or np.linalg.matrix_rank(points-points.mean(axis=0))<3):
        raise ValidationError('INVALID_CORRESPONDENCES','Need finite noncoplanar 3D and 2D points')
    w,h=size
    # Intrinsics are assumed, not calibrated from this photograph.
    focal=float(max(w,h))
    camera=np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]],dtype=float)
    ok,rotation,translation=cv2.solvePnP(points,targets,camera,None,flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise ValidationError('CAMERA_FIT_FAILED','PnP initialization failed')
    rotation,translation=cv2.solvePnPRefineLM(points,targets,camera,None,rotation,translation)
    transformed=points@cv2.Rodrigues(rotation)[0].T+translation.reshape(3)
    if not np.isfinite(transformed).all() or np.any(transformed[:,2]<=0):
        raise ValidationError('CAMERA_FIT_FAILED','Invalid camera depth')
    projected=cv2.projectPoints(points,rotation,translation,camera,None)[0].reshape(-1,2)
    return camera,rotation,translation,projected


def draw_overlay(image_path, vertices, faces, points, observations, destination, camera_solution=None, distortion=None):
    import cv2
    image=Image.open(image_path).convert('RGB')
    target=np.array([[observations[j][0]*image.width,observations[j][1]*image.height] for _,j,_ in PAIRS])
    selected=points[[i for i,_,_ in PAIRS]]
    camera,rotation,translation,projected=(fit_camera(selected,target,image.size)
                                           if camera_solution is None else camera_solution)
    vertex_projection=cv2.projectPoints(vertices,rotation,translation,camera,distortion)[0].reshape(-1,2)
    depth=(vertices@cv2.Rodrigues(rotation)[0].T+translation.reshape(3))[:,2]
    if np.any(depth<=0) or not np.isfinite(vertex_projection).all():
        raise ValidationError('CAMERA_FIT_FAILED','Mesh intersects estimated camera plane')
    # Sparse wireframe is diagnostic only, without hidden-surface removal.
    wire=image.copy(); draw=ImageDraw.Draw(wire)
    for tri in faces[::4]:
        coords=[tuple(p) for p in vertex_projection[tri]]
        draw.line(coords+[coords[0]],fill=(0,200,255),width=1)
    overlay=Image.blend(image,wire,.25); draw=ImageDraw.Draw(overlay)
    fit_errors=np.linalg.norm(projected-target,axis=1)
    # Chin is held out of camera fitting.
    chin=cv2.projectPoints(points[[8]],rotation,translation,camera,distortion)[0].reshape(-1,2)
    projected=np.vstack((projected,chin))
    target=np.vstack((target,[[observations[152][0]*image.width,observations[152][1]*image.height]]))
    errors=np.linalg.norm(projected-target,axis=1)
    rows=[]
    for (_,mp_index,name),a,b,error in zip(PAIRS+((8,152,'chin_held_out'),),target,projected,errors):
        draw.line([tuple(a),tuple(b)],fill='yellow',width=2)
        for p,color in ((a,'lime'),(b,'red')):
            x,y=p;draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
        rows.append({'name':name,'mediapipe_index':mp_index,'observed_px':a.tolist(),
                     'projected_px':b.tolist(),'residual_px':float(error),'used_for_fit':name!='chin_held_out'})
    banner=Image.new('RGB',(max(image.width,700),image.height+65),'#20252c')
    banner.paste(overlay,(0,65)); label=ImageDraw.Draw(banner)
    label.text((8,8),'GREEN: observed | RED: mesh | YELLOW: residual | CYAN: sparse wire',fill='white')
    label.text((8,30),'Approximate camera; same points used for fit and residual. Not a likeness score.',fill='white')
    banner.save(destination)
    return {'camera_matrix':camera.tolist(),'rotation_vector':rotation.reshape(-1).tolist(),
            'translation_mm':translation.reshape(-1).tolist(),'landmarks':rows,
            'mean_fit_residual_px':float(fit_errors.mean()),'max_fit_residual_px':float(fit_errors.max()),
            'intrinsics_calibrated':False,'distortion_coefficients':None if distortion is None else np.asarray(distortion).reshape(-1).tolist(),'mapping':'approximate_semantic_pairs_v1',
            'evaluation':'seven_fit_points_and_one_held_out_chin_not_likeness','hidden_surface_removal':False,
            'geometry_modified':False,'likeness_verified':False}
