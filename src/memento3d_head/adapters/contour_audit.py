"""Pose-indexed FLAME jaw diagnostics and mesh self-occlusion, no shape updates."""
import numpy as np
from PIL import Image,ImageDraw
from memento3d_head.adapters.photo_registration import PAIRS
from memento3d_head.domain.contracts import ValidationError

JAW_MP=(127,234,93,132,58,172,136,150,152,379,365,397,288,361,323,454,356)


def select_contour(rotation_vector,face_table,weight_table):
    import cv2
    face_table=np.asarray(face_table);weight_table=np.asarray(weight_table)
    if face_table.shape!=(79,17) or weight_table.shape!=(79,17,3):
        raise ValidationError('INVALID_CONTOUR_TABLE','Expected FLAME 79x17 dynamic embedding')
    if not np.isfinite(weight_table).all() or not np.allclose(weight_table.sum(axis=-1),1,atol=1e-5):
        raise ValidationError('INVALID_CONTOUR_TABLE','Invalid barycentric weights')
    # OpenCV camera axes: X right,Y down,Z forward. FLAME frontal camera faces +Z.
    rotation=np.diag([1.,-1.,-1.])@cv2.Rodrigues(np.asarray(rotation_vector,dtype=float))[0]
    yaw=float(np.degrees(np.arctan2(-rotation[2,0],np.hypot(rotation[0,0],rotation[1,0]))))
    angle=int(np.rint(np.clip(yaw,-39,39)))
    row=angle if angle>=0 else 39-angle
    return face_table[row].astype(int),weight_table[row],{'camera_yaw_degrees':yaw,'table_row':row,
        'table_yaw_degrees':angle,'table_range_degrees':[-39,39],
        'angle_clamped':abs(yaw)>39,'within_table_support':abs(yaw)<=39,
        'camera_axis_conversion':'diag(1,-1,-1) @ R_opencv',
        'pose_source':'frozen_registration_camera','pose_independently_verified':False}


def interpolate(vertices,faces,indices,weights):
    indices=np.asarray(indices,dtype=int)
    if indices.min()<0 or indices.max()>=len(faces):raise ValidationError('INVALID_CONTOUR_TABLE','Face index outside mesh')
    return np.einsum('nij,ni->nj',vertices[faces[indices]],weights)


def mesh_visibility(vertices,faces,points,rotation_vector,translation,tolerance_mm=1.):
    """Ray/triangle intersections; only mesh self-occlusion, not hair/hands/eyelids."""
    import cv2
    rotation=cv2.Rodrigues(np.asarray(rotation_vector,dtype=float))[0]
    origin=-rotation.T@np.asarray(translation,dtype=float)
    triangles=vertices[faces];a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    s=origin-a;q=np.cross(s,e1)
    result=[]
    for point in points:
        direction=point-origin;distance=float(np.linalg.norm(direction));direction/=max(distance,1e-12)
        h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h)
        inv=np.zeros_like(det);valid=np.abs(det)>1e-10;inv[valid]=1/det[valid]
        u=np.einsum('ij,ij->i',s,h)*inv;v=(q@direction)*inv
        t=np.einsum('ij,ij->i',e2,q)*inv
        hit=valid&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>1e-6)
        closer=hit&(t<distance-tolerance_mm)
        depth=(rotation@point+np.asarray(translation))[2]
        result.append({'positive_depth':bool(depth>0),'self_occluded':bool(closer.any()),
                       'mesh_visible':bool(depth>0 and not closer.any()),
                       'nearest_intersection_mm':float(t[hit].min()) if hit.any() else None,
                       'target_distance_mm':distance})
    return result


def audit_mesh(vertices,faces,embedding,view,observed,size):
    import cv2
    indices,weights,selection=select_contour(view['rotation_vector'],embedding['dynamic_faces'],embedding['dynamic_weights'])
    contour=interpolate(vertices,faces,indices,weights)
    full=interpolate(vertices,faces,embedding['full_faces'],embedding['full_weights'])
    points=np.vstack((full[[i for i,_,_ in PAIRS]],contour))
    k=np.asarray(view['camera_matrix']);r=np.asarray(view['rotation_vector']);t=np.asarray(view['translation_mm'])
    projected=cv2.projectPoints(points,r,t,k,None)[0].reshape(-1,2)
    visibility=mesh_visibility(vertices,faces,points,r,t)
    mp_indices=[j for _,j,_ in PAIRS]+list(JAW_MP)
    targets=np.array([[observed[j][0]*size[0],observed[j][1]*size[1]] for j in mp_indices])
    rows=[]
    for i,(xy,target,vis,mp) in enumerate(zip(projected,targets,visibility,mp_indices)):
        contour_point=i>=7
        rows.append({'name':f'jaw_{i-7:02d}' if contour_point else PAIRS[i][2],
                     'mediapipe_index':mp,'kind':'dynamic_contour' if contour_point else 'static_semantic',
                     'observed_px':target.tolist(),'projected_px':xy.tolist(),
                     'residual_px':float(np.linalg.norm(xy-target)),**vis,
                     'in_frame':bool(0<=xy[0]<size[0] and 0<=xy[1]<size[1]),
                     'used_for_fit':False,'semantic_mapping_verified':False})
    return {'selection':selection,'landmarks':rows,'mesh_visibility_tolerance_mm':1.,
            'external_occlusion_tested':False,'mapping':'approximate_ibug17_to_mediapipe_oval_v1',
            'contour_mean_residual_px':float(np.mean([x['residual_px'] for x in rows[7:]])),
            'contours_used_for_fit':False,'likeness_verified':False}


def draw_audit(photo,destination,baseline,candidate):
    with Image.open(photo) as source:image=source.convert('RGB')
    canvas=Image.new('RGB',(image.width*2,image.height+85),'#20252c');header=ImageDraw.Draw(canvas)
    for panel,(label,audit) in enumerate((('ORIGINAL MICA',baseline),('v0.5 SHAPE CANDIDATE',candidate))):
        im=image.copy();draw=ImageDraw.Draw(im)
        jaw=audit['landmarks'][7:]
        draw.line([tuple(row['observed_px']) for row in jaw],fill='lime',width=2)
        draw.line([tuple(row['projected_px']) for row in jaw],fill='red',width=2)
        for i,row in enumerate(audit['landmarks']):
            x,y=row['projected_px'];ox,oy=row['observed_px']
            color='orange' if row['self_occluded'] else 'red'
            draw.line([(x,y),(ox,oy)],fill='yellow',width=1)
            draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
            draw.ellipse((ox-2,oy-2,ox+2,oy+2),fill='lime')
            if i>=7:draw.text((x+4,y+2),str(i-7),fill=color)
        canvas.paste(im,(panel*image.width,85))
        header.text((panel*image.width+8,8),label,fill='white')
        selection=audit['selection']
        header.text((panel*image.width+8,28),f"Yaw {selection['camera_yaw_degrees']:.1f}; table {selection['table_yaw_degrees']} deg; clamped={selection['angle_clamped']}",fill='white')
        header.text((panel*image.width+8,48),'Green: observed | red: FLAME | orange: mesh self-occluded',fill='white')
        header.text((panel*image.width+8,65),'Approximate contour pairs. Diagnostic only; no fitting or likeness score.',fill='white')
    canvas.save(destination)
