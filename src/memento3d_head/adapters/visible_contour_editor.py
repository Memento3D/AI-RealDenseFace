"""Projected whole-mesh outer boundary and a self-contained offline annotation editor."""
import base64
import json
import numpy as np
from pathlib import Path
from memento3d_head.domain.contracts import ValidationError


def projected_outer_boundary(vertices,faces,view,size):
    import cv2
    k=np.asarray(view['camera_matrix'],dtype=float);r=np.asarray(view['rotation_vector'],dtype=float)
    t=np.asarray(view['translation_mm'],dtype=float)
    z=(vertices@cv2.Rodrigues(r)[0].T+t)[:,2]
    if not np.isfinite(z).all() or np.any(z<=0):raise ValidationError('INVALID_DEPTH','Mesh crosses camera plane')
    xy=cv2.projectPoints(vertices,r,t,k,None)[0].reshape(-1,2)
    w,h=size;mask=np.zeros((h,w),np.uint8)
    # For the outer silhouette, union of projected triangles equals foreground coverage;
    # internal hidden edges are never drawn. This is NOT an anatomical face-only mask.
    for face in faces:
        polygon=xy[face]
        if (polygon[:,0].max()<0 or polygon[:,0].min()>=w or polygon[:,1].max()<0 or polygon[:,1].min()>=h):continue
        if np.max(np.abs(polygon))>1e6:raise ValidationError('INVALID_PROJECTION','Unbounded image projection')
        cv2.fillConvexPoly(mask,np.rint(polygon).astype(np.int32),255)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1,2).tolist() for c in contours if len(c)>=3]


def write_editor(destination,manifest,images):
    payload=dict(manifest)
    payload['images']={slot:'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii') for slot,path in images.items()}
    template=Path(__file__).with_name('contour_editor.html').read_text(encoding='utf-8')
    # Escape script-closing characters even if future metadata contains user-provided text.
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    destination.write_text(template.replace('__CONTOUR_DATA__',encoded),encoding='utf-8')
