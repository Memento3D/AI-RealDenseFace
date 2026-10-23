"""Build a self-contained, shared-camera WebGL comparison without network assets."""
import json
from pathlib import Path
import numpy as np
from memento3d_head.adapters.smooth_preview import vertex_normals
from memento3d_head.domain.contracts import ValidationError


def write_comparison(destination,base,candidate,faces,provenance=None):
    base=np.asarray(base,float);candidate=np.asarray(candidate,float);faces=np.asarray(faces)
    if (base.ndim!=2 or base.shape[1]!=3 or candidate.shape!=base.shape or not np.isfinite(base).all()
        or not np.isfinite(candidate).all() or len(base)>65535 or faces.ndim!=2 or faces.shape[1]!=3
        or len(faces)==0 or faces.dtype.kind not in 'iu' or faces.min()<0 or faces.max()>=len(base)):
        raise ValidationError('INVALID_COMPARISON_MESH','Expected matching finite meshes with <=65535 vertices')
    low=base.min(0);high=base.max(0);center=(low+high)*.5
    radius=float(np.linalg.norm(base-center,axis=1).max())
    if radius<1e-6:raise ValidationError('INVALID_COMPARISON_MESH','Degenerate bounds')
    delta=np.linalg.norm(candidate-base,axis=1)
    payload={'schema_version':1,'units':'millimeters','provenance':provenance or {},'center_mm':center.tolist(),'radius_mm':radius,'faces':faces.tolist(),
             'displacement_mm':delta.tolist(),'max_displacement_mm':float(delta.max()),
             'baseline':{'vertices':base.tolist(),'normals':vertex_normals(base,faces).tolist()},
             'candidate':{'vertices':candidate.tolist(),'normals':vertex_normals(candidate,faces).tolist()},
             'shared_camera':True,'shared_scale':True,'geometry_exaggerated':False,'likeness_verified':False}
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=Path(__file__).with_name('head_compare.html').read_text(encoding='utf-8')
    destination.write_text(template.replace('__HEAD_DATA__',encoded),encoding='utf-8')
    return {'vertices':len(base),'faces':len(faces),'max_displacement_mm':float(delta.max()),
            'shared_camera':True,'shared_scale':True,'geometry_exaggerated':False,
            'renderer':'offline_webgl1_on_demand','external_assets':False}
