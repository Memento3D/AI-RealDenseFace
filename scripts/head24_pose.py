"""Rigid posterior-cranium coordinate frame for head-attached geometry; no mesh edits."""
import numpy as np
from head24_policy import project,InputRejected


def cranial_camera(reference,view,camera,head_corners,eye_centers,max_error_px=5):
    reference=np.asarray(reference,float);view=np.asarray(view,float)
    head=np.asarray(head_corners,bool)
    if reference.shape!=view.shape or reference.ndim!=2 or reference.shape[1]!=3 or head.shape!=(len(reference),):
        raise InputRejected('INVALID_CRANIAL_CORRESPONDENCES')
    if not np.isfinite(reference).all() or not np.isfinite(view).all():raise InputRejected('NONFINITE_CRANIAL_GEOMETRY')
    lo,hi=reference[head].min(0),reference[head].max(0)
    # Reference model convention: +Y up, +Z toward face; no eyeballs or neck in the proxy.
    upper=reference[:,1]>np.asarray(eye_centers).mean(0)[1]+.02*(hi[1]-lo[1])
    posterior=head&upper&(reference[:,2]<(lo[2]+hi[2])/2)
    _,index=np.unique(reference,axis=0,return_index=True)
    index=index[posterior[index]]
    # Hold out one third of the unique posterior points. Repeated triangle corners
    # must not silently increase the evidence count or dominate the rigid fit.
    test=index[::3];train=np.delete(index,np.arange(0,len(index),3))
    if len(train)<30 or len(test)<15:raise InputRejected('INSUFFICIENT_POSTERIOR_CRANIAL_SUPPORT')
    a=reference[train];b=view[train];ca=a.mean(0);cb=b.mean(0)
    if np.linalg.matrix_rank(a-ca)<3 or np.linalg.matrix_rank(b-cb)<3:raise InputRejected('DEGENERATE_CRANIAL_SUPPORT')
    u,_,vt=np.linalg.svd((a-ca).T@(b-cb));d=np.eye(3);d[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@d@u.T
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=cb-rotation@ca
    adjusted=np.asarray(camera)@transform
    original,_=project(reference,camera);expected,_=project(view,camera);aligned,_=project(reference,adjusted)
    raw=np.linalg.norm(original-expected,axis=1);residual=np.linalg.norm(aligned-expected,axis=1)
    moved=reference@rotation.T+transform[:3,3];native=np.linalg.norm(moved-view,axis=1)
    report={'method':'rigid posterior-cranial frame with unique-point holdout; no source geometry deformation',
        'train_unique_vertices':len(train),'holdout_unique_vertices':len(test),
        'native_reference_to_view_cranium':transform.tolist(),
        'raw_upper_all_p95_px':float(np.quantile(raw[upper],.95)),
        'raw_upper_head_p95_px':float(np.quantile(raw[upper&head],.95)),
        'aligned_upper_head_p95_px_diagnostic_only':float(np.quantile(residual[upper&head],.95)),
        'cranial_train_p95_px':float(np.quantile(residual[train],.95)),
        'cranial_holdout_p95_px':float(np.quantile(residual[test],.95)),
        'cranial_holdout_p95_native':float(np.quantile(native[test],.95)),
        'head_height_native':float(hi[1]-lo[1]),'gate_px':max_error_px,
        'camera_calibrated':False,'source_geometry_modified':False,
        'scope':'posterior cranial proxy only; not a full-face or image-ground-truth test'}
    if max(report['cranial_train_p95_px'],report['cranial_holdout_p95_px'])>max_error_px:
        raise InputRejected('CRANIAL_POSE_PROXY_MISMATCH: '+str(report))
    return adjusted,report
