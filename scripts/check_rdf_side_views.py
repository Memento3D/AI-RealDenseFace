"""Inspect held-out photos with a fixed RealDenseFace mesh; no shape refinement."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from uuid import uuid4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_camera(vertices, pixels, k):
    import cv2
    import numpy as np
    object_points=np.ascontiguousarray(vertices,dtype=np.float64)
    image_points=np.ascontiguousarray(pixels,dtype=np.float64)
    ok,rotation,translation,inliers=cv2.solvePnPRansac(object_points,image_points,k,None,
        iterationsCount=300,reprojectionError=6.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<20:
        raise ValueError('Insufficient pose inliers; cannot register this view')
    ids=inliers.ravel()
    rotation,translation=cv2.solvePnPRefineLM(object_points[ids],image_points[ids],k,None,rotation,translation)
    projected=cv2.projectPoints(object_points,rotation,translation,k,None)[0].reshape(-1,2)
    residual=np.linalg.norm(projected-image_points,axis=1)
    camera=object_points@cv2.Rodrigues(rotation)[0].T+translation.ravel()
    if not np.isfinite(projected).all() or np.any(camera[:,2]<=0):raise ValueError('Invalid fitted camera')
    return rotation,translation,{'dense_prior_inliers':len(ids),'dense_prior_points':len(vertices),
        'median_inlier_residual_px':float(np.median(residual[ids])),
        'median_all_residual_px':float(np.median(residual)),
        'residual_is_to_network_predictions_not_ground_truth':True}


def visible_wire(image, vertices, faces, k, rotation, translation):
    import cv2
    import numpy as np
    # Use the same CPU perspective depth rasterizer as the existing preview.
    from memento3d_head.adapters.photo_projection import depth_buffer
    camera=vertices@cv2.Rodrigues(rotation)[0].T+translation.ravel()
    xy,depth=depth_buffer(camera,faces,k,(image.shape[1],image.shape[0]))
    tri=camera[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    facing=np.einsum('ij,ij->i',normal,-tri.mean(1))>0
    result=image.copy();edges=set()
    for face in faces[facing]:
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            edges.add((min(int(a),int(b)),max(int(a),int(b))))
    for a,b in edges:
        t=np.linspace(0,1,9);points=camera[a][None,:]*(1-t[:,None])+camera[b][None,:]*t[:,None]
        pix=(points@k.T);pix=pix[:,:2]/pix[:,2,None];ij=np.floor(pix).astype(int)
        valid=(ij[:,0]>=0)&(ij[:,0]<image.shape[1])&(ij[:,1]>=0)&(ij[:,1]<image.shape[0])
        # Pixel-center ray/edge approximation; reject clearly occluded samples.
        visible=np.zeros(len(points),bool)
        visible[valid]=np.abs(points[valid,2]-depth[ij[valid,1],ij[valid,0]])<.002
        for j in range(8):
            if visible[j] and visible[j+1]:cv2.line(result,tuple(np.rint(pix[j]).astype(int)),tuple(np.rint(pix[j+1]).astype(int)),(80,210,255),1,cv2.LINE_AA)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',required=True,type=Path)
    parser.add_argument('--input-dir',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--repo',type=Path,default=Path('vendor/RealDenseFace'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    repo=args.repo.resolve();params=args.params.resolve();input_dir=args.input_dir.resolve()
    sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(repo))
    import common.fix_chumpy
    import cv2
    import numpy as np
    import torch
    from common.config import FLAMEConfig,RealDenseFaceInferenceConfig,FaceBoxConfig
    from models.flame.model import FLAME
    from tracker import RealDenseFaceInferencer
    from memento3d_head.demo_commands import find_photos
    sources=find_photos(input_dir)
    model_path=repo/'weights/flame/flame2023_Open.pkl';assets=repo/'weights/flame/flame_assets.npz'
    weights=repo/'weights/realdenseface/vits.pth';detector=repo/'weights/facebox/face_box.pth'
    bound={p:digest(p) for p in (params,model_path,assets,weights,detector,*sources.values())}
    with np.load(params,allow_pickle=False) as data:
        identity=data['identity'].copy();state=data['x'].copy();n=int(data['num_expressions'])
        width=int(data['image_width']);height=int(data['image_height']);fov=float(data['camera_fov_y'])
    if identity.shape!=(300,) or state.shape!=(n+18,) or not np.isfinite(identity).all() or not np.isfinite(state).all() or not 0<fov<180:
        raise ValueError('Invalid parameters/FOV')
    # Preserve local articulation and expression; only remove global transform.
    state[n:n+3]=0;state[n+15:n+18]=0
    cfg=FLAMEConfig(flame_model_path=str(model_path),flame_assets_path=str(assets),num_expressions=n)
    model=FLAME(cfg,device='cpu')
    with torch.no_grad():vertices=model.forward(torch.from_numpy(identity),torch.from_numpy(state)).numpy()
    faces=model.faces.cpu().numpy()
    with np.load(assets,allow_pickle=False) as asset:
        if not np.array_equal(asset['faces'],faces):raise ValueError('Open model faces differ from the RealDenseFace asset topology')
    output=(args.output or params.parent/('side_check_'+uuid4().hex[:8])).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'status':'FAILED','pipeline_type':'fixed_mesh_dense_prior_camera_check',
        'geometry_modified':False,'expression_modified_across_views':False,'identity_modified':False,
        'model_variant':'FLAME2023_Open','camera_intrinsics_assumed_shared':True,'fov_y_deg':fov,
        'likeness_verified':False,'photos_not_bound_to_original_inference_report':True,
        'input_sha256':{str(p):h for p,h in bound.items()},'views':{},'error':None}
    try:
        inferencer=RealDenseFaceInferencer(RealDenseFaceInferenceConfig(
            model_config_path=str(repo/'configs/model/vits.yaml'),model_weights_path=str(weights),
            flame=cfg,device='cuda',compile_model=False),FaceBoxConfig(model_weights_path=str(detector),device='cuda'))
        focal=height/(2*math.tan(math.radians(fov)/2));k=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]],float)
        for slot in ('front','left','right'):
            print('Checking '+slot,flush=True)
            raw=np.fromfile(str(sources[slot]),dtype=np.uint8);image=cv2.imdecode(raw,cv2.IMREAD_COLOR)
            if image is None or image.shape[:2]!=(height,width):raise ValueError('All photos must have original resolution '+str((width,height))+': '+slot)
            prediction=inferencer.infer_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),face_bbox=None)
            points=np.asarray(prediction.vertex_coord,float);uncertainty=np.asarray(prediction.vertex_coord_log_var,float).reshape(-1)
            if points.shape!=(len(vertices),2):raise ValueError('Dense-prior vertex mapping mismatch')
            valid=np.isfinite(points).all(1)&np.isfinite(uncertainty)&(points[:,0]>=0)&(points[:,0]<width)&(points[:,1]>=0)&(points[:,1]<height)
            if valid.sum()<30:raise ValueError('Too few valid predicted correspondences')
            valid&=uncertainty<=np.quantile(uncertainty[valid],.5)
            rotation,translation,info=fit_camera(vertices[valid],points[valid],k)
            overlay=visible_wire(image,vertices,faces,k,rotation,translation)
            combined=np.concatenate((image,overlay),axis=1)
            cv2.imencode('.jpg',combined)[1].tofile(str(output/(slot+'.fixed_mesh.jpg')))
            info.update(rotation_vector=rotation.ravel().tolist(),translation_native_units=translation.ravel().tolist(),camera_matrix=k.tolist())
            report['views'][slot]=info
        if any(digest(p)!=h for p,h in bound.items()):raise ValueError('Inputs changed during execution')
        report['status']='SIDE_CHECK_EXPORTED'
    except Exception as error:
        report['error']=str(error);raise
    finally:
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Output: '+str(output),flush=True)
    print('SIDE_CHECK_EXPORTED; visual diagnostic only, not a likeness score',flush=True)


if __name__=='__main__':main()
