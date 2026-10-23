"""Inspect saved joint FLAME states without changing identity, expression, pose or camera."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--repo', type=Path, default=Path('vendor/RealDenseFace'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    repo, run = args.repo.resolve(), args.run_dir.resolve()
    sys.path[:0] = [str(root / 'src'), str(repo)]
    from check_rdf_side_views import digest, visible_wire
    import common.fix_chumpy  # noqa: F401
    import cv2
    import numpy as np
    import torch
    from camera import PerspectiveCamera
    from common.config import FLAMEConfig, GNFlameOptimizerConfig, RealDenseFaceInferenceConfig, FaceBoxConfig
    from common.types import FittingOutput
    from tracker import GNFlameOptimizer, RealDenseFaceInferencer
    from memento3d_head.demo_commands import find_photos
    from memento3d_head.adapters.photo_projection import depth_buffer
    photos = find_photos(args.input_dir.resolve())
    source_report = run / 'report.json'
    source = json.loads(source_report.read_text(encoding='utf-8'))
    if source.get('status') != 'JOINT_CANDIDATE_EXPORTED':
        raise ValueError('Expected a successful joint-fit run')
    cfg = FLAMEConfig(flame_model_path=str(repo/'weights/flame/flame2023_Open.pkl'),
        flame_assets_path=str(repo/'weights/flame/flame_assets.npz'), num_expressions=100)
    weights = repo/'weights/realdenseface/vits.pth'
    detector = repo/'weights/facebox/face_box.pth'
    # Match recorded hashes by filename, independent of Windows drive letter.
    def verify_recorded(path):
        candidates = [value for key, value in source['input_sha256'].items()
                      if key.replace('\\', '/').split('/')[-1].lower() == path.name.lower()]
        if len(candidates) != 1 or digest(path) != candidates[0]:
            raise ValueError('Source binding mismatch: ' + str(path))
    for path in [*photos.values(), Path(cfg.flame_model_path), Path(cfg.flame_assets_path), weights, detector]:
        verify_recorded(path)
    params = run/'joint.flame.npz'
    with np.load(params, allow_pickle=False) as data:
        beta, xs = data['identity'].copy(), data['x'].copy()
        slots = data['slots'].tolist()
        fov, width, height = float(data['camera_fov_y']), int(data['image_width']), int(data['image_height'])
        rot, pos = data['camera_rot'].copy(), data['camera_pos'].copy()
    if slots != ['front', 'left', 'right'] or beta.shape != (300,) or xs.shape != (3,118):
        raise ValueError('Unexpected saved state layout')
    if not all(np.isfinite(a).all() for a in [beta,xs,rot,pos]) or not 0 < fov < 180:
        raise ValueError('Nonfinite parameters or invalid FOV')
    bound = {str(path):digest(path) for path in [params,source_report,*photos.values(),
        Path(cfg.flame_model_path),Path(cfg.flame_assets_path),weights,detector]}
    output = run/('fitted_state_check_'+uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    report = dict(status='FAILED', pipeline_type='saved_fitted_state_diagnostic',
        input_sha256=bound, geometry_modified=False, camera_refitted=False,
        expression_modified=False, pose_modified=False, likeness_verified=False,
        metric_reference='network predictions, not ground truth',
        visible_subset='depth-tested vertices; finite in-frame predictions; lowest 50 percent log variance',
        anatomical_region_metrics_available=False,
        anatomical_region_note='No verified anatomical vertex masks supplied; no invented region labels',
        views={}, error=None)
    try:
        optimizer = GNFlameOptimizer(GNFlameOptimizerConfig(flame=cfg,device='cuda'))
        # Decode with exactly the same CUDA forward model used during fitting.
        fitted = FittingOutput(identity=torch.from_numpy(beta).to('cuda'),
            x=torch.from_numpy(xs).to('cuda'),camera_fov_y=fov)
        with torch.no_grad():
            vertices, _ = optimizer.decode(fitted)
        meshes = vertices.detach().cpu().numpy()
        faces = optimizer.faces.cpu().numpy()
        del optimizer, vertices
        torch.cuda.empty_cache()
        camera = PerspectiveCamera(fov_y=np.radians(fov),rot=rot.astype(np.float32),
            pos=pos.astype(np.float32),width=width,height=height)
        k = camera.intrisics.cpu().numpy().astype(float)
        extrinsic = camera.get_w2v.cpu().numpy()
        rvec = cv2.Rodrigues(extrinsic[:3,:3].astype(float))[0]
        translation = extrinsic[:3,3].astype(float)
        infer = RealDenseFaceInferencer(RealDenseFaceInferenceConfig(
            model_config_path=str(repo/'configs/model/vits.yaml'),model_weights_path=str(weights),
            flame=cfg,device='cuda',compile_model=False),
            FaceBoxConfig(model_weights_path=str(detector),device='cuda'))
        for index, slot in enumerate(slots):
            print('Inspecting saved state: '+slot,flush=True)
            image = cv2.imdecode(np.fromfile(photos[slot],dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height,width):
                raise ValueError('Photo dimensions mismatch: '+slot)
            prediction = infer.infer_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),face_bbox=None)
            points = np.asarray(prediction.vertex_coord,float)
            logvar = np.asarray(prediction.vertex_coord_log_var,float).ravel()
            mesh = meshes[index]
            camera_vertices = mesh@extrinsic[:3,:3].T+translation
            xy, depth = depth_buffer(camera_vertices,faces,k,(width,height))
            projected = (camera_vertices@k.T)
            projected = projected[:,:2]/projected[:,2,None]
            valid = np.isfinite(points).all(1)&np.isfinite(logvar)&np.isfinite(projected).all(1)
            valid &= (points[:,0]>=0)&(points[:,0]<width)&(points[:,1]>=0)&(points[:,1]<height)
            if valid.sum()<30: raise ValueError('Too few predictions: '+slot)
            valid &= logvar<=np.quantile(logvar[valid],.5)
            inside = (projected[:,0]>=0)&(projected[:,0]<width)&(projected[:,1]>=0)&(projected[:,1]<height)&(camera_vertices[:,2]>0)
            ids = np.flatnonzero(inside)
            ij = np.floor(projected[ids]).astype(int)
            visible = np.zeros(len(mesh),bool)
            tolerance = max(float(np.ptp(mesh,axis=0).max())*.005,1e-8)
            visible[ids] = np.abs(camera_vertices[ids,2]-depth[ij[:,1],ij[:,0]])<tolerance
            subset = valid&visible
            if subset.sum()<30: raise ValueError('Too few visible diagnostic vertices: '+slot)
            residual = np.linalg.norm(projected-points,axis=1)
            overlay = visible_wire(image,mesh,faces,k,rvec,translation)
            heat = image.copy()
            for j in np.flatnonzero(subset):
                value = min(float(residual[j])/10.,1.)
                color = (0,int(255*(1-value)),int(255*value))
                cv2.circle(heat,tuple(np.rint(projected[j]).astype(int)),2,color,-1)
            for panel,label in [(overlay,'SAVED POSE + EXPRESSION (NO REFIT)'),(heat,'RESIDUAL: GREEN 0 / RED >=10 PX')]:
                cv2.putText(panel,label,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            cv2.imencode('.jpg',np.concatenate((overlay,heat),axis=1))[1].tofile(str(output/(slot+'.fitted.jpg')))
            np.savez(output/(slot+'.residuals.npz'),projected_px=projected,predicted_px=points,
                residual_px=residual,visible=visible,confidence_selected=valid,evaluated=subset)
            report['views'][slot] = dict(evaluated_vertices=int(subset.sum()),
                median_visible_residual_px=float(np.median(residual[subset])),
                p90_visible_residual_px=float(np.quantile(residual[subset],.9)))
        if any(digest(Path(path))!=value for path,value in bound.items()):
            raise ValueError('Input changed during diagnostic')
        report['status']='FITTED_STATE_DIAGNOSTIC_EXPORTED'
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Output: '+str(output),flush=True)


if __name__=='__main__':
    main()
