"""Front photograph texture preview with projective UV coordinates, no vertex-color averaging."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--front',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path('vendor/RealDenseFace'))
    args=p.parse_args();root=Path(__file__).resolve().parents[1];repo=args.repo.resolve();run=args.run_dir.resolve();front=args.front.resolve()
    sys.path[:0]=[str(root/'src'),str(repo)]
    from check_rdf_side_views import digest
    import common.fix_chumpy
    import numpy as np
    import torch
    from PIL import Image
    from camera import PerspectiveCamera
    from common.config import FLAMEConfig,GNFlameOptimizerConfig
    from common.types import FittingOutput
    from tracker import GNFlameOptimizer
    from memento3d_head.adapters.photo_projection import depth_buffer,write_photo_viewer,export_photo_obj
    source_path=run/'report.json';source=json.loads(source_path.read_text(encoding='utf-8'))
    if source.get('status')!='JOINT_CANDIDATE_EXPORTED':raise ValueError('Expected successful joint fit')
    model=repo/'weights/flame/flame2023_Open.pkl';assets=repo/'weights/flame/flame_assets.npz'
    for path in [front,model,assets]:
        matches=[v for k,v in source['input_sha256'].items() if k.replace('\\','/').split('/')[-1].lower()==path.name.lower()]
        if len(matches)!=1 or digest(path)!=matches[0]:raise ValueError('Source mismatch: '+str(path))
    params=run/'joint.flame.npz';bound={str(path):digest(path) for path in [params,source_path,front,model,assets]}
    with np.load(params,allow_pickle=False) as data:
        beta=data['identity'].copy();states=data['x'].copy();slots=data['slots'].tolist()
        width=int(data['image_width']);height=int(data['image_height']);fov=float(data['camera_fov_y'])
        rot=data['camera_rot'].astype(np.float32);pos=data['camera_pos'].astype(np.float32)
    if slots!=['front','left','right'] or beta.shape!=(300,) or states.shape!=(3,118) or not 0<fov<180 or not all(np.isfinite(a).all() for a in [beta,states,rot,pos]):raise ValueError('Invalid joint parameters')
    with Image.open(front) as image:
        if image.size!=(width,height):raise ValueError('Front resolution mismatch')
        photo=image.convert('RGB')
    output=run/('front_texture_'+uuid4().hex[:8]);output.mkdir(exist_ok=False)
    report=dict(status='FAILED',input_sha256=bound,geometry_refitted=False,
        texture_type='front-only projective photograph; planar UV in OBJ; not full-head UV atlas',
        source_resolution=[width,height],lighting_removed=False,skin_hair_segmentation=False,
        unobserved_regions='gray',display_expression='saved front expression, global pose removed',
        likeness_verified=False,error=None)
    try:
        cfg=FLAMEConfig(flame_model_path=str(model),flame_assets_path=str(assets),num_expressions=100)
        optimizer=GNFlameOptimizer(GNFlameOptimizerConfig(flame=cfg,device='cuda'))
        identity=torch.from_numpy(beta).to('cuda');state=torch.from_numpy(states[0]).to('cuda')
        with torch.no_grad():
            posed,_=optimizer.decode(FittingOutput(identity=identity,x=state))
            local=state.clone();local[100:103]=0;local[115:118]=0
            vertices,_=optimizer.decode(FittingOutput(identity=identity,x=local))
        vertices=vertices.cpu().numpy();posed=posed.cpu().numpy();faces=optimizer.faces.cpu().numpy()
        camera=PerspectiveCamera(fov_y=np.radians(fov),rot=rot,pos=pos,width=width,height=height)
        k=camera.intrisics.cpu().numpy().astype(float);extrinsic=camera.get_w2v.cpu().numpy()
        cam=posed@extrinsic[:3,:3].T+extrinsic[:3,3]
        if not np.isfinite(cam).all() or np.any(cam[:,2]<=0):raise ValueError('Invalid camera depths')
        xy,depth=depth_buffer(cam,faces,k,(width,height))
        triangles=cam[faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);directions=-triangles.mean(1)
        facing=np.sum(normals*directions,axis=1)/np.maximum(np.linalg.norm(normals,axis=1)*np.linalg.norm(directions,axis=1),1e-12)
        covered=facing>.35;inverse=np.linalg.inv(k);tolerance=float(np.ptp(cam,axis=0).max())*.002
        bary=np.array([[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]])
        for index in np.flatnonzero(covered):
            tri=triangles[index];points=np.vstack((tri,bary@tri));pixels=points@k.T;pixels=pixels[:,:2]/pixels[:,2,None];ij=np.floor(pixels).astype(int)
            if np.any(ij<1) or np.any(ij[:,0]>=width-1) or np.any(ij[:,1]>=height-1):covered[index]=False;continue
            rays=np.column_stack((ij[3:,0]+.5,ij[3:,1]+.5,np.ones(4)))@inverse.T
            denominator=rays@normals[index]
            expected=np.divide(np.dot(normals[index],tri[0])*rays[:,2],denominator,out=np.full(4,np.inf),where=np.abs(denominator)>1e-12)
            observed=depth[ij[3:,1],ij[3:,0]]
            covered[index]=bool(np.isfinite(observed).all() and np.all(np.abs(expected-observed)<tolerance))
        if covered.sum()<100:raise ValueError('Too few visible textured triangles')
        coordinates=np.column_stack((xy[:,0]/width*cam[:,2],(1-xy[:,1]/height)*cam[:,2],cam[:,2]))
        photo_path=output/'front.photo.png';photo.save(photo_path)
        viewer=output/'head.front_texture.html'
        write_photo_viewer(viewer,vertices,faces,coordinates,covered,photo_path,report)
        html=viewer.read_text(encoding='utf-8').replace('millimeters','native_model_units').replace(' mm',' model units')
        viewer.write_text(html,encoding='utf-8')
        export_photo_obj(output,vertices,faces,coordinates,covered)
        np.savez(output/'texture.mapping.npz',coordinates=coordinates,covered_faces=covered)
        if any(digest(Path(path))!=value for path,value in bound.items()):raise ValueError('Input changed during execution')
        report.update(status='FRONT_TEXTURE_PREVIEW_EXPORTED',textured_triangles=int(covered.sum()),total_triangles=len(faces),
            note='Original photo pixels retained. Hair/accessories and shadows can be projected. OBJ uses affine UV interpolation; HTML uses perspective-correct projective lookup. Mouth geometry unchanged.')
    except Exception as error:report['error']=str(error);raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Open: '+str(output/'head.front_texture.html'),flush=True)


if __name__=='__main__':main()
