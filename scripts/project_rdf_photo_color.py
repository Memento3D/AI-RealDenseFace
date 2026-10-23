"""Experimental three-photo vertex color projection from saved RDF states; no new fitting."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path('vendor/RealDenseFace'))
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1];repo=args.repo.resolve();run=args.run_dir.resolve()
    sys.path[:0]=[str(root/'src'),str(repo)]
    from check_rdf_side_views import digest
    import common.fix_chumpy
    import numpy as np
    import cv2
    import torch
    import trimesh
    from common.config import FLAMEConfig,GNFlameOptimizerConfig
    from common.types import FittingOutput
    from tracker import GNFlameOptimizer
    from camera import PerspectiveCamera
    from memento3d_head.demo_commands import find_photos
    from memento3d_head.adapters.photo_projection import depth_buffer
    from memento3d_head.adapters.smooth_preview import vertex_normals
    from memento3d_head.adapters.head_compare import write_comparison
    photos=find_photos(args.input_dir.resolve())
    source_report=run/'report.json';source=json.loads(source_report.read_text(encoding='utf-8'))
    if source.get('status')!='JOINT_CANDIDATE_EXPORTED':raise ValueError('Expected successful joint fit')
    cfg=FLAMEConfig(flame_model_path=str(repo/'weights/flame/flame2023_Open.pkl'),flame_assets_path=str(repo/'weights/flame/flame_assets.npz'),num_expressions=100)
    for path in [*photos.values(),Path(cfg.flame_model_path),Path(cfg.flame_assets_path)]:
        hashes=[v for k,v in source['input_sha256'].items() if k.replace('\\','/').split('/')[-1].lower()==path.name.lower()]
        if len(hashes)!=1 or digest(path)!=hashes[0]:raise ValueError('Source mismatch: '+str(path))
    params=run/'joint.flame.npz'
    with np.load(params,allow_pickle=False) as data:
        beta=data['identity'].copy();xs=data['x'].copy();slots=data['slots'].tolist()
        fov=float(data['camera_fov_y']);width=int(data['image_width']);height=int(data['image_height'])
        rot=data['camera_rot'].astype(np.float32);pos=data['camera_pos'].astype(np.float32)
    if slots!=['front','left','right'] or beta.shape!=(300,) or xs.shape!=(3,118) or not all(np.isfinite(a).all() for a in [beta,xs,rot,pos]) or not 0<fov<180:raise ValueError('Invalid saved parameters')
    inputs=[params,source_report,Path(cfg.flame_model_path),Path(cfg.flame_assets_path),*photos.values()]
    bound={str(path):digest(path) for path in inputs}
    out=run/('photo_color_'+uuid4().hex[:8]);out.mkdir(exist_ok=False)
    report=dict(status='FAILED',research_only=True,geometry_refitted=False,color_type='vertex colors sampled from photographs, not UV texture or albedo',
        display_expression='saved frontal expression; global rotation and translation removed',
        lighting_removed=False,unobserved_color='gray',hair_reconstructed=False,likeness_verified=False,
        input_sha256=bound,views={},error=None)
    try:
        optimizer=GNFlameOptimizer(GNFlameOptimizerConfig(flame=cfg,device='cuda'))
        identity=torch.from_numpy(beta).to('cuda');states=torch.from_numpy(xs).to('cuda')
        with torch.no_grad():
            posed,_=optimizer.decode(FittingOutput(identity=identity,x=states))
            display=states[0].clone();display[100:103]=0;display[115:118]=0
            vertices,_=optimizer.decode(FittingOutput(identity=identity,x=display))
        vertices=vertices.cpu().numpy();meshes=posed.cpu().numpy();faces=optimizer.faces.cpu().numpy()
        camera=PerspectiveCamera(fov_y=np.radians(fov),rot=rot,pos=pos,width=width,height=height)
        k=camera.intrisics.cpu().numpy().astype(float);extrinsic=camera.get_w2v.cpu().numpy()
        total=np.zeros((len(vertices),3));weight_sum=np.zeros(len(vertices));observations=np.zeros(len(vertices),int)
        for i,slot in enumerate(slots):
            print('Projecting color: '+slot,flush=True)
            bgr=cv2.imdecode(np.fromfile(photos[slot],dtype=np.uint8),cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2]!=(height,width):raise ValueError('Photo size mismatch: '+slot)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(float)/255
            mesh=meshes[i];cam=mesh@extrinsic[:3,:3].T+extrinsic[:3,3]
            _,depth=depth_buffer(cam,faces,k,(width,height))
            homogeneous=cam@k.T;xy=homogeneous[:,:2]/homogeneous[:,2,None]
            normal=vertex_normals(cam,faces);direction=-cam/np.maximum(np.linalg.norm(cam,axis=1,keepdims=True),1e-9)
            facing=np.maximum(np.sum(normal*direction,axis=1),0)
            good=np.isfinite(xy).all(1)&(cam[:,2]>0)&(xy[:,0]>=1)&(xy[:,0]<width-2)&(xy[:,1]>=1)&(xy[:,1]<height-2)&(facing>.15)
            ids=np.flatnonzero(good);ij=np.floor(xy[ids]).astype(int)
            tolerance=max(float(np.ptp(mesh,axis=0).max())*.003,1e-8)
            good[ids]&=np.abs(cam[ids,2]-depth[ij[:,1],ij[:,0]])<tolerance
            ids=np.flatnonzero(good)
            sample=cv2.remap(rgb.astype(np.float32),xy[:,0].astype(np.float32).reshape(-1,1),xy[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).reshape(-1,3)
            # Blend in linear RGB; frontal sample has modest priority.
            linear=np.where(sample<=.04045,sample/12.92,((sample+.055)/1.055)**2.4)
            weights=np.where(good,facing**4*(2 if i==0 else 1),0)
            total+=linear*weights[:,None];weight_sum+=weights;observations+=good.astype(int)
            report['views'][slot]=dict(sampled_vertices=int(len(ids)))
        covered=weight_sum>1e-8
        if covered.sum()<100:raise ValueError('Too few color samples; inspect camera and normal conventions')
        linear=total/np.maximum(weight_sum[:,None],1e-8)
        srgb=np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)
        colors=np.clip(np.rint(srgb*255),0,255).astype(np.uint8);colors[~covered]=[155,165,180]
        rgba=np.column_stack((colors,np.full(len(colors),255,np.uint8)))
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=rgba,process=False)
        mesh.export(out/'head.photo_color.ply');mesh.export(out/'head.photo_color.glb')
        np.savez(out/'color.coverage.npz',covered=covered,observations=observations,weight_sum=weight_sum,colors_rgb=colors)
        viewer=out/'head.photo_color.html'
        write_comparison(viewer,vertices,vertices,faces,{'units':'native_model_units'})
        html=viewer.read_text(encoding='utf-8')
        replacements={
            'attribute vec3 aPosition;':'attribute vec3 aColor;varying vec3 vColor;attribute vec3 aPosition;',
            'vNormal=uRotation*aNormal;':'vColor=aColor;vNormal=uRotation*aNormal;',
            'precision mediump float;':'precision mediump float;varying vec3 vColor;',
            'vec3 color=vec3(.73,.77,.82);':'vec3 color=vColor;',
            "position:buffer(data[name].vertices.flat()":"color:buffer((name==='baseline'?data[name].vertices.map(()=>[.73,.77,.82]):PHOTO_COLORS).flat(),gl.ARRAY_BUFFER,Float32Array),position:buffer(data[name].vertices.flat()",
            "attribute('aPosition',m.position,3);":"attribute('aColor',m.color,3);attribute('aPosition',m.position,3);",
            'Mesh gốc':'Geometry only','Mesh thử nghiệm':'Projected photo color',
            'Mesh chưa có texture; bản thử nghiệm chưa được tự động chấp nhận.':'Photo colors include lighting; unseen areas are gray. This is a vertex-color preview.',
            ' mm':' model units','millimeters':'native_model_units'}
        for old,new in replacements.items():
            if old not in html:raise ValueError('Viewer template changed: '+old[:60])
            html=html.replace(old,new)
        html=html.replace("'use strict';","'use strict';\nconst PHOTO_COLORS="+json.dumps((colors/255).tolist())+';')
        viewer.write_text(html,encoding='utf-8')
        if any(digest(Path(path))!=value for path,value in bound.items()):raise ValueError('Input changed during execution')
        report.update(status='PHOTO_COLOR_PREVIEW_EXPORTED',covered_vertices=int(covered.sum()),total_vertices=len(vertices),
            note='No skin/hair segmentation or seam optimization; hair/background contamination and color seams may remain')
    except Exception as error:
        report['error']=str(error);raise
    finally:
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Output: '+str(out),flush=True)


if __name__=='__main__':main()
