"""Three source photograph textures on a saved head; preserve original photo detail."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
from uuid import uuid4

SLOTS=('front','left','right')


def visible_faces(cam, faces, k, size, mask=None):
    import numpy as np
    from memento3d_head.adapters.photo_projection import depth_buffer
    width,height=size
    if not np.isfinite(cam).all() or np.any(cam[:,2]<=0):raise ValueError('Invalid camera depth')
    xy,depth=depth_buffer(cam,faces,k,size)
    triangles=cam[faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);direction=-triangles.mean(1)
    facing=np.sum(normals*direction,axis=1)/np.maximum(np.linalg.norm(normals,axis=1)*np.linalg.norm(direction,axis=1),1e-12)
    covered=facing>.35;inverse=np.linalg.inv(k);tolerance=float(np.ptp(cam,axis=0).max())*.002
    bary=np.array([[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]])
    for index in np.flatnonzero(covered):
        tri=triangles[index];points=np.vstack((tri,bary@tri));pixels=points@k.T;pixels=pixels[:,:2]/pixels[:,2,None];ij=np.floor(pixels).astype(int)
        if np.any(ij<1) or np.any(ij[:,0]>=width-1) or np.any(ij[:,1]>=height-1):covered[index]=False;continue
        if mask is not None and not mask[ij[:,1],ij[:,0]].all():covered[index]=False;continue
        rays=np.column_stack((ij[3:,0]+.5,ij[3:,1]+.5,np.ones(4)))@inverse.T
        expected=np.divide(np.dot(normals[index],tri[0])*rays[:,2],rays@normals[index],out=np.full(4,np.inf),where=np.abs(rays@normals[index])>1e-12)
        observed=depth[ij[3:,1],ij[3:,0]]
        covered[index]=bool(np.isfinite(observed).all() and np.all(np.abs(expected-observed)<tolerance))
    coords=np.column_stack((xy[:,0]/width*cam[:,2],(1-xy[:,1]/height)*cam[:,2],cam[:,2]))
    return coords,covered,facing


def choose_sources(covered, facing):
    import numpy as np
    covered=np.asarray(covered,bool);facing=np.asarray(facing,float)
    if covered.shape!=facing.shape or covered.shape[0]!=3:raise ValueError('Expected three source masks')
    scores=np.where(covered,facing,-np.inf)
    chosen=np.argmax(scores,axis=0).astype(np.int8)
    chosen[~covered.any(0)]=-1
    # Stable frontal details: never mix eyes/mouth from multiple photos.
    chosen[covered[0]]=0
    return chosen


def viewer_template(template):
    start=template.index('const texture=gl.createTexture();')
    end=template.index("const models=['baseline','candidate']",start)
    template=template[:start]+'''let photoLoaded=false;
const layers=data.layers.map(layer=>{
 const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);
 gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,1,1,0,gl.RGB,gl.UNSIGNED_BYTE,new Uint8Array([160,170,180]));
 for(const [p,v] of [[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE],[gl.TEXTURE_MIN_FILTER,gl.LINEAR],[gl.TEXTURE_MAG_FILTER,gl.LINEAR]])gl.texParameteri(gl.TEXTURE_2D,p,v);
 const item={texture,loaded:false,photo:buffer(layer.coordinates.flat(),gl.ARRAY_BUFFER,Float32Array),indices:buffer(layer.faces.flat(),gl.ELEMENT_ARRAY_BUFFER,Uint16Array),count:layer.faces.length*3};
 const image=new Image();image.onload=()=>{try{gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,image);item.loaded=true;photoLoaded=layers.every(x=>x.loaded);draw();}catch(e){fail(e.message);}};image.onerror=()=>fail('Cannot load source texture');item.image=image;return item;
});
''' +template[end:]
    old='gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,photoIndex);gl.drawElements(gl.TRIANGLES,data.photo_faces.length*3,gl.UNSIGNED_SHORT,0);'
    new="for(const layer of layers){if(!layer.loaded)continue;gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,layer.texture);attribute('aPhoto',layer.photo,3);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,layer.indices);gl.drawElements(gl.TRIANGLES,layer.count,gl.UNSIGNED_SHORT,0);}"
    if old not in template:raise ValueError('Unsupported viewer draw template')
    template=template.replace(old,new).replace('photoImage.src=data.photo_image;draw();',"layers.forEach((layer,i)=>{layer.image.src=data.layers[i].image;});draw();")
    template=template.replace('millimeters','native_model_units').replace(' mm',' model units')
    template=template.replace('Màu ảnh chính diện','Texture ba ảnh').replace('Thử màu ảnh trên mặt 3D','Texture ba ảnh · Giữ chi tiết ảnh gốc')
    template=template.replace('Hai khung luôn dùng chung góc nhìn, tỷ lệ và ánh sáng.','Cùng hình học và góc nhìn; màu ảnh không chiếu sáng thêm.')
    template=template.replace('Màu được chiếu từ ảnh chính diện, gồm cả ánh sáng và bóng trong ảnh.','Ưu tiên ảnh chính diện; ảnh bên bổ sung vùng nhìn thấy. Giữ cả ánh sáng và bóng trong ảnh; còn có thể có đường nối màu và tóc ảnh.')
    return template


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path('vendor/RealDenseFace'))
    p.add_argument('--masks-dir',type=Path,help='Optional front.png/left.png/right.png: white keeps texture, black excludes hair/background')
    args=p.parse_args();root=Path(__file__).resolve().parents[1];repo=args.repo.resolve();run=args.run_dir.resolve()
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
    from memento3d_head.demo_commands import find_photos
    from memento3d_head.adapters.smooth_preview import vertex_normals
    photos=find_photos(args.input_dir.resolve());source_path=run/'report.json';source=json.loads(source_path.read_text(encoding='utf-8'))
    if source.get('status')!='JOINT_CANDIDATE_EXPORTED':raise ValueError('Expected successful joint fit')
    model=repo/'weights/flame/flame2023_Open.pkl';assets=repo/'weights/flame/flame_assets.npz';params=run/'joint.flame.npz'
    for path in [*photos.values(),model,assets]:
        matches=[v for k,v in source['input_sha256'].items() if k.replace('\\','/').split('/')[-1].lower()==path.name.lower()]
        if len(matches)!=1 or digest(path)!=matches[0]:raise ValueError('Source mismatch: '+str(path))
    inputs=[params,source_path,model,assets,*photos.values()]
    if args.masks_dir:inputs += [args.masks_dir/(s+'.png') for s in SLOTS]
    bound={str(path.resolve()):digest(path) for path in inputs}
    with np.load(params,allow_pickle=False) as data:
        beta=data['identity'].copy();states=data['x'].copy();slots=data['slots'].tolist();width=int(data['image_width']);height=int(data['image_height']);fov=float(data['camera_fov_y']);rot=data['camera_rot'].astype(np.float32);pos=data['camera_pos'].astype(np.float32)
    if slots!=list(SLOTS) or beta.shape!=(300,) or states.shape!=(3,118) or not 0<fov<180 or not all(np.isfinite(a).all() for a in [beta,states,rot,pos]):raise ValueError('Invalid joint parameters')
    output=run/('three_textures_'+uuid4().hex[:8]);output.mkdir(exist_ok=False)
    report=dict(status='FAILED',pipeline_type='three_photo_projective_textures',input_sha256=bound,
        geometry_refitted=False,front_priority=True,color_blending=False,color_calibrated=False,
        masks_supplied=bool(args.masks_dir),hair_reconstructed=False,lighting_removed=False,
        likeness_verified=False,full_head_uv_atlas=False,views={},error=None)
    try:
        cfg=FLAMEConfig(flame_model_path=str(model),flame_assets_path=str(assets),num_expressions=100)
        optimizer=GNFlameOptimizer(GNFlameOptimizerConfig(flame=cfg,device='cuda'))
        identity=torch.from_numpy(beta).to('cuda');state=torch.from_numpy(states).to('cuda')
        with torch.no_grad():
            posed,_=optimizer.decode(FittingOutput(identity=identity,x=state))
            local=state[0].clone();local[100:103]=0;local[115:118]=0
            vertices,_=optimizer.decode(FittingOutput(identity=identity,x=local))
        vertices=vertices.cpu().numpy();posed=posed.cpu().numpy();faces=optimizer.faces.cpu().numpy()
        camera=PerspectiveCamera(fov_y=np.radians(fov),rot=rot,pos=pos,width=width,height=height)
        k=camera.intrisics.cpu().numpy().astype(float);extrinsic=camera.get_w2v.cpu().numpy()
        coordinates=[];visibility=[];angles=[];images=[]
        for i,slot in enumerate(SLOTS):
            print('Texture visibility: '+slot,flush=True)
            with Image.open(photos[slot]) as img:
                if img.size!=(width,height):raise ValueError('Photo resolution mismatch: '+slot)
                image=img.convert('RGB')
            image.save(output/(slot+'.photo.png'));stream=io.BytesIO();image.save(stream,format='PNG')
            images.append('data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii'))
            mask=None
            if args.masks_dir:
                with Image.open(args.masks_dir/(slot+'.png')) as img:
                    if img.size!=(width,height):raise ValueError('Mask resolution mismatch')
                    mask=np.asarray(img.convert('L'))>=128
            cam=posed[i]@extrinsic[:3,:3].T+extrinsic[:3,3]
            coords,covered,facing=visible_faces(cam,faces,k,(width,height),mask)
            coordinates.append(coords);visibility.append(covered);angles.append(facing)
        chosen=choose_sources(visibility,angles)
        layers=[]
        for i,slot in enumerate(SLOTS):
            selected=chosen==i;layers.append(dict(coordinates=coordinates[i].tolist(),faces=faces[selected].tolist(),image=images[i]))
            report['views'][slot]=dict(visible_triangles=int(np.sum(visibility[i])),assigned_triangles=int(selected.sum()))
        if np.sum(chosen>=0)<100:raise ValueError('Too few textured triangles')
        center=(vertices.min(0)+vertices.max(0))*.5;radius=float(np.linalg.norm(vertices-center,axis=1).max());shape=dict(vertices=vertices.tolist(),normals=vertex_normals(vertices,faces).tolist())
        payload=dict(schema_version=1,units='native_model_units',center_mm=center.tolist(),radius_mm=radius,
            faces=faces.tolist(),baseline=shape,candidate=shape,displacement_mm=[0.]*len(vertices),max_displacement_mm=0.,
            photo_coordinates=coordinates[0].tolist(),photo_faces=faces[chosen>=0].tolist(),photo_alpha=[1.]*len(vertices),layers=layers)
        template=viewer_template((root/'src/memento3d_head/adapters/photo_preview.html').read_text(encoding='utf-8'))
        encoded=json.dumps(payload,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        (output/'head.three_textures.html').write_text(template.replace('__HEAD_DATA__',encoded),encoding='utf-8')
        # Standard OBJ with three source-image materials; HTML retains exact projective sampling.
        (output/'head.three_textures.mtl').write_text('newmtl unknown\nKd .65 .68 .72\n\n'+''.join('newmtl '+slot+'\nKd 1 1 1\nmap_Kd '+slot+'.photo.png\n\n' for slot in SLOTS),encoding='ascii')
        with (output/'head.three_textures.obj').open('w',encoding='ascii') as f:
            f.write('mtllib head.three_textures.mtl\n')
            for v in vertices:f.write('v %.9g %.9g %.9g\n'%tuple(v))
            for coords in coordinates:
                for q in coords:f.write('vt %.9g %.9g\n'%(q[0]/q[2],q[1]/q[2]))
            previous=None
            for index,face in enumerate(faces):
                selected=int(chosen[index]);material=SLOTS[selected] if selected>=0 else 'unknown'
                if material!=previous:f.write('usemtl '+material+'\n');previous=material
                offset=max(selected,0)*len(vertices)
                f.write('f '+' '.join('%d/%d'%(v+1,v+1+offset) for v in face)+'\n')
        np.savez(output/'texture.sources.npz',face_source=chosen,source_order=np.array(SLOTS),coordinates=np.array(coordinates))
        if any(digest(Path(path))!=value for path,value in bound.items()):raise ValueError('Input changed during execution')
        report.update(status='THREE_TEXTURES_PREVIEW_EXPORTED',covered_triangles=int(np.sum(chosen>=0)),total_triangles=len(faces),
            note='Hard source selection preserves detail; color seams and hair/background projection can remain. Gray back of head expected. No inferred hidden texture; no mouth correction.')
    except Exception as error:report['error']=str(error);raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Open: '+str(output/'head.three_textures.html'),flush=True)


if __name__=='__main__':main()
