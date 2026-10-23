"""CPU-only texture review workspace built from an existing three-texture export."""
import base64
import hashlib
import io
import json
import re
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image
from memento3d_head.adapters.photo_projection import depth_buffer
from memento3d_head.adapters.smooth_preview import vertex_normals

SLOTS=('front','left','right')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def srgb_linear(value):
    return np.where(value<=.04045,value/12.92,((value+.055)/1.055)**2.4)


def sample(image, coordinates):
    h,w=image.shape[:2];q=np.asarray(coordinates,float)
    x=np.clip(q[:,0]/q[:,2]*w,0,w-1);y=np.clip((1-q[:,1]/q[:,2])*h,0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    dx=(x-x0)[:,None];dy=(y-y0)[:,None]
    return (image[y0,x0]*(1-dx)+image[y0,x1]*dx)*(1-dy)+(image[y1,x0]*(1-dx)+image[y1,x1]*dx)*dy


def source_visibility(coords,faces,k,size,mask):
    w,h=size;q=np.asarray(coords,float);xy=np.column_stack((q[:,0]/q[:,2]*w,(1-q[:,1]/q[:,2])*h))
    cam=np.column_stack((xy,np.ones(len(q))))@np.linalg.inv(k).T*q[:,2,None]
    if not np.isfinite(cam).all() or np.any(cam[:,2]<=0):raise ValueError('Invalid projective coordinates')
    _,depth=depth_buffer(cam,faces,k,size)
    tri=cam[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);direction=-tri.mean(1)
    angle=np.sum(normal*direction,axis=1)/np.maximum(np.linalg.norm(normal,axis=1)*np.linalg.norm(direction,axis=1),1e-12)
    valid=angle>.35;inverse=np.linalg.inv(k);tolerance=float(np.ptp(cam,axis=0).max())*.002
    bary=np.array([[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]])
    for i in np.flatnonzero(valid):
        points=np.vstack((tri[i],bary@tri[i]));pix=points@k.T;pix=pix[:,:2]/pix[:,2,None];ij=np.floor(pix).astype(int)
        if np.any(ij<1) or np.any(ij[:,0]>=w-1) or np.any(ij[:,1]>=h-1) or not mask[ij[:,1],ij[:,0]].all():valid[i]=False;continue
        rays=np.column_stack((ij[3:,0]+.5,ij[3:,1]+.5,np.ones(4)))@inverse.T
        den=rays@normal[i];expected=np.divide(np.dot(normal[i],tri[i,0])*rays[:,2],den,out=np.full(4,np.inf),where=np.abs(den)>1e-12)
        observed=depth[ij[3:,1],ij[3:,0]]
        valid[i]=bool(np.isfinite(observed).all() and np.all(np.abs(expected-observed)<tolerance))
    smooth=vertex_normals(cam,faces);toward=-cam/np.maximum(np.linalg.norm(cam,axis=1,keepdims=True),1e-12)
    return valid,np.maximum(np.sum(smooth*toward,axis=1),0)


def estimate_gains(samples,eligible):
    gains=np.ones((3,3));counts=[]
    linear=srgb_linear(np.asarray(samples))
    for i in (1,2):
        good=eligible[0]&eligible[i]&np.all(linear[0]>.025,axis=1)&np.all(linear[i]>.025,axis=1)&np.all(linear[0]<.9,axis=1)&np.all(linear[i]<.9,axis=1)
        counts.append(int(good.sum()))
        if good.sum()>=40:
            gains[i]=np.clip(np.median(linear[0,good]/linear[i,good],axis=0),.85,1.18)
    return gains,counts


def read_source(texture_run):
    viewer=texture_run/'head.three_textures.html'
    if viewer.stat().st_size>60_000_000:raise ValueError('Viewer exceeds supported size')
    html=viewer.read_text(encoding='utf-8')
    found=re.search(r'<script id="model-data" type="application/json">(.*?)</script>',html,re.S)
    if not found:raise ValueError('Missing embedded model data')
    data=json.loads(found.group(1));vertices=np.asarray(data['baseline']['vertices'],float);faces=np.asarray(data['faces'],int)
    if vertices.shape!=(5023,3) or faces.shape!=(9976,3) or not np.isfinite(vertices).all() or faces.min()<0 or faces.max()>=len(vertices):raise ValueError('Expected finite RDF mesh topology')
    if len(data.get('layers',[]))!=3:raise ValueError('Expected three source layers')
    return viewer,data,vertices,faces


def build_workspace(head_run,texture_run=None,masks_json=None,skin_color=None):
    head_run=Path(head_run).resolve()
    if texture_run is None:
        candidates=sorted(head_run.glob('three_textures_*/report.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        texture_run=next((p.parent for p in candidates if json.loads(p.read_text()).get('status')=='THREE_TEXTURES_PREVIEW_EXPORTED'),None)
        if texture_run is None:raise ValueError('Run preview_rdf_three_textures.py once before optimize-head')
    texture_run=Path(texture_run).resolve();viewer,data,vertices,faces=read_source(texture_run)
    parent_report=head_run/'report.json';texture_report=texture_run/'report.json'
    report0=json.loads(texture_report.read_text(encoding='utf-8'))
    if report0.get('status')!='THREE_TEXTURES_PREVIEW_EXPORTED':raise ValueError('Source texture run is incomplete')
    recorded=report0.get('input_sha256',{})
    for path in [head_run/'joint.flame.npz',parent_report]:
        matches=[v for k,v in recorded.items() if k.replace('\\','/').split('/')[-1]==path.name]
        if len(matches)!=1 or digest(path)!=matches[0]:raise ValueError('Texture run does not belong to this head run: '+path.name)
    source_hash=digest(viewer);images=[];masks=[];coords=[]
    with np.load(head_run/'joint.flame.npz',allow_pickle=False) as state:
        width=int(state['image_width']);height=int(state['image_height']);fov=float(state['camera_fov_y'])
    if not 0<fov<180:raise ValueError('Invalid FOV')
    focal=height/(2*np.tan(np.radians(fov)/2));k=np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1]],float)
    annotations=None
    if masks_json:
        annotations=json.loads(Path(masks_json).read_text(encoding='utf-8'))
        if annotations.get('schema_version')!=1 or annotations.get('source_viewer_sha256')!=source_hash:raise ValueError('Mask review belongs to another texture source')
    for i,slot in enumerate(SLOTS):
        layer=data['layers'][i]
        with Image.open(io.BytesIO(base64.b64decode(layer['image'].split(',',1)[1],validate=True))) as image:
            if image.size!=(width,height):raise ValueError('Embedded image dimensions mismatch')
            images.append(np.asarray(image.convert('RGB'),float)/255)
        q=np.asarray(layer['coordinates'],float)
        if q.shape!=(len(vertices),3) or not np.isfinite(q).all() or np.any(q[:,2]<=0):raise ValueError('Invalid texture coordinates')
        coords.append(q)
        mask=np.ones((height,width),bool)
        if annotations:
            encoded=annotations['masks'][slot]
            if len(encoded)>20_000_000:raise ValueError('Mask too large')
            with Image.open(io.BytesIO(base64.b64decode(encoded.split(',',1)[1],validate=True))) as image:
                if image.size!=(width,height):raise ValueError('Mask dimensions mismatch')
                mask=np.asarray(image.convert('L'))>=128
        masks.append(mask)
    valid=[];angles=[]
    for q,mask in zip(coords,masks):
        v,a=source_visibility(q,faces,k,(width,height),mask);valid.append(v);angles.append(a)
    valid=np.asarray(valid);angles=np.asarray(angles)
    eligible=np.zeros((3,len(vertices)),bool)
    for i in range(3):eligible[i,np.unique(faces[valid[i]])]=True
    sampled=np.array([sample(image,q) for image,q in zip(images,coords)])
    gains,overlap=estimate_gains(sampled,eligible)
    if skin_color:
        value=skin_color.lstrip('#')
        if len(value)!=6 or any(c not in '0123456789abcdefABCDEF' for c in value):raise ValueError('Use --skin-color #RRGGBB')
        skin=np.array([int(value[j:j+2],16)/255 for j in (0,2,4)]);skin_method='explicit CLI sample'
    elif annotations and annotations.get('skin_rgb') is not None:
        skin=np.asarray(annotations['skin_rgb'],float)/255
        if skin.shape!=(3,) or not np.isfinite(skin).all() or np.any(skin<0) or np.any(skin>1):raise ValueError('Invalid skin sample')
        skin_method='automatic multi-region skin median' if annotations.get('origin')=='automatic_face_parsing' else 'user-selected photograph pixel'
    else:
        pixels=sampled[0,eligible[0]]
        good=(pixels[:,0]>pixels[:,1])&(pixels[:,0]>pixels[:,2])&(pixels.mean(1)>.18)&(pixels.mean(1)<.85)
        skin=np.median(pixels[good],axis=0) if good.sum()>=30 else np.array([.65,.48,.4])
        skin_method='heuristic median photo color; requires review'
    # One independent set of vertices per face keeps invalid source faces excluded.
    weights=np.zeros((len(faces),3,3))
    for i in range(3):weights[:,:,i]=np.maximum((angles[i,faces]-.35)/.65,0)**2*valid[i,:,None]
    front_strength=np.clip((angles[0,faces]-.5)/.2,0,1)*valid[0,:,None]
    weights[:,:,0]*=4
    weights[:,:,1:]*=(1-front_strength[:,:,None])
    totals=weights.sum(2);active=totals>1e-10
    original=np.full(len(faces),-1,int)
    # Reconstruct original hard assignments from the exported face batches.
    lookup={tuple(face):index for index,face in enumerate(faces.tolist())}
    for i,layer in enumerate(data['layers']):
        for face in layer['faces']:original[lookup[tuple(face)]]=i
    hard=np.zeros_like(weights)
    for i in range(3):hard[original==i,:,i]=1
    out=head_run/('head_workspace_'+uuid4().hex[:8]);out.mkdir(exist_ok=False)
    report=dict(schema_version=1,pipeline_type='head_texture_workspace',status='TEXTURE_REVIEW_READY',
        source_texture_run=str(texture_run),source_viewer_sha256=source_hash,
        input_sha256={str(path):digest(path) for path in [viewer,texture_report,parent_report,head_run/'joint.flame.npz']},
        masks_reviewed=bool(annotations) and annotations.get('origin')!='automatic_face_parsing',masks_supplied=bool(annotations),mask_origin=annotations.get('origin','manual') if annotations else 'none',human_review_is_declaration=True,mask_semantic_accuracy_verified=False,color_gains_linear=gains.tolist(),color_overlap_vertices=overlap,
        color_gain_limits=[.85,1.18],skin_color_rgb=np.rint(skin*255).astype(int).tolist(),skin_color_method=skin_method,
        photo_covered_triangles=int(np.any(active,axis=1).sum()),total_triangles=len(faces),
        fill_method='flat base material, not recovered hidden texture',geometry_modified=False,
        uv_atlas_baked=False,albedo_recovered=False,hair_reconstructed=False,mouth_corrected=False,
        likeness_verified=False,printability_verified=False,
        review_items=['Remove projected hair, earbuds, shirt with masks','Inspect color seams and eyes/lips for ghosting','Review sampled skin base in unobserved areas','Mouth geometry and eyeballs still require separate review','Hair and graduation cap assets not yet attached'])
    if masks_json:report['input_sha256'][str(Path(masks_json).resolve())]=digest(Path(masks_json))
    center=(vertices.min(0)+vertices.max(0))*.5;radius=float(np.linalg.norm(vertices-center,axis=1).max())
    payload=dict(vertices=vertices[faces].reshape(-1,3).tolist(),normals=vertex_normals(vertices,faces)[faces].reshape(-1,3).tolist(),
        coordinates=[q[faces].reshape(-1,3).tolist() for q in coords],weights=weights.reshape(-1,3).tolist(),
        hard=hard.reshape(-1,3).tolist(),images=[layer['image'] for layer in data['layers']],
        gains=gains.tolist(),skin=skin.tolist(),center=center.tolist(),radius=radius,report=report)
    mask_data=dict(images=payload['images'],slots=SLOTS,source_viewer_sha256=source_hash,
        masks=None if not annotations else annotations['masks'],skin_rgb=None if not annotations else annotations.get('skin_rgb'))
    def write_template(name,target,value):
        text=Path(__file__).with_name(name).read_text(encoding='utf-8')
        encoded=json.dumps(value,allow_nan=False,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        (out/target).write_text(text.replace('__WORKSPACE_DATA__',encoded),encoding='utf-8')
    write_template('head_texture_workspace.html','head.workspace.html',payload)
    write_template('head_mask_review.html','head.masks.html',mask_data)
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    np.savez_compressed(out/'texture.weights.npz',weights=weights,face_original_source=original,skin_rgb=skin,gains=gains)
    if any(digest(Path(path))!=value for path,value in report['input_sha256'].items()):raise ValueError('Input changed during processing')
    return out,report
