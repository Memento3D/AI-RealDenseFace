"""Explicit demo-only inferred neck/scalp colors and photo-colored existing cap."""
import json
from pathlib import Path
import numpy as np
import cv2
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from tools.headapi.service import ROOT,sha,write
from tools.headapi.texture_completion_trial import export_texture

POLICY='harmonized_head_v1'


def atlas_positions(face_index,vertices,faces,uv):
    size=face_index.shape[0];positions=np.full((*face_index.shape,3),np.nan,np.float32)
    yy,xx=np.where(face_index>=0)
    for start in range(0,len(yy),65536):
        y=yy[start:start+65536];x=xx[start:start+65536];fi=face_index[y,x];tri=uv[fi]
        p=np.c_[(x+.5)/size,1-(y+.5)/size];a,b,c=tri.transpose(1,0,2)
        den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        w0=((b[:,1]-c[:,1])*(p[:,0]-c[:,0])+(c[:,0]-b[:,0])*(p[:,1]-c[:,1]))/den
        w1=((c[:,1]-a[:,1])*(p[:,0]-c[:,0])+(a[:,0]-c[:,0])*(p[:,1]-c[:,1]))/den
        positions[y,x]=np.einsum('ni,nij->nj',np.c_[w0,w1,1-w0-w1],vertices[faces[fi]])
    return positions


def presentation_regions(positions,observed,skin,brow,eye,mouth,height):
    valid=np.isfinite(positions).all(-1);x,y,z=np.moveaxis(positions,-1,0)
    lower_rear=(z<eye[2]-.12*height)&(y<eye[1]-.01*height)
    neck=valid&~observed&((y<mouth[1]-.065*height)|lower_rear)
    top=y>brow[1]+.045*height
    posterior=(z<eye[2]-.12*height)&(y>mouth[1]+.10*height)
    # Avoid inferring dark hair into lateral lower ear regions.
    ear_guard=(np.abs(x-eye[0])>.25*height)&(y<eye[1]+.03*height)
    scalp=valid&(top|posterior)&~ear_guard&(~observed|~skin.astype(bool))
    return neck,scalp


def run(job,output):
    job=Path(job).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    bundle=job/'rdf/projection_bundle.npz';atlas_path=job/'rdf_uv/final/face_atlas.png';quality=job/'rdf_uv/raw/quality_maps.npz'
    uv_report=json.loads((job/'rdf_uv/report.json').read_text());raw_report=json.loads((job/'rdf_uv/raw/report.json').read_text())
    asset=next(Path(p) for p in raw_report['input_sha256'] if Path(p).name=='flame_assets.npz')
    if sha(asset)!='922cb76f3afef0c2e9801a63991de269a98e6b67b1728232493f851589151dd3':raise ValueError('UV pin mismatch')
    if uv_report.get('status')!='UV_FACE_EXPORTED' or uv_report['input_sha256'].get(str(bundle))!=sha(bundle):raise ValueError('UV source binding mismatch')
    with np.load(bundle,allow_pickle=False) as b:v=b['vertices'];faces=b['faces'];cameras=b['cameras']
    with np.load(asset,allow_pickle=False) as b:
        if not np.array_equal(faces,b['faces']):raise ValueError('UV topology mismatch')
        uv=b['uvs'][b['uv_faces']]
    from scripts.eye_components import identify
    _,topology=identify(v,faces)
    with np.load(quality,allow_pickle=False) as b:maps={k:b[k] for k in b.files}
    atlas=np.array(Image.open(atlas_path).convert('RGB'));original=atlas.copy();observed=np.asarray(Image.open(job/'rdf_uv/raw/coverage.png'))>0
    positions=atlas_positions(maps['face_index'],v,faces,uv);height=float(np.ptp(v[:,1]))
    eye=np.mean(topology['eye_centers_native'],axis=0)
    brow=eye+np.array([0,.07*height,0]);mouth=eye+np.array([0,-.20*height,0])
    neck,scalp=presentation_regions(positions,observed,maps['skin'],brow,eye,mouth,height)
    reference=observed&np.isfinite(positions).all(-1)&(positions[...,1]<mouth[1]+.015*height)&(positions[...,1]>v[:,1].min()+.04*height)&(maps['confidence']>.35)
    rgb=original.astype(np.int16)
    reference&=(rgb[...,0]>rgb[...,1]+3)&(rgb[...,0]>rgb[...,2]+8)&(rgb.mean(-1)>35)&(rgb.mean(-1)<220)
    ry,rx=np.where(reference)
    if len(ry)<32:raise ValueError('Not enough observed lower-face/neck skin for demo color')
    step=max(1,len(ry)//30000);ry=ry[::step];rx=rx[::step]
    points=positions[ry,rx];colors=original[ry,rx].astype(float);median=np.median(colors,axis=0);tree=cKDTree(points)
    ny,nx=np.where(neck)
    for start in range(0,len(ny),32768):
        y=ny[start:start+32768];x=nx[start:start+32768];dist,index=tree.query(positions[y,x],k=min(16,len(points)))
        weights=1/np.maximum(dist,.001*height)**2;weights/=weights.sum(1,keepdims=True)
        local=(colors[index]*weights[...,None]).sum(1)
        local[dist[:,0]>.35*height]=median
        atlas[y,x]=np.clip(.5*local+.5*median,0,255).astype(np.uint8)
    hair_report=json.loads((job/'hair/report.json').read_text())
    hair_samples=[];images=[];masks=[];inputs={str(p):sha(p) for p in [bundle,atlas_path,quality,asset,job/'rdf_uv/report.json',job/'hair/report.json']}
    for slot in ['front','left','right']:
        photo=job/'rdf_crops'/(slot+'.png');mask_path=job/'hair'/(slot+'.hair-mask.png')
        if sha(photo)!=raw_report['input_sha256'][str(photo)]:raise ValueError('Photo changed')
        if sha(mask_path)!=hair_report['output_sha256'].get(mask_path.name):raise ValueError('Hair mask changed')
        image=np.asarray(Image.open(photo).convert('RGB'));mask=np.asarray(Image.open(mask_path))>0
        mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
        samples=image[mask];samples=samples[(samples.mean(1)>8)&(samples.mean(1)<160)]
        if len(samples):hair_samples.append(samples[::max(1,len(samples)//12000)])
        images.append(image);masks.append(mask);inputs[str(photo)]=sha(photo);inputs[str(mask_path)]=sha(mask_path)
    if not hair_samples:raise ValueError('No observed hair color samples')
    hair_color=np.clip(np.median(np.concatenate(hair_samples),axis=0),12,110).astype(np.uint8)
    atlas[scalp]=hair_color
    provenance=np.asarray(Image.open(job/'rdf_uv/final/provenance-labels.png')).copy();provenance[neck]=3;provenance[scalp]=4
    confidence=np.array(Image.open(job/'rdf_uv/final/confidence.png'));confidence[neck|scalp]=13
    if not np.array_equal(atlas[~(neck|scalp)],original[~(neck|scalp)]):raise ValueError('Changed pixels outside demo ROIs')
    Image.fromarray(atlas).save(output/'head-atlas.png');Image.fromarray(provenance).save(output/'appearance-labels.png');Image.fromarray(confidence).save(output/'appearance-confidence.png')
    export_texture(v,faces,uv,atlas,output/'head.glb')
    hair_source=job/'hair/hair_cap_native.glb';hair=trimesh.load(hair_source,force='mesh',process=False);inputs[str(hair_source)]=sha(hair_source)
    color=np.tile(hair_color,(len(hair.vertices),1)).astype(float);weight_sum=np.zeros(len(hair.vertices));acc=np.zeros_like(color)
    normals=hair.vertex_normals
    for camera,image,mask in zip(cameras,images,masks):
        h=np.c_[hair.vertices,np.ones(len(hair.vertices))]@camera.T;xy=h[:,:2]/h[:,2,None];ij=np.floor(xy).astype(int)
        inside=(h[:,2]>0)&(ij[:,0]>=0)&(ij[:,0]<image.shape[1])&(ij[:,1]>=0)&(ij[:,1]<image.shape[0])
        ix,iy=np.clip(ij,[0,0],[image.shape[1]-1,image.shape[0]-1]).T
        center=-np.linalg.solve(camera[:,:3],camera[:,3]);direction=center-hair.vertices;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
        facing=np.einsum('ij,ij->i',normals,direction)
        valid=inside&mask[iy,ix]&(facing>.2)&(image[iy,ix].mean(1)<160)
        weight=valid*np.maximum(facing,0)**4;acc+=image[iy,ix]*weight[:,None];weight_sum+=weight
    seen=weight_sum>0;color[seen]=.7*acc[seen]/weight_sum[seen,None]+.3*hair_color
    color=np.clip(color,8,140).astype(np.uint8);hair.visual.vertex_colors=np.c_[color,np.full(len(color),255,np.uint8)];hair.export(output/'hair.glb')
    restored=trimesh.load(output/'hair.glb',force='mesh',process=False)
    if not np.array_equal(restored.vertices[restored.faces].astype(np.float32),hair.vertices[hair.faces].astype(np.float32)):raise ValueError('Hair geometry changed')
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Presentation source changed')
    report={'status':'DEMO_APPEARANCE_EXPORTED','policy':POLICY,'geometry_changed':False,'glasses_processed':False,'ready_for_machine':False,
            'neck_inferred_texels':int(neck.sum()),'scalp_appearance_texels':int(scalp.sum()),'scalp_replaced_observed_nonskin_texels':int((scalp&observed).sum()),
            'hair_rgb_from_photos':hair_color.tolist(),'cap_vertices_with_photo_color_support':int(seen.sum()),'cap_vertices_prior_color':int((~seen).sum()),
            'provenance_labels':{'3':'demo inferred neck color, not observed','4':'demo inferred scalp hair background, not recovered hair texture'},
            'limitations':['Demo appearance continuation, not reconstruction of hidden skin','Hair cap geometry unchanged; not recovered strands/parting','Hair color projection uses existing cap/cameras/masks without independent calibration','Glasses and eyes left unchanged by anatomical ROI'],
            'roi_policy':'coarse FLAME geometry ratios relative to connected eyeball centers; not measured anatomy',
            'inferred_confidence':'13/255 heuristic, not probability; do not use presentation colors as observed ground truth',
            'head_sha256':sha(output/'head.glb'),'hair_sha256':sha(output/'hair.glb'),
            'input_sha256':inputs,'code_sha256':sha(Path(__file__))}
    write(output/'report.json',report);return report
