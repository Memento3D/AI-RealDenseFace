"""Face-only UV texture trial: frozen geometry, triangle visibility and semantic masks."""
import hashlib
import json
from pathlib import Path
import sys
import ctypes
import shutil
import numpy as np
import cv2
import trimesh
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from memento3d_head.adapters.photo_projection import depth_buffer
from tools.headapi.hair_parser import HairParser,PARSER_SHA
from tools.headapi.continuous_visibility import ContinuousDepth


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def preview(vertices,faces,uv,atlas,path,size=360):
    # Orthographic front/left/back/right comparison, identical framing per export.
    v=vertices[:,[2,0,1]].copy();v-=(v.min(0)+v.max(0))/2
    scale=.84*size/np.ptp(v,axis=0).max();sheet=Image.new('RGB',(size*4,size+24),'#20252c');draw=ImageDraw.Draw(sheet)
    for index,angle in enumerate([0,90,180,270]):
        a=np.radians(angle);z=v[:,0]*np.cos(a)+v[:,1]*np.sin(a)
        x=-v[:,0]*np.sin(a)+v[:,1]*np.cos(a);xy=np.c_[x*scale+size/2,-v[:,2]*scale+size/2]
        pixels=np.full((size,size,3),[35,40,47],dtype=np.uint8);depth=np.full((size,size),-np.inf)
        for j,face in enumerate(faces):
            tri=xy[face];lo=np.maximum(np.floor(tri.min(0)).astype(int),0);hi=np.minimum(np.ceil(tri.max(0)).astype(int),size-1)
            if (hi<lo).any():continue
            p,q,r=tri;den=(q[1]-r[1])*(p[0]-r[0])+(r[0]-q[0])*(p[1]-r[1])
            if abs(den)<1e-10:continue
            yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
            b0=((q[1]-r[1])*(xx-r[0])+(r[0]-q[0])*(yy-r[1]))/den
            b1=((r[1]-p[1])*(xx-r[0])+(p[0]-r[0])*(yy-r[1]))/den;b2=1-b0-b1
            values=b0*z[face[0]]+b1*z[face[1]]+b2*z[face[2]];section=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
            keep=(b0>=0)&(b1>=0)&(b2>=0)&(values>section)
            if not keep.any():continue
            tex=b0[...,None]*uv[j,0]+b1[...,None]*uv[j,1]+b2[...,None]*uv[j,2]
            tx=np.clip((tex[...,0]*atlas.shape[1]).astype(int),0,atlas.shape[1]-1)
            ty=np.clip(((1-tex[...,1])*atlas.shape[0]).astype(int),0,atlas.shape[0]-1)
            pixels[lo[1]:hi[1]+1,lo[0]:hi[0]+1][keep]=atlas[ty[keep],tx[keep]];section[keep]=values[keep]
        sheet.paste(Image.fromarray(pixels),(index*size,24));draw.text((index*size+8,5),str(angle)+' deg / unlit photo texture',fill='white')
    sheet.save(path)


def robust_skin_gain(reference, sample):
    """Bounded linear-light exposure/color correction from matched skin only."""
    reference=np.asarray(reference,dtype=float)/255
    sample=np.asarray(sample,dtype=float)/255
    valid=((reference>.08)&(reference<.94)&(sample>.08)&(sample<.94)).all(axis=1)
    if valid.sum()<32:return np.ones(3),int(valid.sum())
    def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    ratios=np.log(linear(reference[valid]))-np.log(linear(sample[valid]))
    center=np.median(ratios,axis=0);mad=np.median(np.abs(ratios-center),axis=0)
    keep=(np.abs(ratios-center)<=np.maximum(3*1.4826*mad,.03)).all(axis=1)
    if keep.sum()<32:return np.ones(3),int(keep.sum())
    return np.clip(np.exp(np.median(ratios[keep],axis=0)),.85,1.18),int(keep.sum())


def apply_gain(rgb,gain):
    x=np.asarray(rgb,dtype=float)/255
    x=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)*gain
    return np.clip(np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)*255,0,255)


def run(output=None,harmonize=False,correction_path=None,correction_name=None,cached_parser_report=None,quality_maps=False,geometry_path=None,source_dir=None,expected_bundle_sha=None,visibility_mode='pixel_depth_legacy',visibility_tolerance=.001,parser_path_override=None,assets_path_override=None):
    if visibility_mode not in ['pixel_depth_legacy','continuous_triangle']:raise ValueError('Unknown visibility policy')
    if not np.isfinite(visibility_tolerance) or visibility_tolerance<=0:raise ValueError('Positive finite depth tolerance required')
    if harmonize and correction_path:raise ValueError('Choose one correction method')
    cached=json.loads(Path(cached_parser_report).read_text()) if cached_parser_report else None
    correction=json.loads(Path(correction_path).read_text()) if correction_path else None
    selected=next(v for v in correction['variants'] if v['name']==correction_name) if correction else None
    output=Path(output) if output else ROOT/'runs/experiments/input04-face-texture-01'
    if sys.platform=='win32':
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ['total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','available_extended']]
        memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):raise OSError('Cannot read physical RAM')
        if memory.available_physical<2*2**30:raise RuntimeError('Need at least 2 GiB free physical RAM before starting texture trial')
    if shutil.disk_usage(ROOT).free<2*2**30:raise RuntimeError('Need at least 2 GiB free disk before starting texture trial')
    output.mkdir(parents=True,exist_ok=False)
    fit=ROOT/'runs/experiments/rdf-flame-parameters-40ab0c23fed043eb875537da057027e8/shape_expression_0_05'
    bundle=Path(geometry_path) if geometry_path else fit/'projection_bundle.npz';meta_path=ROOT/'runs/experiments/rdf-flame-parameters-40ab0c23fed043eb875537da057027e8/experiment.json'
    source=Path(source_dir) if source_dir else ROOT/'runs/experiments/input04-validation-01/rdf_crops'
    expected=expected_bundle_sha if geometry_path else json.loads(meta_path.read_text())['variants']['shape_expression_0_05']['bundle_sha256']
    if geometry_path and not expected:raise ValueError('Explicit geometry requires checksum pin')
    if sha(bundle)!=expected:raise ValueError('Geometry checksum changed')
    assets=Path(assets_path_override) if assets_path_override else ROOT/'vendor/RealDenseFace/weights/flame/flame_assets.npz'
    if sha(assets)!='922cb76f3afef0c2e9801a63991de269a98e6b67b1728232493f851589151dd3':raise ValueError('UV asset checksum mismatch')
    with np.load(bundle,allow_pickle=False) as b:V=b['vertices'];F=b['faces'];views=b['view_vertices'];cameras=b['cameras']
    with np.load(assets,allow_pickle=False) as b:
        if not np.array_equal(F,b['faces']):raise ValueError('UV face topology mismatch')
        uv=b['uvs'][b['uv_faces']].astype(float)
    parser_path=Path(parser_path_override) if parser_path_override else Path('D:/AI-cache/face-parsing/resnet18.onnx');parser=None if cached else HairParser(parser_path)
    layers=[];sources={}
    if cached:
        if cached['input_sha256'][str(bundle)]!=sha(bundle):raise ValueError('Cached label bundle mismatch')
        sources[str(Path(cached_parser_report))]=sha(cached_parser_report)
    if correction:
        if correction['input_sha256'][str(bundle)]!=sha(bundle):raise ValueError('Correction bundle mismatch')
        sources[str(Path(correction_path))]=sha(correction_path)
    for i,slot in enumerate(['front','left','right']):
        path=source/(slot+'.png');image=Image.open(path).convert('RGB');rgb=np.asarray(image)
        if cached:
            label_path=Path(cached_parser_report).parent/(slot+'.labels.png')
            for p in [path,label_path]:
                if cached['input_sha256'][str(p)]!=sha(p):raise ValueError('Cached parser provenance mismatch')
                sources[str(p)]=sha(p)
            label=np.asarray(Image.open(label_path))
            if label.shape!=rgb.shape[:2]:raise ValueError('Cached label dimensions mismatch')
        else:label=parser(image)
        if correction and correction['input_sha256'][str(path)]!=sha(path):raise ValueError('Correction image mismatch')
        mask=np.isin(label,[1,2,3,4,5,7,8,10,11,12,13,14]);mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
        Image.fromarray((mask*255).astype(np.uint8)).save(output/(slot+'.accepted-mask.png'))
        K,R,c,*_=cv2.decomposeProjectionMatrix(cameras[i]);K=K/K[2,2];center=(c[:3]/c[3]).reshape(3)
        cam=(views[i]-center)@R.T;_,depth=depth_buffer(cam,F,K,image.size)
        normals=trimesh.Trimesh(views[i],F,process=False).face_normals
        directions=center-views[i][F].mean(1);directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        facing=np.einsum('ij,ij->i',normals,directions)
        layers.append((rgb,label,mask,cam,K,depth,facing));sources[str(path)]=sha(path)
    calibration=[];gains=[np.ones(3) for _ in layers]
    if selected:
        gains=np.asarray(selected['gains'],dtype=float)
        if gains.shape!=(3,3) or not np.isfinite(gains).all() or (gains<=0).any() or not np.array_equal(gains[0],np.ones(3)):raise ValueError('Invalid correction gains')
    if harmonize:
        matches=[]
        for rgb,label,mask,cam,K,depth,facing in layers:
            h=cam@K.T;xy=h[:,:2]/h[:,2,None];ij=np.floor(xy).astype(int)
            valid=(cam[:,2]>0)&(ij[:,0]>=1)&(ij[:,0]<rgb.shape[1]-1)&(ij[:,1]>=1)&(ij[:,1]<rgb.shape[0]-1)
            ix,iy=np.clip(ij,[0,0],[rgb.shape[1]-1,rgb.shape[0]-1]).T
            skin=cv2.erode((label==1).astype(np.uint8),np.ones((5,5),np.uint8))>0
            valid &= skin[iy,ix] & mask[iy,ix] & (np.abs(depth[iy,ix]-cam[:,2])<.001)
            matches.append((rgb[iy,ix],valid))
        for i in [1,2]:
            keep=matches[0][1]&matches[i][1]
            gains[i],used=robust_skin_gain(matches[0][0][keep],matches[i][0][keep])
            calibration.append({'view':i,'matched_skin_vertices':int(keep.sum()),'robust_samples':used,'linear_rgb_gain':gains[i].tolist()})
    size=2048;atlas=np.full((size,size,3),180,dtype=np.uint8);coverage=np.zeros((size,size),dtype=np.uint8)
    engines=[ContinuousDepth(layer[3],F,layer[4],(layer[0].shape[1],layer[0].shape[0])) for layer in layers] if visibility_mode=='continuous_triangle' else None
    if quality_maps:
        interior=np.zeros((size,size),np.uint8);skin_atlas=np.zeros_like(interior);protected=np.zeros_like(interior)
        confidence=np.zeros((size,size),np.float32);view_count=np.zeros_like(interior)
        reason=np.zeros_like(interior);face_index=np.full((size,size),-1,np.int32)
        mask_distances=[cv2.distanceTransform(layer[2].astype(np.uint8),cv2.DIST_L2,5) for layer in layers]
    counts=np.zeros(4,dtype=int);inside_count=0
    for j,face in enumerate(F):
        tri=uv[j]*[size,-size]+[0,size];lo=np.maximum(np.floor(tri.min(0)).astype(int),0);hi=np.minimum(np.ceil(tri.max(0)).astype(int),size-1)
        if (hi<lo).any():continue
        a,b,c=tri;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-10:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
        b0=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        b1=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;b2=1-b0-b1
        inside=(b0>=0)&(b1>=0)&(b2>=0)
        if not inside.any():continue
        bary=np.c_[b0[inside],b1[inside],b2[inside]];inside_count+=len(bary)
        colors=[];scores=[];front_features=None;semantic=[];quality=[];frame_checks=[];mask_checks=[];angle_checks=[]
        for i,(rgb,label,mask,cam,K,depth,facing) in enumerate(layers):
            points=bary@cam[face];h=points@K.T;xy=h[:,:2]/h[:,2,None]
            ij=np.floor(xy).astype(int);valid=(points[:,2]>0)&(ij[:,0]>=1)&(ij[:,0]<rgb.shape[1]-1)&(ij[:,1]>=1)&(ij[:,1]<rgb.shape[0]-1)
            clipped=np.clip(ij,[0,0],[rgb.shape[1]-1,rgb.shape[0]-1]);ix,iy=clipped.T
            if quality_maps:frame_checks.append(valid.copy())
            valid &= mask[iy,ix]
            if quality_maps:mask_checks.append(valid.copy())
            valid &= facing[j]>.3
            if quality_maps:angle_checks.append(valid.copy())
            if engines:
                closest=np.full(len(points),np.inf)
                if valid.any():closest[valid]=engines[i].query(xy[valid])
                valid &= np.abs(closest-points[:,2])<visibility_tolerance
            else:valid &= np.abs(depth[iy,ix]-points[:,2])<visibility_tolerance
            sampled=cv2.remap(rgb,xy[:,0].astype(np.float32)[None],xy[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0].astype(float)
            if harmonize or selected:sampled=apply_gain(sampled,gains[i])
            scores.append(valid.astype(float)*max(float(facing[j]),0)**4*(2 if i==0 else 1));colors.append(sampled)
            if quality_maps:
                semantic.append(label[iy,ix]);quality.append(valid.astype(float)*max(float(facing[j]),0)*np.minimum(mask_distances[i][iy,ix]/5.,1.))
            if i==0:front_features=valid & np.isin(label[iy,ix],[4,5,11,12,13])
        score=np.stack(scores);score[1:,front_features]=0
        total=score.sum(0);observed=total>0;blend=(np.stack(colors)*score[:,:,None]).sum(0)/np.maximum(total[:,None],1e-12)
        target=atlas[lo[1]:hi[1]+1,lo[0]:hi[0]+1];maskview=coverage[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        oy,ox=np.where(inside);target[oy[observed],ox[observed]]=np.clip(blend[observed],0,255).astype(np.uint8)
        owner=np.argmax(score,axis=0)+1;maskview[oy[observed],ox[observed]]=owner[observed]
        if quality_maps:
            yp=lo[1]+oy;xp=lo[0]+ox;interior[yp,xp]=1
            face_index[yp,xp]=j
            failure=np.full(len(yp),5,np.uint8)
            failure[~np.any(angle_checks,axis=0)]=4
            failure[~np.any(mask_checks,axis=0)]=3
            failure[~np.any(frame_checks,axis=0)]=2
            failure[observed]=1;reason[yp,xp]=failure
            weights=score/np.maximum(total[None],1e-12)
            disagreement=np.sqrt((weights[:,:,None]*(np.stack(colors)-blend[None])**2).sum((0,2))/3)
            conf=np.max(np.stack(quality)*(score>0),axis=0)*np.exp(-disagreement/30.)
            winner=owner-1;labels=np.stack(semantic)[winner,np.arange(len(winner))]
            skin_atlas[yp[observed],xp[observed]]=(labels[observed]==1)
            # Union of all visible non-skin semantics protects features even if not dominant.
            protected[yp,xp]=np.any((np.stack(semantic)!=1)&(score>0),axis=0)
            confidence[yp[observed],xp[observed]]=conf[observed]
            view_count[yp[observed],xp[observed]]=(score[:,observed]>0).sum(0)
        for k in range(1,4):counts[k]+=int(((owner==k)&observed).sum())
    Image.fromarray(atlas).save(output/'face_atlas.png');Image.fromarray(coverage*80).save(output/'coverage.png')
    if quality_maps:
        reason[coverage>0]=1
        np.savez_compressed(output/'quality_maps.npz',interior=interior,skin=skin_atlas,protected=protected,confidence=confidence,view_count=view_count,rejection_reason=reason,face_index=face_index)
        Image.fromarray(np.clip(confidence*255,0,255).astype(np.uint8)).save(output/'confidence.png')
    flatV=V[F].reshape(-1,3);flatF=np.arange(len(flatV)).reshape(-1,3)
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(atlas),metallicFactor=0,roughnessFactor=1,doubleSided=True)
    visual=trimesh.visual.texture.TextureVisuals(uv=uv.reshape(-1,2),material=material)
    normals=trimesh.Trimesh(V,F,process=False).vertex_normals[F].reshape(-1,3)
    mesh=trimesh.Trimesh(flatV,flatF,vertex_normals=normals,visual=visual,process=False);mesh.export(output/'face_textured.glb')
    reloaded=trimesh.load(output/'face_textured.glb',force='mesh',process=False)
    if not np.array_equal(reloaded.vertices[reloaded.faces].astype(np.float32),V[F].astype(np.float32)):
        raise ValueError('GLB roundtrip changed triangle coordinates')
    preview(V,F,uv,atlas,output/'face_textured.png')
    report={'schema_version':1,'status':'TEXTURE_EXPORTED_FOR_REVIEW','geometry_changed':False,
        'triangle_coordinates_float32_roundtrip_equal':True,'uv_seams_duplicate_render_vertices':True,
        'manufacturing_mesh':False,'atlas_size':[size,size],'observed_texel_writes':int(counts.sum()),
        'uv_interior_texel_writes':inside_count,'dominant_view_writes':counts.tolist(),
        'unobserved_fill':'explicit neutral gray; not recovered skin','lighting_removed':False,'hair_included':False,
        'visibility':'full triangle depth rasterization; sampled z tolerance 0.001 native units',
        'mask':'pinned semantic parser; exclude background/hair/hat/glasses/clothing/accessories; 1px erosion',
        'blending':'facing^4 weights, front x2; visible front eye/mouth semantics use front only; no exposure harmonization',
        'limitations':['semantic mask and cameras approximate','UV texel overlaps and border seams not independently audited',
            'gray means unobserved or rejected pixels, not model error','2048 atlas does not add source-photo resolution',
            'photographic shading remains; no albedo recovery','texture is display color, not printed geometry'],
        'input_sha256':{**sources,str(bundle):sha(bundle),str(assets):sha(assets),str(parser_path):PARSER_SHA},
        'code_sha256':sha(Path(__file__))}
    report['color_harmonization']={'enabled':harmonize,'reference':'front','method':'robust matched skin log ratios in linear sRGB; gains bounded 0.85..1.18','calibration':calibration,'intrinsic_albedo':False}
    if harmonize:report['blending']=report['blending'].replace('no exposure harmonization','bounded linear-sRGB gains fitted on matched skin vertices')
    if selected:
        report['color_harmonization']={'enabled':True,'method':'external batch train-only exposure/chroma candidate; exploratory','candidate':correction_name,'gains':gains.tolist(),'intrinsic_albedo':False}
        report['blending']=report['blending'].replace('no exposure harmonization','batch exposure/chroma correction; front unchanged')
    report['quality_maps']={'enabled':quality_maps,'confidence':'heuristic max facing * semantic-mask distance /5 capped * exp(-weighted RGB disagreement /30); not calibrated probability','counts':'unique atlas pixels, with last-write ownership at UV overlaps'}
    report['visibility_mode']=visibility_mode
    report['visibility_tolerance_native']=visibility_tolerance
    if engines:report['visibility']='continuous subpixel projected triangle test, conservative spatial bins, perspective 1/z depth; approximate camera and masks remain'
    if quality_maps:report['rejection_reason_labels']={'0':'outside UV','1':'observed','2':'no in-frame view','3':'all semantic masks rejected','4':'all mask-valid views too oblique','5':'all mask/angle-valid views failed depth'}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(output,flush=True)


if __name__=='__main__':run()
