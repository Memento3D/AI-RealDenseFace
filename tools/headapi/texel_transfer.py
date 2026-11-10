"""Experimental texel-level face transfer; no model/geometry modifications."""
import argparse,json,hashlib,sys,struct
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image

def glb_base_image(path):
    with open(path, 'rb') as f:
        f.read(12)  # Skip the GLB header before reading its chunks.
        n, t = struct.unpack('<II', f.read(8))
        doc = json.loads(f.read(n))
        n, t = struct.unpack('<II', f.read(8))
        binary = f.read(n)
    index = doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['index']
    img = doc['images'][doc['textures'][index]['source']]
    view = doc['bufferViews'][img['bufferView']]
    start = view.get('byteOffset', 0)
    raw = binary[start : start + view['byteLength']]
    return Image.open(BytesIO(raw)).convert('RGB'), raw


def bilinear(image, uv):
    h, w = image.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip((1 - uv[:, 1]) * h - 0.5, 0, h - 1)
    x0 = x.astype(int)
    y0 = y.astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    a = (x - x0)[:, None]
    b = (y - y0)[:, None]
    return (
        (1 - a) * (1 - b) * image[y0, x0]
        + a * (1 - b) * image[y0, x1]
        + (1 - a) * b * image[y1, x0]
        + a * b * image[y1, x1]
    )



def raster(uv,faces,depth,size):
    uv=np.asarray(uv,float);faces=np.asarray(faces);depth=np.asarray(depth,float)
    if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all() or depth.shape!=(len(uv),) or not np.isfinite(depth).all():raise ValueError('Invalid raster coordinates')
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or faces.min()<0 or faces.max()>=len(uv) or not 1<=size<=4096:raise ValueError('Invalid raster topology/size')
    pixels=uv*np.array([size,-size])+np.array([0,size])
    z=np.full((size,size),np.inf);ids=np.full((size,size),-1,np.int32);weights=np.zeros((size,size,3),np.float32)
    for i,face in enumerate(faces):
        a,b,c=pixels[face];lo=np.maximum(np.floor(np.minimum(np.minimum(a,b),c)).astype(int),0);hi=np.minimum(np.ceil(np.maximum(np.maximum(a,b),c)).astype(int),size-1)
        if (lo>hi).any():continue
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12:continue
        x,y=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        ba=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        bb=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den;bc=1-ba-bb
        dz=ba*depth[face[0]]+bb*depth[face[1]]+bc*depth[face[2]]
        view=z[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        keep=(ba>=-1e-8)&(bb>=-1e-8)&(bc>=-1e-8)&(dz<view)
        iy,ix=np.nonzero(keep);py=iy+lo[1];px=ix+lo[0]
        z[py,px]=dz[keep];ids[py,px]=i;weights[py,px]=np.column_stack([ba[keep],bb[keep],bc[keep]])
    return ids,weights

def source_classes(uv,labels,confidence):
    h,w=labels.shape;x=np.clip(uv[:,0]*w-.5,0,w-1);y=np.clip((1-uv[:,1])*h-.5,0,h-1)
    x0=x.astype(int);y0=y.astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    values=np.column_stack([labels[y0,x0],labels[y0,x1],labels[y1,x0],labels[y1,x1]])
    scores=np.column_stack([confidence[y0,x0],confidence[y0,x1],confidence[y1,x0],confidence[y1,x1]])
    supported=np.isin(values,[1,5,6,7,8,9,10,11,12,13,14,15]).all(1)
    observed=np.isin(values,[1,7,9,11,15]).all(1)&(scores>=38).all(1)
    result=np.where(supported,np.where(observed,1,2),0).astype(np.uint8)
    score=np.where(observed,scores.min(1),np.minimum(scores.min(1),13)).astype(np.uint8)
    return result,score

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(template,transfer,output,tone_limit=None,runtime=None):
    template,transfer,output=map(lambda p:Path(p).resolve(),[template,transfer,output]);output.mkdir(exist_ok=False)
    manifest=json.loads(template.read_text(encoding='utf-8'));root=Path(runtime).resolve() if runtime is not None else template.parent
    for path,digest in manifest['pinned_files'].items():
        if sha(root/path)!=digest:raise ValueError('Bound input changed')
    with np.load(transfer/'patch_data.npz',allow_pickle=False) as b:
        patch={k:b[k] for k in ['patch_vertices','patch_triangles','patch_uv','old_barycentric','old_faces','source_barycentric','source_faces','weights','atlas_min','atlas_max']}
    with np.load(template.parent/'source_aligned.npz',allow_pickle=False) as b:sv,sf,canonical=b['vertices'],b['triangles'],b['original']
    projection_path=root/manifest['source_projection'];texture_path=root/manifest['source_texture']
    with np.load(projection_path,allow_pickle=False) as b:
        coords,chosen=b['atlas_coordinates'],b['chosen'];kind=str(b['source_kind']) if 'source_kind' in b.files else 'photo_projection'
    if coords.shape!=(len(sf),3,3) or chosen.shape!=(len(sf),):raise ValueError('Projection topology mismatch')
    texture=np.asarray(Image.open(texture_path).convert('RGB'))
    with np.load(template.parent/'baseline.npz',allow_pickle=False) as b:old_uv=np.einsum('ij,ijk->ik',patch['old_barycentric'],b['uv'].reshape(-1,3,2)[patch['old_faces']])
    original,_=glb_base_image(root/manifest['source_glb']);old_color=bilinear(np.asarray(original),old_uv)
    q=np.einsum('ij,ijk->ik',patch['source_barycentric'],coords[patch['source_faces']]);vertex_uv=q[:,:2]/q[:,2,None]
    known=chosen[patch['source_faces']]>=0;source_color=bilinear(texture,vertex_uv)
    if not known.any():raise ValueError('No source donors')
    filled=source_color.copy();filled[~known]=np.median(source_color[known],axis=0)
    tri=patch['patch_triangles'];edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0);a,b=edges.T
    degree=np.maximum(np.bincount(edges.ravel(),minlength=len(filled)),1)
    for _ in range(500):
        for channel in range(3):
            sums=np.bincount(a,weights=filled[b,channel],minlength=len(filled))+np.bincount(b,weights=filled[a,channel],minlength=len(filled))
            filled[~known,channel]=sums[~known]/degree[~known]
    xy=patch['patch_vertices'][:,[0,2]];cheeks=known&(patch['weights']>=1)&(abs(xy[:,0]+.02)>.025)&(xy[:,1]>.83)&(xy[:,1]<.87)
    if cheeks.sum()<16:raise ValueError('Too few cheek donors for common tone policy')
    median=np.median(filled[cheeks],axis=0);raw_shift=.75*(np.median(old_color[cheeks],axis=0)-median);shift=raw_shift.copy();skin_luma=float(median@np.array([.2126,.7152,.0722]))
    if tone_limit is not None:
        if not np.isfinite(tone_limit) or not 0<=tone_limit<=255:raise ValueError('Invalid tone limit')
        shift=np.clip(shift,-tone_limit,tone_limit)
    baseline=np.asarray(Image.open(transfer/'face_atlas.png').convert('RGB'));size=baseline.shape[0]
    pi,pb=raster(patch['patch_uv'],tri,np.zeros(len(patch['patch_uv'])),size)
    syz=sv[:,[0,2]];suv=(syz-patch['atlas_min'])/(patch['atlas_max']-patch['atlas_min'])
    si,sb=raster(suv,sf,sv[:,1],size)
    labels=confidence=None
    if kind=='canonical_rdf_uv':
        labels=np.asarray(Image.open(texture_path.parent/'source-labels.png'));confidence=np.asarray(Image.open(texture_path.parent/'source-confidence.png'))
    result=baseline.copy();provenance=np.zeros((size,size),np.uint8);out_conf=np.zeros((size,size),np.uint8)
    y,x=np.nonzero(pi>=0);provenance[y,x]=3
    candidates=np.flatnonzero(si[y,x]>=0);accepted=observed=inferred=canonical_rejected=0
    for start in range(0,len(candidates),65536):
        select=candidates[start:start+65536];py,px=y[select],x[select];pf=pi[py,px];pbc=pb[py,px];source_fi=si[py,px];bc=sb[py,px]
        weight=np.einsum('ij,ij->i',pbc,patch['weights'][tri[pf]])
        old=np.einsum('ij,ijk->ik',pbc,old_color[tri[pf]])
        qq=np.einsum('ij,ijk->ik',bc,coords[source_fi]);uv=qq[:,:2]/np.maximum(qq[:,2,None],1e-30)
        valid=np.isfinite(uv).all(1)&(qq[:,2]>0)&(uv>=0).all(1)&(uv<=1).all(1)&(weight>0)
        point=np.einsum('ij,ijk->ik',bc,canonical[sf[source_fi]])
        bad=(weight>=1-1e-6)&((point[:,1]>.01)|(point[:,2]<-.105));canonical_rejected+=int(bad.sum());valid&=~bad
        if labels is not None:classes,scores=source_classes(uv,labels,confidence);valid&=classes>0
        else:classes=np.ones(len(uv),np.uint8);scores=np.zeros(len(uv),np.uint8);valid&=chosen[source_fi]>=0
        color=bilinear(texture,uv);luma=color@np.array([.2126,.7152,.0722]);tone=np.clip(luma/max(skin_luma,1),0,1)**2
        color=np.clip(color+tone[:,None]*shift,0,255);mixed=old*(1-weight[:,None])+color*weight[:,None]
        result[py[valid],px[valid]]=np.clip(np.rint(mixed[valid]),0,255).astype(np.uint8)
        provenance[py[valid],px[valid]]=classes[valid];out_conf[py[valid],px[valid]]=scores[valid]
        accepted+=int(valid.sum());observed+=int((valid&(classes==1)).sum());inferred+=int((valid&(classes==2)).sum())
    if not accepted:raise ValueError('No direct texels transferred')
    # Texture gutter only; provenance outside the facial chart remains zero.
    written=pi>=0
    for _ in range(8):
        sums=np.zeros(result.shape,np.float32);count=np.zeros(written.shape,np.uint8)
        for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
            mask=np.roll(written,(dy,dx),(0,1));sums+=np.roll(result,(dy,dx),(0,1))*mask[:,:,None];count+=mask
        fill=~written&(count>0);result[fill]=np.rint(sums[fill]/count[fill,None]).astype(np.uint8);written|=fill
    Image.fromarray(result).save(output/'face_atlas.png');Image.fromarray(provenance).save(output/'provenance.png');Image.fromarray(out_conf).save(output/'confidence.png')
    pins={str(p):sha(p) for p in [template,template.parent/'source_aligned.npz',template.parent/'baseline.npz',transfer/'patch_data.npz',transfer/'face_atlas.png',projection_path,texture_path]}
    if labels is not None:pins.update({str(texture_path.parent/p):sha(texture_path.parent/p) for p in ['source-labels.png','source-confidence.png']})
    report={'policy':'texel_surface_color_transfer_v1','source_kind':kind,'code_sha256':sha(Path(__file__)),'input_sha256':pins,'source_shift_srgb':shift.tolist(),'original_unbounded_shift_srgb':raw_shift.tolist(),'tone_limit_srgb':tone_limit,'source_skin_luma':skin_luma,'direct_texels':accepted,'observed_source_texels':observed,'inferred_or_low_support_source_texels':inferred,'canonical_core_rejected':canonical_rejected,'unchanged_fallback_texels':int((provenance==3).sum()),'labels':{'0':'outside patch/template','1':'observed source with sufficient support','2':'inferred or low support RDF source; original label maps retained','3':'baseline vertex-baked fallback, not newly certified observed'},'confidence_note':'RDF: conservative source four-texel minimum; inferred capped at 13. Photo projection has no texel confidence and encodes 0, not a probability.','geometry_changed':False,'harmonization':'0.75 cheek/luminance policy, optional explicit RGB offset cap; applied per texel','lighting_removed':False,'likeness_verified':False,'ready_for_machine':False,'atlas_sha256':sha(output/'face_atlas.png')}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('DIRECT_TEXELS',accepted,kind,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--transfer',required=True);p.add_argument('--output',required=True);p.add_argument('--tone-limit',type=float);p.add_argument('--runtime',required=True);a=p.parse_args();run(a.template,a.transfer,a.output,a.tone_limit,a.runtime)
