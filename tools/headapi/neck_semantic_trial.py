"""Region-limited neck semantic correction; original face protections remain intact."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import trimesh
from scripts.eye_components import identify
from tools.headapi.demo_appearance import atlas_positions
from tools.headapi.hair_parser import HairParser,PARSER_SHA
from tools.headapi.bald_tone_trial import smoothstep,bounded_rgb
from tools.headapi.bald_seam_band_trial import surface_band_delta,seam_endpoints
from tools.headapi.bald_transition_trial import render_gutter,boundary_metric
from tools.headapi.texture_completion_trial import export_texture
from tools.headapi.service import sha,write

POLICY='neck_semantics_v1'


def neck_certificate(dominant,owner,positions,head,eye_y,height):
    return head & (owner>0) & np.isin(dominant,[1,14]) & (positions[...,1]<eye_y-.30*height)


def barycentric_pixels(y,x,fi,uv,size):
    a,b,c=uv[fi].transpose(1,0,2);p=np.c_[(x+.5)/size,1-(y+.5)/size]
    den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    w0=((b[:,1]-c[:,1])*(p[:,0]-c[:,0])+(c[:,0]-b[:,0])*(p[:,1]-c[:,1]))/den
    w1=((c[:,1]-a[:,1])*(p[:,0]-c[:,0])+(a[:,0]-c[:,0])*(p[:,1]-c[:,1]))/den
    return np.c_[w0,w1,1-w0-w1]


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((source/'report.json').read_text())
    if parent.get('status')!='SEAM_BAND_CANDIDATES_EXPORTED':raise ValueError('Expected seam baseline')
    pins=dict(parent['input_sha256']);pins.update({str(source/p):sha(source/p) for p in ['report.json','gentle/head.glb','gentle/head-atlas.png','gentle/labels.png','gentle/confidence.png']})
    if sha(source/'gentle/head.glb')!=parent['variants']['gentle']['head_sha256']:raise ValueError('Source GLB changed')
    bundle=next(Path(p) for p in pins if Path(p).name=='projection_bundle.npz')
    assets=next(Path(p) for p in pins if Path(p).name=='flame_assets.npz')
    quality=next(Path(p) for p in pins if Path(p).name=='quality_maps.npz')
    raw=json.loads((quality.parent/'report.json').read_text())
    parser_path=next(Path(p) for p in raw['input_sha256'] if Path(p).name=='resnet18.onnx')
    if sha(parser_path)!=PARSER_SHA or raw['input_sha256'][str(parser_path)]!=PARSER_SHA:raise ValueError('Parser pin changed')
    pins[str(parser_path)]=PARSER_SHA
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Source binding changed')
    with np.load(bundle,allow_pickle=False) as b:v,f,views,cameras=b['vertices'],b['faces'],b['view_vertices'],b['cameras']
    with np.load(assets,allow_pickle=False) as b:
        if not np.array_equal(f,b['faces']):raise ValueError('UV topology mismatch')
        uv=b['uvs'][b['uv_faces']]
    with np.load(quality,allow_pickle=False) as b:maps={k:b[k] for k in ['face_index','skin','protected','confidence']}
    owner_path=quality.parent/'coverage.png';pins[str(owner_path)]=sha(owner_path)
    owner=np.asarray(Image.open(owner_path))//80
    atlas=np.array(Image.open(source/'gentle/head-atlas.png').convert('RGB'))
    labels=np.array(Image.open(source/'gentle/labels.png'));confidence=np.array(Image.open(source/'gentle/confidence.png'))
    fi=maps['face_index'];valid=fi>=0;face_labels,anatomy=identify(v,f);head=valid&(face_labels[np.maximum(fi,0)]==0)
    pos=atlas_positions(fi,v,f,uv);eye=np.mean(anatomy['eye_centers_native'],axis=0);height=np.ptp(v[:,1])
    neck=head&(pos[...,1]<eye[1]-.30*height)
    dominant=np.zeros(fi.shape,np.uint8);parser=HairParser(parser_path)
    for i,slot in enumerate(['front','left','right']):
        photo=quality.parents[2]/'rdf_crops'/(slot+'.png')
        if sha(photo)!=raw['input_sha256'][str(photo)]:raise ValueError('Crop changed')
        pins[str(photo)]=sha(photo);image=Image.open(photo).convert('RGB');semantic=parser(image)
        Image.fromarray(semantic).save(output/(slot+'.semantic.png'))
        yy,xx=np.where(neck&(owner==i+1))
        for start in range(0,len(yy),32768):
            y,x=yy[start:start+32768],xx[start:start+32768];tri=fi[y,x]
            bary=barycentric_pixels(y,x,tri,uv,atlas.shape[0]);points=np.einsum('ni,nij->nj',bary,views[i][f[tri]])
            projected=np.c_[points,np.ones(len(points))]@cameras[i].T
            xy=np.floor(projected[:,:2]/projected[:,2,None]).astype(int)
            inside=(projected[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<image.width)&(xy[:,1]>=0)&(xy[:,1]<image.height)
            dominant[y[inside],x[inside]]=semantic[xy[inside,1],xy[inside,0]]
    certified=neck_certificate(dominant,owner,pos,head,eye[1],height)
    inferred=np.isin(labels,[2,5,6,8,10]);known=np.isin(labels,[1,7,9])
    observed=known&certified
    estimated=inferred&neck
    normals=trimesh.Trimesh(v,f,process=False).face_normals
    seam=seam_endpoints(estimated,observed,pos,.01*height,normals,fi)
    if seam.sum()<32:raise ValueError('Insufficient verified neck seams')
    blend=np.nan_to_num(smoothstep((eye[1]-pos[...,1])/height,.30,.40));blend[~neck]=0
    safe_face=head&known&(maps['skin']>0)&(maps['protected']==0)&(maps['confidence']>.35)&((eye[1]-pos[...,1])/height>.15)&((eye[1]-pos[...,1])/height<.30)
    if safe_face.sum()<128 or observed.sum()<128:raise ValueError('Insufficient neck/face references')
    reference=np.median(atlas[safe_face],axis=0);neck_median=np.median(atlas[observed],axis=0)
    offset=np.clip(reference-neck_median,-16,16)*.6
    variants={}
    for name,tone in [('semantic_only',False),('neck_balance',True)]:
        folder=output/name;folder.mkdir();working=atlas.copy()
        if tone:
            y,x=np.where(observed);working[y,x]=bounded_rgb(atlas[y,x],atlas[y,x].astype(float)+offset*blend[y,x,None],limit=12)
        delta=surface_band_delta(working,estimated,observed,pos,normals,fi,height)
        result=working.copy();y,x=np.where(estimated|observed)
        result[y,x]=bounded_rgb(working[y,x],working[y,x].astype(float)+delta[y,x]*.6*blend[y,x,None],limit=18)
        changed=np.any(result!=atlas,axis=-1)
        if not np.array_equal(result[~neck],atlas[~neck]):raise ValueError('Face region changed')
        if (changed&known&~certified).any():raise ValueError('Uncertified observed texture changed')
        new_labels=labels.copy();new_labels[changed&known]=11;new_labels[changed&inferred]=12
        new_conf=confidence.copy();new_conf[changed&inferred]=13
        Image.fromarray(result).save(folder/'head-atlas.png');Image.fromarray(new_labels).save(folder/'labels.png');Image.fromarray(new_conf).save(folder/'confidence.png')
        padded,_=render_gutter(result,valid);export_texture(v,f,uv,padded,folder/'head.glb')
        variants[name]=dict(changed_texels=int(changed.sum()),changed_observed_neck_texels=int((changed&known).sum()),
                            max_rgb_delta=int(np.abs(result.astype(np.int16)-atlas.astype(np.int16)).max()),
                            boundary_before=boundary_metric(atlas,estimated,observed),boundary_after=boundary_metric(result,estimated,observed),head_sha256=sha(folder/'head.glb'))
    Image.fromarray(dominant).save(output/'dominant-neck-semantic.png');Image.fromarray(certified.astype(np.uint8)*255).save(output/'certified-neck.png')
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Source changed during export')
    report=dict(status='NECK_SEMANTIC_CANDIDATES_EXPORTED',policy=POLICY,input_sha256=pins,code_sha256=sha(Path(__file__)),
                certified_neck_texels=int(certified.sum()),previously_protected_neck_texels=int((certified&(maps['protected']>0)).sum()),
                variants=variants,geometry_changed=False,face_above_neck_byte_equal=True,ready_for_machine=False,lighting_removed=False,likeness_verified=False,
                reference_rgb=reference.tolist(),neck_median_rgb=neck_median.tolist(),neck_offset_rgb=offset.tolist(),
                provenance={'11':'observed neck with semantic-aware correction','12':'inferred neck with semantic-aware correction'},
                limitations=['Parser labels and geometry ROI are approximate; not anatomy validation','Dominant-owner reprojection reuses fitted cameras and raster ownership',
                             'Neck tone reference is heuristic, not recovered albedo','Boundary metrics reuse fitted seams; no independent fidelity validation'])
    write(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.source,a.output)
