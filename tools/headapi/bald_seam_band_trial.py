"""Two-sided low-frequency seam correction on a verified combined head."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import trimesh
from scipy.spatial import cKDTree
from scripts.eye_components import identify
from tools.headapi.demo_appearance import atlas_positions
from tools.headapi.bald_transition_trial import render_gutter,boundary_metric
from tools.headapi.bald_tone_trial import smoothstep,bounded_rgb
from tools.headapi.texture_completion_trial import export_texture
from tools.headapi.service import sha,write

POLICY='bald_two_sided_seam_v2'


def seam_endpoints(estimated,observed,positions,max_distance,normals,face_index):
    seam=np.zeros(estimated.shape,np.uint8)
    for dy,dx in [(0,1),(1,0)]:
        a=(slice(None,-1),slice(None)) if dy else (slice(None),slice(None,-1))
        b=(slice(1,None),slice(None)) if dy else (slice(None),slice(1,None))
        y,x=np.where((estimated[a]&observed[b])|(observed[a]&estimated[b]))
        near=np.linalg.norm(positions[y,x]-positions[y+dy,x+dx],axis=1)<max_distance
        facing=np.einsum('ni,ni->n',normals[face_index[y,x]],normals[face_index[y+dy,x+dx]])>.3
        y,x=y[near&facing],x[near&facing]
        seam[y,x]=1;seam[y+dy,x+dx]=1
    return seam


def low_frequency_delta(atlas,estimated,observed,sigma=10):
    """Change the low-frequency mean across a seam, retain source detail residual."""
    eligible=estimated|observed
    def blur(x):return cv2.GaussianBlur(x,(0,0),sigma,borderType=cv2.BORDER_CONSTANT)
    total=blur(eligible.astype(np.float32));a=blur(estimated.astype(np.float32));b=blur(observed.astype(np.float32))
    delta=np.zeros(atlas.shape,np.float32)
    for c in range(3):
        channel=atlas[...,c].astype(np.float32)
        common=blur(channel*eligible)/np.maximum(total,1e-6)
        own=np.where(estimated,blur(channel*estimated)/np.maximum(a,1e-6),blur(channel*observed)/np.maximum(b,1e-6))
        delta[...,c]=common-own
    delta[~eligible]=0
    return delta


def paired_residuals(atlas,estimated,observed,positions,max_distance,normals,face_index):
    width=estimated.shape[1];ei=[];oi=[];differences=[]
    for dy,dx in [(0,1),(1,0)]:
        a=(slice(None,-1),slice(None)) if dy else (slice(None),slice(None,-1))
        b=(slice(1,None),slice(None)) if dy else (slice(None),slice(1,None))
        for reverse in [False,True]:
            mask=(observed[a]&estimated[b]) if reverse else (estimated[a]&observed[b])
            y,x=np.where(mask);ay,ax=y+dy,x+dx
            near=np.linalg.norm(positions[y,x]-positions[ay,ax],axis=1)<max_distance
            same=np.einsum('ni,ni->n',normals[face_index[y,x]],normals[face_index[ay,ax]])>.3
            y,x,ay,ax=y[near&same],x[near&same],ay[near&same],ax[near&same]
            ey,ex,oy,ox=(ay,ax,y,x) if reverse else (y,x,ay,ax)
            ei.append(ey*width+ex);oi.append(oy*width+ox)
            differences.append(atlas[oy,ox].astype(float)-atlas[ey,ex].astype(float))
    difference=np.concatenate(differences)
    def aggregate(parts,correction):
        unique,inverse=np.unique(np.concatenate(parts),return_inverse=True)
        sums=np.zeros((len(unique),3));np.add.at(sums,inverse,correction)
        counts=np.bincount(inverse,minlength=len(unique))
        return unique//width,unique%width,sums/np.maximum(counts[:,None],1)
    return aggregate(ei,.75*difference),aggregate(oi,-.25*difference)


def surface_band_delta(atlas,estimated,observed,positions,normals,face_index,height):
    result=np.zeros(atlas.shape,np.float32)
    anchors=paired_residuals(atlas,estimated,observed,positions,.01*height,normals,face_index)
    for region,(ay,ax,correction) in zip([estimated,observed],anchors):
        if len(ay)<8:raise ValueError('Too few paired surface anchors')
        tree=cKDTree(positions[ay,ax]);anchor_normals=normals[face_index[ay,ax]]
        yy,xx=np.where(region)
        for start in range(0,len(yy),8192):
            y,x=yy[start:start+8192],xx[start:start+8192]
            distance,index=tree.query(positions[y,x],k=min(12,len(ay)))
            accepted=np.einsum('nki,ni->nk',anchor_normals[index],normals[face_index[y,x]])>.4
            accepted&=distance<.03*height
            weight=accepted/np.maximum(distance,.001*height)**2
            weight/=np.maximum(weight.sum(1,keepdims=True),1e-20)
            nearest=np.min(np.where(accepted,distance,np.inf),axis=1)
            fade=(1-smoothstep(nearest,0,.03*height))[:,None]
            result[y,x]=np.einsum('ni,nij->nj',weight,correction[index])*fade
        result[ay,ax]=correction
    return result


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((source/'report.json').read_text());evidence=json.loads((source/'verification.json').read_text())
    if parent.get('status')!='COMBINED_HEAD_REVIEW_READY' or evidence.get('status')!='COMBINED_HEAD_VERIFIED':raise ValueError('Unverified combined baseline')
    if sha(source/'verification.json')!=parent['verification_sha256']:raise ValueError('Verification changed')
    for name,digest in evidence['final_sha256'].items():
        if sha(source/'final'/name)!=digest:raise ValueError('Combined artifact changed')
    tone=json.loads((source/'03_tone/report.json').read_text());pins=dict(tone['input_sha256'])
    pins.update({str(source/p):sha(source/p) for p in ['report.json','verification.json','final/head.glb','final/appearance-atlas.png','final/labels.png','final/confidence.png']})
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Input source changed')
    bundle=next(Path(p) for p in pins if Path(p).name=='projection_bundle.npz')
    asset=next(Path(p) for p in pins if Path(p).name=='flame_assets.npz')
    quality=next(Path(p) for p in pins if Path(p).name=='quality_maps.npz')
    with np.load(bundle,allow_pickle=False) as b:v,f=b['vertices'],b['faces']
    with np.load(asset,allow_pickle=False) as b:
        if not np.array_equal(f,b['faces']):raise ValueError('UV topology mismatch')
        uv=b['uvs'][b['uv_faces']]
    with np.load(quality,allow_pickle=False) as b:maps={k:b[k] for k in ['face_index','skin','protected']}
    atlas=np.array(Image.open(source/'final/appearance-atlas.png').convert('RGB'))
    labels=np.array(Image.open(source/'final/labels.png'));confidence=np.array(Image.open(source/'final/confidence.png'))
    fi=maps['face_index'];valid=fi>=0;face_labels,anatomy=identify(v,f)
    head=valid&(face_labels[np.maximum(fi,0)]==0)
    pos=atlas_positions(fi,v,f,uv);height=np.ptp(v[:,1]);eye=np.mean(anatomy['eye_centers_native'],axis=0)
    x,y,z=np.moveaxis(pos,-1,0)
    central=(np.abs(x-eye[0])<.12*height)&(y>eye[1]-.17*height)&(z>eye[2]-.06*height)
    inferred=np.isin(labels,[2,5,6,8]);observed=np.isin(labels,[1,7])&(maps['skin']>0)
    protected=(maps['protected']>0)&~inferred
    eligible=head&(inferred|observed)&~central&~protected
    estimated=inferred&eligible;observed&=eligible
    normals=trimesh.Trimesh(v,f,process=False).face_normals
    seam=seam_endpoints(estimated,observed,pos,.01*height,normals,fi)
    if seam.sum()<32:raise ValueError('Insufficient spatially valid seam anchors')
    delta=surface_band_delta(atlas,estimated,observed,pos,normals,fi,height)
    weight=eligible.astype(np.float32)
    variants={}
    for name,strength,limit in [('gentle',.6,18),('wide',1.,30)]:
        folder=output/name;folder.mkdir();result=atlas.copy()
        yy,xx=np.where(weight>0)
        result[yy,xx]=bounded_rgb(atlas[yy,xx],atlas[yy,xx].astype(float)+delta[yy,xx]*weight[yy,xx,None]*strength,limit=limit)
        changed=np.any(result!=atlas,axis=-1)
        if not np.array_equal(result[~eligible|central|protected],atlas[~eligible|central|protected]):raise ValueError('Protected texture changed')
        new_labels=labels.copy();new_labels[changed&~inferred]=9;new_labels[changed&inferred]=10
        new_conf=confidence.copy();new_conf[changed&inferred]=13
        padded,_=render_gutter(result,valid)
        Image.fromarray(result).save(folder/'head-atlas.png');Image.fromarray(new_labels).save(folder/'labels.png');Image.fromarray(new_conf).save(folder/'confidence.png')
        Image.fromarray((changed*255).astype(np.uint8)).save(folder/'changed-mask.png')
        export_texture(v,f,uv,padded,folder/'head.glb')
        variants[name]=dict(strength=strength,channel_limit=limit,changed_texels=int(changed.sum()),
                            changed_observed_skin_texels=int((changed&~inferred).sum()),max_rgb_channel_delta=int(np.abs(result.astype(np.int16)-atlas.astype(np.int16)).max()),
                            boundary_before=boundary_metric(atlas,estimated,observed),boundary_after=boundary_metric(result,estimated,observed),
                            head_sha256=sha(folder/'head.glb'))
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Input modified during export')
    report=dict(status='SEAM_BAND_CANDIDATES_EXPORTED',policy=POLICY,input_sha256=pins,code_sha256=sha(Path(__file__)),
                variants=variants,seam_endpoint_texels=int(seam.sum()),geometry_changed=False,protected_central_byte_equal=True,
                ready_for_machine=False,lighting_removed=False,likeness_verified=False,
                provenance={'9':'observed skin with low-frequency seam adjustment','10':'estimated skin with low-frequency seam adjustment'},
                method='paired boundary residuals; estimated side 75%, observed side 25%; 3D-normal guarded continuation',
                limitations=['Euclidean distance and normal guards do not prove geodesic/anatomical correspondence',
                             'Broad observed/inferred skin boundary colors are cosmetic estimates, not albedo',
                             'Boundary metrics reuse optimized edges; no independent fidelity validation',
                             'Feature and central protection can leave unresolved discontinuities'])
    write(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.source,a.output)
