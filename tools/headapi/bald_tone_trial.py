"""Bounded low-frequency cosmetic tone correction; not intrinsic decomposition."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from scripts.eye_components import identify
from tools.headapi.demo_appearance import atlas_positions
from tools.headapi.bald_transition_trial import render_gutter,boundary_metric
from tools.headapi.texture_completion_trial import export_texture
from tools.headapi.service import sha,write

POLICY='bald_low_frequency_tone_v2'


def smoothstep(value,lo,hi):
    t=np.clip((value-lo)/(hi-lo),0,1)
    return t*t*(3-2*t)


def correction_field(vertices,faces,points,lab,confidence,reference):
    ids=cKDTree(vertices).query(points)[1];n=len(vertices)
    count=np.bincount(ids,weights=confidence,minlength=n)
    sums=np.stack([np.bincount(ids,weights=confidence*lab[:,i],minlength=n) for i in range(3)],axis=1)
    mean=np.divide(sums,count[:,None],out=np.zeros_like(sums),where=count[:,None]>0)
    anchor=np.minimum(count/10,1)*4
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    i,j=edges.T
    adjacency=coo_matrix((np.ones(len(i)*2),(np.r_[i,j],np.r_[j,i])),shape=(n,n)).tocsr()
    lap=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    field=spsolve((8*lap+diags(anchor+.03)).tocsc(),anchor[:,None]*(reference-mean))
    if not np.isfinite(field).all():raise ValueError('Invalid tone field')
    return np.clip(field,[-4,-3,-3],[4,3,3])


def bounded_rgb(base,proposed,limit=12):
    return np.clip(np.rint(base.astype(float)+np.clip(proposed.astype(float)-base,-limit,limit)),0,255).astype(np.uint8)


def tone_weights(skin,inferred,envelope):
    observed_distance=cv2.distanceTransform(skin.astype(np.uint8),cv2.DIST_L2,5)
    inferred_distance=cv2.distanceTransform(inferred.astype(np.uint8),cv2.DIST_L2,5)
    # Freeze both sides of existing boundaries; do not undo the previous seam correction.
    feather=np.where(inferred,smoothstep(inferred_distance,2,12),smoothstep(observed_distance,2,12))
    return feather*np.nan_to_num(envelope)


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    prior=json.loads((source/'report.json').read_text())
    if prior.get('status')!='BALD_TRANSITION_EXPORTED_FOR_REVIEW':raise ValueError('Expected verified transition baseline')
    pins=dict(prior['input_sha256'])
    pins.update({str(source/p):sha(source/p) for p in ['report.json','head-atlas.png','head.glb','confidence.png','provenance-labels.png']})
    if sha(source/'head.glb')!=prior['head_sha256']:raise ValueError('Baseline GLB changed')
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Source pin mismatch')
    bundle=next(Path(p) for p in pins if Path(p).name=='projection_bundle.npz')
    assets=next(Path(p) for p in pins if Path(p).name=='flame_assets.npz')
    quality=next(Path(p) for p in pins if Path(p).name=='quality_maps.npz')
    with np.load(bundle,allow_pickle=False) as b:v,faces=b['vertices'],b['faces']
    with np.load(assets,allow_pickle=False) as b:
        if not np.array_equal(faces,b['faces']):raise ValueError('Topology mismatch')
        uv=b['uvs'][b['uv_faces']]
    face_labels,anatomy=identify(v,faces)
    with np.load(quality,allow_pickle=False) as b:maps={k:b[k] for k in ['face_index','skin','protected','confidence']}
    atlas=np.array(Image.open(source/'head-atlas.png').convert('RGB'))
    labels=np.array(Image.open(source/'provenance-labels.png'));confidence=np.array(Image.open(source/'confidence.png'))
    pos=atlas_positions(maps['face_index'],v,faces,uv)
    valid=maps['face_index']>=0;head=valid&(face_labels[np.maximum(maps['face_index'],0)]==0)
    inferred=(labels==5)|(labels==6)
    skin=head&~inferred&(labels>0)&(maps['skin']>0)&(maps['protected']==0)
    height=float(np.ptp(v[:,1]));eye=np.mean(anatomy['eye_centers_native'],axis=0)
    x,y,z=np.moveaxis(pos,-1,0)
    envelope=np.maximum(smoothstep(np.abs(x-eye[0])/height,.12,.25),smoothstep((eye[1]-y)/height,.17,.30))
    # Protect observed feature semantics plus a 12px band; inferred head colors use same spatial field.
    eligible=head&(inferred|skin)
    weight=tone_weights(skin,inferred,envelope)
    weight[~eligible]=0
    donor=skin&(maps['confidence']>=.5)&(atlas.mean(-1)>55)&(atlas.mean(-1)<225)
    dy,dx=np.where(donor);stride=max(1,int(np.ceil(len(dy)/50000)));dy,dx=dy[::stride],dx[::stride]
    if len(dy)<128:raise ValueError('Insufficient skin anchors')
    donor_lab=cv2.cvtColor(atlas[dy,dx].astype(np.float32)[None]/255,cv2.COLOR_RGB2LAB)[0]
    central=(np.abs(pos[dy,dx,0]-eye[0])<.12*height)&(pos[dy,dx,1]>eye[1]-.16*height)&(pos[dy,dx,1]<eye[1]+.025*height)
    if central.sum()<32:raise ValueError('Insufficient central face reference')
    reference=np.median(donor_lab[central],axis=0)
    field=correction_field(v,faces,pos[dy,dx],donor_lab,maps['confidence'][dy,dx],reference)
    tree=cKDTree(v);yy,xx=np.where(weight>0)
    variants={}
    for name,strength in [('mild',.5),('balanced',1.)]:
        folder=output/name;folder.mkdir();result=atlas.copy()
        for start in range(0,len(yy),16384):
            py,px=yy[start:start+16384],xx[start:start+16384]
            distances,ids=tree.query(pos[py,px],k=4)
            w=1/np.maximum(distances,.001*height)**2;w/=w.sum(1,keepdims=True)
            delta=np.einsum('ni,nij->nj',w,field[ids])*weight[py,px,None]*strength
            lab=cv2.cvtColor(atlas[py,px].astype(np.float32)[None]/255,cv2.COLOR_RGB2LAB)[0]
            rgb=cv2.cvtColor((lab+delta.astype(np.float32))[None],cv2.COLOR_LAB2RGB)[0]*255
            result[py,px]=bounded_rgb(atlas[py,px],rgb)
        changed=np.any(result!=atlas,axis=-1)
        if not np.array_equal(result[weight==0],atlas[weight==0]):raise ValueError('Feature/central face protection failed')
        new_labels=labels.copy();new_labels[changed&~inferred]=7;new_labels[changed&inferred]=8
        new_conf=confidence.copy();new_conf[changed&inferred]=13
        Image.fromarray(result).save(folder/'head-atlas.png');Image.fromarray(new_labels).save(folder/'provenance-labels.png');Image.fromarray(new_conf).save(folder/'confidence.png')
        Image.fromarray((changed*255).astype(np.uint8)).save(folder/'changed-mask.png')
        padded,_=render_gutter(result,valid);export_texture(v,faces,uv,padded,folder/'head.glb')
        before=boundary_metric(atlas,inferred,skin);after=boundary_metric(result,inferred,skin)
        rgb_delta=np.abs(result.astype(np.int16)-atlas.astype(np.int16))
        variants[name]=dict(strength=strength,changed_texels=int(changed.sum()),changed_observed_skin_texels=int((changed&~inferred).sum()),
                            max_rgb_channel_delta=int(rgb_delta.max()),protected_and_central_byte_equal=True,
                            boundary_before=before,boundary_after=after,head_sha256=sha(folder/'head.glb'))
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Source modified during experiment')
    report=dict(status='TONE_CANDIDATES_EXPORTED_FOR_REVIEW',policy=POLICY,input_sha256=pins,code_sha256=sha(Path(__file__)),
                geometry_changed=False,triangle_coordinates_float32_roundtrip_equal=True,lighting_removed=False,albedo_recovered=False,
                ready_for_machine=False,likeness_verified=False,reference_lab=reference.tolist(),variants=variants,
                provenance={'7':'observed skin with bounded cosmetic tone adjustment','8':'inferred skin with bounded cosmetic tone adjustment'},
                limitations=['Reference skin tone is a heuristic from the same capture, not calibrated truth','Natural shading may be reduced despite conservative bounds',
                             'Protected observed features and central face remain unchanged; original shadows remain there','UV mask feathering can leave visible transitions',
                             'Boundary metrics reuse fitting data; no independent realism validation'])
    write(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
