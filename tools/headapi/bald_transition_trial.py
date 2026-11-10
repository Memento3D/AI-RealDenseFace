"""Conservative inferred-skin transitions; observed face pixels remain untouched."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

from tools.headapi.demo_appearance import atlas_positions
from tools.headapi.service import sha, write
from tools.headapi.texture_completion_trial import export_texture

POLICY = 'bald_skin_transition_v3'


def bounded_transition(base, target, distance, radius, strength=.75, limit=40):
    if radius<=0 or not 0<=strength<=1:
        raise ValueError('Invalid transition parameters')
    weight=np.clip(1-np.asarray(distance)/radius,0,1)
    weight=weight*weight*(3-2*weight)*strength
    return np.clip(np.rint(np.asarray(base,float)+np.clip(np.asarray(target,float)-base,-limit,limit)*weight[...,None]),0,255).astype(np.uint8)


def render_gutter(atlas, valid, radius=4):
    """Extend colors only outside UV charts for texture filtering, not reconstruction."""
    result=atlas.copy();known=valid.copy()
    for _ in range(radius):
        count=cv2.boxFilter(known.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_CONSTANT)
        fringe=~known & (count>0)
        if not fringe.any():break
        # Process one channel at a time instead of allocating multiple full RGB float arrays.
        for channel in range(3):
            sums=cv2.boxFilter(result[...,channel].astype(np.float32)*known,-1,(3,3),normalize=False,borderType=cv2.BORDER_CONSTANT)
            result[...,channel][fringe]=np.rint(sums[fringe]/count[fringe]).astype(np.uint8)
        known[fringe]=True
    return result,known & ~valid


def boundary_metric(atlas, fill, safe):
    values=[]
    for axis in [0,1]:
        a=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
        b=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
        edge=(fill[a]&safe[b]) | (safe[a]&fill[b])
        values.extend(np.linalg.norm(atlas[a][edge].astype(float)-atlas[b][edge].astype(float),axis=-1).tolist())
    return dict(count=len(values),median=float(np.median(values)) if values else None,
                p90=float(np.percentile(values,90)) if values else None)


def boundary_corrections(atlas, fill, safe, positions=None, max_distance=None):
    """Residual anchors on inferred texels touching retained observed skin."""
    height,width=fill.shape;ids=[];deltas=[]
    for dy,dx in [(0,1),(1,0),(0,-1),(-1,0)]:
        neighbor=np.roll(safe,(-dy,-dx),axis=(0,1))
        if dy==1:neighbor[-1]=False
        if dy==-1:neighbor[0]=False
        if dx==1:neighbor[:,-1]=False
        if dx==-1:neighbor[:,0]=False
        y,x=np.where(fill & neighbor)
        if positions is not None:
            if max_distance is None or max_distance<=0:raise ValueError('Positive surface distance required')
            nearby=np.linalg.norm(positions[y+dy,x+dx]-positions[y,x],axis=1)<max_distance
            y,x=y[nearby],x[nearby]
        ids.append(y*width+x)
        deltas.append(atlas[y+dy,x+dx].astype(float)-atlas[y,x].astype(float))
    ids=np.concatenate(ids);deltas=np.concatenate(deltas)
    unique,inverse=np.unique(ids,return_inverse=True)
    sums=np.zeros((len(unique),3));np.add.at(sums,inverse,deltas)
    counts=np.bincount(inverse,minlength=len(unique))
    return unique//width,unique%width,sums/np.maximum(counts[:,None],1)


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    prior=json.loads((source/'report.json').read_text())
    if prior.get('status')!='BALD_HEAD_EXPORTED_FOR_REVIEW':raise ValueError('Invalid bald head baseline')
    pins=prior['input_sha256']
    for p,digest in pins.items():
        if sha(Path(p))!=digest:raise ValueError('Original source changed')
    bundle=next(Path(p) for p in pins if Path(p).name=='projection_bundle.npz')
    assets=next(Path(p) for p in pins if Path(p).name=='flame_assets.npz')
    quality=next(Path(p) for p in pins if Path(p).name=='quality_maps.npz')
    source_pins={str(source/p):sha(source/p) for p in ['report.json','head.glb','head-atlas.png','provenance-labels.png','confidence.png']}
    if sha(source/'head.glb')!=prior['head_sha256']:raise ValueError('Baseline head mismatch')
    with np.load(bundle,allow_pickle=False) as b:v,faces=b['vertices'],b['faces']
    with np.load(assets,allow_pickle=False) as b:
        if not np.array_equal(faces,b['faces']):raise ValueError('Topology mismatch')
        uv=b['uvs'][b['uv_faces']]
    with np.load(quality,allow_pickle=False) as b:maps={k:b[k] for k in ['face_index','skin','protected','confidence']}
    atlas=np.array(Image.open(source/'head-atlas.png').convert('RGB'));result=atlas.copy()
    labels=np.array(Image.open(source/'provenance-labels.png'));confidence=np.array(Image.open(source/'confidence.png'))
    fill=labels==5;valid=maps['face_index']>=0
    positions=atlas_positions(maps['face_index'],v,faces,uv)
    safe=valid & ~fill & (labels>0) & (maps['skin']>0) & (maps['protected']==0) & (maps['confidence']>=.5)
    evaluation_skin=safe.copy()
    height=float(np.ptp(v[:,1]))
    dy,dx,donor_residual=boundary_corrections(atlas,fill,safe,positions,.01*height)
    if len(dy)<32:raise ValueError('Too few reliable skin boundary anchors')
    tree=cKDTree(positions[dy,dx])
    mesh=trimesh.Trimesh(v,faces,process=False);normals=mesh.face_normals
    donor_normals=normals[maps['face_index'][dy,dx]]
    radius=.045*height
    fy,fx=np.where(fill)
    for start in range(0,len(fy),8192):
        y,x=fy[start:start+8192],fx[start:start+8192]
        distance,index=tree.query(positions[y,x],k=16)
        similarity=np.einsum('nki,ni->nk',donor_normals[index],normals[maps['face_index'][y,x]])
        accepted=(similarity>.4)&(distance<radius)
        weight=accepted/np.maximum(distance,.001*height)**2
        total=weight.sum(1);supported=total>0
        weight/=np.maximum(total[:,None],1e-20)
        target=atlas[y,x].astype(float)+np.einsum('nk,nkj->nj',weight,donor_residual[index])
        nearest=np.min(np.where(accepted,distance,np.inf),axis=1)
        corrected=bounded_transition(atlas[y,x],target,nearest,radius)
        result[y[supported],x[supported]]=corrected[supported]
    # Apply each boundary's own residual, rather than averaging unrelated nearby skin.
    result[dy,dx]=bounded_transition(atlas[dy,dx],atlas[dy,dx].astype(float)+donor_residual,np.zeros(len(dy)),radius)
    changed=np.any(result!=atlas,axis=-1)
    if not np.array_equal(result[~fill],atlas[~fill]):raise ValueError('Observed face changed')
    labels[changed]=6;confidence[changed]=13
    padded,gutter=render_gutter(result,valid)
    Image.fromarray(result).save(output/'head-atlas.png')
    Image.fromarray(padded).save(output/'render-atlas.png')
    Image.fromarray(labels).save(output/'provenance-labels.png')
    Image.fromarray(confidence).save(output/'confidence.png')
    Image.fromarray(gutter.astype(np.uint8)*255).save(output/'gutter-mask.png')
    export_texture(v,faces,uv,padded,output/'head.glb')
    before=boundary_metric(atlas,fill,evaluation_skin);after=boundary_metric(result,fill,evaluation_skin)
    for p,digest in {**pins,**source_pins}.items():
        if sha(Path(p))!=digest:raise ValueError('Source modified during transition')
    report=dict(status='BALD_TRANSITION_EXPORTED_FOR_REVIEW',policy=POLICY,
                geometry_changed=False,triangle_coordinates_float32_roundtrip_equal=True,
                observed_texture_byte_equal=True,changed_inferred_texels=int(changed.sum()),
                max_channel_change=int(np.abs(result.astype(np.int16)-atlas.astype(np.int16)).max()),
                boundary_before=before,boundary_after=after,gutter_texels=int(gutter.sum()),
                boundary_anchors=len(dx),method='observed-boundary residual continuation with normal guard',radius_fraction_height=.045,strength=.75,channel_limit=40,
                label_6='estimated skin transition, not observed or calibrated albedo',
                gutter='4px outside UV only; render filtering aid, not reconstructed skin',
                lighting_removed=False,likeness_verified=False,ready_for_machine=False,
                limitations=['Photographic illumination retained in observed face',
                             'Geometry and semantic masks approximate; normal guard may reject donors',
                             'Boundary metric is color continuity only, not fidelity',
                             'Glasses/occluded eyes unchanged; no new skin detail recovered'],
                input_sha256={**pins,**source_pins},code_sha256=sha(Path(__file__)),head_sha256=sha(output/'head.glb'))
    write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.source,a.output)
