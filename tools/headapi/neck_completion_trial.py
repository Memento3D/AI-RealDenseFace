"""Explicit estimated neck presentation; never relabels synthesized skin as observed."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scripts.eye_components import identify
from tools.headapi.demo_appearance import atlas_positions
from tools.headapi.bald_head_trial import surface_colors
from tools.headapi.bald_tone_trial import smoothstep
from tools.headapi.neck_semantic_trial import barycentric_pixels
from tools.headapi.bald_transition_trial import render_gutter,boundary_metric
from tools.headapi.texture_completion_trial import export_texture
from tools.headapi.service import sha,write

POLICY='estimated_neck_completion_v1'


def neck_blend(fraction,eligible,start,end):
    if not 0<start<end:raise ValueError('Invalid neck blend interval')
    return np.where(eligible,np.nan_to_num(smoothstep(fraction,start,end)),0)


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((source/'report.json').read_text())
    if parent.get('status')!='NECK_SEMANTIC_CANDIDATES_EXPORTED':raise ValueError('Unverified semantic parent')
    selected='neck_balance'
    pins=dict(parent['input_sha256']);pins.update({str(source/p):sha(source/p) for p in ['report.json','certified-neck.png',selected+'/head.glb',selected+'/head-atlas.png',selected+'/labels.png',selected+'/confidence.png']})
    if sha(source/selected/'head.glb')!=parent['variants'][selected]['head_sha256']:raise ValueError('Parent GLB changed')
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Input binding changed')
    bundle=next(Path(p) for p in pins if Path(p).name=='projection_bundle.npz');asset=next(Path(p) for p in pins if Path(p).name=='flame_assets.npz');quality=next(Path(p) for p in pins if Path(p).name=='quality_maps.npz')
    with np.load(bundle,allow_pickle=False) as b:v,f=b['vertices'],b['faces']
    with np.load(asset,allow_pickle=False) as b:
        if not np.array_equal(f,b['faces']):raise ValueError('UV topology changed')
        uv=b['uvs'][b['uv_faces']]
    with np.load(quality,allow_pickle=False) as b:maps={k:b[k] for k in ['face_index','skin','protected','confidence']}
    atlas=np.array(Image.open(source/selected/'head-atlas.png').convert('RGB'))
    labels=np.array(Image.open(source/selected/'labels.png'));confidence=np.array(Image.open(source/selected/'confidence.png'))
    certified=np.asarray(Image.open(source/'certified-neck.png'))>0
    fi=maps['face_index'];valid=fi>=0;face_labels,anatomy=identify(v,f)
    head=valid&(face_labels[np.maximum(fi,0)]==0);pos=atlas_positions(fi,v,f,uv)
    eye=np.mean(anatomy['eye_centers_native'],axis=0);height=np.ptp(v[:,1]);fraction=(eye[1]-pos[...,1])/height
    inferred=np.isin(labels,[2,5,6,8,10,12]);known=np.isin(labels,[1,7,9,11])
    eligible=head&(inferred|certified)&(fraction>.30)
    donors=head&known&(maps['skin']>0)&(maps['protected']==0)&(maps['confidence']>.45)&(fraction>.12)&(fraction<.30)&(atlas.mean(-1)>60)&(atlas.mean(-1)<225)
    yy,xx=np.where(donors);step=max(1,int(np.ceil(len(yy)/50000)));yy,xx=yy[::step],xx[::step]
    if len(yy)<128:raise ValueError('Insufficient jaw skin for estimated neck')
    colors=atlas[yy,xx].astype(float);reference=np.median(colors,axis=0)
    field=surface_colors(v,f,pos[yy,xx],colors,maps['confidence'][yy,xx],reference)
    # Tiny source detail residual only where observed skin has reliable local support.
    detail_support=certified&known&(maps['confidence']>=.5)
    detail_support=cv2.erode(detail_support.astype(np.uint8),np.ones((9,9),np.uint8))>0
    detail=np.clip(atlas.astype(np.float32)-cv2.GaussianBlur(atlas.astype(np.float32),(0,0),2),-3,3)
    detail[~detail_support]=0
    variants={}
    for name,start,end in [('soft',.35,.65),('clean',.30,.52)]:
        folder=output/name;folder.mkdir();result=atlas.copy();alpha=neck_blend(fraction,eligible,start,end)
        py,px=np.where(alpha>0)
        for offset in range(0,len(py),32768):
            y,x=py[offset:offset+32768],px[offset:offset+32768];tri=fi[y,x]
            weights=barycentric_pixels(y,x,tri,uv,atlas.shape[0]);weights=np.maximum(weights,0);weights/=weights.sum(1,keepdims=True)
            target=.7*np.einsum('ni,nij->nj',weights,field[f[tri]])+.3*reference+detail[y,x]
            a=alpha[y,x,None];result[y,x]=np.clip(np.rint((1-a)*atlas[y,x]+a*target),0,255).astype(np.uint8)
        changed=np.any(result!=atlas,axis=-1)
        if not np.array_equal(result[~eligible|(fraction<=.30)],atlas[~eligible|(fraction<=.30)]):raise ValueError('Face or non-neck changed')
        new_labels=labels.copy();new_labels[changed]=13
        new_conf=confidence.copy();new_conf[changed]=13
        Image.fromarray(result).save(folder/'head-atlas.png');Image.fromarray(new_labels).save(folder/'labels.png');Image.fromarray(new_conf).save(folder/'confidence.png')
        Image.fromarray(np.clip(alpha*255,0,255).astype(np.uint8)).save(folder/'estimated-blend.png')
        padded,_=render_gutter(result,valid);export_texture(v,f,uv,padded,folder/'head.glb')
        variants[name]=dict(blend_start_fraction=start,blend_end_fraction=end,changed_texels=int(changed.sum()),
                            changed_previously_observed_texels=int((changed&known).sum()),max_rgb_delta=int(np.abs(result.astype(np.int16)-atlas.astype(np.int16)).max()),
                            boundary_before=boundary_metric(atlas,inferred,certified&known),boundary_after=boundary_metric(result,inferred,certified&known),head_sha256=sha(folder/'head.glb'))
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Source changed during completion')
    report=dict(status='ESTIMATED_NECK_EXPORTED_FOR_REVIEW',policy=POLICY,input_sha256=pins,code_sha256=sha(Path(__file__)),
                variants=variants,reference_rgb=reference.tolist(),jaw_donor_samples=len(yy),geometry_changed=False,face_above_neck_byte_equal=True,
                provenance={'13':'estimated presentation skin, including replacement of observed neck color; not recovered texture'},
                confidence='all changed texels 13/255 heuristic, even if previously observed; not probability',
                ready_for_machine=False,likeness_verified=False,lighting_removed=False,
                limitations=['Observed neck shadows/colors are intentionally replaced for a clean presentation',
                             'Scalp/neck appearance is estimated; no invisible pores or identity detail recovered',
                             'Jaw color prior is not calibrated albedo; original face lighting remains',
                             'Semantic and geometry ROI errors can leave unmodified areas; not print validation'])
    write(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.source,a.output)
