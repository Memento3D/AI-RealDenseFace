"""Pinned template/semantic ear appearance trial, with explicit inferred provenance."""
import argparse,io,json,pickle
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh
from scripts.eye_components import identify
from tools.headapi.service import ROOT,sha,write
from tools.headapi.hair_parser import HairParser,PARSER_SHA
from tools.headapi.demo_appearance import atlas_positions
from tools.headapi.neck_semantic_trial import barycentric_pixels
from tools.headapi.bald_tone_trial import bounded_rgb,smoothstep
from tools.headapi.bald_transition_trial import render_gutter,boundary_metric
from tools.headapi.texture_completion_trial import export_texture

POLICY='template_ear_appearance_v3'
MASK_SHA='ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492'
MASK_PATH=ROOT/'vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl'


class RestrictedMasks(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name) not in {('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype')}:
            raise ValueError('Unsupported mask pickle class')
        return super().find_class(module,name)


def mask_indices(values,count):
    values=np.asarray(values)
    if values.ndim!=1 or values.dtype.kind not in 'iu' or not len(values):raise ValueError('Invalid mask index array')
    if values.min()<0 or values.max()>=count or len(np.unique(values))!=len(values):raise ValueError('Mask index outside topology or duplicated')
    return values.astype(np.int64)


def ear_observation(region,dominant,owner,known,confidence):
    return region & np.isin(dominant,[7,8]) & (owner>0) & known & (confidence>.15)


def feather_fill(original,target,supported,region,observed=None):
    border=cv2.distanceTransform(region.astype(np.uint8),cv2.DIST_L2,5)
    continuity=supported if observed is None else (supported|observed)
    support=cv2.distanceTransform(continuity.astype(np.uint8),cv2.DIST_L2,5)
    alpha=.8*smoothstep(border,0,12)*smoothstep(support,0,8)
    alpha[~supported]=0
    result=original.copy();y,x=np.where(alpha>0)
    result[y,x]=np.clip(np.rint(original[y,x].astype(float)+(target[y,x].astype(float)-original[y,x])*alpha[y,x,None]),0,255).astype(np.uint8)
    return result,alpha


def load_masks(vertices,faces):
    if sha(MASK_PATH)!=MASK_SHA:raise ValueError('Template ear mask pin changed')
    masks=RestrictedMasks(io.BytesIO(MASK_PATH.read_bytes()),encoding='latin1').load()
    keys=['left_ear','right_ear','face','left_eyeball','right_eyeball']
    masks={k:mask_indices(masks[k],len(vertices)) for k in keys}
    face_labels,anatomy=identify(vertices,faces)
    expected={frozenset(np.unique(faces[face_labels==k]).tolist()) for k in [1,2]}
    actual={frozenset(masks[k].tolist()) for k in ['left_eyeball','right_eyeball']}
    if expected!=actual:raise ValueError('Mask/eyeball topology mismatch')
    height=np.ptp(vertices[:,1]);eye=np.mean(anatomy['eye_centers_native'],axis=0)
    for key in ['left_ear','right_ear']:
        center=vertices[masks[key]].mean(0)
        if np.intersect1d(masks[key],masks['face']).size or not .18<abs(center[0]-eye[0])/height<.40:
            raise ValueError('Ear mask overlaps face or has unexpected placement')
    return masks,face_labels,eye,height


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((source/'report.json').read_text())
    if parent.get('status')!='ESTIMATED_NECK_EXPORTED_FOR_REVIEW':raise ValueError('Expected completed neck baseline')
    pins=dict(parent['input_sha256']);pins.update({str(source/p):sha(source/p) for p in ['report.json','clean/head.glb','clean/head-atlas.png','clean/labels.png','clean/confidence.png']})
    pins[str(MASK_PATH)]=MASK_SHA
    if sha(source/'clean/head.glb')!=parent['variants']['clean']['head_sha256']:raise ValueError('Baseline head changed')
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Bound input changed')
    bundle=next(Path(p) for p in pins if Path(p).name=='projection_bundle.npz');assets=next(Path(p) for p in pins if Path(p).name=='flame_assets.npz');quality=next(Path(p) for p in pins if Path(p).name=='quality_maps.npz')
    raw=json.loads((quality.parent/'report.json').read_text())
    parser_path=next(Path(p) for p in raw['input_sha256'] if Path(p).name=='resnet18.onnx')
    if sha(parser_path)!=PARSER_SHA:raise ValueError('Parser pin changed')
    pins[str(parser_path)]=PARSER_SHA
    with np.load(bundle,allow_pickle=False) as b:v,f,views,cameras=b['vertices'],b['faces'],b['view_vertices'],b['cameras']
    with np.load(assets,allow_pickle=False) as b:
        if not np.array_equal(f,b['faces']):raise ValueError('UV topology mismatch')
        uv=b['uvs'][b['uv_faces']]
    with np.load(quality,allow_pickle=False) as b:maps={k:b[k] for k in ['face_index','skin','protected','confidence']}
    masks,face_labels,eye,height=load_masks(v,f)
    fi=maps['face_index'];valid=fi>=0
    regions={}
    for key in ['left_ear','right_ear']:
        face_ear=np.isin(f,masks[key]).sum(1)>=2
        regions[key]=valid&face_ear[np.maximum(fi,0)]
    region=regions['left_ear']|regions['right_ear']
    pos=atlas_positions(fi,v,f,uv)
    region&=pos[...,1]>eye[1]-.30*height
    for key in regions:regions[key]&=region
    atlas=np.array(Image.open(source/'clean/head-atlas.png').convert('RGB'))
    labels=np.array(Image.open(source/'clean/labels.png'));confidence=np.array(Image.open(source/'clean/confidence.png'))
    owner_path=quality.parent/'coverage.png';pins[str(owner_path)]=sha(owner_path)
    owner=np.asarray(Image.open(owner_path))//80;dominant=np.zeros(fi.shape,np.uint8)
    parser=HairParser(parser_path)
    for i,slot in enumerate(['front','left','right']):
        photo=quality.parents[2]/'rdf_crops'/(slot+'.png')
        if sha(photo)!=raw['input_sha256'][str(photo)]:raise ValueError('Photo pin changed')
        pins[str(photo)]=sha(photo);image=Image.open(photo).convert('RGB');semantic=parser(image)
        Image.fromarray(semantic).save(output/(slot+'.semantic.png'))
        yy,xx=np.where(region&(owner==i+1))
        for start in range(0,len(yy),32768):
            y,x=yy[start:start+32768],xx[start:start+32768];tri=fi[y,x]
            bary=barycentric_pixels(y,x,tri,uv,atlas.shape[0]);points=np.einsum('ni,nij->nj',bary,views[i][f[tri]])
            h=np.c_[points,np.ones(len(points))]@cameras[i].T;xy=np.floor(h[:,:2]/h[:,2,None]).astype(int)
            inside=(h[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<image.width)&(xy[:,1]>=0)&(xy[:,1]<image.height)
            dominant[y[inside],x[inside]]=semantic[xy[inside,1],xy[inside,0]]
    known=np.isin(labels,[1,7,9,11]);inferred=np.isin(labels,[0,2,5,6,8,10,12,13])
    observed=ear_observation(region,dominant,owner,known,maps['confidence'])
    fill_only=atlas.copy();tone=atlas.copy();fill_mask=np.zeros(fi.shape,bool);tone_mask=np.zeros(fi.shape,bool);stats={}
    normals=trimesh.Trimesh(v,f,process=False).face_normals
    skin=valid&known&(maps['skin']>0)&(maps['protected']==0)&(maps['confidence']>.5)&~region
    sy,sx=np.where(skin);step=max(1,int(np.ceil(len(sy)/50000)));sy,sx=sy[::step],sx[::step]
    skin_tree=cKDTree(pos[sy,sx]) if len(sy) else None
    for key,ear in regions.items():
        donors=ear&observed&(maps['confidence']>.35)&(atlas.mean(-1)>35)&(atlas.mean(-1)<235)
        dy,dx=np.where(donors);step=max(1,int(np.ceil(len(dy)/20000)));dy,dx=dy[::step],dx[::step]
        if len(dy)<32:
            stats[key]={'status':'SKIPPED_INSUFFICIENT_OBSERVED_EAR','donors':len(dy)};continue
        tree=cKDTree(pos[dy,dx]);colors=atlas[dy,dx].astype(float);donor_normals=normals[fi[dy,dx]]
        ty,tx=np.where(ear&inferred)
        for start in range(0,len(ty),8192):
            y,x=ty[start:start+8192],tx[start:start+8192]
            distance,index=tree.query(pos[y,x],k=min(12,len(dy)))
            accepted=(distance<.05*height)&(np.einsum('nki,ni->nk',donor_normals[index],normals[fi[y,x]])>.25)
            weights=accepted/np.maximum(distance,.001*height)**2;total=weights.sum(1);supported=total>0
            weights/=np.maximum(total[:,None],1e-20)
            target=np.einsum('ni,nij->nj',weights,colors[index])
            fill_only[y[supported],x[supported]]=np.clip(np.rint(target[supported]),0,255).astype(np.uint8)
            fill_mask[y[supported],x[supported]]=True
        offset=np.zeros(3)
        if skin_tree is not None:
            distance,index=skin_tree.query(pos[dy,dx]);near=distance<.12*height
            if near.sum()>=32:
                reference=np.median(atlas[sy[index[near]],sx[index[near]]],axis=0)
                offset=np.clip((reference-np.median(colors[near],axis=0))*.35,-8,8)
        stats[key]={'status':'PROCESSED','donors':len(dy),'supported_texels':int((fill_mask&ear).sum()),'observed_tone_offset_rgb':offset.tolist()}
    fill_only,fill_alpha=feather_fill(atlas,fill_only,fill_mask,region,observed)
    tone=fill_only.copy()
    for key,ear in regions.items():
        if stats[key]['status']!='PROCESSED':continue
        active=ear&(observed|fill_mask)
        feather=smoothstep(cv2.distanceTransform(active.astype(np.uint8),cv2.DIST_L2,5),1,8)
        y,x=np.where(active);offset=np.asarray(stats[key]['observed_tone_offset_rgb'])
        tone[y,x]=bounded_rgb(fill_only[y,x],fill_only[y,x].astype(float)+offset*feather[y,x,None],limit=8)
        tone_mask|=active
    variants={}
    for name,result in [('fill_only',fill_only),('gentle_tone',tone)]:
        # Tone was built ear-by-ear; preserve inferred fill in every processed ear.
        if name=='gentle_tone':result[fill_mask&~tone_mask]=fill_only[fill_mask&~tone_mask]
        changed=np.any(result!=atlas,axis=-1)
        if not np.array_equal(result[~region],atlas[~region]):raise ValueError('Face or neck changed outside ear mask')
        if name=='fill_only' and not np.array_equal(result[known],atlas[known]):raise ValueError('Observed ear changed in fill-only policy')
        new_labels=labels.copy();new_labels[changed&inferred]=14;new_labels[changed&known]=15
        new_conf=confidence.copy();new_conf[changed&inferred]=13
        folder=output/name;folder.mkdir()
        Image.fromarray(result).save(folder/'head-atlas.png');Image.fromarray(new_labels).save(folder/'labels.png');Image.fromarray(new_conf).save(folder/'confidence.png')
        padded,_=render_gutter(result,valid);export_texture(v,f,uv,padded,folder/'head.glb')
        variants[name]=dict(changed_texels=int(changed.sum()),changed_observed_texels=int((changed&known).sum()),
                            max_rgb_delta=int(np.abs(result.astype(np.int16)-atlas.astype(np.int16)).max()),head_sha256=sha(folder/'head.glb'),
                            boundary_before=boundary_metric(atlas,region&inferred,observed),boundary_after=boundary_metric(result,region&inferred,observed))
    Image.fromarray(region.astype(np.uint8)*255).save(output/'ear-region.png');Image.fromarray(observed.astype(np.uint8)*255).save(output/'observed-ear.png');Image.fromarray(fill_mask.astype(np.uint8)*255).save(output/'supported-fill.png')
    Image.fromarray(np.clip(fill_alpha*255,0,255).astype(np.uint8)).save(output/'inferred-fill-blend.png')
    for p,d in pins.items():
        if sha(Path(p))!=d:raise ValueError('Source changed during ear trial')
    report=dict(status='EAR_APPEARANCE_CANDIDATES_EXPORTED',policy=POLICY,input_sha256=pins,code_sha256=sha(Path(__file__)),
                template_mask_sha256=MASK_SHA,ear_stats=stats,variants=variants,supported_inferred_texels=int(fill_mask.sum()),inferred_fill_max_blend=.8,
                geometry_changed=False,outside_ear_byte_equal=True,ready_for_machine=False,likeness_verified=False,lighting_removed=False,
                provenance={'14':'ear color interpolated from same-ear observed skin, not recovered hidden texture','15':'observed ear with bounded tone adjustment'},
                limitations=['Template ear labels are topology priors, not validated personal anatomy','Camera/semantic masks and Euclidean normal guards are approximate',
                             'No ear geometry, invisible folds or identity detail reconstructed','Skipped unsupported ears remain unchanged','Boundary metrics use fitted seams, not independent fidelity'])
    write(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.source,a.output)
