"""Small CPU overlap experiment; spatial holdout, no atlas/geometry modification."""
import json
import ctypes
import sys
from pathlib import Path
import numpy as np
import cv2
import trimesh
from PIL import Image
from tools.headapi.face_texture_trial import ROOT, sha, apply_gain, robust_skin_gain
from memento3d_head.adapters.photo_projection import depth_buffer
from tools.headapi.hair_parser import HairParser, PARSER_SHA


def skin_candidate_mask(rgb, labels):
    """Conservative fixed filters; not a calibrated shadow/specularity detector."""
    lab=cv2.cvtColor((rgb/255.).astype(np.float32),cv2.COLOR_RGB2LAB)
    luminance=lab[...,0];chroma=np.linalg.norm(lab[...,1:],axis=-1)
    skin=cv2.erode((labels==1).astype(np.uint8),np.ones((7,7),np.uint8))>0
    dark=luminance<25
    bright=(luminance>85)|((luminance>75)&(chroma<8))
    return skin&~dark&~bright,{'eroded_skin_pixels':int(skin.sum()),'dark_rejected':int((skin&dark).sum()),'bright_rejected':int((skin&bright).sum())}


def fit_field(x, reference, sample):
    def linear(rgb):
        a=np.clip(rgb/255.,.001,1)
        return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
    target=np.log(linear(reference))-np.log(linear(sample))
    weights=np.ones(len(x));penalty=np.diag([1.,20.,20.])
    for _ in range(8):
        coef=np.linalg.solve(x.T@(weights[:,None]*x)+penalty,x.T@(weights[:,None]*target))
        residual=np.linalg.norm(target-x@coef,axis=1)
        weights=np.minimum(1,.10/np.maximum(residual,1e-9))
    return coef


def metrics(reference, result):
    # OpenCV float Lab uses L 0..100 and conventional a/b units.
    a=cv2.cvtColor((reference/255.).astype(np.float32)[None],cv2.COLOR_RGB2LAB)[0]
    b=cv2.cvtColor((result/255.).astype(np.float32)[None],cv2.COLOR_RGB2LAB)[0]
    delta=np.linalg.norm(a-b,axis=1)
    return {'median_delta_e76':float(np.median(delta)),'p90_delta_e76':float(np.percentile(delta,90)), 'rgb_mae':float(np.mean(np.abs(reference-result)))}


def run(skin_only=False):
    available=None
    if skin_only and sys.platform=='win32':
        m=(ctypes.c_ulonglong*8)();ctypes.cast(m,ctypes.POINTER(ctypes.c_ulong))[0]=ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('Cannot read RAM')
        available=int(m[2])
        if available<2*2**30:raise RuntimeError('Need 2 GiB available physical RAM before parser inference')
    output=ROOT/('runs/experiments/input04-texture-skin-01' if skin_only else 'runs/experiments/input04-texture-overlap-01')
    output.mkdir(exist_ok=False,parents=True)
    bundle=ROOT/'runs/experiments/rdf-flame-parameters-40ab0c23fed043eb875537da057027e8/shape_expression_0_05/projection_bundle.npz'
    previous=json.loads((ROOT/'runs/experiments/input04-face-texture-01/report.json').read_text())
    if previous['input_sha256'][str(bundle)]!=sha(bundle):raise ValueError('Bundle pin changed')
    with np.load(bundle,allow_pickle=False) as b:
        faces=b['faces'];views=b['view_vertices'];cameras=b['cameras']
    matches=[];hashes={str(bundle):sha(bundle)};mask_counts={}
    parser_path=Path('D:/AI-cache/face-parsing/resnet18.onnx')
    parser=HairParser(parser_path) if skin_only else None
    if skin_only:hashes[str(parser_path)]=PARSER_SHA
    for i,slot in enumerate(['front','left','right']):
        photo=ROOT/'runs/experiments/input04-validation-01/rdf_crops'/(slot+'.png')
        rgb=np.asarray(Image.open(photo).convert('RGB'))
        if previous['input_sha256'][str(photo)]!=sha(photo):raise ValueError('Photo pin changed')
        # Reuse stored mask; exact skin labels are regenerated only from cached parsing if available.
        # Conservative smooth/color validity is additional filtering, not a skin classifier.
        maskpath=ROOT/'runs/experiments/input04-face-texture-01'/(slot+'.accepted-mask.png')
        mask=np.asarray(Image.open(maskpath))>0
        if skin_only:
            labels=parser(Image.fromarray(rgb))
            mask,mask_counts[slot]=skin_candidate_mask(rgb,labels)
            labelpath=output/(slot+'.labels.png');Image.fromarray(labels).save(labelpath)
            candidate=output/(slot+'.skin-candidates.png');Image.fromarray((mask*255).astype(np.uint8)).save(candidate)
            hashes[str(labelpath)]=sha(labelpath);hashes[str(candidate)]=sha(candidate)
        hashes[str(photo)]=sha(photo);hashes[str(maskpath)]=sha(maskpath)
        k,r,c,*_=cv2.decomposeProjectionMatrix(cameras[i]);k/=k[2,2];center=(c[:3]/c[3]).reshape(3)
        cam=(views[i]-center)@r.T;_,depth=depth_buffer(cam,faces,k,(rgb.shape[1],rgb.shape[0]))
        h=cam@k.T;xy=h[:,:2]/h[:,2,None];ij=np.floor(xy).astype(int)
        ix,iy=np.clip(ij,[0,0],[rgb.shape[1]-1,rgb.shape[0]-1]).T
        normals=trimesh.Trimesh(views[i],faces,process=False).vertex_normals
        direction=center-views[i];direction/=np.linalg.norm(direction,axis=1,keepdims=True)
        facing=np.einsum('ij,ij->i',normals,direction)
        eroded=mask if skin_only else cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0
        valid=(cam[:,2]>0)&(ij[:,0]>=2)&(ij[:,0]<rgb.shape[1]-2)&(ij[:,1]>=2)&(ij[:,1]<rgb.shape[0]-2)
        valid &= eroded[iy,ix] & (np.abs(depth[iy,ix]-cam[:,2])<.001)&(facing>.5)
        color=cv2.remap(rgb,xy[:,0].astype(np.float32)[None],xy[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0].astype(float)
        valid &= ((color>25)&(color<235)).all(axis=1)
        matches.append((color,valid,xy/np.array([rgb.shape[1],rgb.shape[0]])))
    report={'status':'OVERLAP_DIAGNOSTIC_ONLY','geometry_changed':False,'atlas_exported':False,'input_sha256':hashes,'code_sha256':sha(Path(__file__)),'views':[],'limitations':['Approximate geometry/cameras; no skin ground truth','Masks include facial features; this is accepted face overlap, not skin-only','Spatial checkerboard holdout on same person; no independent subject validation','Delta E76 agreement with front is not intrinsic albedo or likeness']}
    if skin_only:
        report['limitations'][1]='Skin label and fixed Lab thresholds are uncalibrated; moderate shadows/highlights may remain; dark skin can be unfairly rejected'
        report['skin_filter']={'label':1,'erosion':7,'dark_L_below':25,'bright_L_above':85,'low_chroma_bright':{'L_above':75,'chroma_below':8},'counts':mask_counts}
        report['available_physical_ram_at_start_bytes']=available
    for i,slot in [(1,'left'),(2,'right')]:
        valid=matches[0][1]&matches[i][1];ids=np.where(valid)[0]
        a=matches[0][0][ids];b=matches[i][0][ids];xy=matches[i][2][ids]
        cells=np.floor(matches[0][2][ids]*8).astype(int)
        hold=(cells[:,0]+cells[:,1])%3==0;train=~hold
        item={'view':slot,'train_count':int(train.sum()),'holdout_count':int(hold.sum())}
        if min(train.sum(),hold.sum())<32:
            item['status']='INSUFFICIENT_SAMPLES';report['views'].append(item);continue
        gain,n=robust_skin_gain(a[train],b[train]);x=np.c_[np.ones(len(xy)),2*xy-1]
        coef=fit_field(x[train],a[train],b[train]);local_gain=np.clip(np.exp(x@coef),.85,1.18)
        global_color=apply_gain(b,gain);local_color=apply_gain(b,local_gain)
        item.update({'status':'MEASURED','global_gain':gain.tolist(),'local_coefficients':coef.tolist(),'local_gain_clipped_fraction':float(np.mean((local_gain<=.85)|(local_gain>=1.18))),'holdout':{'raw':metrics(a[hold],b[hold]),'global':metrics(a[hold],global_color[hold]),'local':metrics(a[hold],local_color[hold])}})
        if skin_only:
            with np.load(ROOT/'runs/experiments/input04-texture-overlap-01'/(slot+'.samples.npz'),allow_pickle=False) as old:
                old_hold=old['holdout'];old_a=old['reference'][old_hold];old_b=old['sample'][old_hold]
                old_x=np.c_[np.ones(old_hold.sum()),2*old['xy'][old_hold]-1]
                item['fixed_previous_holdout']={'count':int(old_hold.sum()),'raw':metrics(old_a,old_b),'previous_global':metrics(old_a,old['global_color'][old_hold]),'skin_global':metrics(old_a,apply_gain(old_b,gain)),'skin_local':metrics(old_a,apply_gain(old_b,np.clip(np.exp(old_x@coef),.85,1.18)))}
            artifact=ROOT/'runs/experiments/input04-texture-overlap-01'/(slot+'.samples.npz')
            hashes[str(artifact)]=sha(artifact)
        np.savez_compressed(output/(slot+'.samples.npz'),vertex_ids=ids,reference=a,sample=b,xy=xy,holdout=hold,global_color=global_color,local_color=local_color)
        report['views'].append(item)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['views'],indent=2),flush=True)


if __name__=='__main__':run(skin_only='--skin-only' in sys.argv[1:])
