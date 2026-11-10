"""Frozen-split camera/FLAME ablation. Pinned decoder, no model training."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2
import trimesh
from scipy.optimize import least_squares,lsq_linear
from PIL import Image,ImageDraw
from tools.headapi.face_texture_trial import ROOT,sha
from tools.headapi.geometry_refinement_math import camera_update,project,mesh_guard
from tools.headapi.continuous_visibility import ContinuousDepth


def gray_sheet(vertices,faces,reference,path):
    size=360;center=(reference.min(0)+reference.max(0))/2
    v=(vertices-center)[:,[2,0,1]];ref=(reference-center)[:,[2,0,1]]
    scale=.84*size/np.ptp(ref,axis=0).max()
    normals=trimesh.Trimesh(vertices,faces,process=False).vertex_normals[:,[2,0,1]]
    image=Image.new('RGB',(4*size,size+24),'#20252c');draw=ImageDraw.Draw(image)
    for i,angle in enumerate([0,90,180,270]):
        a=np.radians(angle);x=-v[:,0]*np.sin(a)+v[:,1]*np.cos(a);z=v[:,0]*np.cos(a)+v[:,1]*np.sin(a)
        xy=np.c_[x*scale+size/2,-v[:,2]*scale+size/2]
        light=(-.3*(-normals[:,0]*np.sin(a)+normals[:,1]*np.cos(a))+.4*normals[:,2]+normals[:,0]*np.cos(a)+normals[:,1]*np.sin(a))/np.sqrt(1.25)
        color=65+150*np.maximum(light,0)
        pixels=np.full((size,size,3),[35,40,47],np.uint8);depth=np.full((size,size),-np.inf)
        for face in faces:
            tri=xy[face];lo=np.maximum(np.floor(tri.min(0)).astype(int),0);hi=np.minimum(np.ceil(tri.max(0)).astype(int),size-1)
            if (hi<lo).any():continue
            p,q,r=tri;den=(q[1]-r[1])*(p[0]-r[0])+(r[0]-q[0])*(p[1]-r[1])
            if abs(den)<1e-10:continue
            yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
            b0=((q[1]-r[1])*(xx-r[0])+(r[0]-q[0])*(yy-r[1]))/den
            b1=((r[1]-p[1])*(xx-r[0])+(p[0]-r[0])*(yy-r[1]))/den;b2=1-b0-b1
            d=b0*z[face[0]]+b1*z[face[1]]+b2*z[face[2]];section=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
            keep=(b0>=0)&(b1>=0)&(b2>=0)&(d>section)
            value=b0*color[face[0]]+b1*color[face[1]]+b2*color[face[2]]
            pixels[lo[1]:hi[1]+1,lo[0]:hi[0]+1][keep]=np.clip(value[keep,None],0,255).astype(np.uint8);section[keep]=d[keep]
        image.paste(Image.fromarray(pixels),(i*size,24));draw.text((i*size+8,5),str(angle)+' deg / fixed baseline frame',fill='white')
    image.save(path)


def visibility(vertices,faces,camera,landmarks,size):
    k,r,c,*_=cv2.decomposeProjectionMatrix(camera);k/=k[2,2];center=(c[:3]/c[3]).reshape(3)
    cam=(vertices-center)@r.T;points=(landmarks-center)@r.T
    h=points@k.T;xy=h[:,:2]/h[:,2,None]
    frame=(points[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<size[0])&(xy[:,1]>=0)&(xy[:,1]<size[1])
    nearest=ContinuousDepth(cam,faces,k,size).query(xy)
    return frame&(np.abs(nearest-points[:,2])<1e-6),frame


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'vendor/RealDenseFace'))
    import common.fix_chumpy
    import torch
    from tracker import FLAMEConfig,GNFlameOptimizer,GNFlameOptimizerConfig
    from common.types import FittingOutput
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('Pinned RDF experiment runtime requires available CUDA')
    free,total=torch.cuda.mem_get_info()
    if free<1024**3:raise RuntimeError('Need 1 GiB free GPU memory for decoder experiment')
    torch.cuda.set_per_process_memory_fraction(.5);torch.cuda.reset_peak_memory_stats()
    model=ROOT/'vendor/RealDenseFace/weights/flame/flame2023.pkl';assets=ROOT/'vendor/RealDenseFace/weights/flame/flame_assets.npz'
    embedding=ROOT/'runs/experiments/rdf-mediapipe-105-01/mediapipe_landmark_embedding.npz'
    pins={model:'8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0',assets:'922cb76f3afef0c2e9801a63991de269a98e6b67b1728232493f851589151dd3',embedding:'8863363013fc6fe3752d4318ea02a1784970200616a54b785920cdd0568816fc'}
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Pinned model/embedding changed')
    decoder=GNFlameOptimizer(GNFlameOptimizerConfig(flame=FLAMEConfig(flame_model_path=str(model),flame_assets_path=str(assets),num_expressions=100),device='cuda',boost_eye_region=False))
    with np.load(embedding,allow_pickle=False) as b:fi=b['lmk_face_idx'];bary=b['lmk_b_coords'];ids=b['landmark_indices'].astype(int)
    if len(ids)!=105 or not np.isfinite(bary).all() or not np.allclose(bary.sum(1),1) or (bary<-1e-6).any():raise ValueError('Invalid embedding')
    stages=json.loads((ROOT/'runs/experiments/texture-visibility-strict-01/stages.json').read_text())
    results={'status':'EXPLORATORY_CAMERA_GEOMETRY_ABLATION','weights_changed':False,'baseline_promoted':False,
             'split':'landmark id mod 3 == 0 held out of NEW refinement only; baseline already used all photos/landmarks',
             'embedding_semantics_verified':False,'unseen_validation':False,'camera_bounds':'per-axis 1.5deg rotation; camera-space translations 0.5%,0.5%,1% head-height about mesh center; intrinsics fixed',
             'geometry_bounds':'max displacement 0.3% baseline head height, normal change <=30deg, edge change <=15%, area >=20%; no self-intersection test',
             'input_sha256':{str(p):h for p,h in pins.items()},'code_sha256':sha(Path(__file__)),'cases':[]}
    regions=np.array(['brows']*20+['eyes']*32+['nose']*13+['mouth']*40)
    for stage in stages:
        started=time.monotonic();name=stage['case'];source=Path(stage['source']);raw=ROOT/'runs/experiments/texture-visibility-strict-01'/(name+'-raw')
        source_report=json.loads((raw/'report.json').read_text());bundle=next(Path(p) for p in source_report['input_sha256'] if p.endswith('projection_bundle.npz'))
        if sha(bundle)!=stage['bundle_sha256']:raise ValueError('Geometry source changed')
        meta=source/'report.json';observations=json.loads(meta.read_text())
        with np.load(bundle,allow_pickle=False) as b:data={k:b[k].copy() for k in b.files}
        faces=data['faces'];base=data['view_vertices'].astype(float);height=float(np.ptp(base[0,:,1]))
        observed=[];sizes=[]
        for slot in ['front','left','right']:
            photo=source/(slot+'.png')
            if sha(photo)!=source_report['input_sha256'][str(photo)]:raise ValueError('Photo changed')
            observed.append(np.asarray(observations['views'][slot]['landmarks_crop'])[ids]);sizes.append(Image.open(photo).size)
        observed=np.asarray(observed)
        def lm(v):return (v[:,faces[fi]]*bary[None,:,:,None]).sum(2)
        baseline_lm=lm(base);frozen=[]
        for i in range(3):
            vis,frame=visibility(base[i],faces,data['cameras'][i],baseline_lm[i],sizes[i]);frozen.append(vis&frame)
        frozen=np.asarray(frozen);hold=(ids%3==0)[None]&frozen;train=(ids%3!=0)[None]&frozen
        if np.any(train.sum(1)<12) or np.any(hold.sum(1)<4):raise ValueError('Insufficient visible split')
        normalization=float(np.linalg.norm((np.asarray(observations['views']['front']['landmarks_crop'])[33]+np.asarray(observations['views']['front']['landmarks_crop'])[133]-np.asarray(observations['views']['front']['landmarks_crop'])[362]-np.asarray(observations['views']['front']['landmarks_crop'])[263])/2))
        weights=np.zeros_like(train,dtype=float)
        for i in range(3):
            for region in ['eyes','mouth','nose','brows']:
                selected=train[i]&(regions==region);weights[i,selected]=1/np.sqrt(max(int(selected.sum()),1))
        beta=torch.as_tensor(data['identity'],device='cuda');states=torch.as_tensor(data['states'],device='cuda')
        def decode(parameters):
            identity=beta.clone();identity[:16]+=torch.as_tensor(parameters[:16],device='cuda',dtype=torch.float32)
            state=states.clone();state[:,:6]+=torch.as_tensor(np.asarray(parameters[16:]).reshape(3,6),device='cuda',dtype=torch.float32)
            local=state.clone();local[:,100:103]=0;local[:,115:118]=0
            with torch.no_grad():v=decoder.decode(FittingOutput(identity=identity,x=local))[0].cpu().numpy().astype(float)
            return v,identity.cpu().numpy(),state.cpu().numpy()
        reproduced,_,_=decode(np.zeros(34));error=float(np.max(np.abs(reproduced-base)))
        if error>1e-6:raise ValueError('Pinned decoder cannot reproduce baseline')
        def fit_cameras(v):
            landmarks=lm(v);cameras=[];parameters=[]
            for i in range(3):
                chosen=train[i]
                def objective(u):
                    p=camera_update(data['cameras'][i],base[i],u,height)
                    residual=(project(landmarks[i],p)-observed[i])/normalization
                    return np.r_[(residual[chosen]*weights[i,chosen,None]).ravel(),.005*u]
                solution=least_squares(objective,np.zeros(6),bounds=(-np.ones(6),np.ones(6)),loss='soft_l1',f_scale=.02,max_nfev=100)
                if not solution.success:raise ValueError('Camera solve failed')
                cameras.append(camera_update(data['cameras'][i],base[i],solution.x,height));parameters.append(solution.x.tolist())
            return np.asarray(cameras),parameters
        cameras,camera_parameters=fit_cameras(base)
        print(name+' camera solve complete',flush=True)
        basis=[]
        for j in range(34):
            step=np.zeros(34);step[j]=.01;v,_,_=decode(step);basis.append((v-reproduced)/.01)
        basis=np.stack(basis,axis=-1)
        landmark_basis=(basis[:,faces[fi]]*bary[None,:,:,None,None]).sum(2)
        j_columns=[];target=[]
        base_xy=[project(baseline_lm[i],cameras[i]) for i in range(3)]
        for j in range(34):
            column=[]
            for i in range(3):column.extend((((project(baseline_lm[i]+.01*landmark_basis[i,:,:,j],cameras[i])-base_xy[i])/.01/normalization)*weights[i,:,None])[train[i]].ravel())
            j_columns.append(column)
        for i in range(3):target.extend((((observed[i]-base_xy[i])/normalization)*weights[i,:,None])[train[i]].ravel())
        jac=np.stack(j_columns,axis=1);case_dir=output/name;case_dir.mkdir()
        case={'case':name,'source_bundle_sha256':sha(bundle),'source':str(source),'decoder_reproduction_max_native':error,'normalizer_front_interocular_px':normalization,'frozen_train_count':train.sum(1).tolist(),'frozen_holdout_count':hold.sum(1).tolist(),'variants':[]}
        results['input_sha256'].update({str(bundle):sha(bundle),str(meta):sha(meta)})
        variants=[('baseline',[]),('camera_only',[]),('camera_shape',list(range(16))),('camera_expression',list(range(16,34))),('camera_joint',list(range(34)))]
        for variant,indices in variants:
            parameters=np.zeros(34);alpha=1.
            if indices:
                # Coefficient steps are bounded to +/-0.05 native coefficients, not anatomical SDs.
                matrix=np.vstack([jac[:,indices]*.05,.005*np.eye(len(indices))])
                solve=lsq_linear(matrix,np.r_[target,np.zeros(len(indices))],bounds=(-1,1),tol=1e-9,max_iter=100)
                if not solve.success:raise ValueError('Coefficient solve failed')
                parameters[indices]=solve.x*.05
                for _ in range(20):
                    v,identity,state=decode(parameters*alpha);guard=mesh_guard(base,v,faces,height)
                    if guard['passed']:break
                    alpha*=.5
                else:raise ValueError('No coefficient step meets geometry guard')
            else:v=base.copy();identity=data['identity'].copy();state=data['states'].copy();guard=mesh_guard(base,v,faces,height)
            if variant=='baseline':p=data['cameras'].copy();pose=np.zeros((3,6)).tolist()
            elif indices:p,pose=fit_cameras(v)
            else:p=cameras.copy();pose=camera_parameters
            destination=case_dir/variant;destination.mkdir();camera_points=[];intrinsics=[]
            for i in range(3):
                k,r,c,*_=cv2.decomposeProjectionMatrix(p[i]);k/=k[2,2];center=(c[:3]/c[3]).reshape(3)
                camera_points.append((v[i]-center)@r.T);intrinsics.append(k)
            payload={**data,'vertices':v[0],'view_vertices':v,'cameras':p,'camera_points':np.asarray(camera_points),'intrinsics':np.asarray(intrinsics),'identity':identity,'states':state}
            np.savez_compressed(destination/'projection_bundle.npz',**payload)
            mesh=trimesh.Trimesh(v[0],faces,process=False);mesh.visual.vertex_colors=[180,180,180,255];mesh.export(destination/'face_gray.glb')
            gray_sheet(v[0],faces,base[0],destination/'gray.png')
            row={'name':variant,'bundle_sha256':sha(destination/'projection_bundle.npz'),'mesh_guard':guard,'coefficient_step_scale':alpha,'coefficient_delta':(parameters*alpha).tolist(),'camera_normalized_parameters':pose,'views':[]}
            current_lm=lm(v)
            for i,slot in enumerate(['front','left','right']):
                xy=project(current_lm[i],p[i]);baseline_xy=project(baseline_lm[i],data['cameras'][i]);visible,frame=visibility(v[i],faces,p[i],current_lm[i],sizes[i])
                err=np.linalg.norm(xy-observed[i],axis=1);old_err=np.linalg.norm(baseline_xy-observed[i],axis=1)
                view={'view':slot,'visibility_lost':int((frozen[i]&~visible).sum()),'out_of_frame':int((frozen[i]&~frame).sum()),'regions':{},'points':[]}
                for region in ['eyes','mouth','nose','brows']:
                    selected=hold[i]&(regions==region)
                    if selected.any():view['regions'][region]={'count':int(selected.sum()),'baseline_mean_px':float(old_err[selected].mean()),'candidate_mean_px':float(err[selected].mean())}
                view['train_mean_px']=float(err[train[i]].mean());view['holdout_mean_px']=float(err[hold[i]].mean());view['baseline_holdout_mean_px']=float(old_err[hold[i]].mean())
                image=Image.open(source/(slot+'.png')).convert('RGB');draw=ImageDraw.Draw(image)
                for j in np.where(frozen[i])[0]:
                    q=xy[j];target_point=observed[i,j];color='yellow' if hold[i,j] else 'cyan'
                    draw.line([tuple(q),tuple(target_point)],fill=color,width=1)
                    draw.ellipse((q[0]-2,q[1]-2,q[0]+2,q[1]+2),fill='red' if visible[j] else 'orange')
                    view['points'].append({'mediapipe':int(ids[j]),'region':str(regions[j]),'holdout':bool(hold[i,j]),'baseline_px':float(old_err[j]),'candidate_px':float(err[j]),'visible':bool(visible[j]),'in_frame':bool(frame[j])})
                draw.text((8,8),variant+' / '+slot+' | yellow holdout, cyan train | red projected, orange occluded',fill='yellow');image.save(destination/(slot+'.overlay.png'));row['views'].append(view)
            row['review_screen_passed']=bool(guard['passed'] and all(view['visibility_lost']==0 and view['out_of_frame']==0 and all(r['candidate_mean_px']<=1.05*r['baseline_mean_px']+.25 for r in view['regions'].values()) for view in row['views']))
            row['screen_scope']='uncalibrated no-regression screen; not quality, identity or print acceptance'
            (destination/'report.json').write_text(json.dumps(row,indent=2),encoding='utf-8');case['variants'].append(row)
            print(name+' '+variant+' complete',flush=True)
        case['elapsed_seconds']=time.monotonic()-started;results['cases'].append(case)
    results['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated();results['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    results['decoder_only']=True;results['states_pose_fields']='legacy global pose kept; explicit cameras authoritative after correction'
    (output/'experiment.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.output)
