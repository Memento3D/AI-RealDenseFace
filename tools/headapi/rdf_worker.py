"""RDF CUDA worker: explicit ROIs, independent crop cameras, batched per-frame poses."""
import argparse,json,sys,hashlib,base64,io,time,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from head24_policy import eye_gate,rigid_map,project,InputRejected,inference_box,bbox_gate

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(repo,crops,out,selected_views=None,trace_fov=False,fixed_fov=None,prediction_cache=None,warm_fit=None,warm_policy='state'):
    if fixed_fov is not None and not 5 <= fixed_fov <= 90:
        raise ValueError('Experimental crop FOV must be in [5,90] degrees')
    sys.path[:0]=[str(repo),str(ROOT/'src')]
    import common.fix_chumpy
    import cv2,torch
    from tracker import FLAMEConfig,FaceBoxConfig,RealDenseFaceInferenceConfig,RealDenseFaceInferencer,GNFlameOptimizer,GNFlameOptimizerConfig
    from common.types import FittingOutput,InferenceOutput
    from eye_components import identify
    from PIL import Image,ImageDraw
    out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((crops/'report.json').read_text());assert metadata['status']=='CROPS_PREPARED'
    slots=[slot for slot in ('front','left','right') if slot in metadata['views']]
    if selected_views is not None:
        if not selected_views or len(set(selected_views))!=len(selected_views) or any(s not in slots for s in selected_views):
            raise ValueError('Invalid or unavailable experiment views')
        slots=[s for s in slots if s in selected_views]
    if not slots or slots[0]!='front':raise ValueError('FRONT_REQUIRED')
    report={'views_used':slots,'view_count':len(slots),'status':'RUNNING','camera_intrinsics_assumed_shared':False,'photo_quality_gate_executed':True,'gates_calibrated':False,'likeness_verified':False,'input_sha256':{},'camera_maps':[],'single_frame_registration_iterations':20,'joint_registration_iterations':10 if len(slots)>1 else 0,'eye_region_boost_enabled':False,'neural_variant':'vitb for each actually supplied crop','joint_camera_layout':'T=available views,C=1,4,4','vitb_expected_weight_checksum_verified':False}
    report['cuda_launch_blocking']=os.environ.get('CUDA_LAUNCH_BLOCKING')=='1'
    optimizer=None
    started=time.monotonic()
    report['fixed_crop_fov_y_degrees']=fixed_fov
    report['camera_policy']='fixed_fov_experiment_v1' if fixed_fov is not None else 'existing_bounded_search'
    torch.cuda.reset_peak_memory_stats()
    required=[repo/'weights/flame/flame2023.pkl',repo/'weights/flame/flame_assets.npz',repo/'weights/realdenseface/vitb.pth',repo/'weights/facebox/face_box.pth']
    pins={'flame2023.pkl':'8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0','flame_assets.npz':'922cb76f3afef0c2e9801a63991de269a98e6b67b1728232493f851589151dd3','face_box.pth':'3fde8561f059dd60d876021cbd2f67d955d5e02f24c25b297913a4115e433b4d'}
    # Original front model remains vitb; its local hash is recorded, not independently pinned.
    try:
        report['crop_report_sha256']=digest(crops/'report.json')
        for slot in slots:
            path=crops/(slot+'.png')
            if digest(path)!=metadata['views'][slot]['crop_sha256']:raise ValueError('Crop checksum mismatch: '+slot)
            report['input_sha256'][str(path)]=digest(path)
        report['input_sha256'][str(crops/'report.json')]=report['crop_report_sha256']
        for p in required:
            actual=digest(p);report['input_sha256'][str(p)]=actual
            if p.name in pins and actual!=pins[p.name]:raise ValueError('Model pin mismatch: '+p.name)
        cache_fields=('vertex_coord','vertex_depth','vertex_coord_log_var','vertex_depth_log_var','face_bbox')
        cache_signature={**report['input_sha256'],str(repo/'configs/model/vitb.yaml'):digest(repo/'configs/model/vitb.yaml')}
        cache_meta=None
        if prediction_cache is not None:
            prediction_cache=Path(prediction_cache)
            cache_meta=json.loads((prediction_cache/'manifest.json').read_text())
            if cache_meta['inputs']!=cache_signature or cache_meta['views']!=slots:
                raise ValueError('Prediction cache source/model/view mismatch')
            report['prediction_cache_manifest_sha256']=digest(prediction_cache/'manifest.json')
        warm=None
        if warm_fit is not None:
            warm_fit=Path(warm_fit);warm_report=json.loads((warm_fit/'report.json').read_text())
            if warm_report['status']!='FIT_GATES_PASSED' or warm_report['views_used']!=slots or warm_report['input_sha256']!=report['input_sha256']:
                raise ValueError('Warm fit source/model/view mismatch')
            if digest(warm_fit/'projection_bundle.npz')!=warm_report['projection_bundle_sha256']:
                raise ValueError('Warm fit checksum mismatch')
            with np.load(warm_fit/'projection_bundle.npz',allow_pickle=False) as b:
                warm={'identity':b['identity'].copy(),'states':b['states'].copy()}
            report['warm_fit']={'bundle_sha256':warm_report['projection_bundle_sha256'],'path':str(warm_fit),
                                'policy':warm_policy,'translation_rescaled':False}
        cfg=FLAMEConfig(flame_model_path=str(required[0]),flame_assets_path=str(required[1]),num_expressions=100)
        infer=RealDenseFaceInferencer(RealDenseFaceInferenceConfig(model_config_path=str(repo/'configs/model/vitb.yaml'),model_weights_path=str(required[2]),flame=cfg,device='cuda',compile_model=False),FaceBoxConfig(model_weights_path=str(required[3]),device='cuda'))
        # Eye boosting can destabilize FOV search; the independent eye gate still validates alignment.
        from .stable_optimizer import StableRDFOptimizer, POLICY
        report['optimizer_policy']=POLICY
        optimizer=StableRDFOptimizer(GNFlameOptimizerConfig(flame=cfg,device='cuda',registration_iterations=20,boost_eye_region=False))
        if trace_fov:
            report['fov_search_trace']=[]
            report['fov_search_config']={key:getattr(optimizer.config,key) for key in
                ('fov_y_min','fov_y_max','fov_search_iterations','fov_search_epsilon','fov_registration_iterations')}
            build_camera=optimizer._build_config_camera
            evaluate_energy=optimizer._evaluate_energy
            trace_state={}
            def traced_camera(width,height,fov):
                trace_state['fov_y']=float(fov)
                return build_camera(width,height,fov)
            def traced_energy(*args,**kwargs):
                energy=evaluate_energy(*args,**kwargs)
                report['fov_search_trace'].append({'view':trace_state.get('view'),
                    'fov_y':trace_state.get('fov_y'),'energy':float(energy.mean().item())})
                return energy
            optimizer._build_config_camera=traced_camera
            optimizer._evaluate_energy=traced_energy
        predictions=[];fits=[];cameras=[];images=[]
        prediction_dir=out/'predictions';prediction_dir.mkdir()
        prediction_hashes={}
        for view_index,slot in enumerate(slots):
            if trace_fov:trace_state['view']=slot
            path=crops/(slot+'.png');images.append(np.asarray(Image.open(path).convert('RGB')))
            box=inference_box(metadata['views'][slot],slot)
            if cache_meta is None:
                prediction=infer.infer_image(images[-1],face_bbox=box)
            else:
                cache_path=prediction_cache/(slot+'.npz')
                if digest(cache_path)!=cache_meta['artifacts'][cache_path.name]:raise ValueError('Prediction cache checksum mismatch')
                with np.load(cache_path,allow_pickle=False) as b:
                    prediction=InferenceOutput(**{k:b[k].copy() for k in cache_fields},image_width=int(b['image_width']),image_height=int(b['image_height']))
            np.savez_compressed(prediction_dir/(slot+'.npz'),**{k:getattr(prediction,k) for k in cache_fields},
                                image_width=prediction.image_width,image_height=prediction.image_height)
            prediction_hashes[slot+'.npz']=digest(prediction_dir/(slot+'.npz'))
            # Never calls _init_face_bbox or its whole-image default fallback.
            fixed_camera=None if fixed_fov is None else optimizer._build_config_camera(1024,1024,fixed_fov)
            if warm is None:
                result=optimizer.fit_frame(prediction,camera=fixed_camera)
            elif warm_policy=='identity_pose':
                if fixed_camera is None:raise ValueError('Warm experiment requires fixed FOV')
                result=optimizer.fit_frame(prediction,camera=fixed_camera,init_identity=warm['identity'])
            else:
                if fixed_camera is None:raise ValueError('Warm experiment requires fixed FOV')
                vm0,vpm0=optimizer._single_camera_matrices(fixed_camera)
                tensors0=[torch.as_tensor(getattr(prediction,k),device='cuda',dtype=torch.float32) for k in cache_fields[:4]]
                c0,d0,cv0,dv0=tensors0;c0=c0/1024.;cv0=optimizer._apply_eye_region_boost(cv0)
                identity,state=optimizer._register(vm0,vpm0,c0,d0,cv0,dv0,
                    init_identity=torch.as_tensor(warm['identity'],device='cuda'),
                    init_x=torch.as_tensor(warm['states'][view_index],device='cuda'),num_iters=20)
                result=FittingOutput(identity=identity,x=state)
            if fixed_fov is not None:result.camera_fov_y=float(fixed_fov)
            if result.camera_fov_y is None:raise ValueError('No FOV returned: '+slot)
            predictions.append(prediction);fits.append(result)
            cameras.append(optimizer._build_config_camera(1024,1024,result.camera_fov_y))
            print('Head24 independent fit:',slot,'fov:',result.camera_fov_y,flush=True)
        (prediction_dir/'manifest.json').write_text(json.dumps({'schema_version':1,'inputs':cache_signature,'views':slots,'artifacts':prediction_hashes},indent=2),encoding='utf-8')
        faces=optimizer.faces.cpu().numpy()
        def decode(beta,x):
            with torch.no_grad():return optimizer.decode(FittingOutput(identity=beta,x=x))[0].detach().cpu().numpy()
        def check_front(beta,x):
            points=decode(beta,x)
            np.savez_compressed(out/'front_gate_diagnostic.npz',vertices=points,faces=faces)
            components,info=identify(points,faces)
            centers=np.asarray(info['eye_centers_native'])
            E=cameras[0].get_w2v.cpu().numpy();K=cameras[0].intrisics.cpu().numpy()
            projected,_=project(centers,K@E[:3])
            lm=np.asarray(metadata['views']['front']['landmarks_crop'])
            observed=np.array([(lm[33]+lm[133])/2,(lm[362]+lm[263])/2])
            return eye_gate(projected,observed)
        report['front_baseline_gate']=check_front(fits[0].identity,fits[0].x)
        def check_profiles(beta,states):
            checks={}
            ids=infer.key_vertex_ids.detach().cpu().numpy()
            for i,slot in enumerate(slots):
                if slot=='front' or metadata['views'][slot].get('roi_backend')!='facebox':continue
                P=cameras[i].intrisics.cpu().numpy()@cameras[i].get_w2v.cpu().numpy()[:3]
                xy,_=project(decode(beta,states[i])[ids],P)
                checks[slot]=bbox_gate(xy,inference_box(metadata['views'][slot],slot))
            return checks
        # [T=available views,C=1,4,4]: independent poses,
        # with a distinct crop FOV per frame. Shared identity remains a candidate.
        vm,vpm=optimizer._stack_camera_matrices(cameras);vm=vm[:,None];vpm=vpm[:,None]
        tensors=[torch.from_numpy(np.stack([getattr(p,k) for p in predictions]).astype('float32')).to('cuda') for k in ('vertex_coord','vertex_depth','vertex_coord_log_var','vertex_depth_log_var')]
        coords,depths,cv,dv=tensors;coords=coords/1024.;cv=optimizer._apply_eye_region_boost(cv)
        initial=fits[0].identity;states=torch.stack([f.x for f in fits])
        from tools.headapi.fitting_contract import fit_shared_identity
        beta,xs=fit_shared_identity(optimizer,vm,vpm,coords,depths,cv,dv,initial,states,len(slots))
        report['front_joint_gate']=check_front(beta,xs[0])
        report['profile_joint_gates']=check_profiles(beta,xs)
        report['identity_delta_l2']=float(torch.linalg.vector_norm(beta-initial).item())
        posed=decode(beta,xs);locals=[];pixel_cameras=[];camera_points=[]
        for i in range(len(slots)):
            local=xs[i].clone();local[100:103]=0;local[115:118]=0
            local_vertices=decode(beta,local);locals.append(local_vertices)
            transform,error=rigid_map(local_vertices,posed[i])
            K=cameras[i].intrisics.detach().cpu().numpy();E=cameras[i].get_w2v.detach().cpu().numpy()
            P=K@E[:3]@transform
            pixel_cameras.append(P)
            cam=posed[i]@E[:3,:3].T+E[:3,3];camera_points.append(cam)
            report['camera_maps'].append({'view':slots[i],'root_rigid_max_native_error':error,'fov_y':float(fits[i].camera_fov_y)})
            xy,_=project(local_vertices,P)
            im=Image.fromarray(images[i]);draw=ImageDraw.Draw(im)
            for uv in xy[::8]:draw.ellipse((uv[0]-1,uv[1]-1,uv[0]+1,uv[1]+1),fill=(0,255,0))
            im.save(out/(slots[i]+'.fit.png'))
        if not np.isfinite(np.asarray(locals)).all():raise InputRejected('NONFINITE_FIT')
        np.savez_compressed(out/'projection_bundle.npz',vertices=locals[0],view_vertices=np.asarray(locals),faces=faces,cameras=np.asarray(pixel_cameras),camera_points=np.asarray(camera_points),intrinsics=np.array([c.intrisics.cpu().numpy() for c in cameras]),identity=beta.detach().cpu().numpy(),states=xs.detach().cpu().numpy())
        for p,value in report['input_sha256'].items():
            if digest(p)!=value:raise ValueError('Model changed during fit')
        report['projection_bundle_sha256']=digest(out/'projection_bundle.npz')
        report['status']='FIT_GATES_PASSED'
    except Exception as e:
        report['status']='NEEDS_RETAKE' if isinstance(e,InputRejected) else 'FAILED';report['error']=str(e);raise
    finally:
        if optimizer is not None:report['optimizer_audit']=optimizer.audit
        report['resources']={'worker_elapsed_seconds':time.monotonic()-started,
            'gpu_name':torch.cuda.get_device_name(), 'torch_version':torch.__version__,
            'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved(),
            'scope':'PyTorch allocator only; excludes other processes and non-PyTorch CUDA allocations'}
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--crops',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--views',nargs='+',choices=('front','left','right'));p.add_argument('--trace-fov',action='store_true');p.add_argument('--fixed-fov',type=float);p.add_argument('--prediction-cache',type=Path);p.add_argument('--warm-fit',type=Path);p.add_argument('--warm-policy',choices=('state','identity_pose'),default='state');a=p.parse_args();run(a.repo.resolve(),a.crops.resolve(),a.output.resolve(),a.views,a.trace_fov,a.fixed_fov,a.prediction_cache,a.warm_fit,a.warm_policy)
