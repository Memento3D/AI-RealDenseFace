"""RDF CUDA worker: explicit ROIs, independent crop cameras, batched per-frame poses."""
import argparse,json,sys,hashlib,base64,io
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'scripts'),str(Path(__file__).resolve().parents[2]/'src')]
from head24_policy import eye_gate,rigid_map,project,InputRejected,inference_box,bbox_gate

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(repo,crops,out,settings,variant,semantic,prediction_cache):
    sys.path[:0]=[str(repo),str(Path(__file__).resolve().parents[2]/'src')]
    import common.fix_chumpy
    import cv2,torch
    from tracker import FLAMEConfig,FaceBoxConfig,RealDenseFaceInferenceConfig,RealDenseFaceInferencer,GNFlameOptimizer,GNFlameOptimizerConfig
    from common.types import FittingOutput
    from eye_components import identify
    from PIL import Image,ImageDraw
    out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((crops/'report.json').read_text());assert metadata['status']=='CROPS_PREPARED'
    report={'variant':variant,'policy':settings,'status':'RUNNING','camera_intrinsics_assumed_shared':False,'photo_quality_gate_executed':True,'gates_calibrated':False,'likeness_verified':False,'input_sha256':{},'camera_maps':[],'single_frame_registration_iterations':settings['single_iterations'],'joint_registration_iterations':settings['joint_iterations'],'neural_variant':'vitb for all three crops; differs from legacy joint vits','joint_camera_layout':'T=3,C=1,4,4','vitb_expected_weight_checksum_verified':False}
    required=[repo/'weights/flame/flame2023.pkl',repo/'weights/flame/flame_assets.npz',repo/'weights/realdenseface/vitb.pth',repo/'weights/facebox/face_box.pth']
    pins={'flame2023.pkl':'8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0','flame_assets.npz':'922cb76f3afef0c2e9801a63991de269a98e6b67b1728232493f851589151dd3','face_box.pth':'3fde8561f059dd60d876021cbd2f67d955d5e02f24c25b297913a4115e433b4d'}
    # Original front model remains vitb; its local hash is recorded, not independently pinned.
    try:
        report['crop_report_sha256']=digest(crops/'report.json')
        for slot in ('front','left','right'):
            path=crops/(slot+'.png')
            if digest(path)!=metadata['views'][slot]['crop_sha256']:raise ValueError('Crop checksum mismatch: '+slot)
            report['input_sha256'][str(path)]=digest(path)
        report['input_sha256'][str(crops/'report.json')]=report['crop_report_sha256']
        for p in required:
            actual=digest(p);report['input_sha256'][str(p)]=actual
            if p.name in pins and actual!=pins[p.name]:raise ValueError('Model pin mismatch: '+p.name)
        cfg=FLAMEConfig(flame_model_path=str(required[0]),flame_assets_path=str(required[1]),num_expressions=100)
        from common.types import InferenceOutput
        from tools.headlab26.io import key,read,write,sha
        from tools.headlab26.policy import observation_confidence,adjusted_uncertainty,robust_confidence
        from tracker.fitting.gn_flame_optimizer import EYE_REGION_VERTICES
        cache_key=key({'crops':report['crop_report_sha256'],'model_hashes':{p.name:report['input_sha256'][str(p)] for p in required},'inferencer':digest(repo/'tracker/inference/realdenseface_inferencer.py'),'worker':digest(Path(__file__))})
        prediction_cache=Path(prediction_cache);cache_data=None
        if (prediction_cache/'report.json').is_file():
            cache_meta=read(prediction_cache/'report.json')
            if cache_meta.get('key')!=cache_key or sha(prediction_cache/'predictions.npz')!=cache_meta.get('npz_sha256'):raise ValueError('Prediction cache source/model/code mismatch')
            cache_data=np.load(prediction_cache/'predictions.npz',allow_pickle=False)
        infer=None if cache_data is not None else RealDenseFaceInferencer(RealDenseFaceInferenceConfig(model_config_path=str(repo/'configs/model/vitb.yaml'),model_weights_path=str(required[2]),flame=cfg,device='cuda',compile_model=False),FaceBoxConfig(model_weights_path=str(required[3]),device='cuda'))
        optimizer=GNFlameOptimizer(GNFlameOptimizerConfig(flame=cfg,device='cuda',registration_iterations=settings['single_iterations'],boost_eye_region=variant=='rdf_baseline'))
        predictions=[];fits=[];cameras=[];images=[];originals=[];confidence_rows=[];cache_arrays={}
        semantic_report=read(Path(semantic)/'report.json')
        for slot in ('front','left','right'):
            path=crops/(slot+'.png');images.append(np.asarray(Image.open(path).convert('RGB')))
            box=inference_box(metadata['views'][slot],slot)
            fields=('vertex_coord','vertex_depth','vertex_coord_log_var','vertex_depth_log_var')
            if cache_data is None:
                prediction=infer.infer_image(images[-1],face_bbox=box)
                original={k:np.asarray(getattr(prediction,k),dtype='float32').copy() for k in fields}
            else:
                original={k:cache_data[slot+'_'+k].copy() for k in fields}
                prediction=InferenceOutput(**original,face_bbox=box,image_width=1024,image_height=1024)
            cache_arrays.update({slot+'_'+k:v for k,v in original.items()})
            label_path=Path(semantic)/(slot+'.labels.png')
            if sha(path)!=semantic_report['views'][slot]['crop_sha256'] or sha(label_path)!=semantic_report['views'][slot]['labels_sha256']:raise ValueError('Semantic observation source changed')
            confidence,classes=observation_confidence(original['vertex_coord'],np.asarray(Image.open(label_path)),settings)
            confidence_rows.append(confidence);originals.append(original)
            if variant=='rdf_occlusion':
                cv,dv=adjusted_uncertainty(original['vertex_coord_log_var'],original['vertex_depth_log_var'],confidence,EYE_REGION_VERTICES,optimizer._eye_boost_log_var_delta)
                prediction=InferenceOutput(vertex_coord=original['vertex_coord'].copy(),vertex_depth=original['vertex_depth'].copy(),vertex_coord_log_var=cv,vertex_depth_log_var=dv,face_bbox=box,image_width=1024,image_height=1024)
            result=optimizer.fit_frame(prediction,camera=None)
            if result.camera_fov_y is None:raise ValueError('No FOV returned: '+slot)
            predictions.append(prediction);fits.append(result)
            cameras.append(optimizer._build_config_camera(1024,1024,result.camera_fov_y))
            report.setdefault('observation_weights',{})[slot]={'penalized_vertices':int((confidence<.9).sum()),'glasses_vertices':int((classes==6).sum()),'no_hidden_eye_recovery':True}
            print('Head26 independent fit:',slot,'variant:',variant,'fov:',result.camera_fov_y,flush=True)
        if cache_data is None:
            prediction_cache.mkdir(parents=True,exist_ok=False);np.savez_compressed(prediction_cache/'predictions.npz',**cache_arrays);write(prediction_cache/'report.json',{'key':cache_key,'npz_sha256':sha(prediction_cache/'predictions.npz'),'neural_variant':'ViT-B','reused_by':[]})
        else:cache_data.close()
        report['neural_prediction_cache_reused']=cache_data is not None
        faces=optimizer.faces.cpu().numpy()
        def decode(beta,x):
            with torch.no_grad():return optimizer.decode(FittingOutput(identity=beta,x=x))[0].detach().cpu().numpy()
        def check_front(beta,x):
            points=decode(beta,x)
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
            ids=np.arange(3931)
            for i,slot in enumerate(('front','left','right')):
                if slot=='front' or metadata['views'][slot].get('roi_backend')!='facebox':continue
                P=cameras[i].intrisics.cpu().numpy()@cameras[i].get_w2v.cpu().numpy()[:3]
                xy,_=project(decode(beta,states[i])[ids],P)
                checks[slot]=bbox_gate(xy,inference_box(metadata['views'][slot],slot))
            return checks
        # [T=3,C=1,4,4], not [C=3,4,4]: three independent poses,
        # with a distinct crop FOV per frame. Shared identity remains a candidate.
        vm,vpm=optimizer._stack_camera_matrices(cameras);vm=vm[:,None];vpm=vpm[:,None]
        tensor=lambda a:torch.from_numpy(np.asarray(a,dtype='float32')).to('cuda')
        coords=tensor(np.stack([r['vertex_coord'] for r in originals]))/1024.
        depths=tensor(np.stack([r['vertex_depth'] for r in originals]))
        initial=fits[0].identity;states=torch.stack([f.x for f in fits]);robust=np.ones((3,5023));beta=initial;xs=states
        report['joint_history']=[]
        for refit in range(settings['joint_refits']):
            cvs=[];dvs=[]
            for i,original in enumerate(originals):
                if variant=='rdf_occlusion':cv,dv=adjusted_uncertainty(original['vertex_coord_log_var'],original['vertex_depth_log_var'],confidence_rows[i]*robust[i],EYE_REGION_VERTICES,optimizer._eye_boost_log_var_delta)
                else:cv,dv=original['vertex_coord_log_var'],original['vertex_depth_log_var']
                cvs.append(cv);dvs.append(dv)
            cv=tensor(np.stack(cvs));dv=tensor(np.stack(dvs))
            if variant=='rdf_baseline':cv=optimizer._apply_eye_region_boost(cv)
            # Each refit starts from the SAME front identity prior. Do not silently
            # move the regularization centre along with successive shape updates.
            beta,xs=optimizer._register(vm,vpm,coords,depths,cv,dv,init_identity=initial,init_x=xs,num_iters=settings['joint_iterations'])
            current=decode(beta,xs);residuals=[]
            for i in range(3):
                E=cameras[i].get_w2v.cpu().numpy();K=cameras[i].intrisics.cpu().numpy();xy,_=project(current[i],K@E[:3])
                delta=settings['huber_delta_px'];weight,residual=robust_confidence(xy,originals[i]['vertex_coord'],delta)
                if variant=='rdf_occlusion':robust[i]=np.sqrt(np.maximum(weight,settings['minimum_robust_weight']))
                residuals.append(float(np.median(residual[confidence_rows[i]>=.9])) if (confidence_rows[i]>=.9).any() else None)
            report['joint_history'].append({'refit':refit+1,'median_dense_residual_px_by_view':residuals,'scope':'same dense observations used to fit, not independent anatomy validation','identity_prior':'fixed first-frame independent fit'})
        report['front_joint_gate']=check_front(beta,xs[0])
        report['profile_joint_gates']=check_profiles(beta,xs)
        report['identity_delta_l2']=float(torch.linalg.vector_norm(beta-initial).item())
        posed=decode(beta,xs);locals=[];pixel_cameras=[];camera_points=[]
        for i in range(3):
            local=xs[i].clone();local[100:103]=0;local[115:118]=0
            local_vertices=decode(beta,local);locals.append(local_vertices)
            transform,error=rigid_map(local_vertices,posed[i])
            K=cameras[i].intrisics.detach().cpu().numpy();E=cameras[i].get_w2v.detach().cpu().numpy()
            P=K@E[:3]@transform
            pixel_cameras.append(P)
            cam=posed[i]@E[:3,:3].T+E[:3,3];camera_points.append(cam)
            report['camera_maps'].append({'view':('front','left','right')[i],'root_rigid_max_native_error':error,'fov_y':float(fits[i].camera_fov_y)})
            xy,_=project(local_vertices,P)
            im=Image.fromarray(images[i]);draw=ImageDraw.Draw(im)
            for uv in xy[::8]:draw.ellipse((uv[0]-1,uv[1]-1,uv[0]+1,uv[1]+1),fill=(0,255,0))
            im.save(out/(('front','left','right')[i]+'.fit.png'))
        if not np.isfinite(np.asarray(locals)).all():raise InputRejected('NONFINITE_FIT')
        np.savez_compressed(out/'projection_bundle.npz',vertices=locals[0],view_vertices=np.asarray(locals),faces=faces,cameras=np.asarray(pixel_cameras),camera_points=np.asarray(camera_points),intrinsics=np.array([c.intrisics.cpu().numpy() for c in cameras]),identity=beta.detach().cpu().numpy(),states=xs.detach().cpu().numpy())
        for p,value in report['input_sha256'].items():
            if digest(p)!=value:raise ValueError('Model changed during fit')
        report['projection_bundle_sha256']=digest(out/'projection_bundle.npz')
        report['status']='FIT_GATES_PASSED'
    except Exception as e:
        report['status']='NEEDS_RETAKE' if isinstance(e,InputRejected) else 'FAILED';report['error']=str(e);raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--crops',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--settings',type=Path,required=True);p.add_argument('--variant',choices=['rdf_baseline','rdf_occlusion'],required=True);p.add_argument('--semantic',type=Path,required=True);p.add_argument('--prediction-cache',type=Path,required=True);a=p.parse_args();run(a.repo.resolve(),a.crops.resolve(),a.output.resolve(),json.loads(a.settings.read_text()),a.variant,a.semantic.resolve(),a.prediction_cache.resolve())
