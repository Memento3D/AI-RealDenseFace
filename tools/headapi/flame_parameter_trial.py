"""Bounded FLAME coefficient refinement using the existing pinned RDF decoder."""
import json
import argparse
from pathlib import Path
import sys
from uuid import uuid4
import numpy as np
import trimesh
from scipy.optimize import lsq_linear
from tools.headapi.camera_diagnostics import sha, visible
from tools.headapi.bounded_landmark_trial import project
from tools.headapi.face_export import export_face

ROOT=Path(__file__).resolve().parents[2]


def run(source_override=None,fit_override=None):
    output=ROOT/'runs/experiments'/('rdf-flame-parameters-'+uuid4().hex);output.mkdir(parents=True)
    print(output,flush=True)
    sys.path.insert(0,str(ROOT/'vendor/RealDenseFace'))
    import common.fix_chumpy
    import torch,cv2
    from tracker import FLAMEConfig,GNFlameOptimizer,GNFlameOptimizerConfig
    from common.types import FittingOutput
    source=Path(source_override).resolve() if source_override else ROOT/'runs/headapi/9bf00af2-3a4b-40e1-bd08-5706c66ef19d'
    folder=Path(fit_override).resolve() if fit_override else ROOT/'runs/experiments/rdf-ab-696a22fbe38149bf96d9d6b89ec65f5e/B_three'
    bundle=folder/'projection_bundle.npz';fit=json.loads((folder/'report.json').read_text())
    if sha(bundle)!=fit['projection_bundle_sha256']:raise ValueError('Source bundle changed')
    repo=ROOT/'vendor/RealDenseFace';models=[repo/'weights/flame/flame2023.pkl',repo/'weights/flame/flame_assets.npz']
    for p in models:
        if sha(p)!=fit['input_sha256'][str(p)]:raise ValueError('FLAME model pin changed')
    asset=ROOT/'runs/experiments/rdf-mediapipe-105-01/mediapipe_landmark_embedding.npz'
    if sha(asset)!='8863363013fc6fe3752d4318ea02a1784970200616a54b785920cdd0568816fc':raise ValueError('Embedding changed')
    with np.load(bundle,allow_pickle=False) as b:data={k:b[k].copy() for k in b.files}
    with np.load(asset,allow_pickle=False) as b:fi=b['lmk_face_idx'];w=b['lmk_b_coords'];ids=b['landmark_indices']
    meta=json.loads((source/'rdf_crops/report.json').read_text());slots=['front','left','right']
    previous={}
    for i,view in enumerate(slots):
        observed_view=np.asarray(meta['views'][view]['landmarks_crop'])
        if observed_view.ndim!=2:raise ValueError('Observed landmarks unavailable for '+view)
        V=data['view_vertices'][i];P=data['cameras'][i]
        lm=(V[data['faces'][fi]]*w[:,:,None]).sum(1);xy=project(lm,P)
        center=cv2.decomposeProjectionMatrix(P)[2];center=(center[:3]/center[3]).reshape(3)
        previous[view]={'points':[{'region':'brows' if j<20 else ('eyes' if j<52 else ('nose' if j<65 else 'mouth')),
            'mesh_visible':visible(V,data['faces'],point,center),'in_frame':bool(0<=q[0]<1024 and 0<=q[1]<1024),
            'residual_px':float(np.linalg.norm(q-observed_view[ids[j]]))} for j,(point,q) in enumerate(zip(lm,xy))]}
    cfg=FLAMEConfig(flame_model_path=str(models[0]),flame_assets_path=str(models[1]),num_expressions=100)
    decoder=GNFlameOptimizer(GNFlameOptimizerConfig(flame=cfg,device='cuda',boost_eye_region=False))
    beta=torch.as_tensor(data['identity'],device='cuda');states=torch.as_tensor(data['states'],device='cuda')
    def decode(parameters):
        identity=beta.clone();identity[:24]+=torch.as_tensor(parameters[:24],device='cuda',dtype=torch.float32)
        state=states.clone()
        if len(parameters)>24:state[:,:12]+=torch.as_tensor(parameters[24:],device='cuda',dtype=torch.float32)
        local=state.clone();local[:,100:103]=0;local[:,115:118]=0
        with torch.no_grad():vertices=decoder.decode(FittingOutput(identity=identity,x=local))[0].cpu().numpy().astype(float)
        return vertices,identity.cpu().numpy(),state.cpu().numpy()
    base,_,_=decode(np.zeros(36));faces=data['faces'];height=float(np.ptp(base[0,:,1]))
    reproduction=float(np.abs(base-data['view_vertices']).max())
    if reproduction>1e-6:raise ValueError('Decoder cannot reproduce baseline local geometry')
    indices=[j for j,p in enumerate(previous['front']['points']) if p['region'] in ('eyes','mouth') and p['mesh_visible'] and p['in_frame']]
    observed=np.asarray(meta['views']['front']['landmarks_crop']);norm=float(np.linalg.norm((observed[33]+observed[133]-observed[362]-observed[263])/2))
    def landmarks(V):return (V[faces[fi]]*w[:,:,None]).sum(1)
    baseline_xy=project(landmarks(base[0]),data['cameras'][0])[indices]
    target=((observed[ids[indices]]-baseline_xy)/norm).ravel()
    jac=[];basis=[]
    for j in range(36):
        step=np.zeros(36);step[j]=.1;V,_,_=decode(step)
        jac.append(((project(landmarks(V[0]),data['cameras'][0])[indices]-baseline_xy)/(.1*norm)).ravel())
        basis.append((V[0]-base[0])/.1)
    J=np.stack(jac,axis=1);B=np.stack(basis,axis=-1)
    mesh=trimesh.Trimesh(base[0],faces,process=False);edges=np.unique(mesh.edges_sorted,axis=0)
    cross0=np.cross(base[0,faces[:,1]]-base[0,faces[:,0]],base[0,faces[:,2]]-base[0,faces[:,0]])
    area0=np.linalg.norm(cross0,axis=1);edge0=np.linalg.norm(base[0,edges[:,0]]-base[0,edges[:,1]],axis=1)
    result={'schema_version':1,'algorithm':'one bounded linearized FLAME coefficient step, exact decoder reevaluation and backtracking',
        'camera_changed':False,'weights_changed':False,'training_views':['front'],'evaluation_views':['left','right'],
        'held_out_scope':'held out of refinement only; baseline used all views','decoder_reproduction_max_native':reproduction,
        'embedding_semantics_verified':False,'self_intersections_verified':False,'ready_for_machine':False,
        'coefficient_units':'native decoder coefficients; bounds are not anatomical standard deviations',
        'regularization':'0.01 coefficient L2 plus 0.5 RMS global displacement/head height',
        'guard':'<=0.3% head-height displacement, <=30deg normal change, >=20% area, <=15% edge length change',
        'input_sha256':{str(p):sha(p) for p in [bundle,asset,*models,source/'rdf_crops/report.json',Path(__file__)]},'variants':{}}
    for mode,count in [('shape',24),('shape_expression',36)]:
        A=np.vstack([J[:,:count],.01*np.eye(count),.5*B[:,:,:count].reshape(-1,count)/(height*np.sqrt(len(base[0])))])
        rhs=np.r_[target,np.zeros(count+len(base[0])*3)]
        for bound in [.05,.15,.3]:
            name=mode+'_'+str(bound).replace('.','_');dest=output/name;dest.mkdir()
            solution=lsq_linear(A,rhs,bounds=(-bound,bound),tol=1e-8,max_iter=100)
            if not solution.success:raise ValueError('Parameter solve failed')
            alpha=1.
            for _ in range(20):
                params=solution.x*alpha;V,identity,state=decode(params)
                cross=np.cross(V[0,faces[:,1]]-V[0,faces[:,0]],V[0,faces[:,2]]-V[0,faces[:,0]])
                cosine=np.einsum('ij,ij->i',cross0,cross)/np.maximum(area0*np.linalg.norm(cross,axis=1),1e-20)
                edge1=np.linalg.norm(V[0,edges[:,0]]-V[0,edges[:,1]],axis=1)
                displacement=float(np.linalg.norm(V-base,axis=2).max())
                if displacement<=.003*height and np.all(cosine>=np.cos(np.radians(30))) and np.all(np.linalg.norm(cross,axis=1)>=.2*area0) and np.max(np.abs(edge1/np.maximum(edge0,1e-15)-1))<=.15:break
                alpha*=.5
            else:raise ValueError('No parameter step meets geometry guards')
            payload={**data,'vertices':V[0],'view_vertices':V,'identity':identity,'states':state}
            np.savez_compressed(dest/'projection_bundle.npz',**payload)
            export_face(dest/'projection_bundle.npz',source/'rdf_crops',slots,dest)
            row={'coefficient_bound':bound,'step_scale':alpha,'coefficient_delta':params.tolist(),
                'max_displacement_native':displacement,'normal_reversals':int((cosine<=0).sum()),
                'max_normal_change_deg':float(np.degrees(np.arccos(np.clip(cosine,-1,1))).max()),
                'bundle_sha256':sha(dest/'projection_bundle.npz'),'views':{}}
            for i,view in enumerate(slots):
                projected=project(landmarks(V[i]),data['cameras'][i]);obs=np.asarray(meta['views'][view]['landmarks_crop'])
                c=cv2.decomposeProjectionMatrix(data['cameras'][i])[2];c=(c[:3]/c[3]).reshape(3)
                records=[];lm=landmarks(V[i])
                for j,p in enumerate(previous[view]['points']):
                    if p['mesh_visible'] and p['in_frame']:
                        records.append({'mp':int(ids[j]),'region':p['region'],'baseline_px':p['residual_px'],
                            'candidate_px':float(np.linalg.norm(projected[j]-obs[ids[j]])),
                            'visible':visible(V[i],faces,lm[j],c)})
                row['views'][view]={'regions':{r:{'baseline_px':float(np.mean([p['baseline_px'] for p in records if p['region']==r])),
                    'candidate_px':float(np.mean([p['candidate_px'] for p in records if p['region']==r])),
                    'visibility_lost':sum(not p['visible'] for p in records if p['region']==r)} for r in ['eyes','mouth','nose','brows']},'points':records}
            result['variants'][name]=row
            (output/'experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
            print(name+' completed',flush=True)
    print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--fit',type=Path)
    a=p.parse_args();run(a.source,a.fit)
