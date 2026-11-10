"""Conservative camera-only backtracking using TRAIN regional errors only."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2,trimesh
from PIL import Image,ImageDraw
from tools.headapi.face_texture_trial import ROOT,sha
from tools.headapi.geometry_refinement_math import camera_update,project,mesh_guard
from tools.headapi.geometry_camera_trial import gray_sheet,visibility


def run(root):
    root=Path(root).resolve();experiment=json.loads((root/'fit/experiment.json').read_text())
    embedding=ROOT/'runs/experiments/rdf-mediapipe-105-01/mediapipe_landmark_embedding.npz'
    if sha(embedding)!=experiment['input_sha256'][str(embedding)]:raise ValueError('Embedding changed')
    with np.load(embedding,allow_pickle=False) as b:fi=b['lmk_face_idx'];bary=b['lmk_b_coords'];ids=b['landmark_indices'].astype(int)
    rows=[]
    for case in experiment['cases']:
        name=case['case'];source=Path(case['source']);folder=root/'fit'/name
        with np.load(folder/'baseline/projection_bundle.npz',allow_pickle=False) as b:data={k:b[k].copy() for k in b.files}
        v=data['view_vertices'];f=data['faces'];height=float(np.ptp(v[0,:,1]));lm=(v[:,f[fi]]*bary[None,:,:,None]).sum(2)
        original=next(r for r in case['variants'] if r['name']=='baseline');camera=next(r for r in case['variants'] if r['name']=='camera_only')
        observations=json.loads((source/'report.json').read_text());cameras=[];alphas=[];row={'name':'camera_guarded','case':name,'selection_uses_holdout_targets':False,'rule':'backtrack each camera until ALL train-region mean <= baseline*1.05+0.10px and total train mean does not increase','views':[]}
        destination=folder/'camera_guarded';destination.mkdir()
        for i,slot in enumerate(['front','left','right']):
            observed=np.asarray(observations['views'][slot]['landmarks_crop'])[ids]
            points=original['views'][i]['points'];frozen=np.array([int(mp) in {p['mediapipe'] for p in points} for mp in ids])
            hold=frozen&(ids%3==0);train=frozen&~hold
            regions=np.array(['brows']*20+['eyes']*32+['nose']*13+['mouth']*40)
            baseline_xy=project(lm[i],data['cameras'][i]);before=np.linalg.norm(baseline_xy-observed,axis=1)
            u=np.asarray(camera['camera_normalized_parameters'][i]);alpha=1.;accepted=False
            for _ in range(20):
                p=camera_update(data['cameras'][i],v[i],u*alpha,height)
                err=np.linalg.norm(project(lm[i],p)-observed,axis=1)
                regional=all(err[train&(regions==r)].mean()<=before[train&(regions==r)].mean()*1.05+.10 for r in ['eyes','mouth','nose','brows'] if (train&(regions==r)).any())
                if regional and err[train].mean()<=before[train].mean():accepted=True;break
                alpha*=.5
            if not accepted:alpha=0.;p=data['cameras'][i].copy()
            cameras.append(p);alphas.append(alpha);xy=project(lm[i],p);err=np.linalg.norm(xy-observed,axis=1)
            image=Image.open(source/(slot+'.png')).convert('RGB');vis,frame=visibility(v[i],f,p,lm[i],image.size);draw=ImageDraw.Draw(image)
            view={'view':slot,'visibility_lost':int((frozen&~vis).sum()),'out_of_frame':int((frozen&~frame).sum()),'train_mean_px':float(err[train].mean()),'holdout_mean_px':float(err[hold].mean()),'baseline_holdout_mean_px':float(before[hold].mean()),'regions':{},'points':[]}
            for region in ['eyes','mouth','nose','brows']:
                mask=hold&(regions==region)
                if mask.any():view['regions'][region]={'count':int(mask.sum()),'baseline_mean_px':float(before[mask].mean()),'candidate_mean_px':float(err[mask].mean())}
            for j in np.where(frozen)[0]:
                a=xy[j];b=observed[j];draw.line([tuple(a),tuple(b)],fill='yellow' if hold[j] else 'cyan',width=1);draw.ellipse((a[0]-2,a[1]-2,a[0]+2,a[1]+2),fill='red' if vis[j] else 'orange')
                view['points'].append({'mediapipe':int(ids[j]),'region':str(regions[j]),'holdout':bool(hold[j]),'baseline_px':float(before[j]),'candidate_px':float(err[j]),'visible':bool(vis[j]),'in_frame':bool(frame[j])})
            draw.text((8,8),'camera_guarded '+slot+' | train-only backtracking',fill='yellow');image.save(destination/(slot+'.overlay.png'));row['views'].append(view)
        cameras=np.asarray(cameras);cam_points=[];intrinsics=[]
        for i,p in enumerate(cameras):
            k,r,c,*_=cv2.decomposeProjectionMatrix(p);k/=k[2,2];center=(c[:3]/c[3]).reshape(3);cam_points.append((v[i]-center)@r.T);intrinsics.append(k)
        payload={**data,'cameras':cameras,'camera_points':np.asarray(cam_points),'intrinsics':np.asarray(intrinsics)}
        np.savez_compressed(destination/'projection_bundle.npz',**payload)
        mesh=trimesh.Trimesh(data['vertices'],f,process=False);mesh.visual.vertex_colors=[180,180,180,255];mesh.export(destination/'face_gray.glb')
        gray_sheet(data['vertices'],f,data['vertices'],destination/'gray.png')
        row['camera_step_scales']=alphas;row['coefficient_delta']=[0.]*34;row['mesh_guard']=mesh_guard(v,v,f,height)
        row['bundle_sha256']=sha(destination/'projection_bundle.npz')
        row['review_screen_passed']=all(w['visibility_lost']==0 and w['out_of_frame']==0 and all(r['candidate_mean_px']<=1.05*r['baseline_mean_px']+.25 for r in w['regions'].values()) for w in row['views'])
        (destination/'report.json').write_text(json.dumps(row,indent=2),encoding='utf-8');rows.append(row)
        print(name+' guarded camera '+str(alphas),flush=True)
    (root/'guarded-camera.json').write_text(json.dumps({'baseline_promoted':False,'input_experiment_sha256':sha(root/'fit/experiment.json'),'code_sha256':sha(Path(__file__)),'cases':rows},indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();run(args.root)
