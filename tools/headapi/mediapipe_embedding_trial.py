"""Apply pinned direct MediaPipe embedding to existing meshes, without fitting."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tools.headapi.camera_diagnostics import sha, visible

ROOT=Path(__file__).resolve().parents[2]
PIN='8863363013fc6fe3752d4318ea02a1784970200616a54b785920cdd0568816fc'


def run():
    output=ROOT/'runs/experiments/rdf-mediapipe-105-01'
    report_path=output/'comparison.json'
    if report_path.exists():raise ValueError('Use a fresh attempt; comparison already exists')
    embedding=output/'mediapipe_landmark_embedding.npz'
    if sha(embedding)!=PIN:raise ValueError('Embedding checksum mismatch')
    with np.load(embedding,allow_pickle=False) as b:
        ids=b['landmark_indices'].astype(int);fi=b['lmk_face_idx'].astype(int);weights=b['lmk_b_coords']
    if ids.shape!=(105,) or len(set(ids))!=105 or fi.shape!=(105,) or weights.shape!=(105,3):raise ValueError('Invalid embedding shape')
    if not np.isfinite(weights).all() or not np.allclose(weights.sum(1),1) or (weights < -1e-6).any():raise ValueError('Invalid barycentric weights')
    source=ROOT/'runs/headapi/9bf00af2-3a4b-40e1-bd08-5706c66ef19d'
    meta_path=source/'rdf_crops/report.json';meta=json.loads(meta_path.read_text())
    previous_path=ROOT/'runs/experiments/rdf-camera-01/projections/diagnostics.json'
    old=json.loads(previous_path.read_text())['variants']['B_three']['views']
    result={'schema_version':1,'embedding_sha256':PIN,
        'source_url':'https://raw.githubusercontent.com/georgeretsi/smirk/main/assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz',
        'readme_url':'https://raw.githubusercontent.com/georgeretsi/smirk/main/assets/mediapipe_landmark_embedding/readme',
        'source_project_model_version':'SMIRK installer uses FLAME2020; not a certified FLAME2023 embedding',
        'semantic_mapping_verified':False,'geometry_modified':False,'production_policy_changed':False,
        'limitations':['template transfer remains unverified anatomically','MediaPipe predictions are not ground truth',
                       'self-occlusion is approximate; external occlusion not tested','regional means across different landmark sets are not directly comparable'],
        'input_sha256':{str(meta_path):sha(meta_path),str(previous_path):sha(previous_path)},
        'code_sha256':sha(Path(__file__)),'variants':{}}
    for name,folder in [('baseline',ROOT/'runs/experiments/rdf-ab-696a22fbe38149bf96d9d6b89ec65f5e/B_three'),
                        ('continued_40',ROOT/'runs/experiments/rdf-warm-pose-01/continuation_40')]:
        fit=json.loads((folder/'report.json').read_text());bundle=folder/'projection_bundle.npz'
        if sha(bundle)!=fit['projection_bundle_sha256']:raise ValueError('Bundle changed')
        result['input_sha256'][str(bundle)]=sha(bundle)
        rows={};result['variants'][name]=rows
        with np.load(bundle,allow_pickle=False) as b:
            faces=b['faces']
            if faces.shape!=(9976,3) or fi.min()<0 or fi.max()>=len(faces):raise ValueError('Embedding topology bounds invalid')
            for i,view in enumerate(fit['views_used']):
                vertices=b['view_vertices'][i];P=b['cameras'][i]
                if not np.isfinite(vertices).all():raise ValueError('Invalid geometry')
                import cv2
                center=cv2.decomposeProjectionMatrix(P)[2];center=(center[:3]/center[3]).reshape(3)
                lm=(vertices[faces[fi]]*weights[:,:,None]).sum(1)
                observed=np.asarray(meta['views'][view]['landmarks_crop'])
                norm=np.linalg.norm((observed[33]+observed[133]-observed[362]-observed[263])/2)
                image=Image.open(source/'rdf_crops'/(view+'.png')).convert('RGB');draw=ImageDraw.Draw(image)
                records=[]
                for j,(mp,point) in enumerate(zip(ids,lm)):
                    h=P@np.r_[point,1];xy=h[:2]/h[2];target=observed[mp]
                    region='brows' if j<20 else ('eyes' if j<52 else ('nose' if j<65 else 'mouth'))
                    vis=visible(vertices,faces,point,center)
                    frame=bool(h[2]>0 and 0<=xy[0]<1024 and 0<=xy[1]<1024)
                    error=float(np.linalg.norm(xy-target))
                    records.append({'mp':int(mp),'region':region,'residual_px':error,'residual_interocular':error/norm,
                        'mesh_visible':vis,'in_frame':frame,'projected_px':xy.tolist(),'observed_px':target.tolist()})
                    if vis and frame:
                        draw.line([tuple(xy),tuple(target)],fill='yellow',width=1)
                        for q,color in [(xy,'red'),(target,'lime')]:
                            x,y=q;draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
                        if mp in (4,33,133,263,362,61,291):draw.text(tuple(xy+4),'MP'+str(mp),fill='red')
                rows[view]={'points':records,'visible_regions':{r:{'count':len(vals),'mean_px':float(np.mean(vals))} for r in ('brows','eyes','nose','mouth') if (vals:=[p['residual_px'] for p in records if p['region']==r and p['mesh_visible'] and p['in_frame']])}}
                if name=='baseline':
                    nose_old=next(p for p in old[view]['points'] if p['name']=='nose_tip')
                    xy=np.asarray(nose_old['projected_px']);direct=next(p for p in records if p['mp']==4)
                    rows[view]['nose_definition_comparison']={'old_F30_to_MP1_px':nose_old['residual_px'],
                        'old_F30_to_MP4_px':float(np.linalg.norm(xy-observed[4])),
                        'direct_MP4_to_MP4_px':direct['residual_px'],
                        'direct_MP4_to_MP1_px':float(np.linalg.norm(np.asarray(direct['projected_px'])-observed[1]))}
                draw.text((8,8),name+' '+view+' | direct MP105; red projected / green observed',fill='yellow')
                image.save(output/(name+'_'+view+'.png'))
    report_path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({n:{v:{k:r[k] for k in ('visible_regions','nose_definition_comparison') if k in r} for v,r in rows.items()} for n,rows in result['variants'].items()},indent=2))


if __name__=='__main__':run()
