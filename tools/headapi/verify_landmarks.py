"""Read-only template transfer audit, labeled overlays and relative regression screen."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor/RealDenseFace')]
import common.fix_chumpy
from memento3d_head.domain.projection_regression import ProjectionPoint
from memento3d_head.pipelines.projection_regression_v1 import screen
from memento3d_head.adapters.photo_registration import PAIRS


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def points(views):
    return tuple(ProjectionPoint(view,p['name'],p['residual_interocular'],p['mesh_visible'],p['in_frame'])
                 for view,row in views.items() for p in row.get('points',[]))


def run():
    output=ROOT/'runs/experiments/rdf-landmark-verification-01';output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'runs/headapi/9bf00af2-3a4b-40e1-bd08-5706c66ef19d'
    base_path=ROOT/'runs/experiments/rdf-camera-01/projections/diagnostics.json'
    base=json.loads(base_path.read_text())['variants']['B_three']['views']
    files=[base_path,Path(__file__),ROOT/'src/memento3d_head/pipelines/projection_regression_v1.py',
           ROOT/'src/memento3d_head/domain/projection_regression.py',ROOT/'src/memento3d_head/adapters/photo_registration.py']
    result={'schema_version':1,'semantic_mapping_verified':False,'production_gate_changed':False,
        'policy':'projection_regression_v1; relative only; multiplier=1.5, additive allowance=0.005 interocular units; uncalibrated',
        'sources':['https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md',
                   'https://github.com/TimoBolkart/TF_FLAME/blob/master/fit_2D_landmarks.py'],
        'screening':{},'landmark_pairs':[{'flame_68_zero_based':a,'mediapipe_zero_based':m,'name':label} for a,m,label in PAIRS]}
    datasets={'baseline':base}
    for name,relative,variant in [
        ('same_fov_refined','rdf-warm-pose-01/control_diagnostics/diagnostics.json','state_control_21'),
        ('warm_30','rdf-warm-pose-01/state_30_diagnostics/diagnostics.json','state_30'),
        ('continued_40','rdf-warm-pose-01/continuation_40_diagnostics/diagnostics.json','continuation_40')]:
        path=ROOT/'runs/experiments'/relative;files.append(path)
        views=json.loads(path.read_text())['variants'][variant]['views'];datasets[name]=views
        result['screening'][name]=asdict(screen(points(base),points(views)))
    for name,views in datasets.items():
        for view,row in views.items():
            image=Image.open(source/'rdf_crops'/(view+'.png')).convert('RGB');draw=ImageDraw.Draw(image)
            for p in row.get('points',[]):
                if p['kind']!='static_semantic':continue
                xy=p['projected_px'];target=p['observed_px'];label=p['name']
                pair=next((a,m) for a,m,n in PAIRS if n==label)
                color='red' if p['mesh_visible'] else 'orange'
                draw.line([tuple(xy),tuple(target)],fill='yellow',width=2)
                for q,c in ((xy,color),(target,'lime')):
                    x,y=q;draw.ellipse((x-4,y-4,x+4,y+4),fill=c)
                draw.text((xy[0]+5,xy[1]-14),'F'+str(pair[0])+' '+label,fill=color)
                draw.text((target[0]+5,target[1]+5),'MP'+str(pair[1]),fill='lime')
            draw.text((8,8),name+' '+view+' | F: FLAME68, MP: MediaPipe | approximate mapping',fill='yellow')
            image.save(output/(name+'_'+view+'.png'))
    embedding=ROOT/'vendor/MICA/data/FLAME2020/landmark_embedding.npy';files.append(embedding)
    if digest(embedding)!='8095348eeafce5a02f6bd8765146307f9567a3f03b316d788a2e47336d667954':raise ValueError('Embedding pin mismatch')
    data=np.load(embedding,allow_pickle=True,encoding='latin1').item()
    models=[]
    for path in [ROOT/'vendor/MICA/data/FLAME2020/generic_model.pkl',ROOT/'vendor/RealDenseFace/weights/flame/flame2023.pkl']:
        files.append(path)
        with path.open('rb') as f:models.append(pickle.load(f,encoding='latin1'))
    if not np.array_equal(models[0]['f'],models[1]['f']):raise ValueError('Topology ordering mismatch')
    face_ids=np.asarray(data['full_lmk_faces_idx']).reshape(68);weights=np.asarray(data['full_lmk_bary_coords']).reshape(68,3)
    if not np.isfinite(weights).all() or not np.allclose(weights.sum(1),1) or (weights < -1e-6).any():raise ValueError('Invalid embedding weights')
    normalized=[]
    for model in models:
        vertices=np.asarray(model['v_template']);faces=np.asarray(model['f'])
        lm=(vertices[faces[face_ids]]*weights[:,:,None]).sum(1)
        eye_a=(lm[36]+lm[39])/2;eye_b=(lm[42]+lm[45])/2
        normalized.append((lm-(eye_a+eye_b)/2)/np.linalg.norm(eye_a-eye_b))
    delta=np.linalg.norm(normalized[0]-normalized[1],axis=1)
    result['template_transfer']={'triangle_order_equal':True,'barycentric_weights_valid':True,
        'normalization':'eye-midpoint translation and eye-center distance scale only; no rotation fit',
        'paired_landmark_template_displacements':{label:float(delta[a]) for a,_,label in PAIRS},
        'scope':'template transfer consistency only; not anatomical equivalence or measured person shape'}
    result['input_sha256']={str(p):digest(p) for p in files}
    (output/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('template_transfer','screening')},indent=2))


if __name__=='__main__':run()
