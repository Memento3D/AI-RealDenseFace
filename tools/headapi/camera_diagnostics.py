"""Read-only projection diagnostics. Approximate landmark transfer, no shape fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'vendor/RealDenseFace'),str(ROOT/'src')]
import common.fix_chumpy
from memento3d_head.adapters.contour_audit import select_contour, interpolate, JAW_MP
from memento3d_head.adapters.photo_registration import PAIRS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible(vertices,faces,point,center):
    tri=vertices[faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    direction=point-center;distance=np.linalg.norm(direction);direction/=distance
    h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h)
    valid=np.abs(det)>1e-12;inv=np.zeros_like(det);inv[valid]=1/det[valid]
    s=center-tri[:,0];u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1)
    v=q@direction*inv;t=np.einsum('ij,ij->i',e2,q)*inv
    return not bool(np.any(valid&(u>=0)&(v>=0)&(u+v<=1)&(t>1e-8)&(t<distance-1e-5)))


def run(experiment,source,output,variants=('A_front','B_three')):
    output.mkdir(parents=True,exist_ok=False)
    embedding=ROOT/'vendor/MICA/data/FLAME2020/landmark_embedding.npy'
    models=[ROOT/'vendor/MICA/data/FLAME2020/generic_model.pkl',ROOT/'vendor/RealDenseFace/weights/flame/flame2023.pkl']
    topology=[]
    for path in models:
        with path.open('rb') as f:topology.append(np.asarray(pickle.load(f,encoding='latin1')['f']))
    if not np.array_equal(*topology):raise ValueError('FLAME2020/2023 face ordering differs; transfer rejected')
    if sha(embedding)!='8095348eeafce5a02f6bd8765146307f9567a3f03b316d788a2e47336d667954':
        raise ValueError('Unsupported landmark embedding checksum')
    data=np.load(embedding,allow_pickle=True,encoding='latin1').item()
    full_faces=np.asarray(data['full_lmk_faces_idx']).reshape(68)
    full_weights=np.asarray(data['full_lmk_bary_coords']).reshape(68,3)
    meta=json.loads((source/'rdf_crops/report.json').read_text())
    result={'schema_version':1,'variants':{},'likeness_verified':False,'geometry_modified':False,
        'topology_faces_equal':True,'semantic_mapping_verified':False,
        'limitations':['FLAME2020 landmark embedding transferred to matching FLAME2023 face topology; semantic positions not independently verified',
                      'MediaPipe observations are estimates, not measured ground truth',
                      'Visibility checks mesh self-occlusion only, not hair or accessories',
                      'Jaw correspondences approximate; dynamic table unsupported above 39 degrees',
                      'Each view uses its saved expression and fitted camera; errors mix camera, shape, expression and landmark errors'],
        'source_hashes':{str(p):sha(p) for p in [embedding,*models,source/'rdf_crops/report.json']},
        'code_sha256':sha(Path(__file__))}
    for name in variants:
        folder=experiment/name;fit=json.loads((folder/'report.json').read_text())
        bundle=folder/'projection_bundle.npz'
        if sha(bundle)!=fit['projection_bundle_sha256']:raise ValueError('Bundle checksum changed')
        rows={};result['variants'][name]={'bundle_sha256':sha(bundle),'views':rows}
        with np.load(bundle,allow_pickle=False) as b:
            faces=b['faces']
            if not np.array_equal(faces,topology[1]):raise ValueError('Output topology mismatch')
            for i,slot in enumerate(fit['views_used']):
                vertices=b['view_vertices'][i];P=b['cameras'][i]
                observed=meta['views'][slot]['landmarks_crop']
                if observed is None:
                    rows[slot]={'status':'NO_OBSERVED_LANDMARKS'};continue
                observed=np.asarray(observed)
                _,rotation,center,*_=cv2.decomposeProjectionMatrix(P)
                center=(center[:3]/center[3]).reshape(3)
                full=interpolate(vertices,faces,full_faces,full_weights)
                pairs=[(full[a],m,label,'static_semantic') for a,m,label in PAIRS]
                rv=cv2.Rodrigues(rotation)[0].reshape(3)
                indices,weights,selection=select_contour(rv,data['dynamic_lmk_faces_idx'],data['dynamic_lmk_bary_coords'])
                if selection['within_table_support']:
                    jaw=interpolate(vertices,faces,indices,weights)
                    pairs += [(p,m,'jaw_'+str(k),'dynamic_contour') for k,(p,m) in enumerate(zip(jaw,JAW_MP))]
                image=Image.open(source/'rdf_crops'/(slot+'.png')).convert('RGB');draw=ImageDraw.Draw(image)
                points=[];normalizer=float(np.linalg.norm((observed[33]+observed[133]-observed[362]-observed[263])/2))
                for p,m,label,kind in pairs:
                    h=P@np.r_[p,1.];xy=h[:2]/h[2];target=observed[m]
                    in_frame=bool(h[2]>0 and 0<=xy[0]<image.width and 0<=xy[1]<image.height)
                    mesh_visible=visible(vertices,faces,p,center)
                    residual=float(np.linalg.norm(xy-target))
                    points.append({'name':label,'kind':kind,'mediapipe_index':m,'projected_px':xy.tolist(),
                        'observed_px':target.tolist(),'residual_px':residual,
                        'residual_interocular':residual/max(normalizer,1e-8),
                        'mesh_visible':mesh_visible,'in_frame':in_frame})
                    color='red' if mesh_visible else 'orange'
                    draw.line([tuple(xy),tuple(target)],fill='yellow',width=2)
                    for q,c in ((xy,color),(target,'lime')):
                        x,y=q;draw.ellipse((x-3,y-3,x+3,y+3),fill=c)
                draw.text((10,10),name+' '+slot+' | green: observed; red: projected; orange: occluded',fill='yellow')
                image.save(output/(name+'_'+slot+'.png'))
                rows[slot]={'contour_selection':selection,'points':points,
                    'visible_region_mean_px':{group:float(np.mean([p['residual_px'] for p in points if p['name'].startswith(prefix) and p['mesh_visible'] and p['in_frame']]))
                        for group,prefix in [('eyes','eye'),('nose','nose'),('mouth','mouth'),('jaw','jaw')]
                        if any(p['name'].startswith(prefix) and p['mesh_visible'] and p['in_frame'] for p in points)}}
    (output/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['A_front','B_three'])
    a=p.parse_args();run(a.experiment,a.source,a.output,a.variants)
