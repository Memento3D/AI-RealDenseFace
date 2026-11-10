"""Subprocess-only MVP accessory fitting. Never imports SDKs into the API process."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from tools.customization.contract import library,POLICY,confined
from tools.customization.fit_trial import (world_parts,ear_support_anchors,fit_front_clearance,
    refine_temples,surface_samples)
from tools.customization.fit_math import head_frame,profile_for_head,glasses_transform,transform_points
from tools.headapi.assembly_math import AXES
from tools.headapi.service import load_config,sha,write


def export(parts,path):
    scene=trimesh.Scene()
    for name,mesh in parts.items():scene.add_geometry(mesh,node_name=name,geom_name=name)
    scene.apply_transform(np.linalg.inv(AXES));scene.export(path)


def fit(job,output,config):
    job=Path(job).resolve();output=Path(output).resolve()
    root,catalog,profile=library(config)
    output.mkdir(parents=True,exist_ok=False)
    row=json.loads((job/'job.json').read_text(encoding='utf8'))
    report=json.loads((job/'render/report.json').read_text(encoding='utf8'))
    if report.get('output_kind')!='customization_base':raise ValueError('Expected customization base output')
    base=job/'render/scene.glb'
    if sha(base)!=report['scene_sha256']:raise ValueError('Base checksum mismatch')
    head=world_parts(base)['Personalized_Complete_Head']
    with np.load(job/'rdf/projection_bundle.npz',allow_pickle=False) as bundle:
        vertices=bundle['vertices'];faces=bundle['faces']
    embedding=Path(config['root'])/config['assembly']['assets']['embedding']['path']
    if sha(embedding)!=config['assembly']['assets']['embedding']['sha256']:raise ValueError('Landmark embedding pin changed')
    with np.load(embedding,allow_pickle=False) as e:
        landmarks=np.einsum('ij,ijk->ik',e['full_weights'],vertices[faces[e['full_faces']]])
    M=np.asarray(report['matrix_native_to_working']);landmarks=transform_points(landmarks,M)
    origin,basis,_=head_frame(landmarks);ears=ear_support_anchors(vertices,M,np.zeros(3),landmarks)
    reference_basis=np.asarray(profile['reference_head_basis'])
    local=(head.vertices-origin)@basis
    crown=float(local[:,2].max());scale=crown/float(profile['reference_eye_to_crown_m'])
    lower,upper=profile['cap_crown_scale_bounds']
    if not lower<=scale<=upper:raise ValueError('Cap crown scale requires manual review')
    entries=[];diagnostics={}
    for asset in catalog['assets']:
        aid=asset['id'];source=confined(root,asset['preview_file'])
        if aid=='graduation-cap':
            working=np.eye(4);working[:3,:3]=basis@reference_basis.T*scale;working[:3,3]=origin
            gltf=np.linalg.inv(AXES)@working@AXES
            entries.append({'asset_id':aid,'asset_version':asset['version'],'geometry_mode':'shared_transform',
                'sha256':sha(source),'source_sha256':asset['preview_sha256'],
                'asset_to_base_matrix_column_major':gltf.T.ravel().tolist()})
            diagnostics[aid]={'crown_scale':scale,'print_ready':False}
            continue
        parts=world_parts(source)
        if aid=='hair-side-part':
            all_vertices=np.vstack([mesh.vertices for mesh in parts.values()])@reference_basis
            lo,hi=all_vertices.min(0),all_vertices.max(0)
            scalp=local[local[:,2]>.012]
            if len(scalp)<3 or np.any(hi-lo<1e-6):raise ValueError('Invalid scalp envelope')
            target_min=np.array([scalp[:,0].min()-.003,scalp[:,1].min()-.004,-.018])
            target_max=np.array([scalp[:,0].max()+.003,scalp[:,1].max()+.004,crown+.008])
            factors=(target_max-target_min)/(hi-lo)
            if np.any(factors<.5) or np.any(factors>2):raise ValueError('Hair scale requires manual review')
            for mesh in parts.values():
                mesh.vertices=((mesh.vertices@reference_basis-lo)*factors+target_min)@basis.T+origin
            diagnostics[aid]={'axis_scale':factors.tolist(),'scope':'Scalp-envelope initialization, visual review required'}
        else:
            glasses_profile,width=profile_for_head(profile['glasses'],ears,landmarks,np.vstack([m.vertices for m in parts.values()]))
            if width['requires_wider_asset']:raise ValueError('Glasses require a wider asset')
            G,initial=glasses_transform(landmarks,glasses_profile)
            G,front=fit_front_clearance(parts,head,landmarks,glasses_profile,G)
            if front['requires_manual_review']:raise ValueError('Glasses front clearance requires manual review')
            temples=[]
            for mesh in parts.values():
                source_vertices=mesh.vertices.copy();mesh.apply_transform(G)
                temples.append(refine_temples(mesh,source_vertices,head,landmarks,glasses_profile,ears))
            clearance=surface_samples(head,np.vstack([m.vertices for m in parts.values()]))
            if clearance['samples_below_minus_1mm']>0:raise ValueError('Glasses sampled clearance requires manual review')
            diagnostics[aid]={'initial':initial,'width':width,'front':front,'temples':temples,'sampled_clearance':clearance}
        target=output/(aid+'.glb');export(parts,target)
        entries.append({'asset_id':aid,'asset_version':asset['version'],'geometry_mode':'fitted_geometry',
            'sha256':sha(target),'source_sha256':asset['preview_sha256'],
            'asset_to_base_matrix_column_major':np.eye(4).T.ravel().tolist()})
    # Revalidate inputs before publishing the manifest.
    library(config)
    if sha(base)!=report['scene_sha256']:raise ValueError('Base changed during fit')
    manifest={'schema_version':1,'policy':POLICY,'job_id':row['job_id'],'attempt_id':row['attempt_id'],
        'catalog_version':catalog['selection'],'catalog_sha256':config['customization']['catalog_sha256'],
        'profile_sha256':config['customization']['profile_sha256'],
        'base_model':{'sha256':sha(base),'output_kind':'customization_base'},
        'default_selection':catalog['default_selection'],'groups':catalog['groups'],
        'assets':entries,'coordinate_system':'glTF Y-up; column-major matrices; one common model root',
        'quality_status':'NEEDS_VISUAL_REVIEW','ready_for_machine':False,
        'scope':'Per-job preview fit; not a full collision or print certificate'}
    write(output/'fit-report.json',{'policy':POLICY,'job_id':row['job_id'],'assets':diagnostics,'ready_for_machine':False})
    write(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();fit(args.job,args.output,load_config(args.job/'config.snapshot.json'))
