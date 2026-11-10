"""Read-only sampled thin-feature locations and horizontal contour checks."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from tools.triposr_probe.audit import render

ROOT=Path(__file__).resolve().parents[2]


def run(study_override=None):
    study=Path(study_override).resolve() if study_override else ROOT/'runs/experiments/input04-print-study-01'
    destination=study/'diagnostics';destination.mkdir(exist_ok=False)
    study_report=json.loads((study/'report.json').read_text());results={}
    embedding=ROOT/'runs/experiments/rdf-mediapipe-105-01/mediapipe_landmark_embedding.npz'
    with np.load(embedding,allow_pickle=False) as e:fi=e['lmk_face_idx'];weights=e['lmk_b_coords']
    for name in ['baseline','refined']:
        job=study/name;path=job/'head_hair_pedestal_mm.stl';mesh=trimesh.load(path,force='mesh',process=True)
        view=mesh.copy();view.vertices=np.column_stack((-view.vertices[:,1],view.vertices[:,0],view.vertices[:,2]));render(view,destination/(name+'.png'),gray=True)
        samples,ids=trimesh.sample.sample_surface_even(mesh,3000,seed=456)
        normals=mesh.face_normals[ids];origins=samples-.001*normals
        hits,rays,_=mesh.ray.intersects_location(origins,-normals,multiple_hits=True)
        distances=np.linalg.norm(hits-origins[rays],axis=1);nearest=np.full(len(samples),np.inf)
        # Discard near-origin numerical/self hits before taking nearest exit.
        usable=distances>.002;np.minimum.at(nearest,rays[usable],distances[usable])
        flags=np.flatnonzero(nearest<.8)
        with np.load(job/'rdf/projection_bundle.npz',allow_pickle=False) as b:
            vertices=b['vertices'];faces=b['faces']
        scale=study_report['variants'][name]['scale_mm_per_native_unit']
        # Existing export transform: source Y-up -> Z-up; base starts head_ymin-2.
        shift=float((vertices*scale)[:,1].min()-2)+study_report['variants'][name].get('additional_export_z_shift_mm',0)
        def transform(points):
            return np.column_stack((points[:,0],-points[:,2],points[:,1]-shift))
        hair=trimesh.load(job/'hair/hair_cap_native.glb',force='mesh',process=False);hair.vertices=transform(hair.vertices*scale)
        source_mesh=trimesh.Trimesh(transform(vertices*scale),faces,process=False)
        if len(flags):
            hd=hair.nearest.on_surface(samples[flags])[1];fd=source_mesh.nearest.on_surface(samples[flags])[1]
        else:hd=fd=[]
        records=[{'position_mm':samples[j].tolist(),'normal':normals[j].tolist(),'chord_mm':float(nearest[j]),
                  'nearest_source_component':'hair' if a<b else 'head_or_eyes',
                  'hair_distance_mm':float(a),'source_face_distance_mm':float(b)} for j,a,b in zip(flags,hd,fd)]
        lm=(vertices[faces[fi]]*weights[:,:,None]).sum(1)*scale;lm=transform(lm)
        distances_to_union=mesh.nearest.on_surface(lm)[1]
        heights=np.arange(.025,float(mesh.bounds[1,2]),.05)
        sections=mesh.section_multiplane(plane_origin=[0,0,0],plane_normal=[0,0,1],heights=heights)
        bad=[{'z_mm':float(h),'entity_count':len(section.entities)} for h,section in zip(heights,sections) if section is not None and not section.is_closed]
        results[name]={'stl_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'thin_feature_proxy':{
            'sample_count':len(samples),'threshold_mm':.8,'flagged_count':len(records),'locations':records,
            'scope':'inward-normal chord screen; not exact minimum thickness, resin calibration or mechanical strength'},
            'embedded_facial_surface_distance_mm':{'p95':float(np.quantile(distances_to_union,.95)),'max':float(distances_to_union.max()),
                'scope':'105 embedding samples including eye/lip regions; not a whole-surface deviation bound or likeness score'},
            'horizontal_contours':{'step_mm':.05,'planes':len(heights),'nonempty_planes':sum(s is not None for s in sections),
                'open_contour_planes':bad,'scope':'trimesh cross-sections only; no exposure, support, island or printer-file validation'},
            'physical_print_verified':False,'ready_for_machine':False}
        print(name+': '+str(len(records))+' thin-proxy flags; '+str(len(bad))+' open contour planes',flush=True)
    (destination/'report.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path);a=p.parse_args();run(a.study)
