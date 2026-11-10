"""Experimental local image-to-solid CLI. Exports evidence; never certifies printing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import numpy as np
import trimesh
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/memento3d_hair_on_head/runtime')]
from eye_components import identify
from figurine_geometry import close_boundaries
from figurine2_boolean import solid_union
from tools.headapi.neck_print_trial import clip_above


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(input_dir,output,height):
    if not np.isfinite(height) or not 20<=height<=60:raise ValueError('Experimental head+neck design height must be 20..60mm')
    config=json.loads((ROOT/'configs/headapi.json').read_text(encoding='utf-8-sig'))
    def path(name):
        p=Path(config['paths'][name]);return p if p.is_absolute() else ROOT/p
    for slot in ['front','left','right']:
        matches=[p for p in input_dir.iterdir() if p.stem.lower()==slot and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp')]
        if len(matches)!=1:raise ValueError('Exactly one supported image required for '+slot)
    output.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':1,'status':'RUNNING','ready_for_machine':False,'physical_print_verified':False,
        'slicing_verified':False,'design_head_neck_height_mm':height,'stages':[],
        'policy':'local_image_to_solid_v1_experimental; no refinement; no back image; solid head; connected pedestal',
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tools/headapi/rdf_worker.py',
            ROOT/'tools/headapi/prepare_faces.py',ROOT/'tools/headapi/hair_cap_worker.py',ROOT/'tools/headapi/neck_print_trial.py',
            ROOT/'src/memento3d_hair_on_head/runtime/figurine2_boolean.py',ROOT/'src/memento3d_hair_on_head/runtime/figurine_geometry.py']},
        'blockers':[]}
    def save(): (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    def stage(name,python,args):
        started=time.monotonic();command=[str(python),*map(str,args)]
        with (output/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'},stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        manifest['stages'].append({'name':name,'command':command,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-started});save()
        if result.returncode:raise RuntimeError(name+' failed; see its log')
    save()
    try:
        (output/'inputs').mkdir();observations=[]
        for slot in ['front','left','right']:
            source=next(p for p in input_dir.iterdir() if p.stem.lower()==slot and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
            with Image.open(source) as image:ImageOps.exif_transpose(image).convert('RGB').save(output/'inputs'/(slot+'.png'))
            observations.append({'slot':slot,'source_path':str(source),'original_sha256':sha(source),
                                 'normalized_sha256':sha(output/'inputs'/(slot+'.png'))})
        manifest['observations']=observations
        (output/'job.json').write_text(json.dumps({'observations':observations}),encoding='utf-8')
        stage('01_crops',path('cpu_python'),['-m','tools.headapi.prepare_faces','--input',output/'inputs','--task',path('landmarker'),
            '--output',output/'rdf_crops','--rdf-python',path('rdf_python'),'--repo',path('rdf_repo')])
        stage('02_rdf',path('rdf_python'),['-m','tools.headapi.rdf_worker','--repo',path('rdf_repo'),'--crops',output/'rdf_crops','--output',output/'rdf'])
        stage('03_export',path('haircap_python'),['-m','tools.headapi.face_export','--fit',output/'rdf','--crops',output/'rdf_crops'])
        profile=json.loads((ROOT/'configs/hair-cap.json').read_text());profile['design_mesh_height_mm']=height
        (output/'hair-profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
        stage('04_hair',path('haircap_python'),['-m','tools.headapi.hair_cap_worker','--job',output,'--output',output/'hair',
            '--parser',path('parser'),'--profile',output/'hair-profile.json'])
        with np.load(output/'rdf/projection_bundle.npz',allow_pickle=False) as b:V=b['vertices'];F=b['faces']
        labels,_=identify(V,F);head=trimesh.Trimesh(V,F[labels==0],process=True);scale=height/np.ptp(head.vertices[:,1]);head.vertices*=scale
        head,caps=close_boundaries(head)
        hair=trimesh.load(output/'hair/hair_cap_native.glb',force='mesh',process=False);hair.vertices*=scale
        eyes=[trimesh.Trimesh(V*scale,F[labels==i],process=True) for i in (1,2)]
        solid=solid_union([head,hair,*eyes]);solid.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
        # Explicit provisional crop budget: remove lower 20% of head+neck height.
        z0=float(solid.bounds[0,2]);cut_z=z0+.2*height
        cut=clip_above(solid,cut_z);cut.remove_unreferenced_vertices();cut,cut_caps=close_boundaries(cut)
        ring=cut.vertices[cut.vertices[:,2]<cut_z+.001];center=ring[:,:2].mean(0)
        radius=float(np.linalg.norm(ring[:,:2]-center,axis=1).max()+1.5)
        pedestal=trimesh.creation.cylinder(radius=radius,height=3,sections=96);pedestal.apply_translation([*center,cut_z-.5])
        solid=solid_union([cut,pedestal]);solid.apply_translation([0,0,-solid.bounds[0,2]])
        final=output/'head_hair_pedestal_mm.stl';solid.export(final);solid.export(output/'head_hair_pedestal_mm.glb')
        reread=trimesh.load(final,force='mesh',process=True)
        if not reread.is_volume or not reread.is_winding_consistent or len(reread.split())!=1:raise ValueError('Final STL topology validation failed')
        samples,face_ids=trimesh.sample.sample_surface_even(reread,3000,seed=456)
        normals=reread.face_normals[face_ids];origins=samples-.001*normals
        hits,rays,_=reread.ray.intersects_location(origins,-normals,multiple_hits=True)
        distance=np.linalg.norm(hits-origins[rays],axis=1);nearest=np.full(len(samples),np.inf)
        mask=distance>.002;np.minimum.at(nearest,rays[mask],distance[mask])
        flagged=np.flatnonzero(nearest<.8)
        manifest['geometry']={'watertight':bool(reread.is_watertight),'positive_volume':bool(reread.is_volume),
            'components':len(reread.split()),'bounds_mm':reread.bounds.tolist(),'volume_mm3':float(reread.volume),
            'scale_mm_per_native_unit':float(scale),'head_closure_caps':caps,'print_crop_caps':cut_caps,
            'self_intersections_verified':False,'minimum_thickness_verified':False}
        manifest['thin_proxy']={'threshold_mm':.8,'sample_count':len(samples),'flagged_count':len(flagged),
            'locations':[{'position_mm':samples[i].tolist(),'chord_mm':float(nearest[i])} for i in flagged],
            'scope':'uncalibrated sampled normal chords, not exact thickness'}
        manifest['blockers']=['SLICER_SUPPORT_ISLAND_REVIEW_NOT_RUN','RESIN_PROFILE_NOT_VALIDATED','PHYSICAL_PRINT_NOT_VERIFIED',
                              'GLOBAL_SELF_INTERSECTION_NOT_VERIFIED','EXACT_MINIMUM_THICKNESS_NOT_VERIFIED']
        if len(flagged):manifest['blockers'].append('SHORT_CHORD_LOCATIONS_REQUIRE_REVIEW')
        manifest['artifacts']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [final,output/'head_hair_pedestal_mm.glb']}
        for row in observations:
            if sha(row['source_path'])!=row['original_sha256']:raise ValueError('Source image changed during preparation')
        manifest['status']='GEOMETRY_EXPORTED_REQUIRES_REVIEW'
    except Exception as error:
        manifest['status']='FAILED';manifest['error']=str(error);raise
    finally:save()
    print(json.dumps({'status':manifest['status'],'output':str(output),'blockers':manifest['blockers']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--head-height-mm',type=float,default=30);a=p.parse_args();prepare(a.input.resolve(),a.output.resolve(),a.head_height_mm)
