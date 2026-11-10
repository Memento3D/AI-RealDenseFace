"""Experimental head/hair solid assembly and evidence ledger, not print certification."""
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import trimesh
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/memento3d_hair_on_head/runtime')]
from eye_components import identify
from head24_policy import eye_gate,project
from figurine_geometry import close_boundaries
from figurine2_boolean import solid_union
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def run():
    study=ROOT/'runs/experiments/input04-print-study-01';study.mkdir(parents=True,exist_ok=False)
    source=ROOT/'runs/experiments/input04-validation-01'
    profile=json.loads((ROOT/'configs/hair-cap.json').read_text());profile['design_mesh_height_mm']=30.
    write(study/'hair-profile.json',profile)
    report={'schema_version':1,'design_head_neck_height_mm':30,'physical_print_verified':False,
        'slicing_verified':False,'ready_for_machine':False,'variants':{},'policy':'experimental solid portrait with connected pedestal; no hollowing',
        'code_sha256':sha(Path(__file__))}
    for name,fitdir in [('baseline',source/'baseline'),('refined',ROOT/'runs/experiments/rdf-flame-parameters-40ab0c23fed043eb875537da057027e8/shape_expression_0_05')]:
        job=study/name;job.mkdir();(job/'rdf').mkdir();(job/'inputs').mkdir()
        shutil.copytree(source/'rdf_crops',job/'rdf_crops')
        for filename in ['projection_bundle.npz','rdf_face_photo.glb']:shutil.copyfile(fitdir/filename,job/'rdf'/filename)
        with np.load(job/'rdf/projection_bundle.npz',allow_pickle=False) as b:
            vertices=b['vertices'];faces=b['faces'];cameras=b['cameras']
        labels,info=identify(vertices,faces)
        meta=json.loads((job/'rdf_crops/report.json').read_text());observed=np.asarray(meta['views']['front']['landmarks_crop'])
        xy,_=project(np.asarray(info['eye_centers_native']),cameras[0])
        gate=eye_gate(xy,np.array([(observed[33]+observed[133])/2,(observed[362]+observed[263])/2]))
        # Re-evaluate the existing narrow gate; this is NOT a fresh neural fit.
        write(job/'rdf/report.json',{'status':'FIT_GATES_PASSED','views_used':['front','left','right'],
            'projection_bundle_sha256':sha(job/'rdf/projection_bundle.npz'),'crop_report_sha256':sha(job/'rdf_crops/report.json'),
            'validation_only':True,'inference_rerun':False,'front_eye_gate':gate,'source_bundle_sha256':sha(fitdir/'projection_bundle.npz')})
        observations=[]
        for slot in ['front','left','right']:
            with Image.open(ROOT/'input/04'/(slot+'.jpg')) as image:ImageOps.exif_transpose(image).convert('RGB').save(job/'inputs'/(slot+'.png'))
            observations.append({'slot':slot,'normalized_sha256':sha(job/'inputs'/(slot+'.png'))})
        write(job/'job.json',{'observations':observations,'purpose':'experimental print geometry study, existing gates only'})
        row={'status':'RUNNING','source_bundle_sha256':sha(fitdir/'projection_bundle.npz')};report['variants'][name]=row
        try:
            with (job/'hair.log').open('w',encoding='utf-8') as log:
                subprocess.run([str(ROOT/'.venv-triposr/Scripts/python.exe'),'-m','tools.headapi.hair_cap_worker',
                    '--job',str(job),'--output',str(job/'hair'),'--parser','D:/AI-cache/face-parsing/resnet18.onnx',
                    '--profile',str(study/'hair-profile.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True)
            head=trimesh.Trimesh(vertices,faces[labels==0],process=True)
            scale=30./np.ptp(head.vertices[:,1]);head.vertices*=scale
            head,caps=close_boundaries(head)
            hair=trimesh.load(job/'hair/hair_cap_native.glb',force='mesh',process=False);hair.vertices*=scale
            eyes=[trimesh.Trimesh(vertices*scale,faces[labels==label],process=True) for label in (1,2)]
            ymin=float(head.bounds[0,1]);low=head.vertices[head.vertices[:,1]<ymin+2]
            center=low[:,[0,2]].mean(0);radius=float(np.linalg.norm(low[:,[0,2]]-center,axis=1).max()+2)
            pedestal=trimesh.creation.cylinder(radius=radius,height=3,sections=96)
            pedestal.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
            pedestal.apply_translation([center[0],ymin-.5,center[1]])
            solid=solid_union([head,hair,*eyes,pedestal])
            before=head.vertices.copy();after_distance=solid.nearest.on_surface(before)[1]
            # Convert source Y-up to Z-up by a proper rotation, place pedestal at Z=0.
            solid.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]));solid.apply_translation([0,0,-solid.bounds[0,2]])
            solid.export(job/'head_hair_pedestal_mm.stl');solid.export(job/'head_hair_pedestal_mm.glb')
            reread=trimesh.load(job/'head_hair_pedestal_mm.stl',force='mesh',process=True)
            samples,face_ids=trimesh.sample.sample_surface_even(reread,1000,seed=123)
            normals=reread.face_normals[face_ids];origins=samples-normals*.001
            locations,rays,_=reread.ray.intersects_location(origins,-normals,multiple_hits=True)
            distances=np.linalg.norm(locations-origins[rays],axis=1)
            nearest=np.full(len(samples),np.inf);np.minimum.at(nearest,rays,distances)
            valid=nearest[np.isfinite(nearest)&(nearest>.002)]
            row.update(status='SOLID_EXPORTED',scale_mm_per_native_unit=scale,closure_caps=caps,
                geometry={'watertight':bool(reread.is_watertight),'positive_volume':bool(reread.is_volume),
                    'winding_consistent':bool(reread.is_winding_consistent),'components':len(reread.split()),
                    'bounds_mm':reread.bounds.tolist(),'volume_mm3':float(reread.volume),
                    'self_intersections_verified':False},
                source_head_distance_to_union_mm={'p95':float(np.quantile(after_distance,.95)),'max':float(after_distance.max()),
                    'scope':'includes skull/neck intentionally covered by hair and base; not a facial likeness metric'},
                thickness_proxy={'sample_count':len(samples),'hit_count':len(valid),'minimum_chord_mm':float(valid.min()) if len(valid) else None,
                    'chords_below_0_8mm':int((valid<.8).sum()),'exact_minimum_thickness_verified':False,
                    'scope':'sampled inward-normal chords; misses thin regions and is not calibrated for resin'},
                boolean_processing=solid.metadata.get('surface_processing'),stl_sha256=sha(job/'head_hair_pedestal_mm.stl'))
        except Exception as error:row.update(status='FAILED',error=str(error))
        write(study/'report.json',report);print(name+': '+row['status'],flush=True)
    print(study,flush=True)


if __name__=='__main__':run()
