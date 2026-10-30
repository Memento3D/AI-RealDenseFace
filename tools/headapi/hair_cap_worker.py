"""Fit an experimental solid hair cap to RDF geometry and supplied photos."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'src/memento3d_hair_on_head/runtime')]
from eye_components import identify
from headlab_raster import surface_depth_fast
from scalp_cap import clean_mask,infer_boundary,cap_patch,fit_thickness,close_cap,radial_samples,scalp_origin


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def export_render_scene(head_path,hair_path,scene_path):
    head=trimesh.load(head_path,force='mesh',process=False)
    hair=trimesh.load(hair_path,force='mesh',process=False)
    if not isinstance(head,trimesh.Trimesh) or not isinstance(hair,trimesh.Trimesh):
        raise ValueError('Render scene components must be triangle meshes')
    if not len(head.vertices) or not len(hair.vertices):
        raise ValueError('Render scene components must not be empty')
    scene=trimesh.Scene()
    scene.add_geometry(head,geom_name='RDF_Head',node_name='RDF_Head')
    scene.add_geometry(hair,geom_name='Hair_Cap',node_name='Hair_Cap')
    scene.export(scene_path)
    loaded=trimesh.load(scene_path,force='scene',process=False)
    if not {'RDF_Head','Hair_Cap'}<=set(loaded.graph.nodes_geometry):
        raise ValueError('Render scene is missing a component node')
    return {
        'schema_version':1,
        'asset':'scene.glb',
        'coordinate_frame':'RDF native coordinates',
        'units':'native RDF units; not calibrated millimeters',
        'components':[
            {'id':'head','kind':'head','node':'RDF_Head'},
            {'id':'hair-cap','kind':'hair_cap','node':'Hair_Cap'},
        ],
        'geometry_union_performed':False,
    }


def load_profile(path):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data.get('schema_version')!=1:
        raise ValueError('Unsupported hair-cap policy version')
    if not np.isfinite(data.get('design_mesh_height_mm',np.nan)) or not 1<=data['design_mesh_height_mm']<=100:
        raise ValueError('Invalid assumed design mesh height')
    scaffold=data.get('scaffold',{})
    if any(not isinstance(scaffold.get(key),int) or scaffold[key]<minimum
           for key,minimum in (('theta_samples',64),('phi_samples',16),('cap_rings',8))):
        raise ValueError('Invalid hair-cap scaffold sampling')
    volume=data.get('volume',{})
    if not all(np.isfinite(volume.get(key,np.nan)) for key in ('base_mm','edge_mm','min_mm','max_mm')):
        raise ValueError('Invalid hair-cap volume policy')
    if not 0<volume['edge_mm']<=volume['min_mm']<=volume['base_mm']<=volume['max_mm']<=10:
        raise ValueError('Hair-cap thickness bounds are inconsistent')
    if not np.isfinite(data.get('inset_mm',np.nan)) or not 0<=data['inset_mm']<=2:
        raise ValueError('Invalid hair-cap inset')
    return data


def write_fit_overlays(output,slots,images,masks,mask_valid,view_surface_points,
                       directions,offsets,patch,cameras):
    rows=[];names=[]
    for slot,image,mask,reliable,surface,camera in zip(
            slots,images,masks,mask_valid,view_surface_points,cameras):
        mesh=trimesh.Trimesh(vertices=surface+directions*offsets[:,None],
            faces=patch.faces,process=False)
        depth=surface_depth_fast(mesh.vertices,mesh.faces,camera,image.size)
        predicted=np.isfinite(depth)
        if reliable:
            iou=float((predicted&mask).sum()/max((predicted|mask).sum(),1))
            coverage=float((predicted&mask).sum()/max(mask.sum(),1))
        else:
            iou=None;coverage=None
        rgb=np.asarray(image.convert('RGB')).copy()
        rgb[mask]=(rgb[mask]*.55+np.array([35,210,75])*.45).astype(np.uint8)
        edge=predicted&~ndi.binary_erosion(predicted)
        rgb[edge]=[235,55,205]
        name=slot+'.fit.png'
        Image.fromarray(rgb).save(output/name);names.append(name)
        rows.append({'view':slot,'hair_mask_reliable':bool(reliable),
            'mask_iou':iou,'mask_coverage':coverage,
            'scope':'projected cap/mask consistency under fitted cameras; not independent likeness ground truth'})
    return rows,names


def run(job,output,parser_path,profile_path):
    job=Path(job);output=Path(output);parser_path=Path(parser_path)
    if output.exists():
        raise ValueError('Hair-cap output already exists; attempts are immutable')
    rdf=job/'rdf';bundle_path=rdf/'projection_bundle.npz'
    rdf_report=json.loads((rdf/'report.json').read_text(encoding='utf-8'))
    crop_report_path=job/'rdf_crops/report.json'
    crop_report=json.loads(crop_report_path.read_text(encoding='utf-8'))
    if rdf_report.get('status')!='FIT_GATES_PASSED':
        raise ValueError('Hair cap requires a successful RDF fit')
    job_record=json.loads((job/'job.json').read_text(encoding='utf-8'))
    observation_hashes={}
    for entry in job_record.get('observations',[]):
        path=job/'inputs'/(entry['slot']+'.png')
        observation_hashes[str(path)]=entry['normalized_sha256']
        if digest(path)!=entry['normalized_sha256']:
            raise ValueError('Normalized input checksum mismatch: '+entry['slot'])
    if digest(bundle_path)!=rdf_report.get('projection_bundle_sha256'):
        raise ValueError('RDF bundle checksum mismatch')
    if digest(crop_report_path)!=rdf_report.get('crop_report_sha256'):
        raise ValueError('RDF crop-report checksum mismatch')
    slots=rdf_report.get('views_used',[])
    if slots!=['front','left','right']:
        raise ValueError('Hair-cap MVP requires RDF front, left and right views')
    profile=load_profile(profile_path)
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','policy_version':profile['schema_version'],
        'source_views':list(slots),'back_view_used':False,'ready_for_machine':False,
        'physical_scale_verified':False,'hair_flow_or_part_inferred':False,
        'source_head_modified':False,'input_sha256':{},'units':'assumed design millimeters; physical scale unverified',
        'policy_sha256':digest(profile_path)}
    try:
        from tools.headapi.hair_parser import HairParser
        parser=HairParser(parser_path)
        report['parser_sha256']=digest(parser_path)
        report['rdf_bundle_sha256']=digest(bundle_path)
        report['crop_report_sha256']=digest(crop_report_path)
        with np.load(bundle_path,allow_pickle=False) as data:
            vertices=data['vertices'].copy()
            view_vertices=data['view_vertices'].copy()
            faces=data['faces'].copy()
            cameras=data['cameras'].copy()
        if view_vertices.shape[0]!=3 or cameras.shape[0]!=3:
            raise ValueError('RDF view geometry does not match the three-view MVP')
        face_labels,topology=identify(vertices,faces)
        head_faces=faces[face_labels==0]
        head_ids=np.unique(head_faces)
        remap=np.full(len(vertices),-1,dtype=np.int64)
        remap[head_ids]=np.arange(len(head_ids))
        head_faces=remap[head_faces]
        source_views=view_vertices[:,head_ids]
        source_head=trimesh.Trimesh(vertices=source_views[0],faces=head_faces,process=False)
        height=float(np.ptp(source_head.vertices[:,1]))
        if not np.isfinite(height) or height<=0:
            raise ValueError('RDF head has invalid vertical extent')
        scale=float(profile['design_mesh_height_mm'])/height
        source_views*=scale
        view_heads=[trimesh.Trimesh(vertices=v.copy(),faces=head_faces,process=False) for v in source_views]
        head=view_heads[0]
        eye_vertex_ids=[np.unique(faces[face_labels==label]) for label in (1,2)]
        view_eye_centers=np.asarray([[view_vertices[i,ids].mean(0)*scale for ids in eye_vertex_ids] for i in range(3)])
        eye_centers=view_eye_centers[0]
        scale_to_native=np.diag([1/scale,1/scale,1/scale,1.])
        image_matrices=[];images=[];labels=[];masks=[];hair_view_valid=[];mask_status=[]
        for index,slot in enumerate(slots):
            image_path=job/'rdf_crops'/(slot+'.png')
            crop_row=crop_report['views'].get(slot)
            if crop_row is None:
                raise ValueError('RDF crop metadata is missing view: '+slot)
            if digest(job/'rdf_crops'/(slot+'.png'))!=crop_row.get('crop_sha256'):
                raise ValueError('RDF crop checksum mismatch: '+slot)
            source_path=job/'inputs'/(slot+'.png')
            if digest(source_path)!=next((row['normalized_sha256'] for row in job_record['observations'] if row['slot']==slot),None):
                raise ValueError('Normalized input checksum mismatch: '+slot)
            image=Image.open(image_path).convert('RGB')
            label=parser(image)
            if int((label==17).sum())<100:
                mask=np.zeros(label.shape,dtype=bool);hair_view_valid.append(False)
                mask_status.append({'view':slot,'used_for_hair_fit':False,'hair_pixels':int((label==17).sum()),
                    'reason':'parser returned insufficient hair pixels'})
            else:
                mask,cleaning=clean_mask(image,label);hair_view_valid.append(True)
                mask_status.append({'view':slot,'used_for_hair_fit':True,'hair_pixels':int((label==17).sum()),
                    'cleaning':cleaning})
            camera=cameras[index]@scale_to_native
            if camera.shape!=(3,4) or not np.isfinite(camera).all():
                raise ValueError('Invalid camera transform for hair view: '+slot)
            image_matrices.append(camera);images.append(image);labels.append(label);masks.append(mask)
            report['input_sha256'][str(source_path)]=digest(source_path)
            report['input_sha256'][str(image_path)]=digest(image_path)
            Image.fromarray(mask.astype(np.uint8)*255).save(output/(slot+'.hair-mask.png'))
        face={'head':head,'view_heads':view_heads,'eye_centers':eye_centers,
            'view_eye_centers':view_eye_centers,'cameras':image_matrices,'images':images,
            'labels':labels,'hair_view_valid':hair_view_valid,'points':head.vertices,'head_corners':np.ones(len(head.vertices),dtype=bool),
            'data':{'weights':np.ones((len(head.vertices),3),dtype=float)}}
        back_path=job/'inputs/back.png'
        if back_path.is_file():
            if digest(back_path)!=observation_hashes.get(str(back_path)):
                raise ValueError('Normalized input checksum mismatch: back')
            from back_view import prepare_back
            back_image,back_label,back_camera,back_report=prepare_back(back_path,parser_path,face)
            back_mask,back_cleaning=clean_mask(back_image,back_label)
            face['images'].append(back_image);face['labels'].append(back_label)
            face['cameras'].append(back_camera);face['view_heads'].append(head)
            face['view_eye_centers']=np.concatenate((face['view_eye_centers'],eye_centers[None]),axis=0)
            face['hair_view_valid'].append(True)
            face['points']=head.vertices
            masks.append(back_mask)
            report['back_view_used']=True
            report['back_view_alignment']=back_report
            report['input_sha256'][str(back_path)]=digest(back_path)
            mask_status.append({'view':'back','used_for_hair_fit':True,
                'hair_pixels':int((back_label==17).sum()),'cleaning':back_cleaning})
            Image.fromarray(back_mask.astype(np.uint8)*255).save(output/'back.hair-mask.png')
        if not hair_view_valid[0]:
            raise ValueError('Front photo has insufficient parser hair pixels; cannot fit the personalized hairline')
        if sum(face['hair_view_valid'])<2:
            raise ValueError('At least two views must provide reliable hair masks')
        report['hair_mask_views']=mask_status
        field=infer_boundary(face,masks,profile['scaffold'])
        patch,directions,radial,boundary,fraction=cap_patch(face,field,profile['scaffold'])
        view_surface_points=[]
        for view_head,view_eye in zip(face['view_heads'],face['view_eye_centers']):
            origin=scalp_origin(view_head,view_eye)
            radius,_=radial_samples(view_head,origin,directions)
            view_surface_points.append(origin+directions*radius[:,None])
        offsets,volume=fit_thickness(face,masks,patch,directions,radial,fraction,
            profile['volume'],view_surface_points=view_surface_points)
        hair=close_cap(patch,directions,offsets,boundary,float(profile['inset_mm']))
        if not hair.is_volume:
            raise ValueError('Generated hair cap is not a closed positive volume')
        photo_consistency,overlay_files=write_fit_overlays(output,
            ['front','left','right']+(['back'] if report['back_view_used'] else []),
            face['images'],masks,face['hair_view_valid'],view_surface_points,
            directions,offsets,patch,image_matrices)
        native_hair=hair.copy();native_hair.vertices/=scale
        native_hair.export(output/'hair_cap_native.glb')
        hair.export(output/'hair_cap_design_mm.stl')
        manifest=export_render_scene(job/'rdf/rdf_face_photo.glb',
            output/'hair_cap_native.glb',output/'scene.glb')
        write(output/'scene.manifest.json',manifest)
        for path,expected in report['input_sha256'].items():
            if digest(path)!=expected:
                raise ValueError('Input changed during hair-cap fit: '+path)
        for path,expected in observation_hashes.items():
            if digest(path)!=expected:
                raise ValueError('Uploaded input changed during hair-cap fit: '+path)
        if digest(bundle_path)!=report['rdf_bundle_sha256'] or digest(crop_report_path)!=report['crop_report_sha256']:
            raise ValueError('RDF source artifacts changed during hair-cap fit')
        if digest(parser_path)!=report['parser_sha256']:
            raise ValueError('Parser changed during hair-cap fit')
        if digest(profile_path)!=report['policy_sha256']:
            raise ValueError('Hair-cap policy changed during fit')
        output_files=['hair_cap_native.glb','hair_cap_design_mm.stl','scene.glb','scene.manifest.json']+overlay_files+[
            slot+'.hair-mask.png' for slot in (['front','left','right']+
                (['back'] if report['back_view_used'] else []))]
        report['output_sha256']={name:digest(output/name) for name in output_files}
        report.update(status='HAIR_CAP_EXPORTED',scale_mm_per_native_unit=scale,
            design_mesh_height_mm=float(profile['design_mesh_height_mm']),
            source_head_height_native=height,head_topology=topology,
            hair_boundary=field['report'],volume_fit=volume,
            photo_consistency=photo_consistency,
            geometry={'vertices':int(len(hair.vertices)),'triangles':int(len(hair.faces)),
                'watertight':bool(hair.is_watertight),'positive_volume':bool(hair.is_volume),
                'connected_components':len(hair.split(only_watertight=False)),
                'bounds_design_mm':hair.bounds.tolist(),'self_intersections_verified':False,
                'minimum_wall_verified':False,'physical_scale_verified':False},
            hidden_region_policy=field['report']['posterior_policy'],
            output_files=output_files)
    except Exception as exc:
        report.update(status='FAILED',error=str(exc))
        raise
    finally:
        write(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parser',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    a=p.parse_args()
    result=run(a.job.resolve(),a.output.resolve(),a.parser.resolve(),a.profile.resolve())
    print(json.dumps({'status':result['status'],'output':str(a.output)},indent=2),flush=True)


if __name__=='__main__':
    main()
