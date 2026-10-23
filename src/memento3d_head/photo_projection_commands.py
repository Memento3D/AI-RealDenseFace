"""Compose a source-bound single-photo projection preview with no geometry changes."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError


def add_commands(commands):
    parser=commands.add_parser('preview-photo-color',help='Experimental frontal photo projection; unknown surfaces stay gray')
    for name in ('candidate-run','registration-run','output'):parser.add_argument('--'+name,required=True,type=Path)


def execute(args):
    output=None
    try:
        from memento3d_head.adapters.smooth_preview import read_geometry
        from memento3d_head.adapters.photo_projection import project_photo,export_photo_obj,write_photo_viewer
        candidate_report=args.candidate_run/'report.json';registration_report=args.registration_run/'report.json'
        mesh=args.candidate_run/'head.candidate.obj';photo=args.registration_run/'front.normalized.png'
        sources={p:sha256(p) for p in (candidate_report,registration_report,mesh,photo)}
        c=json.loads(candidate_report.read_text(encoding='utf-8'));r=json.loads(registration_report.read_text(encoding='utf-8'))
        if (c.get('pipeline_type')!='bounded_frontal_jaw_proxy_fit' or c.get('status') not in ('JAW_EXPERIMENT_EXPORTED','NO_SHAPE_CHANGE')
            or r.get('pipeline_type')!='three_view_camera_registration' or r.get('status')!='REGISTRATIONS_EXPORTED'
            or c['input_sha256']['registration']!=sources[registration_report]
            or c['artifacts']['head.candidate.obj']['sha256']!=sources[mesh]
            or c['input_sha256']['photo_front']!=sources[photo]
            or r['artifacts']['front.normalized.png']['sha256']!=sources[photo]):
            raise ValidationError('SOURCE_MISMATCH','Candidate, frozen registration and normalized photo must match linked reports')
        assessment=next(p for p in r['assessment']['photos'] if p['slot']=='front')
        with Image.open(photo) as image:size=image.size
        if size!=(assessment['oriented_width'],assessment['oriented_height']):
            raise ValidationError('SOURCE_MISMATCH','Photo dimensions differ from registered landmarks')
        vertices,faces=read_geometry(mesh)
        coordinates,covered,info=project_photo(vertices,faces,r['views']['front'],size,assessment['face']['landmarks'])
        if not covered.any():raise ValidationError('NO_PHOTO_COVERAGE','No triangles passed photo mask/facing/depth tests')
        try:args.output.resolve().mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=args.output.resolve();shutil.copyfile(photo,output/'front.photo.png')
        export_photo_obj(output,vertices,faces,coordinates,covered)
        np.savez(output/'photo_projection.npz',coordinates=coordinates,covered_faces=covered)
        provenance={'candidate_obj_sha256':sources[mesh],'candidate_report_sha256':sources[candidate_report],
                    'registration_report_sha256':sources[registration_report],'photo_sha256':sources[photo]}
        write_photo_viewer(output/'photo_preview.html',vertices,faces,coordinates,covered,output/'front.photo.png',provenance)
        # A coverage mask in the photo coordinate system; not a skin-segmentation claim.
        xy=coordinates[:,:2]/coordinates[:,2,None];xy[:,0]*=size[0];xy[:,1]=(1-xy[:,1])*size[1]
        coverage=Image.new('L',size);draw=ImageDraw.Draw(coverage)
        for face in faces[covered]:draw.polygon([tuple(p) for p in xy[face]],fill=255)
        coverage.save(output/'front.coverage.png')
        if any(sha256(p)!=h for p,h in sources.items()):raise ValidationError('SOURCE_CHANGED','Input changed during photo projection')
        report={'schema_version':1,'pipeline_type':'frontal_photo_color_projection','pipeline_version':1,
                'status':'PHOTO_PROJECTION_EXPORTED','error':None,'projection':info,'input':provenance,
                'geometry_modified':False,'cameras_modified':False,'photo_projection_only':True,
                'albedo_recovered':False,'uv_atlas_generated':False,'multi_view_texture':False,
                'likeness_verified':False,'printability_verified':False,'automatic_acceptance':False,
                'obj_uv_mapping':'per_vertex_projected_uv_approximation; viewer uses homogeneous projective lookup',
                'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in output.iterdir() if p.is_file()}}
        LocalStore().write_json(output/'report.json',report)
        print(json.dumps({'status':report['status'],'report_path':str(output/'report.json'),
                         'covered_triangles':info['covered_triangles'],'error':None},indent=2));return 0
    except Exception as error:
        report={'schema_version':1,'status':'FAILED','error':{'code':error.code if isinstance(error,ValidationError) else 'PHOTO_PROJECTION_FAILED','message':str(error)}}
        if output is not None:LocalStore().write_json(output/'report.json',report)
        print(json.dumps(report,indent=2));return 1
