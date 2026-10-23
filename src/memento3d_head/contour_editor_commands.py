"""Prepare an offline visible-contour editor and validate source-bound annotations."""
import json
from pathlib import Path
from uuid import uuid4
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError
from memento3d_head.domain.contour_annotations import validate_annotations


def add_commands(commands):
    prepare=commands.add_parser('prepare-contours',help='Create offline image contour editor with projected outer mesh boundary')
    for flag in ('registration-run','head-run','shape-run','output'):
        prepare.add_argument('--'+flag,required=True,type=Path)
    validate=commands.add_parser('validate-contours',help='Check exported manual curves and source binding; no fitting')
    for flag in ('workspace','annotations','output'):
        validate.add_argument('--'+flag,required=True,type=Path)


def execute(args):
    output=None
    try:
        if args.command=='prepare-contours':
            import numpy as np
            from PIL import Image
            from memento3d_head.adapters.smooth_preview import read_geometry
            from memento3d_head.adapters.visible_contour_editor import projected_outer_boundary,write_editor
            registration=args.registration_run/'report.json';head_path=args.head_run/'report.json';shape_path=args.shape_run/'report.json'
            sources={p:sha256(p) for p in (registration,head_path,shape_path)}
            r=json.loads(registration.read_text(encoding='utf-8'));h=json.loads(head_path.read_text(encoding='utf-8'));s=json.loads(shape_path.read_text(encoding='utf-8'))
            mesh=args.head_run/'head.obj';candidate=args.shape_run/'head.candidate.obj'
            if (r.get('status')!='REGISTRATIONS_EXPORTED' or r.get('pipeline_type')!='three_view_camera_registration'
                    or s.get('status') not in {'SHAPE_CANDIDATE_EXPORTED','NO_SHAPE_CHANGE'}
                    or s.get('pipeline_type')!='bounded_multiview_identity_fit'
                    or s['input']['registration_report_sha256']!=sources[registration]
                    or r['input']['head_report']!=sources[head_path]
                    or sha256(mesh)!=r['input']['mesh'] or sha256(mesh)!=h['artifacts']['head.obj']['sha256']
                    or s['input']['head_obj_sha256']!=sha256(mesh)
                    or sha256(candidate)!=s['artifacts']['head.candidate.obj']['sha256']):
                raise ValidationError('SOURCE_MISMATCH','Original/candidate meshes and linked reports do not match')
            sources[mesh]=sha256(mesh);sources[candidate]=sha256(candidate)
            images={slot:args.registration_run/f'{slot}.normalized.png' for slot in ('front','left','right')}
            for p in images.values():
                expected=r['artifacts'][p.name]['sha256']
                if sha256(p)!=expected:raise ValidationError('SOURCE_MISMATCH','Photo changed')
                sources[p]=expected
            base,faces=read_geometry(mesh);after,after_faces=read_geometry(candidate)
            if not np.array_equal(faces,after_faces):raise ValidationError('TOPOLOGY_MISMATCH','Face ordering changed')
            candidate_output=args.output.resolve()
            try:candidate_output.mkdir(parents=True,exist_ok=False)
            except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
            output=candidate_output
            binding={'registration_report_sha256':sources[registration],'head_report_sha256':sources[head_path],
                     'shape_report_sha256':sources[shape_path],'head_obj_sha256':sources[mesh],
                     'candidate_obj_sha256':sources[candidate],'photos':{slot:sources[p] for slot,p in images.items()}}
            manifest={'schema_version':1,'workspace_id':str(uuid4()),'source_binding':binding,'views':{},
                      'boundary_type':'projected_whole_head_outer_silhouette',
                      'face_only_mask':False,'camera_source':'original_registration_unchanged',
                      'automatic_contour_correspondences':False}
            for slot,p in images.items():
                with Image.open(p) as image:size=image.size
                manifest['views'][slot]={'image_size':list(size),'image_sha256':sources[p],
                    'outlines':{'baseline':projected_outer_boundary(base,faces,r['views'][slot],size),
                                'candidate':projected_outer_boundary(after,faces,r['views'][slot],size)}}
            LocalStore().write_json(output/'manifest.json',manifest)
            write_editor(output/'contour_editor.html',manifest,images)
            if any(sha256(p)!=expected for p,expected in sources.items()):raise ValidationError('SOURCE_CHANGED','A source changed during preparation')
            report={'schema_version':1,'status':'CONTOUR_EDITOR_EXPORTED','error':None,'geometry_modified':False,
                    'cameras_modified':False,'manual_annotations_created':False,'likeness_verified':False,
                    'workspace_id':manifest['workspace_id'],'artifacts':{name:{'path':name,'sha256':sha256(output/name)} for name in ('manifest.json','contour_editor.html')}}
        else:
            workspace_report=json.loads((args.workspace/'report.json').read_text(encoding='utf-8'))
            manifest_path=args.workspace/'manifest.json'
            if (workspace_report.get('status')!='CONTOUR_EDITOR_EXPORTED'
                    or sha256(manifest_path)!=workspace_report['artifacts']['manifest.json']['sha256']):
                raise ValidationError('WORKSPACE_MISMATCH','Editor manifest changed or workspace invalid')
            if args.annotations.stat().st_size>2_000_000:raise ValidationError('INVALID_ANNOTATIONS','Annotation JSON exceeds 2 MB')
            annotations_hash=sha256(args.annotations)
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'));data=json.loads(args.annotations.read_text(encoding='utf-8'))
            result=validate_annotations(data,manifest)
            try:args.output.resolve().mkdir(parents=True,exist_ok=False)
            except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
            output=args.output.resolve()
            LocalStore().write_json(output/'visible_contours.snapshot.json',data)
            if sha256(args.annotations)!=annotations_hash:raise ValidationError('SOURCE_CHANGED','Annotations changed during validation')
            report={'schema_version':1,**result,'error':None,'workspace_id':manifest['workspace_id'],
                    'source_annotations_sha256':annotations_hash,'source_manifest_sha256':sha256(manifest_path),
                    'geometry_modified':False,'artifacts':{'visible_contours.snapshot.json':{
                        'path':'visible_contours.snapshot.json','sha256':sha256(output/'visible_contours.snapshot.json')}}}
        LocalStore().write_json(output/'report.json',report)
        print(json.dumps({'status':report['status'],'report_path':str(output/'report.json'),'error':None},indent=2))
        return 0
    except Exception as error:
        report={'schema_version':1,'status':'FAILED','error':{'code':error.code if isinstance(error,ValidationError) else 'CONTOUR_EDITOR_FAILED','message':str(error)}}
        if output is not None:LocalStore().write_json(output/'report.json',report)
        print(json.dumps(report,indent=2));return 1
