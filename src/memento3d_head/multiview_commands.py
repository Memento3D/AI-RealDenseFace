"""Compose diagnostic three-camera registration without modifying head geometry."""
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import Photo, ValidationError
from memento3d_head.settings import load_settings

SLOTS=('front','left','right')


def add_commands(commands):
    parser=commands.add_parser('register-views',help='Fit three cameras to one unchanged head; no shape fitting')
    for flag in ('head-run','validation-run','front','left','right','model','mica-config','output'):
        parser.add_argument('--'+flag,required=True,type=Path)
    parser.add_argument('--model-sha256',required=True)
    parser.add_argument('--config',type=Path)


def bind_sources(head_run, validation_run, paths):
    """Reject swapped photos, modified meshes and a validation report from another attempt."""
    head_path=head_run/'report.json'; validation_path=validation_run/'report.json'
    head=json.loads(head_path.read_text(encoding='utf-8'))
    validation=json.loads(validation_path.read_text(encoding='utf-8'))
    mesh=head_run/'head.obj'
    hashes={slot:sha256(path) for slot,path in paths.items()}
    if (head.get('status')!='GEOMETRY_EXPORTED'
            or head.get('input',{}).get('validation_report_sha256')!=sha256(validation_path)
            or head['input'].get('source_sha256')!=hashes['front']
            or head.get('artifacts',{}).get('head.obj',{}).get('sha256')!=sha256(mesh)):
        raise ValidationError('SOURCE_MISMATCH','Head, front photo or validation report differs from original run')
    assessment=validation.get('assessment',{})
    observations=assessment.get('photos',[])
    if (validation.get('status')!='NEEDS_HUMAN_REVIEW' or assessment.get('issues')
            or len(observations)!=3 or {item['slot'] for item in observations}!=set(SLOTS)
            or any(hashes[item['slot']]!=item['source_sha256'] for item in observations)):
        raise ValidationError('PHOTO_SET_MISMATCH','Use the exact three photos from the accepted validation run')
    return mesh, {'mesh':sha256(mesh),'photos':hashes,'head_report':sha256(head_path),
                  'validation_report':sha256(validation_path)}


def execute(args):
    output=None
    report={'schema_version':1,'pipeline_type':'three_view_camera_registration','pipeline_version':1,
            'status':'FAILED','error':None,'research_only':True,'geometry_modified':False,
            'multi_view_fitting':False,'likeness_verified':False,'views':{},'artifacts':{}}
    try:
        import numpy as np
        from memento3d_head.adapters.mediapipe_inspector import MediaPipeInspector
        from memento3d_head.adapters.mica_assets import verify_config
        from memento3d_head.adapters.smooth_preview import read_geometry
        from memento3d_head.adapters.photo_registration import embedded_points,draw_overlay
        from memento3d_head.pipelines.photo_set_validation_v1 import PhotoSetValidationV1
        paths={slot:getattr(args,slot).resolve() for slot in SLOTS}
        mesh,evidence=bind_sources(args.head_run,args.validation_run,paths)
        config=verify_config(args.mica_config)
        settings=load_settings(args.config)
        candidate=args.output.resolve()
        try:
            candidate.mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:
            raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=candidate
        report['input']=evidence
        inspector=MediaPipeInspector(args.model,args.model_sha256)
        try:
            assessment=PhotoSetValidationV1(inspector).execute(
                tuple(Photo(slot,paths[slot]) for slot in SLOTS),output,settings)
        finally:
            inspector.close()
        report['assessment']=asdict(assessment)
        if assessment.issues:
            raise ValidationError('PHOTO_REJECTED',str(assessment.issues))
        worker=Path(__file__).parent/'adapters'/'embedding_worker.py'
        item=config['assets']['data/FLAME2020/landmark_embedding.npy']
        result=subprocess.run([config['python'],str(worker.resolve()),item['path'],str(output/'embedding.npz')],
                              capture_output=True,text=True,timeout=120)
        (output/'embedding_worker.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:
            raise ValidationError('EMBEDDING_FAILED','See embedding_worker.log')
        vertices,triangles=read_geometry(mesh)
        with np.load(output/'embedding.npz',allow_pickle=False) as data:
            points=embedded_points(vertices,triangles,data['faces'],data['barycentric'])
        for observation in assessment.photos:
            slot=observation.slot
            if observation.source_sha256!=evidence['photos'][slot]:
                raise ValidationError('SOURCE_CHANGED','Photograph changed while being inspected')
            face=observation.face
            fit=draw_overlay(output/f'{slot}.normalized.png',vertices,triangles,points,face.landmarks,
                             output/f'{slot}.registration.png')
            fit['detected_yaw_degrees']=face.yaw
            fit['side_visibility_verified']=False
            report['views'][slot]=fit
            for name in (f'{slot}.registration.png',f'{slot}.normalized.png'):
                report['artifacts'][name]={'path':name,'sha256':sha256(output/name)}
        if (sha256(mesh)!=evidence['mesh']
                or any(sha256(path)!=evidence['photos'][slot] for slot,path in paths.items())
                or sha256(args.head_run/'report.json')!=evidence['head_report']
                or sha256(args.validation_run/'report.json')!=evidence['validation_report']):
            raise ValidationError('SOURCE_CHANGED','Source changed during registration')
        report['mediapipe_model_sha256']=args.model_sha256.lower()
        report['embedding_sha256']=item['sha256']
        report['status']='REGISTRATIONS_EXPORTED'
    except KeyboardInterrupt:
        report['status']='CANCELLED';report['error']={'code':'INTERRUPTED','message':'Interrupted'}
    except Exception as error:
        report['error']={'code':error.code if isinstance(error,ValidationError) else 'REGISTRATION_FAILED',
                         'message':str(error)}
    if output is not None:
        LocalStore().write_json(output/'report.json',report)
    print(json.dumps({'status':report['status'],'report_path':str(output/'report.json') if output else None,
                      'error':report['error'],
                      'mean_fit_residual_px':{k:v['mean_fit_residual_px'] for k,v in report['views'].items()}},indent=2))
    return 0 if report['status']=='REGISTRATIONS_EXPORTED' else 1
