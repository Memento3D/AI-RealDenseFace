"""Composition for diagnostic frontal photo registration."""
import json
import subprocess
from pathlib import Path
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.experiment_input import LocalExperimentStore
from memento3d_head.domain.contracts import Photo, ValidationError
from memento3d_head.settings import load_settings


def add_commands(commands):
    from memento3d_head.multiview_commands import add_commands as add_views
    add_views(commands)
    parser=commands.add_parser('register-head',help='Fit approximate camera; preserve head geometry')
    for flag in ('head-run','front','model','mica-config','output'):
        parser.add_argument('--'+flag,required=True,type=Path)
    parser.add_argument('--model-sha256',required=True)
    parser.add_argument('--config',type=Path)


def execute(args):
    if args.command == "register-views":
        from memento3d_head.multiview_commands import execute as register_views
        return register_views(args)
    output=None
    try:
        import numpy as np
        from memento3d_head.adapters.mediapipe_inspector import MediaPipeInspector
        from memento3d_head.adapters.mica_assets import verify_config
        from memento3d_head.adapters.smooth_preview import read_geometry
        from memento3d_head.adapters.photo_registration import embedded_points,draw_overlay
        config=verify_config(args.mica_config)
        previous=json.loads((args.head_run/'report.json').read_text(encoding='utf-8'))
        mesh=args.head_run/'head.obj'
        if (previous.get('status')!='GEOMETRY_EXPORTED'
                or previous['input']['source_sha256']!=sha256(args.front)
                or previous['artifacts']['head.obj']['sha256']!=sha256(mesh)):
            raise ValidationError('SOURCE_MISMATCH','Photo or mesh differs from successful reconstruction report')
        before=sha256(mesh)
        args.output.resolve().mkdir(parents=True,exist_ok=False)
        output=args.output.resolve()
        settings=load_settings(args.config)
        inspector=MediaPipeInspector(args.model,args.model_sha256)
        try:
            observation=inspector.inspect(Photo('front',args.front),output,settings)
        finally:
            inspector.close()
        if observation.face is None or observation.issues:
            raise ValidationError('PHOTO_REJECTED',str(observation.issues))
        face=observation.face
        if abs(face.yaw)>settings.front_max_yaw or abs(face.pitch)>settings.max_pitch or abs(face.roll)>settings.max_roll:
            raise ValidationError('PHOTO_REJECTED','Registration v1 requires a near-frontal photograph')
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
        fit=draw_overlay(output/'front.normalized.png',vertices,triangles,points,face.landmarks,
                         output/'front.registration.png')
        if sha256(mesh)!=before or sha256(args.front)!=observation.source_sha256:
            raise ValidationError('SOURCE_CHANGED','Source changed during registration')
        report={'schema_version':1,'pipeline_type':'frontal_camera_registration','pipeline_version':1,
                'status':'REGISTRATION_EXPORTED','error':None,'research_only':True,
                'source_mesh_sha256':before,'source_photo_sha256':observation.source_sha256,
                'source_report_sha256':sha256(args.head_run/'report.json'),
                'mediapipe_model_sha256':args.model_sha256.lower(),'embedding_sha256':item['sha256'],
                'geometry_modified':False,'multi_view_fitting':False,'registration':fit,
                'artifacts':{name:{'path':name,'sha256':sha256(output/name)}
                             for name in ('front.registration.png','front.normalized.png')}}
        LocalExperimentStore().write_json(output/'report.json',report)
        print(json.dumps({'status':report['status'],'report_path':str(output/'report.json'),
                          'mean_fit_residual_px':fit['mean_fit_residual_px']},indent=2))
        return 0
    except Exception as error:
        report={'schema_version':1,'status':'FAILED','error':{
            'code':error.code if isinstance(error,ValidationError) else 'REGISTRATION_FAILED','message':str(error)}}
        if output is not None:
            LocalExperimentStore().write_json(output/'report.json',report)
        print(json.dumps(report,indent=2));return 1
