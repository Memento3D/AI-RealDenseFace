"""Compose a contour/visibility audit before adding contour optimization."""
import json
import subprocess
from pathlib import Path
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError


def add_commands(commands):
    parser=commands.add_parser('audit-contours',help='Inspect pose-dependent jaw and self-occlusion on original/candidate meshes')
    for flag in ('registration-run','head-run','shape-run','mica-config','output'):
        parser.add_argument('--'+flag,required=True,type=Path)


def execute(args):
    output=None
    report={'schema_version':1,'pipeline_type':'dynamic_contour_audit','pipeline_version':1,
            'status':'FAILED','error':None,'research_only':True,'geometry_modified':False,
            'cameras_modified':False,'contours_used_for_fit':False,'likeness_verified':False,'views':{},'artifacts':{}}
    try:
        import numpy as np
        from PIL import Image
        from memento3d_head.adapters.mica_assets import verify_config
        from memento3d_head.adapters.smooth_preview import read_geometry
        from memento3d_head.adapters.contour_audit import audit_mesh,draw_audit
        registration=args.registration_run/'report.json';shape_path=args.shape_run/'report.json'
        head_path=args.head_run/'report.json'
        sources={p:sha256(p) for p in (registration,shape_path,head_path)}
        previous=json.loads(registration.read_text(encoding='utf-8'));shape=json.loads(shape_path.read_text(encoding='utf-8'))
        head=json.loads(head_path.read_text(encoding='utf-8'))
        mesh=args.head_run/'head.obj';candidate=args.shape_run/'head.candidate.obj'
        if (previous.get('status')!='REGISTRATIONS_EXPORTED' or previous.get('pipeline_type')!='three_view_camera_registration'
                or shape.get('status') not in {'SHAPE_CANDIDATE_EXPORTED','NO_SHAPE_CHANGE'}
                or shape.get('pipeline_type')!='bounded_multiview_identity_fit'
                or shape['input']['registration_report_sha256']!=sources[registration]
                or previous['input']['head_report']!=sources[head_path]
                or sha256(mesh)!=previous['input']['mesh'] or sha256(mesh)!=head['artifacts']['head.obj']['sha256']
                or shape['input']['head_obj_sha256']!=sha256(mesh)
                or sha256(candidate)!=shape['artifacts']['head.candidate.obj']['sha256']):
            raise ValidationError('SOURCE_MISMATCH','Head, candidate or registration differs from linked reports')
        sources[mesh]=sha256(mesh);sources[candidate]=sha256(candidate)
        photos={slot:args.registration_run/f'{slot}.normalized.png' for slot in ('front','left','right')}
        for p in photos.values():
            if sha256(p)!=previous['artifacts'][p.name]['sha256']:raise ValidationError('SOURCE_MISMATCH','Photo changed')
            sources[p]=sha256(p)
        config=verify_config(args.mica_config);sources[args.mica_config]=sha256(args.mica_config)
        try:args.output.resolve().mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=args.output.resolve()
        item=config['assets']['data/FLAME2020/landmark_embedding.npy']
        worker=Path(__file__).parent/'adapters'/'contour_worker.py'
        result=subprocess.run([config['python'],str(worker.resolve()),item['path'],str(output/'contours.npz')],
                              capture_output=True,text=True,timeout=120)
        (output/'contour_worker.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise ValidationError('CONTOUR_WORKER_FAILED','See contour_worker.log')
        with np.load(output/'contours.npz',allow_pickle=False) as data:embedding={k:data[k] for k in data.files}
        base,faces=read_geometry(mesh);changed,changed_faces=read_geometry(candidate)
        if not np.array_equal(faces,changed_faces):raise ValidationError('TOPOLOGY_MISMATCH','Source/candidate face order differs')
        observed={p['slot']:p['face']['landmarks'] for p in previous['assessment']['photos']}
        for slot,photo in photos.items():
            with Image.open(photo) as image:size=image.size
            base_audit=audit_mesh(base,faces,embedding,previous['views'][slot],observed[slot],size)
            candidate_audit=audit_mesh(changed,faces,embedding,previous['views'][slot],observed[slot],size)
            report['views'][slot]={'baseline':base_audit,'candidate':candidate_audit}
            path=output/f'{slot}.contour_audit.png';draw_audit(photo,path,base_audit,candidate_audit)
            report['artifacts'][path.name]={'path':path.name,'sha256':sha256(path)}
        if any(sha256(p)!=expected for p,expected in sources.items()):raise ValidationError('SOURCE_CHANGED','Source changed during audit')
        report['input']={'registration_sha256':sources[registration],'shape_report_sha256':sources[shape_path],
                         'head_obj_sha256':sources[mesh],'candidate_obj_sha256':sources[candidate]}
        report['embedding_sha256']=item['sha256']
        report['warning']='Dynamic contour may clamp outside +/-39 deg. Semantic pairs approximate. Ray tests only mesh self-occlusion; no contour fitting or automatic acceptance.'
        report['status']='CONTOUR_AUDIT_EXPORTED'
    except KeyboardInterrupt:
        report['status']='CANCELLED';report['error']={'code':'INTERRUPTED','message':'Interrupted'}
    except Exception as error:
        report['error']={'code':error.code if isinstance(error,ValidationError) else 'CONTOUR_AUDIT_FAILED','message':str(error)}
    if output is not None:LocalStore().write_json(output/'report.json',report)
    print(json.dumps({'status':report['status'],'report_path':str(output/'report.json') if output else None,'error':report['error'],
                     'views':{s:{'angle_clamped':v['baseline']['selection']['angle_clamped'],
                                 'camera_yaw_degrees':v['baseline']['selection']['camera_yaw_degrees']} for s,v in report['views'].items()}},indent=2))
    return 0 if report['status']=='CONTOUR_AUDIT_EXPORTED' else 1
