"""Run a shared-focal experiment from an immutable three-view registration attempt."""
import json
import subprocess
from pathlib import Path
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError


def add_commands(commands):
    parser=commands.add_parser('refine-cameras',help='Shared-focal camera experiment; unchanged mesh')
    for flag in ('registration-run','head-run','mica-config','output'):
        parser.add_argument('--'+flag,required=True,type=Path)


def execute(args):
    output=None
    report={'schema_version':1,'pipeline_type':'shared_camera_experiment','pipeline_version':1,
            'status':'FAILED','error':None,'research_only':True,'geometry_modified':False,
            'multi_view_fitting':False,'likeness_verified':False,'views':{},'artifacts':{}}
    try:
        import numpy as np
        from PIL import Image
        from memento3d_head.adapters.mica_assets import verify_config
        from memento3d_head.adapters.photo_registration import embedded_points,draw_overlay
        from memento3d_head.adapters.smooth_preview import read_geometry
        from memento3d_head.adapters.shared_camera import search_shared_focal
        source=args.registration_run/'report.json';before=sha256(source)
        original=json.loads(source.read_text(encoding='utf-8'))
        if (original.get('status')!='REGISTRATIONS_EXPORTED'
                or original.get('pipeline_type')!='three_view_camera_registration'
                or set(original.get('views',{}))!={'front','left','right'}):
            raise ValidationError('INVALID_REGISTRATION','Need a successful register-views run')
        mesh=args.head_run/'head.obj';mesh_hash=sha256(mesh)
        if mesh_hash!=original['input']['mesh']:
            raise ValidationError('SOURCE_MISMATCH','Mesh differs from registration input')
        images={slot:args.registration_run/f'{slot}.normalized.png' for slot in original['views']}
        sizes={}
        for slot,path in images.items():
            if sha256(path)!=original['artifacts'][path.name]['sha256']:
                raise ValidationError('SOURCE_MISMATCH','Normalized photo checksum mismatch')
            with Image.open(path) as image:sizes[slot]=image.size
        if len(set(sizes.values()))!=1:
            raise ValidationError('INCOMPATIBLE_IMAGES','Use images captured with identical dimensions')
        config=verify_config(args.mica_config)
        candidate=args.output.resolve()
        try:candidate.mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=candidate
        item=config['assets']['data/FLAME2020/landmark_embedding.npy']
        worker=Path(__file__).parent/'adapters'/'embedding_worker.py'
        result=subprocess.run([config['python'],str(worker.resolve()),item['path'],str(output/'embedding.npz')],
                              capture_output=True,text=True,timeout=120)
        (output/'embedding_worker.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise ValidationError('EMBEDDING_FAILED','See embedding_worker.log')
        vertices,faces=read_geometry(mesh)
        with np.load(output/'embedding.npz',allow_pickle=False) as data:
            points=embedded_points(vertices,faces,data['faces'],data['barycentric'])
        solutions,weights,experiment=search_shared_focal(points,original['views'],sizes)
        report['experiment']=experiment
        observations={row['slot']:row['face']['landmarks'] for row in original['assessment']['photos']}
        for slot in images:
            overlay=output/f'{slot}.candidate.png'
            fit=draw_overlay(images[slot],vertices,faces,points,observations[slot],overlay,solutions[slot])
            for index,row in enumerate(fit['landmarks']):
                row['fit_weight']=float(weights[slot][index]) if index<7 else 0.
            report['views'][slot]={'baseline':original['views'][slot],'candidate':fit}
            with Image.open(args.registration_run/f'{slot}.registration.png') as baseline, Image.open(overlay) as after:
                # Check baseline artifact before copying it into a comparison.
                if sha256(args.registration_run/f'{slot}.registration.png')!=original['artifacts'][f'{slot}.registration.png']['sha256']:
                    raise ValidationError('SOURCE_MISMATCH','Baseline overlay checksum mismatch')
                comparison=Image.new('RGB',(baseline.width+after.width,max(baseline.height,after.height)))
                comparison.paste(baseline,(0,0));comparison.paste(after,(baseline.width,0))
                comparison.save(output/f'{slot}.comparison.png')
            for name in (f'{slot}.candidate.png',f'{slot}.comparison.png'):
                report['artifacts'][name]={'path':name,'sha256':sha256(output/name)}
        if sha256(mesh)!=mesh_hash or sha256(source)!=before or any(sha256(p)!=original['artifacts'][p.name]['sha256'] for p in images.values()):
            raise ValidationError('SOURCE_CHANGED','Source changed during experiment')
        report['source_registration_sha256']=before;report['source_mesh_sha256']=mesh_hash
        report['embedding_sha256']=item['sha256']
        report['warning']='Candidate cameras are uncalibrated; compare held-out diagnostics. No shape improvement established.'
        report['status']='CAMERA_EXPERIMENT_EXPORTED'
    except Exception as error:
        report['error']={'code':error.code if isinstance(error,ValidationError) else 'CAMERA_EXPERIMENT_FAILED','message':str(error)}
    if output is not None:LocalStore().write_json(output/'report.json',report)
    summary={k:report[k] for k in ('status','error')}
    summary['report_path']=str(output/'report.json') if output else None
    if 'experiment' in report:
        summary.update({k:report['experiment'][k] for k in ('focal_px','at_search_boundary','candidate_reduces_fit_objective')})
    print(json.dumps(summary,indent=2));return 0 if report['status']=='CAMERA_EXPERIMENT_EXPORTED' else 1
