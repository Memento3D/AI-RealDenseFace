"""Compare original camera and two experimental checkerboard camera models."""
import json
import subprocess
from pathlib import Path
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError


def add_commands(commands):
    parser=commands.add_parser('compare-calibration',help='Compare both experimental checkerboard models; unchanged mesh')
    for flag in ('registration-run','head-run','mica-config','calibration','output'):
        parser.add_argument('--'+flag,required=True,type=Path)


def execute(args):
    output=None
    report={'schema_version':1,'pipeline_type':'checkerboard_camera_comparison','pipeline_version':1,
            'status':'FAILED','error':None,'research_only':True,'geometry_modified':False,
            'multi_view_fitting':False,'likeness_verified':False,'automatic_model_selection':False,
            'views':{},'artifacts':{}}
    try:
        import numpy as np
        from PIL import Image,ImageDraw
        from memento3d_head.adapters.mica_assets import verify_config
        from memento3d_head.adapters.photo_registration import PAIRS,embedded_points,draw_overlay
        from memento3d_head.adapters.smooth_preview import read_geometry
        from memento3d_head.adapters.calibrated_camera import load_calibration,fit_fixed_camera
        source=args.registration_run/'report.json';source_hash=sha256(source)
        original=json.loads(source.read_text(encoding='utf-8'))
        if (original.get('status')!='REGISTRATIONS_EXPORTED'
                or original.get('pipeline_type')!='three_view_camera_registration'
                or set(original.get('views',{}))!={'front','left','right'}):
            raise ValidationError('INVALID_REGISTRATION','Need the successful register-views run, not refine-cameras')
        mesh=args.head_run/'head.obj';mesh_hash=sha256(mesh)
        if mesh_hash!=original['input']['mesh']:
            raise ValidationError('SOURCE_MISMATCH','Mesh differs from registration input')
        source_files={source:source_hash,mesh:mesh_hash}
        observations={row['slot']:row['face']['landmarks'] for row in original['assessment']['photos']}
        sizes={};images={};baselines={}
        for slot in ('front','left','right'):
            images[slot]=args.registration_run/f'{slot}.normalized.png'
            baselines[slot]=args.registration_run/f'{slot}.registration.png'
            for path in (images[slot],baselines[slot]):
                expected=original['artifacts'][path.name]['sha256']
                if sha256(path)!=expected:
                    raise ValidationError('SOURCE_MISMATCH','Source photo or overlay checksum mismatch')
                source_files[path]=expected
            with Image.open(images[slot]) as image:sizes[slot]=image.size
        calibration_hash=sha256(args.calibration)
        calibration=load_calibration(args.calibration,sizes['front'])
        if any(list(size)!=calibration['image_size'] for size in sizes.values()):
            raise ValidationError('CALIBRATION_SIZE_MISMATCH','All images must match calibration resolution')
        source_files[args.calibration]=calibration_hash
        config=verify_config(args.mica_config)
        try:args.output.resolve().mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=args.output.resolve()
        report['calibration']=calibration
        LocalStore().write_json(output/'calibration.snapshot.json',calibration)
        item=config['assets']['data/FLAME2020/landmark_embedding.npy']
        worker=Path(__file__).parent/'adapters'/'embedding_worker.py'
        result=subprocess.run([config['python'],str(worker.resolve()),item['path'],str(output/'embedding.npz')],
                              capture_output=True,text=True,timeout=120)
        (output/'embedding_worker.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise ValidationError('EMBEDDING_FAILED','See embedding_worker.log')
        vertices,faces=read_geometry(mesh)
        with np.load(output/'embedding.npz',allow_pickle=False) as data:
            points=embedded_points(vertices,faces,data['faces'],data['barycentric'])
        selected=points[[i for i,_,_ in PAIRS]]
        for slot in ('front','left','right'):
            w,h=sizes[slot]
            target=np.array([[observations[slot][j][0]*w,observations[slot][j][1]*h] for _,j,_ in PAIRS])
            baseline=original['views'][slot]
            expected=np.array([row['observed_px'] for row in baseline['landmarks'][:7]])
            if not np.allclose(target,expected,atol=1e-5,rtol=0):
                raise ValidationError('SOURCE_MISMATCH','Stored landmarks disagree with baseline')
            report['views'][slot]={'baseline':baseline}
            panels=[]
            with Image.open(baselines[slot]) as image:panels.append(image.convert('RGB'))
            for name in ('standard5','radial2'):
                model=calibration['models'][name]
                solution=fit_fixed_camera(selected,target,model['camera_matrix'],model['distortion'],baseline,vertices)
                destination=output/f'{slot}.{name}.png'
                fit=draw_overlay(images[slot],vertices,faces,points,observations[slot],destination,
                                 camera_solution=solution,distortion=np.asarray(model['distortion']))
                fit['intrinsics_source']='checkerboard_experimental'
                fit['fixed_intrinsics']=True
                fit['fit_policy']='same_seven_points_uniform_least_squares_as_original_registration'
                fit['delta_mean_fit_residual_px']=fit['mean_fit_residual_px']-baseline['mean_fit_residual_px']
                fit['delta_chin_residual_px']=fit['landmarks'][-1]['residual_px']-baseline['landmarks'][-1]['residual_px']
                report['views'][slot][name]=fit
                with Image.open(destination) as image:panels.append(image.convert('RGB'))
            comparison=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)+30),'#20252c')
            draw=ImageDraw.Draw(comparison);x=0
            for label,panel in zip(('BASELINE: assumed intrinsics','STANDARD5: experimental fixed intrinsics','RADIAL2: experimental fixed intrinsics'),panels):
                draw.text((x+8,8),label,fill='white');comparison.paste(panel,(x,30));x+=panel.width
            comparison.save(output/f'{slot}.calibration_comparison.png')
            for name in (f'{slot}.standard5.png',f'{slot}.radial2.png',f'{slot}.calibration_comparison.png'):
                report['artifacts'][name]={'path':name,'sha256':sha256(output/name)}
        if any(sha256(path)!=expected for path,expected in source_files.items()):
            raise ValidationError('SOURCE_CHANGED','A source changed during comparison')
        report['source_registration_sha256']=source_hash;report['source_mesh_sha256']=mesh_hash
        report['source_calibration_sha256']=calibration_hash;report['embedding_sha256']=item['sha256']
        report['opencv_version']=__import__('cv2').__version__
        report['warning']='Experimental calibration; focus/zoom match and board flatness unverified. Chin contour is approximate. No shape or identity improvement established.'
        report['artifacts']['calibration.snapshot.json']={'path':'calibration.snapshot.json','sha256':sha256(output/'calibration.snapshot.json')}
        report['status']='CALIBRATION_COMPARISON_EXPORTED'
    except KeyboardInterrupt:
        report['status']='CANCELLED';report['error']={'code':'INTERRUPTED','message':'Interrupted'}
    except Exception as error:
        report['error']={'code':error.code if isinstance(error,ValidationError) else 'CALIBRATION_COMPARISON_FAILED','message':str(error)}
    if output is not None:LocalStore().write_json(output/'report.json',report)
    print(json.dumps({'status':report['status'],'report_path':str(output/'report.json') if output else None,
                     'error':report['error'],'mean_fit_residual_px':{
                         slot:{name:fit['mean_fit_residual_px'] for name,fit in view.items()} for slot,view in report['views'].items()}},indent=2))
    return 0 if report['status']=='CALIBRATION_COMPARISON_EXPORTED' else 1
