"""Compose bounded shared-identity fitting against the original three-view cameras."""
import json
import subprocess
from pathlib import Path
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError


def add_commands(commands):
    parser=commands.add_parser('fit-head',help='Experimental bounded FLAME shape fit; original cameras frozen')
    for flag in ('registration-run','head-run','mica-config','output'):
        parser.add_argument('--'+flag,required=True,type=Path)


def execute(args):
    output=None
    report={'schema_version':1,'pipeline_type':'bounded_multiview_identity_fit','pipeline_version':1,
            'status':'FAILED','error':None,'research_only':True,'likeness_verified':False,
            'printability_verified':False,'has_texture':False,'cameras_modified':False,
            'automatic_acceptance':False,'artifacts':{},'views':{}}
    try:
        import numpy as np
        import cv2
        from PIL import Image,ImageDraw
        from memento3d_head.adapters.mica_assets import verify_config
        from memento3d_head.adapters.smooth_preview import read_geometry,render_smooth
        from memento3d_head.adapters.photo_registration import PAIRS,draw_overlay
        from memento3d_head.adapters.shape_fitting import fit_shape,barycentric_points
        source=args.registration_run/'report.json';before=sha256(source)
        original=json.loads(source.read_text(encoding='utf-8'))
        head_path=args.head_run/'report.json';head=json.loads(head_path.read_text(encoding='utf-8'))
        if (original.get('status')!='REGISTRATIONS_EXPORTED' or set(original.get('views',{}))!={'front','left','right'}
                or original.get('pipeline_type')!='three_view_camera_registration'):
            raise ValidationError('INVALID_REGISTRATION','Use successful original register-views run')
        mesh=args.head_run/'head.obj';identity=args.head_run/'identity.npy'
        if (head.get('status')!='GEOMETRY_EXPORTED' or sha256(head_path)!=original['input']['head_report']
                or sha256(mesh)!=original['input']['mesh'] or sha256(mesh)!=head['artifacts']['head.obj']['sha256']
                or sha256(identity)!=head['artifacts']['identity.npy']['sha256']):
            raise ValidationError('SOURCE_MISMATCH','Head, identity or original report changed')
        sources={source:before,head_path:sha256(head_path),mesh:sha256(mesh),identity:sha256(identity)}
        photos={};observations={row['slot']:row['face']['landmarks'] for row in original['assessment']['photos']}
        for slot in ('front','left','right'):
            for suffix in ('normalized.png','registration.png'):
                path=args.registration_run/f'{slot}.{suffix}'
                if sha256(path)!=original['artifacts'][path.name]['sha256']:
                    raise ValidationError('SOURCE_MISMATCH','Normalized photo or baseline overlay changed')
                sources[path]=sha256(path)
            photos[slot]=args.registration_run/f'{slot}.normalized.png'
            with Image.open(photos[slot]) as im:w,h=im.size
            actual=np.array([[observations[slot][j][0]*w,observations[slot][j][1]*h] for _,j,_ in PAIRS])
            expected=np.array([p['observed_px'] for p in original['views'][slot]['landmarks'][:7]])
            if not np.allclose(actual,expected,atol=1e-5,rtol=0):raise ValidationError('SOURCE_MISMATCH','Landmarks differ from camera fit')
        config=verify_config(args.mica_config)
        sources[args.mica_config]=sha256(args.mica_config)
        try:args.output.resolve().mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=args.output.resolve()
        worker=Path(__file__).parent/'adapters'/'shape_basis_worker.py'
        result=subprocess.run([config['python'],str(worker.resolve()),'--config',str(args.mica_config.resolve()),
                              '--identity',str(identity.resolve()),'--output',str(output/'shape_basis.npz')],
                              cwd=config['repo'],capture_output=True,text=True,timeout=180)
        (output/'shape_worker.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise ValidationError('SHAPE_WORKER_FAILED','See shape_worker.log; no fallback to arbitrary mesh deformation')
        base,faces=read_geometry(mesh)
        with np.load(output/'shape_basis.npz',allow_pickle=False) as data:
            decoder_base=data['base'];basis=data['basis'];indices=data['landmark_faces'];weights=data['barycentric']
            if not np.array_equal(faces,data['faces']):raise ValidationError('TOPOLOGY_MISMATCH','Decoder and OBJ face ordering differ')
        disagreement=float(np.linalg.norm(base-decoder_base,axis=1).max())
        if disagreement>.1:
            raise ValidationError('DECODER_MISMATCH','Pinned FLAME decoder differs from source MICA mesh by over 0.1 mm; see source checkpoint')
        candidate,delta,fit=fit_shape(base,basis,faces,indices,weights,original['views'])
        report['fitting']=fit;report['decoder_source_max_difference_mm']=disagreement
        points=barycentric_points(candidate,faces,indices,weights)
        # Explicit audit of actual correspondences; contour/occluded points are not claimed verified.
        audit=[{'flame68_index':i,'mediapipe_index':j,'name':name,'approximate':True,'used_for_fit':True} for i,j,name in PAIRS]
        audit.append({'flame68_index':8,'mediapipe_index':152,'name':'chin_fixed_contour_reference','approximate':True,'used_for_fit':False})
        LocalStore().write_json(output/'correspondences.json',{'pairs':audit,'dynamic_contour_implemented':False,
                                                           'landmark_visibility_verified':False})
        # Pure triangle OBJ with source topology; keep original artifacts untouched.
        with (output/'head.candidate.obj').open('w',encoding='utf-8') as stream:
            for v in candidate:stream.write('v %.8f %.8f %.8f\n'%tuple(v))
            for f in faces:stream.write('f %d %d %d\n'%tuple(f+1))
        beta=np.load(identity,allow_pickle=False).copy();beta[:8]+=delta.astype(beta.dtype)
        np.save(output/'identity.candidate.npy',beta,allow_pickle=False)
        render_smooth(output/'head.candidate.obj',output/'head.candidate.preview.png')
        for slot in ('front','left','right'):
            v=original['views'][slot]
            k=np.array(v['camera_matrix']);r=np.array(v['rotation_vector']);t=np.array(v['translation_mm'])
            projected=cv2.projectPoints(points[[i for i,_,_ in PAIRS]],r,t,k,None)[0].reshape(-1,2)
            candidate_report=draw_overlay(photos[slot],candidate,faces,points,observations[slot],output/f'{slot}.candidate.png',
                                          camera_solution=(k,r,t,projected))
            candidate_report['cameras_modified']=False
            report['views'][slot]={'baseline':v,'candidate':candidate_report}
            with Image.open(args.registration_run/f'{slot}.registration.png') as old,Image.open(output/f'{slot}.candidate.png') as new:
                comparison=Image.new('RGB',(old.width+new.width,max(old.height,new.height)+30),'#20252c')
                comparison.paste(old,(0,30));comparison.paste(new,(old.width,30))
                draw=ImageDraw.Draw(comparison);draw.text((8,8),'ORIGINAL MICA',fill='white')
                draw.text((old.width+8,8),'BOUNDED SHAPE CANDIDATE - SAME CAMERA',fill='white')
                comparison.save(output/f'{slot}.shape_comparison.png')
        if any(sha256(path)!=expected for path,expected in sources.items()):raise ValidationError('SOURCE_CHANGED','Input changed during fitting')
        report['input']={'registration_report_sha256':before,'head_report_sha256':sources[head_path],
                         'head_obj_sha256':sources[mesh],'identity_sha256':sources[identity]}
        report['assets_sha256']={key:item['sha256'] for key,item in config['assets'].items()}
        report['geometry_modified']=bool(np.max(np.abs(delta))>1e-12)
        report['multi_view_fitting']=True
        report['camera_source']='original_register_views_assumed_intrinsics_no_distortion'
        report['warning']='Candidate fit to 21 approximate landmarks. Dynamic contours and occlusion filtering remain unimplemented; no claim of improved likeness.'
        report['status']='SHAPE_CANDIDATE_EXPORTED' if report['geometry_modified'] else 'NO_SHAPE_CHANGE'
        for path in output.iterdir():
            if path.is_file() and path.name!='report.json':report['artifacts'][path.name]={'path':path.name,'sha256':sha256(path),'size_bytes':path.stat().st_size}
    except KeyboardInterrupt:
        report['status']='CANCELLED';report['error']={'code':'INTERRUPTED','message':'Interrupted'}
    except Exception as error:
        report['error']={'code':error.code if isinstance(error,ValidationError) else 'SHAPE_FIT_FAILED','message':str(error)}
    if output is not None:LocalStore().write_json(output/'report.json',report)
    print(json.dumps({'status':report['status'],'report_path':str(output/'report.json') if output else None,
                     'error':report['error'],'max_vertex_displacement_mm':report.get('fitting',{}).get('max_vertex_displacement_mm')},indent=2))
    return 0 if report['status'] in {'SHAPE_CANDIDATE_EXPORTED','NO_SHAPE_CHANGE'} else 1
