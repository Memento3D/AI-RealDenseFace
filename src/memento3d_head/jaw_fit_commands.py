"""Source-bound jaw experiment from existing CPU basis; never downloads models."""
import json
from pathlib import Path
import numpy as np
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError
from memento3d_head.domain.contour_annotations import validate_annotations


def add_commands(commands):
    parser=commands.add_parser('fit-jaw',help='Bounded frontal jaw-proxy experiment; guards landmarks in three views')
    for name in ('head-run','shape-run','registration-run','workspace','validation-run','jaw-comparison-run','output'):
        parser.add_argument('--'+name,required=True,type=Path)


def execute(args):
    output=None
    try:
        from memento3d_head.adapters.smooth_preview import read_geometry,render_smooth
        from memento3d_head.adapters.jaw_fitting import fit_jaw
        from memento3d_head.adapters.anatomical_jaw import project_anatomical_jaw,compare_anatomical_arc
        from memento3d_head.adapters.manual_contour_comparison import render_comparison
        load=lambda p:json.loads(p.read_text(encoding='utf-8'))
        paths={'head':args.head_run/'report.json','shape':args.shape_run/'report.json',
               'registration':args.registration_run/'report.json','workspace':args.workspace/'report.json',
               'manifest':args.workspace/'manifest.json','validation':args.validation_run/'report.json',
               'comparison':args.jaw_comparison_run/'report.json',
               'annotations':args.validation_run/'visible_contours.snapshot.json',
               'mesh':args.head_run/'head.obj','identity':args.head_run/'identity.npy',
               'basis':args.shape_run/'shape_basis.npz'}
        hashes={key:sha256(path) for key,path in paths.items()}
        h,s,r,w,m,v,c,data=[load(paths[key]) for key in ('head','shape','registration','workspace','manifest','validation','comparison','annotations')]
        binding=m['source_binding']
        expected={'head':binding['head_report_sha256'],'shape':binding['shape_report_sha256'],
                  'registration':binding['registration_report_sha256'],'mesh':binding['head_obj_sha256'],
                  'manifest':w['artifacts']['manifest.json']['sha256'],
                  'annotations':v['artifacts']['visible_contours.snapshot.json']['sha256'],
                  'identity':h['artifacts']['identity.npy']['sha256'],
                  'basis':s['artifacts']['shape_basis.npz']['sha256']}
        if (any(hashes[key]!=value for key,value in expected.items())
            or hashes['mesh']!=h['artifacts']['head.obj']['sha256']
            or w.get('status')!='CONTOUR_EDITOR_EXPORTED'
            or v.get('status')!='ANNOTATIONS_STRUCTURALLY_VALID'
            or v['source_manifest_sha256']!=hashes['manifest']
            or r.get('status')!='REGISTRATIONS_EXPORTED'
            or r['input']['mesh']!=hashes['mesh']
            or r['input']['head_report']!=hashes['head']
            or s.get('status') not in ('SHAPE_CANDIDATE_EXPORTED','NO_SHAPE_CHANGE')
            or c.get('status')!='MANUAL_CONTOUR_COMPARISON_EXPORTED'
            or c.get('pipeline_version')!=4
            or c.get('boundary_source')!='lower_semantic_face_patch_boundary_proxy'
            or c['input']['manifest_sha256']!=hashes['manifest']
            or c['input']['validation_report_sha256']!=hashes['validation']
            or c['input']['annotations_snapshot_sha256']!=hashes['annotations']
            or c['input']['registration_report_sha256']!=hashes['registration']):
            raise ValidationError('SOURCE_MISMATCH','Expected linked original head, basis, validation and v0.6.4 comparison')
        validate_annotations(data,m)
        curves=data['views']['front']['curves']
        if len(curves)!=2 or {x['kind'] for x in curves}!={'jaw_left_image','jaw_right_image'}:
            raise ValidationError('INVALID_FRONT_JAW','Require left/right frontal curves')
        if len(c['views']['front']['curves'])!=2 or any(x['baseline']['status']!='ANATOMICAL_PROXY_SUPPORTED' for x in c['views']['front']['curves']):
            raise ValidationError('UNSUPPORTED_FRONT_JAW','Baseline frontal curves must pass comparison guards')
        ids=c['surface_extraction']['front']['baseline']['boundary_vertices']
        base,faces=read_geometry(paths['mesh']);beta=np.load(paths['identity'],allow_pickle=False)
        with np.load(paths['basis'],allow_pickle=False) as b:
            decoder=b['base'];basis=b['basis'];indices=b['landmark_faces'];weights=b['barycentric']
            if not np.array_equal(faces,b['faces']):raise ValidationError('TOPOLOGY_MISMATCH','Basis face order differs')
        if beta.shape!=(300,) or not np.isfinite(beta).all() or decoder.shape!=base.shape or not np.isfinite(decoder).all():
            raise ValidationError('INVALID_BASIS','Invalid identity/base')
        difference=float(np.linalg.norm(base-decoder,axis=1).max())
        if difference>.1:raise ValidationError('DECODER_MISMATCH','Basis base differs by over 0.1 mm')
        if not isinstance(ids,list) or len(ids)<3 or any(type(i)!=int or i<0 or i>=len(base) for i in ids) or len(set(ids))!=len(ids):
            raise ValidationError('INVALID_JAW_PATH','Invalid frozen boundary indices')
        # Prevent use of a diagnostic path that jumps between disconnected vertices.
        edges={tuple(sorted((int(a),int(b)))) for face in faces for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0]))}
        if any(tuple(sorted((a,b))) not in edges for a,b in zip(ids,ids[1:])):
            raise ValidationError('INVALID_JAW_PATH','Boundary path crosses a non-edge')
        photos={slot:args.registration_run/f'{slot}.normalized.png' for slot in ('front','left','right')}
        for slot,p in photos.items():
            hashes['photo_'+slot]=sha256(p);paths['photo_'+slot]=p
            if hashes['photo_'+slot]!=binding['photos'][slot]:raise ValidationError('SOURCE_MISMATCH','Photo changed')
        try:args.output.resolve().mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=args.output.resolve()
        mesh,delta,info=fit_jaw(base,basis,faces,indices,weights,r['views'],ids,curves)
        with (output/'head.candidate.obj').open('w',encoding='utf-8') as stream:
            for point in mesh:stream.write('v %.8f %.8f %.8f\n'%tuple(point))
            for face in faces:stream.write('f %d %d %d\n'%tuple(face+1))
        new_beta=beta.copy();new_beta[:8]+=delta.astype(beta.dtype)
        np.save(output/'identity.candidate.npy',new_beta,allow_pickle=False)
        render_smooth(output/'head.candidate.obj',output/'head.candidate.preview.png')
        boundary=(ids,dict(c['surface_extraction']['front']['baseline']));views={}
        for slot in photos:
            results=[]
            for curve in data['views'][slot]['curves']:
                item={'kind':curve['kind'],'id':curve['id']}
                for name,vertices in (('baseline',base),('candidate',mesh)):
                    chains,_=project_anatomical_jaw(vertices,faces,np.arange(len(base)),r['views'][slot],boundary=boundary)
                    item[name]=compare_anatomical_arc(curve['points_px'],chains)
                results.append(item)
            render_comparison(photos[slot],output/f'{slot}.jaw_comparison.png',data['views'][slot]['curves'],results)
            views[slot]={'curves':results}
        if any(sha256(paths[key])!=value for key,value in hashes.items()):raise ValidationError('SOURCE_CHANGED','Input changed while fitting')
        report={'schema_version':1,'pipeline_type':'bounded_frontal_jaw_proxy_fit','pipeline_version':1,
                'status':'JAW_EXPERIMENT_EXPORTED' if np.any(delta) else 'NO_SHAPE_CHANGE',
                'error':None,'fitting':info,'views':views,'input_sha256':hashes,
                'basis_base_max_difference_mm':difference,'geometry_modified':bool(np.any(delta)),
                'cameras_modified':False,'automatic_acceptance':False,'likeness_verified':False,
                'printability_verified':False,'texture_generated':False,
                'warning':'Fit to unverified frontal anatomical proxy; guards use 21 approximate landmarks, not independent identity validation',
                'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in output.iterdir() if p.is_file()}}
        LocalStore().write_json(output/'report.json',report)
        print(json.dumps({'status':report['status'],'report_path':str(output/'report.json'),'error':None},indent=2));return 0
    except Exception as error:
        report={'schema_version':1,'status':'FAILED','error':{'code':error.code if isinstance(error,ValidationError) else 'JAW_FIT_FAILED','message':str(error)}}
        if output is not None:LocalStore().write_json(output/'report.json',report)
        print(json.dumps(report,indent=2));return 1
