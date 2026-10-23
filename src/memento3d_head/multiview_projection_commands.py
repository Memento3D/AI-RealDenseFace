"""Source-bound multiview color experiment CLI."""
import json
from pathlib import Path
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError


def add_commands(commands):
    parser=commands.add_parser('preview-multiview-color',help='Inspect three photo projections and best-view composition')
    for name in ('candidate-run','registration-run','output'):parser.add_argument('--'+name,required=True,type=Path)


def execute(args):
    output=None
    try:
        from memento3d_head.adapters.multiview_projection import build_preview,SLOTS
        from memento3d_head.adapters.smooth_preview import read_geometry
        cp=args.candidate_run/'report.json';rp=args.registration_run/'report.json';mesh=args.candidate_run/'head.candidate.obj'
        photos={slot:args.registration_run/(slot+'.normalized.png') for slot in SLOTS}
        hashes={p:sha256(p) for p in (cp,rp,mesh,*photos.values())}
        c=json.loads(cp.read_text(encoding='utf-8'));r=json.loads(rp.read_text(encoding='utf-8'))
        valid=(c.get('pipeline_type')=='bounded_frontal_jaw_proxy_fit' and c.get('status') in ('JAW_EXPERIMENT_EXPORTED','NO_SHAPE_CHANGE') and r.get('pipeline_type')=='three_view_camera_registration' and r.get('status')=='REGISTRATIONS_EXPORTED' and c['input_sha256']['registration']==hashes[rp] and c['artifacts']['head.candidate.obj']['sha256']==hashes[mesh])
        valid=valid and all(c['input_sha256']['photo_'+slot]==hashes[p] and r['artifacts'][slot+'.normalized.png']['sha256']==hashes[p] for slot,p in photos.items())
        if not valid:raise ValidationError('SOURCE_MISMATCH','Candidate, registration and all three normalized photos must match linked reports')
        vertices,faces=read_geometry(mesh)
        try:args.output.resolve().mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=args.output.resolve();provenance={str(p):h for p,h in hashes.items()}
        info=build_preview(output,vertices,faces,r,photos,provenance)
        if any(sha256(p)!=h for p,h in hashes.items()):raise ValidationError('SOURCE_CHANGED','Input changed during projection')
        report={'schema_version':1,'pipeline_type':'multiview_photo_color_projection','pipeline_version':1,'status':'MULTIVIEW_COLOR_EXPORTED','error':None,'input_sha256':provenance,'projection':info,'geometry_modified':False,'cameras_modified':False,'research_only':True,'automatic_acceptance':False,'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in output.iterdir() if p.is_file()}}
        LocalStore().write_json(output/'report.json',report)
        print(json.dumps({'status':report['status'],'report_path':str(output/'report.json'),'covered_triangles':info['covered_triangles']},indent=2));return 0
    except Exception as error:
        report={'status':'FAILED','error':{'code':error.code if isinstance(error,ValidationError) else 'MULTIVIEW_PROJECTION_FAILED','message':str(error)}}
        if output is not None:LocalStore().write_json(output/'report.json',report)
        print(json.dumps(report,indent=2));return 1
