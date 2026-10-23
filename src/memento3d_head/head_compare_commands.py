"""Source-bound comparison viewer composition; never changes mesh geometry."""
import json
from pathlib import Path
import numpy as np
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError


def add_commands(commands):
    parser=commands.add_parser('compare-heads',help='Create offline 3D comparison with shared orbit/zoom/light')
    for name in ('head-run','candidate-run','output'):parser.add_argument('--'+name,required=True,type=Path)


def execute(args):
    output=None
    try:
        from memento3d_head.adapters.smooth_preview import read_geometry
        from memento3d_head.adapters.head_compare import write_comparison
        hp=args.head_run/'report.json';cp=args.candidate_run/'report.json'
        base_path=args.head_run/'head.obj';candidate_path=args.candidate_run/'head.candidate.obj'
        sources={p:sha256(p) for p in (hp,cp,base_path,candidate_path)}
        h=json.loads(hp.read_text(encoding='utf-8'));c=json.loads(cp.read_text(encoding='utf-8'))
        if (h.get('status')!='GEOMETRY_EXPORTED'
            or c.get('status') not in ('JAW_EXPERIMENT_EXPORTED','NO_SHAPE_CHANGE')
            or c.get('pipeline_type')!='bounded_frontal_jaw_proxy_fit'
            or c['input_sha256']['head']!=sources[hp]
            or c['input_sha256']['mesh']!=sources[base_path]
            or h['artifacts']['head.obj']['sha256']!=sources[base_path]
            or c['artifacts']['head.candidate.obj']['sha256']!=sources[candidate_path]):
            raise ValidationError('SOURCE_MISMATCH','Expected linked original head and jaw experiment OBJ/report')
        base,faces=read_geometry(base_path);candidate,candidate_faces=read_geometry(candidate_path)
        if not np.array_equal(faces,candidate_faces):raise ValidationError('TOPOLOGY_MISMATCH','Mesh topology/order changed')
        try:args.output.resolve().mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=args.output.resolve();info=write_comparison(output/'head_comparison.html',base,candidate,faces,provenance={
            'head_obj_sha256':sources[base_path],'candidate_obj_sha256':sources[candidate_path],
            'head_report_sha256':sources[hp],'candidate_report_sha256':sources[cp]})
        if any(sha256(p)!=value for p,value in sources.items()):raise ValidationError('SOURCE_CHANGED','Input changed during viewer export')
        report={'schema_version':1,'pipeline_type':'offline_head_comparison','pipeline_version':1,
                'status':'HEAD_COMPARISON_EXPORTED','error':None,'viewer':info,'geometry_modified':False,
                'likeness_verified':False,'automatic_acceptance':False,
                'input':{'head_report_sha256':sources[hp],'candidate_report_sha256':sources[cp],
                         'head_obj_sha256':sources[base_path],'candidate_obj_sha256':sources[candidate_path]},
                'artifacts':{'head_comparison.html':{'path':'head_comparison.html','sha256':sha256(output/'head_comparison.html')}}}
        LocalStore().write_json(output/'report.json',report)
        print(json.dumps({'status':report['status'],'report_path':str(output/'report.json'),'error':None},indent=2));return 0
    except Exception as error:
        report={'schema_version':1,'status':'FAILED','error':{'code':error.code if isinstance(error,ValidationError) else 'VIEWER_EXPORT_FAILED','message':str(error)}}
        if output is not None:LocalStore().write_json(output/'report.json',report)
        print(json.dumps(report,indent=2));return 1
