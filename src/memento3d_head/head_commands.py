"""Composition helpers for external MICA environment and head CLI commands."""
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from memento3d_head.adapters.head_store import LocalHeadStore
from memento3d_head.adapters.mica_assets import MICA_COMMIT, inspect_repo, pin_assets
from memento3d_head.adapters.mica_backend import MicaBackend, run_worker
from memento3d_head.application.reconstruct_head import reconstruct_head
from memento3d_head.domain.contracts import ValidationError
from memento3d_head.domain.head import HeadRequest


def add_commands(commands) -> None:
    """Register only implemented commands, without changing validation v1 options."""
    from memento3d_head.research_commands import add_commands as add_research_commands
    add_research_commands(commands)
    status=commands.add_parser('mica-status')
    status.add_argument('--repo',required=True,type=Path)
    status.add_argument('--worker-python',type=Path)
    pin=commands.add_parser('mica-pin')
    pin.add_argument('--repo',required=True,type=Path)
    pin.add_argument('--worker-python',required=True,type=Path)
    pin.add_argument('--detector',required=True,type=Path)
    pin.add_argument('--output-config',required=True,type=Path)
    for command in ('mica-doctor','reconstruct-head'):
        child=commands.add_parser(command)
        child.add_argument('--mica-config',required=True,type=Path)
        child.add_argument('--output',required=True,type=Path)
        if command=='reconstruct-head':
            child.add_argument('--validation-run',required=True,type=Path)
            child.add_argument('--front',required=True,type=Path)
            child.add_argument('--reviewed-by',required=True)
            child.add_argument('--allow-synthetic',action='store_true')


def execute(args) -> int:
    """Compose preflight, isolated model doctor or one geometry attempt."""
    from memento3d_head.research_commands import COMMANDS, execute as execute_research
    if args.command in COMMANDS:
        return execute_research(args)
    output=None
    try:
        if args.command=='mica-status':
            repo=args.repo.resolve()
            if not (repo/'.git').exists():
                result={'status':'SETUP_REQUIRED','expected_commit':MICA_COMMIT,
                        'repo_found':False,'message':'Clone official MICA to this path first'}
            else:
                result={'status':'SETUP_REQUIRED','repo_found':True,**inspect_repo(repo)}
            if args.worker_python:
                if not args.worker_python.is_file():
                    result['worker_environment']={'error':'Worker Python does not exist'}
                else:
                    probe="import importlib.util,json,sys; print(json.dumps({'python':sys.version.split()[0],'missing':[n for n in ['torch','numpy','cv2','trimesh','insightface','onnxruntime','yacs','loguru','chumpy'] if importlib.util.find_spec(n) is None]}))"
                    info=subprocess.run([str(args.worker_python.resolve()),'-c',probe],capture_output=True,text=True,timeout=30)
                    result['worker_environment']=json.loads(info.stdout) if info.returncode==0 else {'error':info.stderr[-1500:]}
            result['inference_tested']=False
            print(json.dumps(result,indent=2))
            return 0
        if args.command=='mica-pin':
            pin_assets(args.repo.resolve(),args.worker_python.resolve(),args.detector.resolve(),args.output_config.resolve())
            print(json.dumps({'status':'ASSETS_PINNED','config_path':str(args.output_config.resolve()),'inference_tested':False},indent=2))
            return 0
        candidate=args.output.resolve()
        try:
            candidate.mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:
            raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=candidate
        if args.command=='mica-doctor':
            result=run_worker(args.mica_config.resolve(),output)
            result.update(schema_version=1,pipeline_type='mica_doctor',pipeline_version=1)
            LocalHeadStore().write_json(output/'report.json',result)
        else:
            request=HeadRequest(args.validation_run.resolve(),args.front.resolve(),args.reviewed_by,args.allow_synthetic)
            result=reconstruct_head(request,output,MicaBackend(args.mica_config.resolve()),LocalHeadStore())
        print(json.dumps({'status':result['status'],'report_path':str(output/'report.json'),
                          'error':result.get('error')},ensure_ascii=False,indent=2))
        return 0 if result['status'] in {'MODEL_LOADED','GEOMETRY_EXPORTED'} else (130 if result['status']=='CANCELLED' else 1)
    except KeyboardInterrupt:
        failure={'schema_version':1,'status':'CANCELLED','error':{'code':'INTERRUPTED','message':'Interrupted'}}
    except Exception as error:
        failure={'schema_version':1,'status':'FAILED','error':{
            'code':error.code if isinstance(error,ValidationError) else 'SETUP_FAILED','message':str(error)}}
    if output is not None:
        LocalHeadStore().write_json(output/'report.json',failure)
    print(json.dumps(failure,ensure_ascii=False,indent=2))
    return 130 if failure['status']=='CANCELLED' else 1
