"""Composition of research-only experiments and mesh previews."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from memento3d_head import __version__
from memento3d_head.adapters.experiment_input import LocalExperimentStore, OneShotMediaPipeInspector
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.mica_backend import MicaBackend
from memento3d_head.application.experiment_head import experiment_head
from memento3d_head.domain.contracts import Photo, ValidationError
from memento3d_head.pipelines.single_photo_validation_v1 import SinglePhotoValidationV1
from memento3d_head.settings import load_settings


COMMANDS={'experiment-head','render-head','register-head','register-views','refine-cameras','compare-calibration','fit-head','audit-contours','prepare-contours','validate-contours','compare-manual-contours','compare-jaw-contours','fit-jaw','compare-heads','preview-photo-color','preview-multiview-color','demo'}


def add_commands(commands) -> None:
    """Expose separate research commands without weakening the photo-set gate."""
    from memento3d_head.demo_commands import add_commands as add_demo
    add_demo(commands)
    from memento3d_head.multiview_projection_commands import add_commands as add_multiview
    add_multiview(commands)
    from memento3d_head.photo_projection_commands import add_commands as add_photo_color
    add_photo_color(commands)
    from memento3d_head.head_compare_commands import add_commands as add_compare_heads
    add_compare_heads(commands)
    from memento3d_head.jaw_fit_commands import add_commands as add_jaw_fit
    add_jaw_fit(commands)
    from memento3d_head.manual_comparison_commands import add_commands as add_manual
    add_manual(commands)
    from memento3d_head.contour_editor_commands import add_commands as add_editor
    add_editor(commands)
    from memento3d_head.contour_commands import add_commands as add_contour
    add_contour(commands)
    from memento3d_head.shape_commands import add_commands as add_shape
    add_shape(commands)
    from memento3d_head.calibration_comparison_commands import add_commands as add_calibration
    add_calibration(commands)
    from memento3d_head.camera_experiment_commands import add_commands as add_camera
    add_camera(commands)
    from memento3d_head.registration_commands import add_commands as add_registration
    add_registration(commands)
    experiment=commands.add_parser('experiment-head',help='Research-only single-photo geometry trial')
    experiment.add_argument('--input',required=True,type=Path)
    experiment.add_argument('--source-kind',required=True,choices=('real_capture','synthetic','unknown'))
    experiment.add_argument('--model',required=True,type=Path)
    experiment.add_argument('--model-sha256',required=True)
    experiment.add_argument('--mica-config',required=True,type=Path)
    experiment.add_argument('--config',type=Path)
    experiment.add_argument('--output',required=True,type=Path)
    render=commands.add_parser('render-head',help='Smooth preview of existing OBJ, no inference')
    render.add_argument('--mesh',required=True,type=Path)
    render.add_argument('--output',required=True,type=Path)


def execute(args) -> int:
    """Create a fresh workspace, retain failures and emit only measured results."""
    if args.command == "demo":
        from memento3d_head.demo_commands import execute as demo
        return demo(args)
    if args.command == "preview-multiview-color":
        from memento3d_head.multiview_projection_commands import execute as multiview
        return multiview(args)
    if args.command == "preview-photo-color":
        from memento3d_head.photo_projection_commands import execute as preview_photo
        return preview_photo(args)
    if args.command == "compare-heads":
        from memento3d_head.head_compare_commands import execute as compare_heads
        return compare_heads(args)
    if args.command == "fit-jaw":
        from memento3d_head.jaw_fit_commands import execute as jaw_fit
        return jaw_fit(args)
    if args.command in {"compare-manual-contours", "compare-jaw-contours"}:
        from memento3d_head.manual_comparison_commands import execute as compare_manual
        return compare_manual(args)
    if args.command in {"prepare-contours", "validate-contours"}:
        from memento3d_head.contour_editor_commands import execute as editor
        return editor(args)
    if args.command == "audit-contours":
        from memento3d_head.contour_commands import execute as audit
        return audit(args)
    if args.command == "fit-head":
        from memento3d_head.shape_commands import execute as fit
        return fit(args)
    if args.command == "compare-calibration":
        from memento3d_head.calibration_comparison_commands import execute as compare
        return compare(args)
    if args.command == "refine-cameras":
        from memento3d_head.camera_experiment_commands import execute as refine
        return refine(args)
    if args.command in {"register-head", "register-views"}:
        from memento3d_head.registration_commands import execute as register
        return register(args)
    output=None
    try:
        settings=None
        if args.command=='experiment-head':
            settings=load_settings(args.config)
            if not args.input.is_file():
                raise ValidationError('INPUT_NOT_FOUND','Single-image input does not exist')
            if len(args.model_sha256)!=64 or any(c not in '0123456789abcdefABCDEF' for c in args.model_sha256):
                raise ValidationError('INVALID_MODEL_HASH','Expected a 64-character SHA256')
        elif not args.mesh.is_file():
            raise ValidationError('INPUT_NOT_FOUND','OBJ file does not exist')
        candidate=args.output.resolve()
        try:
            candidate.mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:
            raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=candidate
        store=LocalExperimentStore()
        if args.command=='experiment-head':
            inspector=OneShotMediaPipeInspector(args.model.resolve(),args.model_sha256)
            provenance={'package_version':__version__,'backend':'mediapipe','device':'CPU',
                        'model_sha256':args.model_sha256.lower(),'calibrated':False,
                        'pipeline_type':'single_photo_validation','pipeline_version':1}
            report=experiment_head(Photo('front',args.input.resolve()),output,settings,args.source_kind,
                                   SinglePhotoValidationV1(inspector),
                                   MicaBackend(args.mica_config.resolve(),smooth_preview=True),store,provenance)
        else:
            from memento3d_head.adapters.smooth_preview import render_smooth
            start=time.monotonic()
            before=sha256(args.mesh)
            preview=output/'head.preview.smooth.png'
            rendering=render_smooth(args.mesh,preview)
            if sha256(args.mesh)!=before:
                raise ValidationError('SOURCE_CHANGED','OBJ changed while rendering')
            report={'schema_version':1,'attempt_id':str(uuid4()),'pipeline_type':'mesh_inspection',
                    'pipeline_version':1,'status':'PREVIEW_EXPORTED','error':None,
                    'started_at':datetime.now(timezone.utc).isoformat(),'inference_run':False,
                    'source_mesh_sha256':before,'geometry_modified':False,'rendering':rendering,
                    'artifacts':{preview.name:{'path':preview.name,'sha256':sha256(preview),
                                               'size_bytes':preview.stat().st_size}},
                    'elapsed_seconds':round(time.monotonic()-start,3)}
            store.write_json(output/'report.json',report)
        print(json.dumps({'status':report['status'],'report_path':str(output/'report.json'),
                          'error':report.get('error')},ensure_ascii=False,indent=2))
        return {'PREVIEW_EXPORTED':0,'GEOMETRY_EXPORTED':0,'NEEDS_RETAKE':2,'CANCELLED':130}.get(report['status'],1)
    except KeyboardInterrupt:
        failure={'schema_version':1,'status':'CANCELLED','error':{'code':'INTERRUPTED','message':'Interrupted'}}
    except Exception as error:
        failure={'schema_version':1,'status':'FAILED','error':{
            'code':error.code if isinstance(error,ValidationError) else 'SETUP_FAILED','message':str(error)}}
    if output is not None:
        LocalExperimentStore().write_json(output/'report.json',failure)
    print(json.dumps(failure,ensure_ascii=False,indent=2))
    return 130 if failure['status']=='CANCELLED' else 1
