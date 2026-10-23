"""Composition boundary for the CPU texture workspace."""
import json
from pathlib import Path


def add_command(commands):
    child=commands.add_parser('optimize-head',help='Review masks, balance seams and add a base material to an existing RDF head')
    child.add_argument('--head-run',type=Path,required=True)
    child.add_argument('--texture-run',type=Path)
    mask_group=child.add_mutually_exclusive_group()
    mask_group.add_argument('--masks-json',type=Path)
    mask_group.add_argument('--auto-mask',action='store_true',help='Generate masks automatically before optimizing texture')
    child.add_argument('--face-parser-model',type=Path,default=Path('D:/AI-cache/face-parsing/resnet18.onnx'))
    child.add_argument('--skin-color',help='Optional reviewed base color #RRGGBB')


def execute(args):
    from memento3d_head.adapters.head_texture_workspace import build_workspace
    try:
        auto_report=None
        masks=args.masks_json
        texture=args.texture_run
        if args.auto_mask:
            from uuid import uuid4
            from memento3d_head.adapters.automatic_face_masks import generate_masks
            if texture is None:
                candidates=sorted(args.head_run.glob('three_textures_*/report.json'),key=lambda p:p.stat().st_mtime,reverse=True)
                texture=next((p.parent for p in candidates if json.loads(p.read_text()).get('status')=='THREE_TEXTURES_PREVIEW_EXPORTED'),None)
                if texture is None:raise ValueError('No successful three-texture export found')
            masks,auto_report=generate_masks(texture,args.face_parser_model,args.head_run/('auto_masks_'+uuid4().hex[:8]))
        output,report=build_workspace(args.head_run,texture,masks,args.skin_color)
        if auto_report is not None:
            report['automatic_parsing_report']=str(masks.parent/'report.json')
            report['automatic_parsing_warnings']=auto_report['warnings']
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'status':report['status'],'workspace':str(output/'head.workspace.html'),
            'mask_editor':str(output/'head.masks.html'),'report':str(output/'report.json'),
            'source_texture_run':report['source_texture_run'],'masks_reviewed':report['masks_reviewed'],'mask_origin':report['mask_origin'],
            'automatic_masks':str(masks) if auto_report else None},indent=2))
        return 0
    except Exception as error:
        print(json.dumps({'status':'FAILED','error':str(error)},indent=2))
        return 1
