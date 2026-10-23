"""Source-bound manual/mesh contour diagnostics; no geometry fitting."""
import json
from pathlib import Path
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contour_annotations import validate_annotations
from memento3d_head.domain.contracts import ValidationError


def add_commands(commands):
    parser=commands.add_parser('compare-manual-contours',help='Diagnostic ordered silhouette arcs; no fitting')
    for name in ('workspace','validation-run','registration-run','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    jaw=commands.add_parser('compare-jaw-contours',help='Semantic face surface edges; exclude neck/ears; no fitting')
    for name in ('workspace','validation-run','registration-run','head-run','shape-run','mica-config','output'):
        jaw.add_argument('--'+name,required=True,type=Path)


def execute(args):
    output=None
    try:
        from memento3d_head.adapters.manual_contour_comparison import compare_arc,render_comparison
        manifest_path=args.workspace/'manifest.json';workspace_report=args.workspace/'report.json'
        validation_path=args.validation_run/'report.json';snapshot=args.validation_run/'visible_contours.snapshot.json'
        registration_path=args.registration_run/'report.json'
        sources={p:sha256(p) for p in (manifest_path,workspace_report,validation_path,snapshot,registration_path)}
        load=lambda p:json.loads(p.read_text(encoding='utf-8'))
        manifest=load(manifest_path);workspace=load(workspace_report);validation=load(validation_path);data=load(snapshot)
        if (workspace.get('status')!='CONTOUR_EDITOR_EXPORTED'
                or sources[manifest_path]!=workspace['artifacts']['manifest.json']['sha256']
                or validation.get('status')!='ANNOTATIONS_STRUCTURALLY_VALID'
                or validation['source_manifest_sha256']!=sources[manifest_path]
                or sources[snapshot]!=validation['artifacts'][snapshot.name]['sha256']
                or sources[registration_path]!=manifest['source_binding']['registration_report_sha256']):
            raise ValidationError('SOURCE_MISMATCH','Workspace, validated snapshot and registration do not match')
        validate_annotations(data,manifest)
        front_kinds=[c['kind'] for c in data['views']['front']['curves']]
        if front_kinds.count('jaw_left_image')!=1 or front_kinds.count('jaw_right_image')!=1 or len(front_kinds)!=2:
            raise ValidationError('JAW_LABELS_REQUIRED','Comparison requires one left and one right frontal jaw curve')
        photos={slot:args.registration_run/f'{slot}.normalized.png' for slot in data['views']}
        for slot,p in photos.items():
            sources[p]=sha256(p)
            if sources[p]!=manifest['views'][slot]['image_sha256']:
                raise ValidationError('SOURCE_MISMATCH','Registration photo changed')
        try:args.output.resolve().mkdir(parents=True,exist_ok=False)
        except FileExistsError as error:raise ValidationError('OUTPUT_EXISTS','Use a new output directory') from error
        output=args.output.resolve();views={}
        anatomical=getattr(args,'command',None)=='compare-jaw-contours'
        outlines={slot:manifest['views'][slot]['outlines'] for slot in data['views']}
        diagnostics={}
        if anatomical:
            from memento3d_head.adapters.jaw_sources import prepare_jaw_outlines
            from memento3d_head.adapters.anatomical_jaw import compare_anatomical_arc
            outlines,diagnostics=prepare_jaw_outlines(args,manifest,load(registration_path),output,sources)
        for slot,view in data['views'].items():
            results=[]
            for curve in view['curves']:
                result={'id':curve['id'],'kind':curve['kind']}
                for name in ('baseline','candidate'):
                    result[name]=(compare_anatomical_arc if anatomical else compare_arc)(curve['points_px'],outlines[slot][name])
                result['candidate_minus_baseline_px']=(result['candidate']['mean_ordered_distance_px']-result['baseline']['mean_ordered_distance_px']
                    if all('mean_ordered_distance_px' in result[n] for n in ('baseline','candidate')) else None)
                results.append(result)
            render_comparison(photos[slot],output/f'{slot}.manual_comparison.png',view['curves'],results)
            views[slot]={'curves':results}
        if any(sha256(p)!=expected for p,expected in sources.items()):
            raise ValidationError('SOURCE_CHANGED','Source changed during comparison')
        report={'schema_version':1,'pipeline_type':'manual_contour_diagnostic','pipeline_version':4 if anatomical else 1,
                'status':'MANUAL_CONTOUR_COMPARISON_EXPORTED','error':None,'views':views,
                'geometry_modified':False,'cameras_modified':False,'likeness_verified':False,
                'correspondences_verified':False,'ready_for_automatic_fitting':False,
                'boundary_source':'lower_semantic_face_patch_boundary_proxy' if anatomical else 'whole_head_silhouette',
                'surface_extraction':diagnostics,
                'limitations':(['Anatomical patch boundary is a proxy, not an occluding silhouette',
                                'Visibility and anatomical correspondence remain unverified; unsupported curves have no fitting metric'] if anatomical else ['Endpoint-bounded whole-head arcs can include ear/neck'])+[
                               'Equal arclength distances are diagnostic, not anatomical error or likeness'],
                'input':{'manifest_sha256':sources[manifest_path],'validation_report_sha256':sources[validation_path],
                         'annotations_snapshot_sha256':sources[snapshot],'registration_report_sha256':sources[registration_path]},
                'artifacts':{f'{slot}.manual_comparison.png':{'path':f'{slot}.manual_comparison.png',
                             'sha256':sha256(output/f'{slot}.manual_comparison.png')} for slot in views}}
        if anatomical:
            report['input'].update({key:manifest['source_binding'][key] for key in ('head_obj_sha256','candidate_obj_sha256','head_report_sha256','shape_report_sha256')})
            report['input']['mica_config_sha256']=sources[args.mica_config]
            report['input']['face_mask_sha256']='ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492'
            report['supported_curve_counts']={slot:{name:sum('mean_ordered_distance_px' in curve[name] for curve in view['curves']) for name in ('baseline','candidate')} for slot,view in views.items()}
            report['all_curves_supported']=all('mean_ordered_distance_px' in curve[name] for view in views.values() for curve in view['curves'] for name in ('baseline','candidate'))
            for name in ('face_regions.npz','face_mask_worker.log','surface_chains.json'):
                report['artifacts'][name]={'path':name,'sha256':sha256(output/name)}
        LocalStore().write_json(output/'report.json',report)
        print(json.dumps({'status':report['status'],'report_path':str(output/'report.json'),'error':None},indent=2));return 0
    except Exception as error:
        report={'schema_version':1,'status':'FAILED','error':{'code':error.code if isinstance(error,ValidationError) else 'COMPARISON_FAILED','message':str(error)}}
        if output is not None:LocalStore().write_json(output/'report.json',report)
        print(json.dumps(report,indent=2));return 1
