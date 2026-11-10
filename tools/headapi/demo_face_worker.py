"""Versioned UV render stage for real RDF jobs, independent of experiment folders."""
import argparse,json
from pathlib import Path
from tools.headapi.service import sha,write

POLICY='strict_uv_v1'


def run(job,output,parser,assets):
    job=Path(job).resolve();output=Path(output).resolve();parser=Path(parser).resolve();assets=Path(assets).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','policy':POLICY,'geometry_changed':False,'camera_refinement_enabled':False,
            'ready_for_machine':False,'likeness_verified':False,'input_sha256':{}}
    try:
        fit=job/'rdf/report.json';crops=job/'rdf_crops/report.json';bundle=job/'rdf/projection_bundle.npz'
        rdf=json.loads(fit.read_text(encoding='utf-8'));metadata=json.loads(crops.read_text(encoding='utf-8'))
        if rdf.get('status')!='FIT_GATES_PASSED' or rdf.get('views_used')!=['front','left','right']:raise ValueError('UV stage requires a successful three-view RDF fit')
        if sha(bundle)!=rdf.get('projection_bundle_sha256') or sha(crops)!=rdf.get('crop_report_sha256'):raise ValueError('RDF bundle/crop report checksum mismatch')
        report['input_sha256']={str(path):sha(path) for path in [fit,crops,bundle,parser,assets]}
        for slot in ['front','left','right']:
            path=job/'rdf_crops'/(slot+'.png')
            if sha(path)!=metadata['views'][slot].get('crop_sha256'):raise ValueError('Crop checksum mismatch: '+slot)
            report['input_sha256'][str(path)]=sha(path)
        from tools.headapi.face_texture_trial import run as bake
        from tools.headapi.texture_completion_trial import finalize
        bake(output=output/'raw',geometry_path=bundle,expected_bundle_sha=sha(bundle),source_dir=job/'rdf_crops',
             quality_maps=True,harmonize=True,visibility_mode='continuous_triangle',visibility_tolerance=1e-6,
             parser_path_override=parser,assets_path_override=assets)
        finalize(output/'raw',output/'final',bundle,assets_path=assets)
        final=json.loads((output/'final/report.json').read_text())
        if not final.get('triangle_coordinates_float32_roundtrip_equal'):raise ValueError('UV render geometry verification failed')
        for path,digest in report['input_sha256'].items():
            if sha(path)!=digest:raise ValueError('Input changed during UV render stage')
        report.update(status='UV_FACE_EXPORTED',head_artifact='final/face_textured.glb',
                      color_policy='per-job robust matched-skin RGB gains bounded 0.85..1.18; front anchor; no frozen dataset gains',
                      raw_report_sha256=sha(output/'raw/report.json'),final_report_sha256=sha(output/'final/report.json'),
                      head_sha256=sha(output/'final/face_textured.glb'),manufacturing_mesh=False)
    except Exception as exc:
        report.update(status='FAILED',error=str(exc));raise
    finally:write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parser',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    a=p.parse_args();run(a.job,a.output,a.parser,a.assets)
