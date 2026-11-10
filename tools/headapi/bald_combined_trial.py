"""Guarded, immutable combined bald-head review from existing RDF/UV jobs."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
from tools.headapi.service import ROOT,sha,write
from tools.headapi.runtime_budget import available_ram,run_guarded

POLICY='bald_combined_review_v1'
STAGES=[('01_bald','tools.headapi.bald_head_trial','BALD_HEAD_EXPORTED_FOR_REVIEW','bald_head_surface_v3'),
        ('02_transition','tools.headapi.bald_transition_trial','BALD_TRANSITION_EXPORTED_FOR_REVIEW','bald_skin_transition_v3'),
        ('03_tone','tools.headapi.bald_tone_trial','TONE_CANDIDATES_EXPORTED_FOR_REVIEW','bald_low_frequency_tone_v2')]


def validate_stage(report,status,policy):
    if report.get('status')!=status or report.get('policy')!=policy:
        raise ValueError('Stage did not satisfy status/policy contract')
    if report.get('geometry_changed') is not False:
        raise ValueError('Stage changed geometry or omitted evidence')
    for p,digest in report.get('input_sha256',{}).items():
        if sha(Path(p))!=digest:raise ValueError('Stage input changed')


def verify(job,output):
    """Independent final file checks and review renders, in isolated SDK process."""
    import numpy as np
    from PIL import Image
    import trimesh
    from scripts.eye_components import identify
    from tools.headapi.demo_appearance import atlas_positions
    from tools.headapi.appearance_preview import preview_scene
    from tools.headapi.demo_scene_worker import inspect_glb
    job,output=Path(job).resolve(),Path(output).resolve()
    reports=[]
    for name,_,status,policy in STAGES:
        row=json.loads((output/name/'report.json').read_text())
        validate_stage(row,status,policy);reports.append(row)
    first,transition,tone=reports
    source=output/'03_tone/mild'
    if sha(source/'head.glb')!=tone['variants']['mild']['head_sha256']:raise ValueError('Selected GLB changed')
    with np.load(job/'rdf/projection_bundle.npz',allow_pickle=False) as b:v,f=b['vertices'],b['faces']
    assets=next(Path(p) for p in first['input_sha256'] if Path(p).name=='flame_assets.npz')
    quality=job/'rdf_uv/raw/quality_maps.npz'
    with np.load(assets,allow_pickle=False) as b:uv=b['uvs'][b['uv_faces']]
    with np.load(quality,allow_pickle=False) as b:fi,protected=b['face_index'],b['protected']
    original=np.asarray(Image.open(job/'rdf_uv/final/face_atlas.png').convert('RGB'))
    baseline=np.asarray(Image.open(output/'01_bald/head-atlas.png').convert('RGB'))
    after=np.asarray(Image.open(source/'head-atlas.png').convert('RGB'))
    early_labels=np.asarray(Image.open(output/'01_bald/provenance-labels.png'))
    labels=np.asarray(Image.open(source/'provenance-labels.png'))
    confidence=np.asarray(Image.open(source/'confidence.png'))
    inherited_conf=np.asarray(Image.open(output/'01_bald/confidence.png'))
    _,anatomy=identify(v,f);eye=np.mean(anatomy['eye_centers_native'],axis=0);height=np.ptp(v[:,1])
    pos=atlas_positions(fi,v,f,uv)
    central=(np.abs(pos[...,0]-eye[0])<=.12*height)&((eye[1]-pos[...,1])<=.17*height)
    retained=early_labels!=5
    guard=retained & ((protected>0)|central)
    if not np.array_equal(after[guard],original[guard]):raise ValueError('End-to-end observed feature protection failed')
    if not np.array_equal(baseline[retained],original[retained]):raise ValueError('Bald stage changed retained UV')
    if not np.array_equal(confidence[retained],inherited_conf[retained]):raise ValueError('Observed confidence increased/changed')
    if not np.isin(labels,[0,1,2,5,6,7,8]).all():raise ValueError('Unexpected provenance label')
    if not (confidence[np.isin(labels,[5,6,8])]==13).all():raise ValueError('Inferred confidence mismatch')
    mesh=trimesh.load(source/'head.glb',force='mesh',process=False)
    if not np.array_equal(mesh.vertices[mesh.faces].astype(np.float32),v[f].astype(np.float32)):
        raise ValueError('Final geometry differs from source RDF')
    info=inspect_glb(source/'head.glb')
    if not info['embedded_base_color_texture'] or info['external_resources']:raise ValueError('GLB texture packaging failed')
    if tone['variants']['mild']['boundary_before']!=tone['variants']['mild']['boundary_after']:
        raise ValueError('Tone stage changed established boundary metric')
    final=output/'final';final.mkdir(exist_ok=False)
    for src,dst in [(source/'head.glb','head.glb'),(output/'01_bald/head-gray.glb','head-gray.glb'),
                    (source/'head-atlas.png','appearance-atlas.png'),(source/'provenance-labels.png','labels.png'),
                    (source/'confidence.png','confidence.png')]:shutil.copyfile(src,final/dst)
    mesh.visual.material.baseColorTexture.save(final/'embedded-texture.png')
    review=output/'review';review.mkdir(exist_ok=False)
    preview_scene(job/'rdf_uv/final/face_textured.glb',None,review/'before.png')
    preview_scene(final/'head.glb',None,review/'after.png')
    preview_scene(final/'head-gray.glb',None,review/'gray.png',shade_head=True)
    preview_scene(final/'head.glb',None,review/'top.png',elevation=55)
    preview_scene(final/'head.glb',None,review/'under-chin.png',elevation=-35)
    evidence=dict(status='COMBINED_HEAD_VERIFIED',policy=POLICY,selected_variant='mild',
                  scope='Existing RDF and UV reused; no fresh image reconstruction, web API or physical print test',
                  original_retained_observed_features_byte_equal=True,geometry_float32_byte_equal=True,
                  provenance_and_confidence_verified=True,glb=info,
                  observed_tone_max_rgb_delta=tone['variants']['mild']['max_rgb_channel_delta'],
                  transition_boundary_before=transition['boundary_before'],transition_boundary_after=transition['boundary_after'],
                  topology=first['topology'],hair_included=False,glasses_processed=False,
                  lighting_removed=False,likeness_verified=False,ready_for_machine=False,
                  final_sha256={p.name:sha(p) for p in final.iterdir()},
                  stage_report_sha256={name:sha(output/name/'report.json') for name,_,_,_ in STAGES})
    write(output/'verification.json',evidence);return evidence


def run(job,output,python):
    job,output,python=Path(job).resolve(),Path(output).resolve(),Path(python).resolve()
    if not python.is_file():raise ValueError('Worker Python missing')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',policy=POLICY,job=str(job),stages=[],
                fresh_reconstruction_run=False,web_api_tested=False,ready_for_machine=False,
                code_sha256={p.name:sha(p) for p in [Path(__file__),*(ROOT/'tools/headapi'/(module.rsplit('.',1)[1]+'.py') for _,module,_,_ in STAGES)]})
    try:
        previous=job
        for name,module,status,policy in STAGES:
            stage=dict(name=name,available_before_bytes=available_ram(),peak_process_memory_measured=False)
            report['stages'].append(stage);started=time.monotonic()
            args=[str(python),'-m',module,'--job' if name=='01_bald' else '--source',str(previous),'--output',str(output/name)]
            with (output/(name+'.log')).open('x',encoding='utf-8') as log:
                result=run_guarded(args,cwd=ROOT,env=os.environ.copy(),stdout=log,timeout=240,minimum=2*2**30,abort=768*2**20)
            stage.update(elapsed_seconds=time.monotonic()-started,exit_code=result.returncode)
            if result.returncode:raise RuntimeError('Worker failed: '+name)
            validate_stage(json.loads((output/name/'report.json').read_text()),status,policy)
            stage['status']='VERIFIED';previous=output/name
            write(output/'report.json',report)
        with (output/'04_verify.log').open('x',encoding='utf-8') as log:
            result=run_guarded([str(python),'-m','tools.headapi.bald_combined_trial','--verify','--job',str(job),'--output',str(output)],
                               cwd=ROOT,env=os.environ.copy(),stdout=log,timeout=240,minimum=2*2**30,abort=768*2**20)
        if result.returncode:raise RuntimeError('Final verification failed')
        evidence=json.loads((output/'verification.json').read_text())
        if evidence.get('status')!='COMBINED_HEAD_VERIFIED':raise ValueError('Missing final verification')
        for filename,digest in evidence['final_sha256'].items():
            if sha(output/'final'/filename)!=digest:raise ValueError('Final artifact changed')
        report.update(status='COMBINED_HEAD_REVIEW_READY',verification_sha256=sha(output/'verification.json'),
                      final_head_sha256=sha(output/'final/head.glb'))
    except Exception as exc:
        report.update(status='FAILED',error=str(exc));raise
    finally:write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--python',type=Path,default=ROOT/'.venv-triposr/Scripts/python.exe');p.add_argument('--verify',action='store_true')
    a=p.parse_args();verify(a.job,a.output) if a.verify else run(a.job,a.output,a.python)
