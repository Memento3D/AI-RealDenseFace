"""Real image-to-figurine worker; retains master separately from web transport."""
import argparse,json,os,shutil,sys
from pathlib import Path
from .service import sha,write
from .runtime_budget import run_guarded

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args()
    job=a.job.resolve();config=json.loads((job/'config.snapshot.json').read_text(encoding='utf-8'));paths=config['paths']
    root=Path(config['root']);runtime=Path(paths['runtime']);python=paths['cpu_python'];blender=paths['blender']
    env=os.environ.copy();env.update(PYTHONPATH=str(runtime)+os.pathsep+str(root),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MEMENTO_SKIP_PREVIEWS='1')
    def step(name,args):
        print('STAGE',name,flush=True)
        with (job/(name+'.log')).open('x',encoding='utf-8') as log:
            result=run_guarded([str(x) for x in args],cwd=root,env=env,stdout=log,timeout=900,minimum=3*2**30,abort=1024**3)
        if result.returncode:raise RuntimeError(name+' failed; see '+name+'.log')
    production=job/'production'
    step('01_images',[python,'-X','utf8',runtime/'scripts/run_face_on_glb.py','--glb',runtime/'GLB/men.glb','--photos',job/'inputs','--output',production,'--face-python',python,'--split-python',python,'--model',runtime/'Face/models/face_landmarker.task','--pose-model',runtime/'HeadBodySplit/models/pose_landmarker_full.task','--mica-config',paths['mica_config'],'--blender',blender,'--skip-previews'])
    state=json.loads((production/'run.json').read_text(encoding='utf-8'))
    if state['status']!='needs_visual_review':raise ValueError('Image reconstruction did not export')
    template=production/'template/template.json';transfer=production/'transfer';texture=job/'texture'
    step('02_texel',[python,'-m','tools.headapi.texel_transfer','--runtime',runtime,'--template',template,'--transfer',transfer,'--output',texture,'--tone-limit','12'])
    for name in ('patch_data.npz','candidate_geometry.blend'):shutil.copyfile(transfer/name,texture/name)
    env.update(CHIBI_TEMPLATE_DIR=str(template.parent),CHIBI_OUTPUT_DIR=str(texture))
    def blend(script):return [blender,'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','-t','4','--python',runtime/'scripts'/script]
    step('03_finalize',blend('finalize_face_patch.py'))
    step('04_textures',[python,runtime/'scripts/compare_protected_textures.py','--source',runtime/'GLB/men.glb','--result',texture/'face_transferred.glb','--report',texture/'protected_textures_report.json'])
    step('05_validation',blend('validate_face_patch.py'))
    sys.path.insert(0,str(runtime));from chibi_face_transfer.cli import validate_result
    validate_result(json.loads((texture/'validation_report.json').read_text()),sha(runtime/'GLB/men.glb'))
    protected=json.loads((texture/'protected_textures_report.json').read_text())
    if len(protected)!=3 or not all(x['decoded_pixels_identical'] for x in protected.values()):raise ValueError('Master body textures changed')
    step('06_web',[blender,'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','-t','4','--python',root/'tools/headapi/production_web_export.py','--','--source',texture/'candidate.blend','--output',job/'render','--body-ratio',str(config['web_body_ratio'])])
    web=job/'render/scene.glb';report=json.loads((job/'render/report.json').read_text())
    if web.stat().st_size>=50*1024**2 or web.read_bytes()[:4]!=b'glTF':raise ValueError('Web GLB exceeds backend contract or has invalid header')
    if not report['personalized_face_geometry_unchanged'] or not report['personalized_face_uv_unchanged']:raise ValueError('Personalized face changed during web export')
    from .production_audit import audit
    write(job/'face-web-audit.json',audit(texture/'face_transferred.glb',web))
    write(job/'production-report.json',{'policy':config['pipeline_policy'],'status':'EXPORTED_NEEDS_VISUAL_REVIEW','master_artifact':'texture/face_transferred.glb','master_sha256':sha(texture/'face_transferred.glb'),'web_artifact':'render/scene.glb','web_sha256':sha(web),'web_bytes':web.stat().st_size,'display_only':True,'ready_for_machine':False,'likeness_verified':False,'tone_limit_srgb':12})
    print('EXPORTED_NEEDS_VISUAL_REVIEW',flush=True)

if __name__=='__main__':main()
