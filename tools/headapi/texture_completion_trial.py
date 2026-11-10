"""Sequential confidence, seam/hole finishing and second-capture review."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
import trimesh
from PIL import Image,ImageDraw
from tools.headapi.face_texture_trial import ROOT,sha,preview
from tools.headapi.texture_batch_trial import available_ram
from tools.headapi.texture_finish import finish_atlas


def supervised(code,logpath,python=None,timeout=300):
    ram=available_ram()
    if ram is not None and ram<2*2**30:return {'status':'SKIPPED_LOW_RAM','available_bytes':ram}
    if shutil.disk_usage(ROOT).free<2*2**30:return {'status':'SKIPPED_LOW_DISK'}
    minimum=ram;started=time.monotonic();status=None
    with logpath.open('x',encoding='utf-8') as log:
        child=subprocess.Popen([str(python or sys.executable),'-u','-c',code],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            ram=available_ram()
            if ram is not None:minimum=min(minimum,ram)
            if ram is not None and ram<768*2**20:status='TERMINATED_LOW_RAM'
            if time.monotonic()-started>timeout:status='TERMINATED_TIMEOUT'
            if status:child.terminate();child.wait();break
            time.sleep(.5)
    return {'status':status or ('COMPLETED' if child.returncode==0 else 'FAILED'),'exit_code':child.returncode,'elapsed_seconds':time.monotonic()-started,'minimum_host_available_ram_bytes_sampled':minimum,'peak_process_memory_measured':False}


def export_texture(vertices,faces,uv,atlas,path):
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(atlas),metallicFactor=0,roughnessFactor=1,doubleSided=True)
    visual=trimesh.visual.texture.TextureVisuals(uv=uv.reshape(-1,2),material=material)
    normals=trimesh.Trimesh(vertices,faces,process=False).vertex_normals[faces].reshape(-1,3)
    mesh=trimesh.Trimesh(vertices[faces].reshape(-1,3),np.arange(faces.size).reshape(-1,3),vertex_normals=normals,visual=visual,process=False)
    mesh.export(path)
    restored=trimesh.load(path,force='mesh',process=False)
    if not np.array_equal(restored.vertices[restored.faces].astype(np.float32),vertices[faces].astype(np.float32)):raise ValueError('Geometry roundtrip failed')


def artificial_holes(atlas,owner,maps):
    safe=(owner>0)&(maps['skin']>0)&(maps['protected']==0)&(maps['confidence']>=.5)
    safe=cv2.erode(safe.astype(np.uint8),np.ones((25,25),np.uint8))>0
    points=[]
    for y in range(0,atlas.shape[0],128):
        for x in range(0,atlas.shape[1],128):
            yy,xx=np.where(safe[y:y+128,x:x+128])
            if len(yy):points.append((int(y+yy[len(yy)//2]),int(x+xx[len(xx)//2])))
            if len(points)>=20:break
        if len(points)>=20:break
    damaged=atlas.copy();owners=owner.copy();removed=np.zeros(owner.shape,bool)
    for y,x in points:removed[y:y+2,x:x+2]=True
    owners[removed]=0;damaged[removed]=180
    if not points:return {'status':'NO_SAFE_TEST_PATCHES','removed_texels':0}
    repaired,_,provenance,_=finish_atlas(damaged,owners,maps['interior'],maps['skin'],maps['protected'],maps['confidence'])
    restored=removed&(provenance==2)
    if not restored.any():return {'status':'NONE_REPAIRED','removed_texels':int(removed.sum())}
    error=np.linalg.norm(repaired[restored].astype(float)-atlas[restored].astype(float),axis=1)
    return {'status':'MEASURED','removed_texels':int(removed.sum()),'repaired_texels':int(restored.sum()),'median_rgb_l2':float(np.median(error)),'p90_rgb_l2':float(np.percentile(error,90)),'scope':'artificial 2x2 holes in already observed smooth skin; not hidden-region ground truth'}


def finalize(raw,output,bundle,assets_path=None):
    raw=Path(raw);output=Path(output);bundle=Path(bundle)
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads((raw/'report.json').read_text())
    if sha(bundle)!=report['input_sha256'][str(bundle)]:raise ValueError('Geometry provenance mismatch')
    atlas=np.asarray(Image.open(raw/'face_atlas.png').convert('RGB'));owner=np.asarray(Image.open(raw/'coverage.png'))//80
    with np.load(raw/'quality_maps.npz',allow_pickle=False) as q:maps={k:q[k] for k in q.files}
    result,confidence,provenance,statistics=finish_atlas(atlas,owner,maps['interior'],maps['skin'],maps['protected'],maps['confidence'])
    with np.load(bundle,allow_pickle=False) as b:vertices=b['vertices'];faces=b['faces']
    assets=Path(assets_path) if assets_path else ROOT/'vendor/RealDenseFace/weights/flame/flame_assets.npz'
    if sha(assets)!=report['input_sha256'][str(assets)]:raise ValueError('UV assets changed')
    with np.load(assets,allow_pickle=False) as b:
        if not np.array_equal(faces,b['faces']):raise ValueError('UV topology mismatch')
        uv=b['uvs'][b['uv_faces']].astype(float)
    Image.fromarray(result).save(output/'face_atlas.png')
    export_texture(vertices,faces,uv,result,output/'face_textured.glb')
    preview(vertices,faces,uv,result,output/'face_textured.png')
    Image.fromarray(np.clip(confidence*255,0,255).astype(np.uint8)).save(output/'confidence.png')
    Image.fromarray(provenance).save(output/'provenance-labels.png')
    palette=np.array([[80,80,80],[65,155,210],[255,160,0]],np.uint8)
    Image.fromarray(palette[provenance]).save(output/'provenance.png')
    heat=cv2.cvtColor(cv2.applyColorMap(np.clip(confidence*255,0,255).astype(np.uint8),cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
    heat[provenance==0]=[100,100,100];heat[provenance==2]=[255,160,0]
    Image.fromarray(heat).save(output/'confidence-color.png');preview(vertices,faces,uv,heat,output/'confidence-preview.png')
    delta=np.max(np.abs(result.astype(np.int16)-atlas.astype(np.int16)),axis=-1)
    Image.fromarray(np.clip(delta*20,0,255).astype(np.uint8)).save(output/'change-map.png')
    validation=artificial_holes(atlas,owner,maps)
    final={'status':'FINISHED_TEXTURE_REVIEW_ONLY','geometry_changed':False,'triangle_coordinates_float32_roundtrip_equal':True,
           'baseline_promoted':False,'manufacturing_mesh':False,'likeness_verified':False,'skin_realism_verified':False,
           'unique_interior_texels':int((maps['interior']>0).sum()),'statistics':statistics,'artificial_hole_check':validation,
           'provenance_labels':{'0':'unsupported or outside UV; unchanged','1':'photo observed, possibly color/seam adjusted','2':'local interpolation, not observed'},
           'confidence':'heuristic facing/mask-border distance/color agreement, not calibrated probability; inferred fill <=0.15',
           'limitations':['Only source-owner seams within UV chart interiors; actual UV chart cuts are preserved','Microholes only; large unknown regions remain gray','Eyes/lips and other non-skin semantics plus 8px border protected','Photographic shadows and geometry errors remain','Artificial-hole test is not independent capture validation'],
           'input_sha256':{str(raw/p):sha(raw/p) for p in ['report.json','face_atlas.png','coverage.png','quality_maps.npz']},
           'code_sha256':sha(Path(__file__)),'finishing_policy_sha256':sha(ROOT/'tools/headapi/texture_finish.py')}
    final['input_sha256'].update({str(bundle):sha(bundle),str(assets):sha(assets)})
    final['artifact_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output/'report.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print(json.dumps(statistics,indent=2),flush=True)


def run(output_root=None,reuse_raw=None,visibility_mode='pixel_depth_legacy',visibility_tolerance=.001):
    output=Path(output_root) if output_root else ROOT/'runs/experiments/texture-completion-01'
    output.mkdir(parents=True,exist_ok=False)
    current=ROOT/'runs/experiments/rdf-flame-parameters-40ab0c23fed043eb875537da057027e8/shape_expression_0_05/projection_bundle.npz'
    old=ROOT/'runs/headapi/9bf00af2-3a4b-40e1-bd08-5706c66ef19d'
    cases=[('input04',current,ROOT/'runs/experiments/input04-validation-01/rdf_crops'),('capture02',old/'rdf/projection_bundle.npz',old/'rdf_crops')]
    stages=[]
    for name,bundle,source in cases:
        raw=(Path(reuse_raw) if reuse_raw else output)/(name+'-raw');final=output/(name+'-finished')
        kwargs={'output':str(raw),'quality_maps':True,'geometry_path':str(bundle),'source_dir':str(source),'expected_bundle_sha':sha(bundle),'visibility_mode':visibility_mode,'visibility_tolerance':visibility_tolerance}
        if name=='input04':kwargs.update(correction_path=str(ROOT/'runs/experiments/input04-texture-review-01/fit.json'),correction_name='adaptive',cached_parser_report=str(ROOT/'runs/experiments/input04-texture-skin-01/report.json'))
        else:
            # Historical crop checksums, not newly fitted camera/shape or imported input04 gains.
            pins=json.loads((old/'rdf/report.json').read_text())['input_sha256']
            for slot in ['front','left','right']:
                photo=source/(slot+'.png')
                if sha(photo)!=pins[str(photo)]:raise ValueError('Second capture crop provenance mismatch')
        code='from tools.headapi.face_texture_trial import run; run(**'+repr(kwargs)+')'
        if reuse_raw:
            raw_report=json.loads((raw/'report.json').read_text())
            if raw_report['input_sha256'][str(bundle)]!=sha(bundle):raise ValueError('Reused raw geometry pin mismatch')
            bake={'status':'COMPLETED','reused':True,'raw_report_sha256':sha(raw/'report.json')}
        else:bake=supervised(code,output/(name+'.bake.log'))
        entry={'case':name,'bundle_sha256':sha(bundle),'source':str(source),'raw':str(raw),'bake':bake}
        if bake['status']=='COMPLETED':
            code='from tools.headapi.texture_completion_trial import finalize; finalize('+repr(str(raw))+','+repr(str(final))+','+repr(str(bundle))+')'
            entry['finish']=supervised(code,output/(name+'.finish.log'))
        stages.append(entry)
        if bake['status'].startswith('TERMINATED'):break
    (output/'stages.json').write_text(json.dumps(stages,indent=2),encoding='utf-8')
    cards=[];summary=[]
    for entry in stages:
        name=entry['case'];raw=Path(entry['raw']);final=output/(name+'-finished')
        if entry.get('finish',{}).get('status')!='COMPLETED':continue
        report=json.loads((final/'report.json').read_text());summary.append({'case':name,**report['statistics'],'artificial_hole_check':report['artificial_hole_check']})
        canvas=Image.new('RGB',(1440,3*414),'#20252c');draw=ImageDraw.Draw(canvas)
        for i,(label,path) in enumerate([('Before',raw/'face_textured.png'),('After',final/'face_textured.png'),('Confidence; gray = unsupported; orange = inferred',final/'confidence-preview.png')]):
            draw.text((10,i*414+6),name+' / '+label,fill='white');canvas.paste(Image.open(path),(0,i*414+30))
        canvas.save(output/(name+'.comparison.png'))
        stats=report['statistics'];before=stats['seam_before'];after=stats['seam_after']
        raw_link=Path(__import__('os').path.relpath(raw,output)).as_posix()
        cards.append('<section><h2>'+html.escape(name)+'</h2><p>Same geometry; '+str(stats['filled_texels'])+' locally inferred texels. RGB seam jump '+str(before['median_rgb_jump'])+' → '+str(after['median_rgb_jump'])+' ('+str(before['count'])+' pairs). Not a likeness score.</p><p><a href="'+raw_link+'/face_textured.glb">Before GLB</a> | <a href="'+name+'-finished/face_textured.glb">After GLB</a> | <a href="'+name+'-finished/face_atlas.png">Atlas</a> | <a href="'+name+'-finished/provenance.png">Provenance</a> | <a href="'+name+'-finished/change-map.png">Changes</a> | <a href="'+name+'-finished/report.json">Report</a></p><img src="'+name+'.comparison.png"></section>')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Texture completion review</title><style>body{font:16px system-ui;background:#20252c;color:#eee;margin:24px}a{color:#8cf}img{width:100%;max-width:1440px}section{border-top:1px solid #777;padding-top:12px}</style><h1>Texture completion — before / after / confidence</h1><p>Review candidates only. Large unsupported areas remain gray. Orange marks locally interpolated pixels; confidence is heuristic, not probability. Same geometry, no albedo/likeness/printability claim.</p><p>Second capture uses its own existing mesh/cameras and raw photo colors, without input04 gain transfer. This checks finishing behavior, not generalization of the color model.</p><a href="summary.json">Summary</a> | <a href="stages.json">Resources and stages</a>'+''.join(cards)+'</html>',encoding='utf-8')
    print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--reuse-raw');parser.add_argument('--visibility',choices=['pixel_depth_legacy','continuous_triangle'],default='pixel_depth_legacy');parser.add_argument('--visibility-tolerance',type=float,default=.001)
    arguments=parser.parse_args();run(arguments.output,arguments.reuse_raw,arguments.visibility,arguments.visibility_tolerance)
