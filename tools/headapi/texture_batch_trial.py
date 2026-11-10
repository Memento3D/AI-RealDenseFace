"""Sequential exposure/chroma study and review artifacts; no baseline promotion."""
import csv
import ctypes
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from tools.headapi.face_texture_trial import ROOT, sha, apply_gain, robust_skin_gain
from tools.headapi.texture_exposure_trial import linear, fit_exposure, clipping
from tools.headapi.texture_overlap_trial import metrics


class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','extended']]


def available_ram():
    if sys.platform!='win32':return None
    m=MemoryStatus();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('RAM query failed')
    return int(m.available_physical)


def fit_combined(a,b,skin_pixels,ev_limit,chroma_shrink):
    ev,n=fit_exposure(a,b)
    if n<32:raise ValueError('Insufficient robust exposure samples')
    ratios=np.log(np.maximum(linear(a),1e-6))-np.log(np.maximum(linear(b),1e-6))
    center=np.median(ratios,axis=0);mad=np.median(np.abs(ratios-center),axis=0)
    keep=(np.abs(ratios-center)<=np.maximum(3*1.4826*mad,.03)).all(1)
    if keep.sum()<32:raise ValueError('Insufficient chroma samples')
    chroma=np.median(ratios[keep],axis=0);chroma-=chroma.mean()
    chroma*=chroma_shrink
    # Scale the zero-sum vector as a whole, preserving product of chroma gains = 1.
    chroma*=min(1.,.12/max(float(np.max(np.abs(chroma))),1e-12))
    log_gain=np.clip(ev,-ev_limit,ev_limit)*np.log(2)+chroma
    # A single attenuation along log-gain controls clipping on source candidate skin.
    low,high=0.,1.
    for _ in range(18):
        mid=(low+high)/2
        if clipping(skin_pixels,np.exp(log_gain*mid))<=.01:low=mid
        else:high=mid
    alpha=1. if clipping(skin_pixels,np.exp(log_gain))<=.01 else low
    gain=np.exp(log_gain*alpha)
    return gain,{'train_ev':ev,'robust_exposure_count':n,'robust_chroma_count':int(keep.sum()),'chroma_log_vector':chroma.tolist(),'clip_attenuation':alpha,'source_skin_clipped_fraction':clipping(skin_pixels,gain)}


def child_export(output,manifest,name,cache):
    free=available_ram()
    if free is not None and free<2*2**30:return {'status':'SKIPPED_LOW_RAM','available_bytes':free}
    if shutil.disk_usage(ROOT).free<2*2**30:return {'status':'SKIPPED_LOW_DISK'}
    code='from tools.headapi.face_texture_trial import run; run(output='+repr(str(output))+', correction_path='+repr(str(manifest))+', correction_name='+repr(name)+', cached_parser_report='+repr(str(cache))+')'
    started=time.monotonic();minimum=free;status=None
    with (manifest.parent/(name+'.export.log')).open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u','-c',code],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            ram=available_ram()
            if ram is not None:minimum=min(minimum,ram)
            if ram is not None and ram<768*2**20:status='TERMINATED_LOW_RAM'
            if time.monotonic()-started>300:status='TERMINATED_TIMEOUT'
            if status:process.terminate();process.wait();break
            time.sleep(.5)
    return {'status':status or ('EXPORTED' if process.returncode==0 else 'FAILED'),'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-started,'minimum_host_available_ram_bytes_sampled':minimum,'peak_process_memory_measured':False}


def run():
    output=ROOT/'runs/experiments/input04-texture-batch-01';output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'runs/experiments/input04-texture-skin-01';cache=source/'report.json'
    pins=json.loads(cache.read_text());parent=ROOT/'runs/experiments/input04-texture-boundary-01/report.json';prior=json.loads(parent.read_text())
    policies=[('balanced',.5,.5),('strong_chroma',.5,1.),('wider_exposure',.75,.5)]
    report={'status':'EXPLORATORY_BATCH','baseline_promoted':False,'geometry_changed':False,
            'primary_policy':'balanced','policies':[{'name':n,'ev_limit':e,'chroma_shrink':s,'chroma_log_abs_max':.12,'source_skin_clip_max':.01} for n,e,s in policies],
            'input_sha256':{str(cache):sha(cache),str(parent):sha(parent)},'code_sha256':sha(Path(__file__)),
            'dependency_sha256':{str(p):sha(p) for p in [ROOT/'tools/headapi/face_texture_trial.py',ROOT/'tools/headapi/texture_exposure_trial.py',ROOT/'tools/headapi/texture_overlap_trial.py']},
            'limitations':['Repeated holdout and blocked CV reuse the same person/photos; no unseen validation','Masks/cameras approximate; no albedo, likeness or printability claim','Clipping control covers candidate skin only, not every exported facial feature','All candidates exported for comparison, not automatic acceptance'],
            'views':[],'variants':[{'name':n,'gains':[[1.,1.,1.]]} for n,_,_ in policies]}
    bundle=ROOT/'runs/experiments/rdf-flame-parameters-40ab0c23fed043eb875537da057027e8/shape_expression_0_05/projection_bundle.npz'
    for path,expected in pins['input_sha256'].items():
        # Validate sources, including parser pin for provenance; no parser inference.
        if sha(path)!=expected:raise ValueError('Source changed: '+path)
        report['input_sha256'][path]=expected
    with np.load(bundle,allow_pickle=False) as geometry:
        vertices=geometry['view_vertices'];cameras=geometry['cameras']
    csv_rows=[]
    for vi,slot in [(1,'left'),(2,'right')]:
        path=source/(slot+'.samples.npz')
        if sha(path)!=prior['input_sha256'][str(path)]:raise ValueError('Sample checksum changed')
        report['input_sha256'][str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as s:
            ids=s['vertex_ids'];a=s['reference'];b=s['sample'];hold=s['holdout'];old=s['global_color']
        rgb=np.asarray(Image.open(ROOT/'runs/experiments/input04-validation-01/rdf_crops'/(slot+'.png')).convert('RGB'))
        mask=np.asarray(Image.open(source/(slot+'.skin-candidates.png')))>0
        skin_pixels=rgb[mask]
        # Deterministic bounded subset for clipping search, followed by full-mask check.
        row={'view':slot,'holdout_count':int(hold.sum()),'previous_global':metrics(a[hold],old[hold]),'raw':metrics(a[hold],b[hold]),'candidates':[]}
        h=np.c_[vertices[0][ids],np.ones(len(ids))]@cameras[0].T;front_xy=h[:,:2]/h[:,2,None]
        size=np.asarray(Image.open(ROOT/'runs/experiments/input04-validation-01/rdf_crops/front.png').size)
        grid=np.floor(front_xy/size*8).astype(int);fold=(grid[:,0]+grid[:,1])%3
        for index,(name,ev_limit,shrink) in enumerate(policies):
            gain,fit=fit_combined(a[~hold],b[~hold],skin_pixels,ev_limit,shrink)
            corrected=apply_gain(b,gain);metric=metrics(a[hold],corrected[hold])
            folds=[]
            for k in range(3):
                test=fold==k;train=~test
                if min(int(test.sum()),int(train.sum()))<32:
                    folds.append({'fold':k,'status':'INSUFFICIENT_SAMPLES'});continue
                g,_=fit_combined(a[train],b[train],skin_pixels,ev_limit,shrink)
                control,_=robust_skin_gain(a[train],b[train])
                folds.append({'fold':k,'status':'MEASURED','test_count':int(test.sum()),'control':metrics(a[test],apply_gain(b[test],control)),'combined':metrics(a[test],apply_gain(b[test],g))})
            candidate={'name':name,'gain':gain.tolist(),'fit':fit,'holdout':metric,'blocked_folds':folds,
                       'source_full_photo_clipped_fraction':clipping(rgb,gain),'source_accepted_face_clipped_fraction':clipping(rgb[np.asarray(Image.open(ROOT/'runs/experiments/input04-face-texture-01'/(slot+'.accepted-mask.png')))>0],gain)}
            row['candidates'].append(candidate);report['variants'][index]['gains'].append(gain.tolist())
            csv_rows.append({'view':slot,'candidate':name,**metric,'skin_clipped_fraction':fit['source_skin_clipped_fraction']})
        report['views'].append(row)
    manifest=output/'fit.json';manifest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    exports=[]
    baseline=ROOT/'runs/experiments/input04-face-texture-01'
    base_coverage=np.asarray(Image.open(baseline/'coverage.png'));base_atlas=np.asarray(Image.open(baseline/'face_atlas.png'))
    for name,_,_ in policies:
        destination=output/name;result=child_export(destination,manifest,name,cache);result['name']=name
        if result['status']=='EXPORTED':
            cover=np.asarray(Image.open(destination/'coverage.png'));atlas=np.asarray(Image.open(destination/'face_atlas.png'))
            result['coverage_identical_to_baseline']=bool(np.array_equal(cover,base_coverage))
            result['unobserved_texels_unchanged']=bool(np.array_equal(atlas[cover==0],base_atlas[cover==0]))
            result['changed_observed_texel_fraction']=float(np.mean(np.any(atlas[cover>0]!=base_atlas[cover>0],axis=1)))
            result['artifact_sha256']={p.name:sha(p) for p in destination.iterdir() if p.is_file()}
        exports.append(result)
        if result['status'].startswith('TERMINATED'):break
    (output/'exports.json').write_text(json.dumps(exports,indent=2),encoding='utf-8')
    sheets=[('baseline',baseline/'face_textured.png')]+[(r['name'],output/r['name']/'face_textured.png') for r in exports if r['status']=='EXPORTED']
    sheet=Image.new('RGB',(1440,414*len(sheets)),'#20252c');draw=ImageDraw.Draw(sheet)
    for i,(name,path) in enumerate(sheets):
        draw.text((12,i*414+6),name+' | same geometry; unlit photographic texture',fill='white')
        sheet.paste(Image.open(path).convert('RGB'),(0,i*414+30))
    sheet.save(output/'comparison.png')
    rows=''.join('<tr><td>'+html.escape(str(row['view']))+'</td><td>'+html.escape(str(row['candidate']))+'</td><td>%.3f</td><td>%.3f</td><td>%.2f%%</td></tr>'%(row['median_delta_e76'],row['p90_delta_e76'],100*row['skin_clipped_fraction']) for row in csv_rows)
    links=''.join('<li>'+html.escape(r['name'])+': '+('<a href="'+r['name']+'/face_textured.glb">GLB</a> | <a href="'+r['name']+'/face_atlas.png">atlas</a>' if r['status']=='EXPORTED' else html.escape(r['status']))+'</li>' for r in exports)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Texture batch review</title><style>body{font:16px system-ui;background:#20252c;color:#eee;margin:24px}img{width:100%;max-width:1440px}td,th{padding:8px;border:1px solid #666}a{color:#8cf}</style><h1>Texture batch review</h1><p>Exploratory candidates, same geometry. No baseline promotion or printability claim. Unknown regions remain gray.</p><table><tr><th>View</th><th>Candidate</th><th>Median DE76</th><th>P90</th><th>Skin clipping</th></tr>'+rows+'</table><ul>'+links+'</ul><a href="fit.json">Fit / folds / provenance</a> | <a href="exports.json">Export verification</a><p><img src="comparison.png" alt="Baseline and exported candidate views"></p>',encoding='utf-8')
    print(output,flush=True)


if __name__=='__main__':run()
