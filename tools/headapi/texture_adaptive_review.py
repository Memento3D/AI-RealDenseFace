"""Train-only per-view selection plus a single consolidated offline review page."""
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tools.headapi.face_texture_trial import ROOT,sha,apply_gain,robust_skin_gain
from tools.headapi.texture_batch_trial import fit_combined,child_export
from tools.headapi.texture_exposure_trial import clipping
from tools.headapi.texture_overlap_trial import metrics


def run():
    batch=ROOT/'runs/experiments/input04-texture-batch-01';source=ROOT/'runs/experiments/input04-texture-skin-01'
    output=ROOT/'runs/experiments/input04-texture-review-01';output.mkdir(parents=True,exist_ok=False)
    parent=batch/'fit.json';prior=json.loads(parent.read_text());pins=prior['input_sha256']
    bundle=ROOT/'runs/experiments/rdf-flame-parameters-40ab0c23fed043eb875537da057027e8/shape_expression_0_05/projection_bundle.npz'
    if sha(bundle)!=pins[str(bundle)]:raise ValueError('Geometry changed')
    with np.load(bundle,allow_pickle=False) as g:views=g['view_vertices'];cameras=g['cameras']
    report={'status':'EXPLORATORY_TRAIN_SELECTED_REVIEW','baseline_promoted':False,'primary_policy':'balanced or old global, selected per view',
            'selection_rule':'Two inner spatial folds restricted to original train. Choose balanced only if median and P90 improve >=2% in BOTH inner folds; otherwise old bounded per-channel global. Holdout evaluated after selection.',
            'unseen_validation':False,'input_sha256':{**pins,str(parent):sha(parent)},'code_sha256':sha(Path(__file__)),
            'dependency_sha256':{str(p):sha(p) for p in [ROOT/'tools/headapi/texture_batch_trial.py',ROOT/'tools/headapi/face_texture_trial.py']},
            'limitations':['Policy designed after prior exploratory results; not a preregistered unseen benchmark','Same subject/photos and approximate correspondences','No albedo, likeness, texture completeness or printability claim'],
            'variants':[{'name':'adaptive','gains':[[1.,1.,1.]]}],'views':[]}
    for vi,slot in [(1,'left'),(2,'right')]:
        path=source/(slot+'.samples.npz')
        if sha(path)!=pins[str(path)]:raise ValueError('Samples changed')
        with np.load(path,allow_pickle=False) as s:ids=s['vertex_ids'];a=s['reference'];b=s['sample'];hold=s['holdout'];old=s['global_color']
        photo=ROOT/'runs/experiments/input04-validation-01/rdf_crops'/(slot+'.png');maskpath=source/(slot+'.skin-candidates.png')
        for p in [photo,maskpath]:
            if sha(p)!=pins[str(p)]:raise ValueError('Photo/mask changed')
        rgb=np.asarray(Image.open(photo).convert('RGB'));pixels=rgb[np.asarray(Image.open(maskpath))>0]
        h=np.c_[views[0][ids],np.ones(len(ids))]@cameras[0].T;xy=h[:,:2]/h[:,2,None]
        size=np.asarray(Image.open(ROOT/'runs/experiments/input04-validation-01/rdf_crops/front.png').size)
        cells=np.floor(xy/size*8).astype(int);fold=(cells[:,0]+cells[:,1])%3
        if not np.array_equal(hold,fold==0):raise ValueError('Original split mismatch')
        records=[];choose=True
        for k in [1,2]:
            test=(~hold)&(fold==k);train=(~hold)&(fold!=k)
            if min(int(test.sum()),int(train.sum()))<32:
                records.append({'fold':k,'status':'INSUFFICIENT_SAMPLES'});choose=False;continue
            gain,details=fit_combined(a[train],b[train],pixels,.5,.5);control,n=robust_skin_gain(a[train],b[train])
            if n<32:raise ValueError('Control had insufficient robust data')
            control_metrics=metrics(a[test],apply_gain(b[test],control));new=metrics(a[test],apply_gain(b[test],gain))
            passed=all(new[m]<=.98*control_metrics[m] for m in ['median_delta_e76','p90_delta_e76'])
            records.append({'fold':k,'train_count':int(train.sum()),'test_count':int(test.sum()),'control':control_metrics,'balanced':new,'passes_rule':passed});choose &= passed
        if choose:gain,details=fit_combined(a[~hold],b[~hold],pixels,.5,.5)
        else:
            gain,n=robust_skin_gain(a[~hold],b[~hold]);details={'robust_samples':n}
            if n<32:raise ValueError('Insufficient final control data')
        report['variants'][0]['gains'].append(gain.tolist())
        report['views'].append({'view':slot,'selected':'balanced' if choose else 'old_global','inner_train_folds':records,'fit':details,'gain':gain.tolist(),'holdout':metrics(a[hold],apply_gain(b[hold],gain)),'previous_global':metrics(a[hold],old[hold]),'source_skin_clipped_fraction':clipping(pixels,gain)})
    manifest=output/'fit.json';manifest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    export=child_export(output/'adaptive',manifest,'adaptive',source/'report.json')
    baseline=ROOT/'runs/experiments/input04-face-texture-01'
    if export['status']=='EXPORTED':
        cover=np.asarray(Image.open(output/'adaptive/coverage.png'));base=np.asarray(Image.open(baseline/'coverage.png'))
        atlas=np.asarray(Image.open(output/'adaptive/face_atlas.png'));old=np.asarray(Image.open(baseline/'face_atlas.png'))
        export['coverage_identical_to_baseline']=bool(np.array_equal(cover,base))
        export['unobserved_texels_unchanged']=bool(np.array_equal(atlas[cover==0],old[cover==0]))
        export['artifact_sha256']={p.name:sha(p) for p in (output/'adaptive').iterdir() if p.is_file()}
    (output/'export.json').write_text(json.dumps(export,indent=2),encoding='utf-8')
    exports=json.loads((batch/'exports.json').read_text())
    choices=[('baseline',baseline)]+[(r['name'],batch/r['name']) for r in exports if r['status']=='EXPORTED']
    if export['status']=='EXPORTED':choices.append(('adaptive',output/'adaptive'))
    sheet=Image.new('RGB',(1440,414*len(choices)),'#20252c');draw=ImageDraw.Draw(sheet)
    for i,(name,directory) in enumerate(choices):
        draw.text((12,414*i+6),name+' | same geometry; unlit photographic texture',fill='white');sheet.paste(Image.open(directory/'face_textured.png'),(0,414*i+30))
    sheet.save(output/'comparison-all.png')
    def relative(directory):return Path(__import__('os').path.relpath(directory,output)).as_posix()
    cards=''.join('<article><h2>'+html.escape(name)+'</h2><a href="'+relative(d)+'/face_textured.glb">Download GLB</a> | <a href="'+relative(d)+'/face_atlas.png">Atlas</a><p><img loading="lazy" src="'+relative(d)+'/face_textured.png"></p></article>' for name,d in choices)
    rows=[]
    for view in prior['views']:
        rows.append((view['view'],'old_global',view['previous_global']))
        rows += [(view['view'],c['name'],c['holdout']) for c in view['candidates']]
    rows += [(v['view'],'adaptive',v['holdout']) for v in report['views']]
    table=''.join('<tr><td>'+html.escape(s)+'</td><td>'+html.escape(n)+'</td><td>%.3f</td><td>%.3f</td></tr>'%(m['median_delta_e76'],m['p90_delta_e76']) for s,n,m in rows)
    (output/'index.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Texture review</title><style>body{background:#20252c;color:#eee;font:16px system-ui;margin:24px}a{color:#8cf}img{width:100%;max-width:1440px}td,th{padding:8px;border:1px solid #777}article{border-top:1px solid #777;margin-top:24px}</style><h1>So sánh texture — input/04</h1><p>5 bản để review: baseline và các candidate. Geometry giữ nguyên; vùng không quan sát vẫn xám. Chưa xác nhận độ giống, albedo hoặc khả năng in.</p><p>Adaptive chọn phương pháp bằng inner folds trong train. Bảng vẫn là dữ liệu cùng người/ảnh đã dùng nhiều lần; không phải kiểm định độc lập.</p><table><tr><th>View</th><th>Method</th><th>Median DE76</th><th>P90</th></tr>'+table+'</table><p><a href="fit.json">Adaptive fit / provenance</a> | <a href="export.json">Export checks</a> | <a href="../input04-texture-batch-01/fit.json">Batch fit / folds</a> | <a href="comparison-all.png">Ảnh tổng hợp</a></p>'+cards+'</html>',encoding='utf-8')
    print(output,flush=True)


if __name__=='__main__':run()
