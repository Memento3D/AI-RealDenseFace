"""Train-only achromatic exposure study; fixed EV bounds and cached masks."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tools.headapi.face_texture_trial import ROOT, sha, apply_gain
from tools.headapi.texture_overlap_trial import metrics


def linear(rgb):
    x=np.asarray(rgb,dtype=float)/255.
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def fit_exposure(reference, sample):
    """Robust median log2 luminance ratio; return EV and retained count."""
    reference=np.asarray(reference,dtype=float);sample=np.asarray(sample,dtype=float)
    valid=np.isfinite(reference).all(1)&np.isfinite(sample).all(1)
    valid &= ((reference>20)&(reference<240)&(sample>20)&(sample<240)).all(1)
    if valid.sum()<32:return 0.,0
    weights=np.array([.2126,.7152,.0722])
    ratios=np.log2((linear(reference[valid])@weights)/(linear(sample[valid])@weights))
    center=np.median(ratios);mad=np.median(np.abs(ratios-center))
    keep=np.abs(ratios-center)<=max(3*1.4826*mad,.05)
    if keep.sum()<32:return 0.,int(keep.sum())
    return float(np.median(ratios[keep])),int(keep.sum())


def clipping(rgb,gain):
    return float(np.mean(np.any(linear(rgb)*gain>1,axis=-1)))


def run():
    source=ROOT/'runs/experiments/input04-texture-skin-01'
    parent=ROOT/'runs/experiments/input04-texture-boundary-01/report.json'
    prior=json.loads(parent.read_text());pins=json.loads((source/'report.json').read_text())
    output=ROOT/'runs/experiments/input04-texture-exposure-01';output.mkdir(parents=True,exist_ok=False)
    hashes={str(parent):sha(parent),str(source/'report.json'):sha(source/'report.json')}
    report={'status':'EXPOSURE_DIAGNOSTIC_ONLY','geometry_changed':False,'atlas_exported':False,
            'primary_ev_limit':.5,'sensitivity_ev_limits':[.25,1.],
            'fit':'median log2 ratio of linear-sRGB luminance; MAD rejection; train only',
            'review_gate':'primary median and P90 below previous per-channel global on both fixed holdouts; source candidate skin clipping <=1%; not a calibrated quality gate',
            'input_sha256':hashes,'code_sha256':sha(Path(__file__)),
            'dependency_sha256':{str(p):sha(p) for p in [ROOT/'tools/headapi/face_texture_trial.py',ROOT/'tools/headapi/texture_overlap_trial.py']},
            'limitations':['Repeated holdout on same capture is exploratory, not unseen validation','One scalar preserves linear RGB ratios until clipping, not perceptual skin color','No shading removal, ground-truth albedo or registration validation','Candidate skin masks exclude many facial features; full photo clipping is reported separately'],
            'views':[]}
    for slot in ['left','right']:
        path=source/(slot+'.samples.npz')
        if sha(path)!=prior['input_sha256'][str(path)]:raise ValueError('Sample checksum changed')
        hashes[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as s:
            a=s['reference'];b=s['sample'];hold=s['holdout'];old=s['global_color']
        ev,count=fit_exposure(a[~hold],b[~hold])
        if count<32:raise ValueError('Insufficient exposure training samples')
        photo=ROOT/'runs/experiments/input04-validation-01/rdf_crops'/(slot+'.png')
        maskpath=source/(slot+'.skin-candidates.png')
        for path in [photo,maskpath]:
            if sha(path)!=pins['input_sha256'][str(path)]:raise ValueError('Photo/mask pin changed')
            hashes[str(path)]=sha(path)
        rgb=np.asarray(Image.open(photo).convert('RGB'));mask=np.asarray(Image.open(maskpath))>0
        row={'view':slot,'train_count':int((~hold).sum()),'robust_samples':count,'unbounded_train_ev':ev,'holdout_count':int(hold.sum()),'previous_global':metrics(a[hold],old[hold]),'policies':[]}
        for limit in [.25,.5,1.]:
            applied=float(np.clip(ev,-limit,limit));gain=2**applied;color=apply_gain(b,gain)
            item={'ev_limit':limit,'applied_ev':applied,'gain':gain,'bound_hit':abs(ev)>limit,'fixed_skin_holdout':metrics(a[hold],color[hold]),'holdout_clipped_pixel_fraction':clipping(b[hold],gain),'source_skin_clipped_pixel_fraction':clipping(rgb[mask],gain),'source_full_photo_clipped_pixel_fraction':clipping(rgb,gain)}
            row['policies'].append(item)
        primary=row['policies'][1];new=primary['fixed_skin_holdout'];old_metrics=row['previous_global']
        row['primary_review_gate']=new['median_delta_e76']<old_metrics['median_delta_e76'] and new['p90_delta_e76']<old_metrics['p90_delta_e76'] and primary['source_skin_clipped_pixel_fraction']<=.01
        corrected=np.clip(apply_gain(rgb,primary['gain']),0,255).astype(np.uint8)
        Image.fromarray(corrected).save(output/(slot+'.exposure-review.png'))
        sheet=Image.new('RGB',(720,390),'#20252c');draw=ImageDraw.Draw(sheet)
        for i,im in enumerate([Image.fromarray(rgb),Image.fromarray(corrected)]):sheet.paste(im.resize((360,360)),(i*360,30))
        draw.text((8,8),slot+' | original / scalar exposure %.3f EV | photo review only'%primary['applied_ev'],fill='white')
        sheet.save(output/(slot+'.comparison.png'))
        report['views'].append(row)
    report['primary_review_gate_both_views']=all(v['primary_review_gate'] for v in report['views'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['views'],indent=2),flush=True)


if __name__=='__main__':run()
