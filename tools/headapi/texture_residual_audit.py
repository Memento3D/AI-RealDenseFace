"""Read-only spatial residual review from pinned cached skin samples."""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tools.headapi.face_texture_trial import ROOT, sha


def lab(rgb):
    return cv2.cvtColor((rgb/255.).astype(np.float32)[None],cv2.COLOR_RGB2LAB)[0]


def summary(delta):
    magnitude=np.linalg.norm(delta,axis=1)
    return {'count':len(delta),'median_delta_e76':float(np.median(magnitude)),
            'p90_delta_e76':float(np.percentile(magnitude,90)),
            'median_signed_Lab_reference_minus_sample':np.median(delta,axis=0).tolist()}


def run():
    source=ROOT/'runs/experiments/input04-texture-skin-01'
    output=ROOT/'runs/experiments/input04-texture-residual-01';output.mkdir(parents=True,exist_ok=False)
    previous=json.loads((source/'report.json').read_text())
    bundle=ROOT/'runs/experiments/rdf-flame-parameters-40ab0c23fed043eb875537da057027e8/shape_expression_0_05/projection_bundle.npz'
    if sha(bundle)!=previous['input_sha256'][str(bundle)]:raise ValueError('Geometry pin mismatch')
    with np.load(bundle,allow_pickle=False) as b:views=b['view_vertices'];cameras=b['cameras']
    hashes={str(bundle):sha(bundle),str(source/'report.json'):sha(source/'report.json')}
    result={'status':'SPATIAL_REVIEW_ONLY','geometry_changed':False,'correspondence_accuracy_verified':False,
            'grid':'3x3 image-space skin bounding box, not anatomical regions','views':[],
            'limitations':['Same capture and approximate mesh correspondence','Residual is relative to lit front photo, not ground-truth skin','Cannot distinguish geometry, registration and lighting causes','Charts include spatially correlated samples; no significance claim'],
            'code_sha256':sha(Path(__file__)),'input_sha256':hashes}
    for i,slot in [(1,'left'),(2,'right')]:
        sample=source/(slot+'.samples.npz');hashes[str(sample)]=sha(sample)
        with np.load(sample,allow_pickle=False) as s:
            ids=s['vertex_ids'];a=s['reference'];b=s['sample'];corrected=s['global_color'];hold=s['holdout'];xy=s['xy']
        maskpath=source/(slot+'.skin-candidates.png')
        if sha(maskpath)!=previous['input_sha256'][str(maskpath)]:raise ValueError('Mask pin mismatch')
        mask=np.asarray(Image.open(maskpath));height,width=mask.shape
        yy,xx=np.where(mask>0);lo=np.array([xx.min(),yy.min()]);hi=np.array([xx.max()+1,yy.max()+1])
        grid=np.clip(np.floor((xy*[width,height]-lo)/(hi-lo)*3),0,2).astype(int)
        raw_delta=lab(a)-lab(b);delta=lab(a)-lab(corrected)
        row={'view':slot,'raw_holdout':summary(raw_delta[hold]),'global_holdout':summary(delta[hold]),'cells':[]}
        for gy in range(3):
            for gx in range(3):
                selected=hold&(grid[:,0]==gx)&(grid[:,1]==gy)
                count=int(selected.sum())
                cell={'x':gx,'y':gy,'count':count,'status':'MEASURED' if count>=8 else 'SPARSE'}
                if count>=8:cell.update({'raw':summary(raw_delta[selected]),'global':summary(delta[selected])})
                row['cells'].append(cell)
        photo_paths=[ROOT/'runs/experiments/input04-validation-01/rdf_crops'/(s+'.png') for s in ['front',slot]]
        photos=[];points=[]
        for vi,p in zip([0,i],photo_paths):
            if sha(p)!=previous['input_sha256'][str(p)]:raise ValueError('Photo pin mismatch')
            hashes[str(p)]=sha(p);photos.append(Image.open(p).convert('RGB'))
            homogeneous=np.c_[views[vi][ids],np.ones(len(ids))]@cameras[vi].T
            points.append(homogeneous[:,:2]/homogeneous[:,2,None])
        # Choose separated high-residual held-out points for review only, never fit from them.
        ranked=np.where(hold)[0][np.argsort(np.linalg.norm(delta[hold],axis=1))[::-1]];chosen=[]
        for index in ranked:
            if all(np.linalg.norm(points[1][index]-points[1][other])>35 for other in chosen):chosen.append(int(index))
            if len(chosen)==6:break
        sheet=Image.new('RGB',(720,400),'#20252c');draw=ImageDraw.Draw(sheet)
        for view_index,photo in enumerate(photos):
            sheet.paste(photo.resize((360,360)),(360*view_index,40))
            for n,index in enumerate(chosen):
                q=points[view_index][index]/photo.size*360+[view_index*360,40];x,y=q
                draw.ellipse((x-4,y-4,x+4,y+4),outline='yellow',width=2)
                draw.text((x+5,y-8),str(n+1),fill='yellow')
        draw.text((8,8),'Front / '+slot+' | same mesh vertex IDs; positions are NOT verified',fill='white')
        sheet.save(output/(slot+'.correspondences.png'))
        heat=Image.new('RGB',(660,400),'#20252c');draw=ImageDraw.Draw(heat)
        draw.text((10,8),slot+' | held-out residual: raw -> global | image-space grid',fill='white')
        for cell in row['cells']:
            x=cell['x']*220;y=40+cell['y']*120
            value=cell.get('global',{}).get('median_delta_e76',0)
            color=(min(200,int(value*8)),45,55) if cell['status']=='MEASURED' else (55,55,55)
            draw.rectangle((x+2,y+2,x+218,y+118),fill=color)
            text='n='+str(cell['count'])
            if cell['status']=='MEASURED':text+=' | %.2f -> %.2f'%(cell['raw']['median_delta_e76'],value)
            else:text+=' | sparse'
            draw.text((x+9,y+25),text,fill='white')
        heat.save(output/(slot+'.grid.png'))
        row['review_points']=[{'number':n+1,'vertex_id':int(ids[index]),'reference_px':points[0][index].tolist(),'sample_px':points[1][index].tolist(),'global_delta_e76':float(np.linalg.norm(delta[index]))} for n,index in enumerate(chosen)]
        result['views'].append(row)
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in row.items() if k not in ['cells','review_points']} for row in result['views']],indent=2))


if __name__=='__main__':run()
