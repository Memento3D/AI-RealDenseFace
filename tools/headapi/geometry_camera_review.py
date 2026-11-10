"""Batch texture export and consolidated gray/photo/landmark review."""
import argparse,csv,html,json,shutil
from pathlib import Path
import numpy as np
import cv2,trimesh
from PIL import Image,ImageDraw
from tools.headapi.face_texture_trial import ROOT,sha
from tools.headapi.hair_parser import HairParser,PARSER_SHA
from tools.headapi.texture_completion_trial import supervised


def gray_overlay(vertices,faces,camera,photo,destination):
    image=Image.open(photo).convert('RGB');size=512;scale=size/image.width
    out=np.asarray(image.resize((size,int(image.height*scale)))).copy();depth=np.full(out.shape[:2],np.inf);rendered=np.zeros_like(out)
    k,r,c,*_=cv2.decomposeProjectionMatrix(camera);center=(c[:3]/c[3]).reshape(3)
    v=(vertices-center)@r.T;h=np.c_[vertices,np.ones(len(vertices))]@camera.T;xy=h[:,:2]/h[:,2,None]*scale
    normals=trimesh.Trimesh(vertices,faces,process=False).vertex_normals@r.T
    light=np.array([-.3,-.4,-1.]);light/=np.linalg.norm(light);gray=65+150*np.maximum(normals@light,0)
    for face in faces:
        tri=xy[face];lo=np.maximum(np.floor(tri.min(0)).astype(int),0);hi=np.minimum(np.ceil(tri.max(0)).astype(int),[out.shape[1]-1,out.shape[0]-1])
        if (hi<lo).any():continue
        a,b,c=tri;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-10:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
        b0=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den;b1=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;b2=1-b0-b1
        z=1/np.maximum(b0/v[face[0],2]+b1/v[face[1],2]+b2/v[face[2],2],1e-12)
        section=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1];keep=(b0>=0)&(b1>=0)&(b2>=0)&(z<section)
        color=b0*gray[face[0]]+b1*gray[face[1]]+b2*gray[face[2]];patch=rendered[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        patch[keep]=np.clip(color[keep,None],0,255).astype(np.uint8);section[keep]=z[keep]
    visible=np.isfinite(depth);out[visible]=np.clip(.45*out[visible]+.55*rendered[visible],0,255).astype(np.uint8)
    Image.fromarray(out).save(destination)


def run(root):
    root=Path(root).resolve();experiment=json.loads((root/'fit/experiment.json').read_text());guarded=json.loads((root/'guarded-camera.json').read_text())
    if json.loads((root/'fit-stage.json').read_text())['status']!='COMPLETED':raise ValueError('Refinement did not complete')
    stages=[];cards=[];csv_rows=[];parser=None
    adaptive_path=ROOT/'runs/experiments/input04-texture-review-01/fit.json';adaptive=json.loads(adaptive_path.read_text());fixed_gains=adaptive['variants'][0]['gains']
    for case,guard in zip(experiment['cases'],guarded['cases']):
        name=case['case'];source=Path(case['source']);folder=root/'fit'/name;variants=case['variants']+[guard]
        cache=root/(name+'-cache');cache.mkdir();hashes={str(adaptive_path):sha(adaptive_path)} if name=='input04' else {}
        # Labels are image-dependent, not geometry-dependent; bind all candidate bundles explicitly.
        for row in variants:
            bundle=folder/row['name']/'projection_bundle.npz'
            if sha(bundle)!=row['bundle_sha256']:raise ValueError('Candidate bundle changed')
            hashes[str(bundle)]=sha(bundle)
        for slot in ['front','left','right']:
            photo=source/(slot+'.png');hashes[str(photo)]=sha(photo)
            label=cache/(slot+'.labels.png')
            if name=='input04':
                original=ROOT/'runs/experiments/input04-texture-skin-01'/(slot+'.labels.png')
                pins=json.loads((original.parent/'report.json').read_text())['input_sha256']
                if sha(original)!=pins[str(original)] or sha(photo)!=pins[str(photo)]:raise ValueError('Cached labels/photo changed')
                shutil.copyfile(original,label);hashes[str(original)]=sha(original)
            else:
                if parser is None:parser=HairParser(Path('D:/AI-cache/face-parsing/resnet18.onnx'))
                Image.fromarray(parser(Image.open(photo))).save(label)
            hashes[str(label)]=sha(label)
        cache_report=cache/'report.json';cache_report.write_text(json.dumps({'input_sha256':hashes,'parser_sha256':PARSER_SHA,'scope':'reused or pinned-parser image labels; independent of candidate geometry'},indent=2),encoding='utf-8')
        correction=root/(name+'.correction.json');correction.write_text(json.dumps({'input_sha256':hashes,'variants':[{'name':'frozen','gains':fixed_gains if name=='input04' else [[1.,1.,1.]]*3}],'scope':'Frozen baseline gains for controlled geometry/camera comparison; not refitted or calibrated on candidates'},indent=2),encoding='utf-8')
        for variant in ['camera_guarded','camera_joint']:
            geometry=folder/variant/'projection_bundle.npz';raw=root/(name+'-'+variant+'-raw');final=root/(name+'-'+variant+'-texture')
            kwargs={'output':str(raw),'quality_maps':True,'geometry_path':str(geometry),'source_dir':str(source),'expected_bundle_sha':sha(geometry),'visibility_mode':'continuous_triangle','visibility_tolerance':1e-6,'correction_path':str(correction),'correction_name':'frozen','cached_parser_report':str(cache_report)}
            code='from tools.headapi.face_texture_trial import run; run(**'+repr(kwargs)+')'
            bake=supervised(code,root/(name+'-'+variant+'.bake.log'));entry={'case':name,'variant':variant,'bake':bake}
            if bake['status']=='COMPLETED':
                code='from tools.headapi.texture_completion_trial import finalize; finalize('+repr(str(raw))+','+repr(str(final))+','+repr(str(geometry))+')'
                entry['finish']=supervised(code,root/(name+'-'+variant+'.finish.log'))
            stages.append(entry)
            print(name+' '+variant+' texture '+str(entry),flush=True)
        gray=Image.new('RGB',(1440,414*len(variants)),'#20252c');draw=ImageDraw.Draw(gray)
        overlays=Image.new('RGB',(3*360,390*len(variants)),'#20252c');overlay_draw=ImageDraw.Draw(overlays)
        table=[];links=[]
        for index,row in enumerate(variants):
            variant=row['name'];dest=folder/variant;passed=row['review_screen_passed']
            draw.text((10,index*414+6),variant+' | '+('no regression flagged' if passed else 'REGRESSION FLAGGED - diagnostic only'),fill='white' if passed else 'orange');gray.paste(Image.open(dest/'gray.png'),(0,index*414+30))
            with np.load(dest/'projection_bundle.npz',allow_pickle=False) as b:v=b['view_vertices'];f=b['faces'];p=b['cameras']
            for i,slot in enumerate(['front','left','right']):
                gray_overlay(v[i],f,p[i],source/(slot+'.png'),dest/(slot+'.gray-overlay.png'))
                overlays.paste(Image.open(dest/(slot+'.overlay.png')).resize((360,360)),(i*360,index*390+30))
            overlay_draw.text((8,index*390+6),variant+' / front, left, right | residuals, not ground truth',fill='white')
            errors=[w['holdout_mean_px'] for w in row['views']]
            table.append('<tr><td>'+variant+'</td>'+''.join('<td>%.3f</td>'%value for value in errors)+'<td>'+str(sum(w['visibility_lost'] for w in row['views']))+'</td><td>'+str(passed)+'</td></tr>')
            relative='fit/'+name+'/'+variant
            links.append('<li>'+variant+': <a href="'+relative+'/face_gray.glb">Gray GLB</a> | '+ ' | '.join('<a href="'+relative+'/'+slot+'.gray-overlay.png">'+slot+' gray-on-photo</a>' for slot in ['front','left','right'])+'</li>')
            for view in row['views']:
                for region,metrics in view['regions'].items():csv_rows.append({'case':name,'variant':variant,'view':view['view'],'region':region,**metrics,'visibility_lost':view['visibility_lost'],'screen_pass':passed})
        gray.save(root/(name+'.gray-comparison.png'));overlays.save(root/(name+'.landmark-comparison.png'))
        texture_sheet=Image.new('RGB',(1440,414*3),'#20252c');texture_draw=ImageDraw.Draw(texture_sheet)
        texture_choices=[('Previous strict baseline',ROOT/'runs/experiments/texture-visibility-strict-01'/(name+'-finished')/'face_textured.png')]+[(variant,root/(name+'-'+variant+'-texture')/'face_textured.png') for variant in ['camera_guarded','camera_joint']]
        texture_links=[]
        for i,(label,path) in enumerate(texture_choices):
            texture_draw.text((10,i*414+6),label+' | photo-textured comparison; auto frame, not pixel calibration',fill='white')
            if path.exists():texture_sheet.paste(Image.open(path),(0,i*414+30))
            if i and path.exists():texture_links.append('<a href="'+path.parent.name+'/face_textured.glb">'+label+' textured GLB</a>')
        texture_sheet.save(root/(name+'.texture-comparison.png'))
        cards.append('<section><h2>'+name+'</h2><table><tr><th>Variant</th><th>Front holdout px</th><th>Left</th><th>Right</th><th>Visibility lost</th><th>No-regression screen</th></tr>'+''.join(table)+'</table><p>False means regression flagged. Even True is NOT quality/likeness/print acceptance.</p><ul>'+''.join(links)+'</ul><p>'+' | '.join(texture_links)+'</p><h3>Gray: same baseline frame and lighting</h3><img src="'+name+'.gray-comparison.png"><h3>Photo landmarks</h3><img src="'+name+'.landmark-comparison.png"><h3>Texture, frozen gains</h3><img src="'+name+'.texture-comparison.png"></section>')
    (root/'texture-stages.json').write_text(json.dumps(stages,indent=2),encoding='utf-8')
    with (root/'regional-metrics.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><title>Camera / geometry batch review</title><style>body{font:16px system-ui;background:#20252c;color:#eee;margin:24px}a{color:#8cf}img{width:100%;max-width:1440px}td,th{border:1px solid #777;padding:7px}section{border-top:1px solid #777}</style><h1>Camera / geometry / expression — một đợt so sánh</h1><p>Candidate experiments, baseline giữ nguyên. Camera-only không sửa mesh. Shape/expression dùng pinned FLAME decoder, không train weights. Gray có frame cố định theo baseline; texture dùng gain cũ để so sánh có kiểm soát.</p><p>Holdout chỉ tách khỏi tối ưu mới; baseline đã dùng tất cả ảnh/landmarks. Mapping và detector chưa được xác nhận bằng ground truth. Không suy ra likeness, fidelity hay khả năng in từ residual.</p><p>Guarded camera chọn step bằng TRAIN regional errors; holdout chỉ đánh giá. Candidate bị regression vẫn có ảnh/GLB để xem, không tự promote.</p><p><a href="README.md">Báo cáo</a> | <a href="fit/experiment.json">Fit and provenance</a> | <a href="guarded-camera.json">Train-only fallback</a> | <a href="regional-metrics.csv">Regional metrics</a> | <a href="texture-stages.json">Texture stages/resources</a></p>'+''.join(cards)+'</html>'
    with (root/'review.html').open('x',encoding='utf-8') as file:file.write(page)
    print(root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();run(args.root)
