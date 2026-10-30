"""Complete CPU postprocess: source-owned 3D hair cap, source face preserved."""
import argparse,json,sys,time,hashlib,shutil,html,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import trimesh
from PIL import Image,ImageDraw
from figurine2_face import load_face,sha
from figurine2_boolean import solid_union
from figurine2_evaluation import mesh_audit
from figurine_geometry import weld
from run_pipeline_v2 import glb,front_surface_guard,hair_assessment,write
from figurine_preview import add_mesh,accessor
from add_graduation_cap import load,save
from figurine21_render import srgb_to_linear,render_fast
from headlab_raster import surface_depth_fast
from run_figurine_v3 import export_print_stl
from scalp_cap import clean_mask,infer_boundary,cap_patch,fit_thickness,close_cap
from native_detail import relief
from back_view import prepare_back
from headlab_support5 import prepare_supports

def preview(path,destination,span):
    size=360;sheet=Image.new('RGB',(size*4,size+24),(23,28,35));draw=ImageDraw.Draw(sheet)
    for i,angle in enumerate([0,65,-65,180]):sheet.paste(render_fast(path,angle,size,([0,0,0],span)),(i*size,24));draw.text((i*size+6,5),str(angle)+' deg',fill='white')
    sheet.save(destination)

def overlay(hair,face,masks,out):
    rows=[]
    for j,(cam,image,target) in enumerate(zip(face['cameras'],face['images'],masks)):
        headdepth=surface_depth_fast(face['head'].vertices,face['head'].faces,cam,image.size)
        hairdepth=surface_depth_fast(hair.vertices,hair.faces,cam,image.size)
        pred=np.isfinite(hairdepth)&((hairdepth<=headdepth+.08*np.linalg.norm(cam[2,:3]))|~np.isfinite(headdepth))
        iou=float((pred&target).sum()/max((pred|target).sum(),1));coverage=float((pred&target).sum()/max(target.sum(),1))
        rgb=np.asarray(image).copy();rgb[target]=(rgb[target]*.5+np.array([30,210,70])*.5).astype('uint8');rgb[pred]=(rgb[pred]*.5+np.array([230,40,210])*.5).astype('uint8')
        Image.fromarray(rgb).save(out/('view_'+str(j)+'.fit.png'));rows.append({'view':j,'mask_iou':iou,'mask_coverage':coverage,'scope':'consistency with recovered source camera/semantic mask, not independent likeness ground truth'})
    return rows

def review(out,r):
    e=html.escape;body='<!doctype html><meta charset="utf-8"><title>Hair on source head</title><style>body{background:#171c24;color:#eee;font:16px system-ui;margin:24px}img{max-width:100%}a{color:#9cd5ff}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Tóc dựng trên đầu nguồn — không ghép mảng phẳng</h1><p>Giữ nguyên tọa độ và texture mặt. Tóc dùng sọ làm khung, fit chân tóc và độ phồng theo mask. Native TripoSR chỉ cung cấp relief phía trước có giới hạn.</p>'
    if (out/'appearance.png').exists():body+='<h2>Bản ghép hoàn chỉnh</h2><p><a href="viewer/index.html">Xem 3D tương tác (mở qua launcher)</a></p><img src="appearance.png"><p><a href="head_with_hair.glb">GLB đầu và tóc</a> · <a href="hair_on_head.glb">GLB tóc đúng vị trí đầu</a> · <a href="head_mm.stl">STL đầu–tóc, mm</a></p><h2>Hình học</h2><img src="geometry.png">'
    for j in range(r.get('views',3)):body+='<img width="440" src="view_'+str(j)+'.fit.png">'
    body+='<p>Xanh: mask tóc nguồn. Hồng: tóc 3D chiếu lên ảnh. Vùng sau chưa quan sát được ghi trong report; không coi prior là tái dựng đúng. Chưa kiểm chứng resin, slicer hoặc in vật lý. ready_for_machine=false.</p><pre>'+e(json.dumps(r,indent=2))+'</pre>'
    (out/'review.html').write_text(body,encoding='utf-8')

def run(bench,workspace,model,out,profile,back=None):
    if out.exists():raise ValueError('Output exists; refuse overwrite')
    subject_dir=bench/'subject' if (bench/'subject/subject.json').exists() else bench
    subject=json.loads((subject_dir/'subject.json').read_text(encoding='utf-8-sig'));source=subject_dir/'original_face.glb'
    if sha(workspace)!=subject['workspace_sha256'] or sha(source)!=subject['face_sha256']:raise ValueError('Workspace or face SHA differs from benchmark subject')
    out.mkdir(parents=True);shared=out/'shared';shared.mkdir();start=time.perf_counter()
    report={'status':'RUNNING','ready_for_machine':False,'physical_print_verified':False,'views':3,'workspace_sha256':sha(workspace),'source_face_sha256':sha(source)}
    write(out/'profile.used.json',profile)
    try:
        face=load_face(workspace,model,profile,shared)
        original_doc,original_blob=load(source);original_positions=accessor(original_doc,original_blob,0)
        def texture_hashes(d,b):
            values=[]
            for image in d.get('images',[]):
                view=d['bufferViews'][image['bufferView']];start=view.get('byteOffset',0);values.append(hashlib.sha256(b[start:start+view['byteLength']]).hexdigest())
            return values
        original_textures=texture_hashes(original_doc,original_blob)
        if original_positions.shape!=face['points'].shape or np.max(abs(original_positions*1000-face['points']))>2e-5:raise ValueError('Original face does not match the recovered workspace head scale')
        if back is not None:
            image,labels,camera,back_report=prepare_back(back,model,face)
            face['images'].append(image);face['labels'].append(labels);face['cameras'].append(camera);face['masks'].append(labels==17)
            report['back_view']=dict(back_report,source_sha256=sha(back));Image.fromarray(labels).save(shared/'view_3.labels.png')
        else:report['back_view']={'used':False,'reason':'No posterior photo supplied; posterior roots use bounded prior, never a side-view sliver'}
        masks=[];cleaning=[]
        for i,(im,label) in enumerate(zip(face['images'],face['labels'])):
            mask,c=clean_mask(im,label);masks.append(mask);cleaning.append(c);Image.fromarray(mask.astype('uint8')*255).save(shared/('view_'+str(i)+'.hair_mask.png'))
        report['views']=len(masks);report['mask_cleaning']=cleaning
        field=infer_boundary(face,masks,profile['scaffold']);patch,ds,radius,boundary,fraction=cap_patch(face,field,profile['scaffold'])
        offsets,volume=fit_thickness(face,masks,patch,ds,radius,fraction,profile['volume'])
        native=bench/'triposr/native.glb'
        if native.exists():detail,detail_report=relief(native,subject,masks[0],patch.vertices,face['cameras'][0],field['origin'],ds,fraction,profile['detail_amplitude_mm']);offsets=np.maximum(offsets+detail,profile['volume']['edge_mm'])
        else:detail_report={'used':False,'reason':'No native mesh present; scaffold remains valid'}
        hair=close_cap(patch,ds,offsets,boundary,profile['inset_mm']);report.update(hair_boundary=field['report'],hair_volume_fit=volume,native_detail=detail_report)
        face['masks']=masks;attachment,_=hair_assessment(hair,face['head'],face,profile)
        if not attachment['attachment_and_front_proxy_pass']:raise ValueError('Source face/contact guard failed: '+str(attachment))
        report['attachment']=attachment
        theta=np.deg2rad(np.repeat([0,90,180,270],2));phi=np.tile(np.deg2rad([20,40]),4)
        rays=np.c_[np.sin(phi)*np.sin(theta),np.cos(phi),np.sin(phi)*np.cos(theta)]
        _,ids,_=hair.ray.intersects_location(np.tile(field['origin'],(8,1)),rays,multiple_hits=False)
        report['scalp_wrap_screen']={'required_rays':8,'hit_rays':len(np.unique(ids)),'directions':'front, both sides and back at 20/40 degrees from crown','pass':len(np.unique(ids))==8,'scope':'sparse radial coverage, not exact thickness certificate'}
        if not report['scalp_wrap_screen']['pass']:raise ValueError('Hair failed full-head coverage screen')
        eyes=[weld(face['points'][np.repeat(face['components']==i,3)]) for i in [1,2]]
        solid=solid_union([face['head'],hair]+eyes);report['face_surface_guard']=front_surface_guard(solid,face['head'],face)
        base=float(face['head'].bounds[0,1]-.1);export_print_stl(solid,out/'head_mm.stl',base);export_print_stl(hair,out/'hair_mm.stl',base)
        audit=mesh_audit(trimesh.load(out/'head_mm.stl',force='mesh',process=True),'mm Z up')
        if not audit['closed_surface_manifold']:raise ValueError('Final STL topology gate failed')
        report['geometry']=audit;report['hair_geometry']=mesh_audit(hair,'mm Y up')
        inside=__import__('scipy').ndimage.binary_erosion(masks[0],iterations=3);rgb=np.asarray(face['images'][0])[inside];hair_rgb=np.clip(np.median(rgb,axis=0)*.75,15,110).astype(int)
        colors=np.tile(np.r_[srgb_to_linear(hair_rgb/255),1.],(len(hair.vertices),1))
        add_mesh(original_doc,original_blob,'ScalpOwnedHair',hair,colors);save(out/'head_with_hair.glb',original_doc,original_blob)
        outdoc,outblob=load(out/'head_with_hair.glb')
        if not np.array_equal(accessor(outdoc,outblob,0),original_positions):raise ValueError('Source face vertex buffer changed')
        if texture_hashes(outdoc,outblob)[:len(original_textures)]!=original_textures:raise ValueError('Source face texture changed')
        report['face_texture_bytes_identical']=True
        shutil.copytree(ROOT/'viewer/hair-on-head',out/'viewer')
        report['face_vertices_byte_identical']=True;report['face_texture_recolored']=False;report['hair_rgb_design']=hair_rgb.tolist()
        glb(out/'hair_on_head.glb',[('ScalpOwnedHair',hair,np.r_[hair_rgb/255,1])]);glb(out/'geometry.glb',[('PrintShape',solid,[.7,.7,.7,1])])
        span=profile['head_neck_mm']*1.4;preview(out/'head_with_hair.glb',out/'appearance.png',span);preview(out/'geometry.glb',out/'geometry.png',span)
        report['photo_consistency']=overlay(hair,face,masks,out)
        report['photo_fit_gate_pass']=report['photo_consistency'][0]['mask_iou']>=.55 and all(x['mask_iou']>=(.45 if x['view']==3 else .35) for x in report['photo_consistency'][1:])
        if not report['photo_fit_gate_pass']:raise ValueError('Photo-consistency gate failed; inspect overlays, not a printable success')
        if profile['support_preparation']['enabled']:
            try:report['experimental_supports']=prepare_supports(out/'head_mm.stl',out/'experimental_supports',profile['support_preparation'])
            except Exception as error:report['support_preparation_error']=str(error)
        if sha(source)!=subject['face_sha256']:raise ValueError('Input face changed')
        report['status']='FITTED_ON_SOURCE_HEAD';report['source_face_input_unchanged']=True
    except Exception as error:
        report['status']='REJECTED';report['error']=str(error);(out/'traceback.txt').write_text(traceback.format_exc())
    report['seconds']=time.perf_counter()-start;write(out/'report.json',report);review(out,report)
    write(out/'checksums.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='checksums.json'})
    print(json.dumps({'status':report['status'],'seconds':report['seconds'],'output':str(out),'error':report.get('error')},indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bench',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--parser-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--back-image',type=Path);p.add_argument('--profile',type=Path,default=ROOT/'configs/hair-on-head.json');a=p.parse_args()
    profile=json.loads(a.profile.read_text(encoding='utf-8-sig'));r=run(a.bench.resolve(),a.workspace,a.parser_model,a.output,profile,a.back_image);return 0 if r['status']=='FITTED_ON_SOURCE_HEAD' else 2
if __name__=='__main__':raise SystemExit(main())
