"""V2.1 correction: source face texture, bright skin reference, compact short hair."""
import argparse,html,json,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
import trimesh
from figurine2_face import load_face,sha
from figurine2_hair import read_library,fit_hair
from figurine2_boolean import solid_union
from figurine2_repair import repair_hair
from figurine2_evaluation import mesh_audit,audit_input,vertex_fans
from figurine21_appearance import source_appearance,assemble_appearance
from figurine21_fit import compact_fit
from figurine21_render import render_fast
from figurine_geometry import weld
from run_pipeline_v2 import validate,write,hair_assessment,front_surface_guard,glb
from run_figurine_v3 import export_print_stl
from headlab_diagnostics5 import diagnose
from headlab_support5 import prepare_supports

def preview(source,destination,frame,size=360):
    sheet=Image.new('RGB',(size*3,size+24),(23,28,35));draw=ImageDraw.Draw(sheet)
    for i,yaw in enumerate([0,65,180]):
        sheet.paste(render_fast(source,yaw,size,frame),(i*size,24));draw.text((i*size+8,5),str(yaw)+' deg',fill='white')
    sheet.save(destination)

def review(out,report):
    esc=html.escape
    text='<!doctype html><meta charset="utf-8"><title>Memento3D V2.1</title><style>body{background:#171c24;color:#eee;font:16px system-ui;margin:24px}img{max-width:100%;height:auto}section{background:#252f3b;padding:16px;margin:18px 0}a{color:#9cd0ff}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>V2.1 · giữ nét nguồn, da sáng tham chiếu, tóc ôm đầu</h1><p>Không vẽ lại mắt/mày/môi, không đổi biểu cảm hoặc hình dáng mặt. Vùng da sáng có hỗ trợ được đánh dấu; không chọn pixel cháy sáng. Mẫu tóc ngắn được giảm khối ngoài theo sọ.</p><h2>Đầu nguồn đã bake</h2><img src="shared/source_reference.png"><h2>Đầu nguồn sau điều chỉnh sáng có giới hạn</h2><img src="shared/bright_reference.png"><h2>Vùng da dùng làm tham chiếu</h2><img width="640" src="shared/bright_skin_patch.png">'
    if report.get('selected'):text+='<h2>Bản đề xuất</h2><img src="selected/appearance.png"><p><a href="selected/appearance.glb">GLB màu</a> · <a href="selected/head_mm.stl">STL</a></p>'
    for c in report['candidates']:
        text+='<section><h2>'+esc(c['id'])+'</h2>'
        if c['status']=='FAILED':text+='<pre>'+esc(c['error'])+'</pre>'
        else:
            text+='<img src="'+esc(c['id'])+'/appearance.png"><details><summary>Hình học, mask và số đo</summary><img src="'+esc(c['id'])+'/geometry.png">'
            for view in ['front','left','right']:
                text+='<img width="360" src="'+esc(c['id'])+'/'+view+'.fit.png">'
            text+='<pre>'+esc(json.dumps(c,indent=2))+'</pre></details>'
        text+='</section>'
    text+='<p>Tham chiếu da là lựa chọn thiết kế từ ảnh chưa hiệu chuẩn. Mẫu tóc vẫn là mẫu thư viện, không phải tóc cá nhân hóa hoàn toàn. Chưa kiểm chứng độ giống độc lập, in vật lý hoặc mã máy. ready_for_machine=false.</p><pre>'+esc(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2))+'</pre>'
    (out/'review.html').write_text(text,encoding='utf-8')

def run(workspace,model,out,profile):
    validate(profile)
    if out.exists():raise ValueError('Output exists; refusing overwrite')
    out.mkdir(parents=True);shared=out/'shared';shared.mkdir();started=time.perf_counter()
    write(out/'profile.used.json',profile)
    face=load_face(workspace,model,profile,shared)
    original_skin=face['report']['skin'].copy()
    native,skin=source_appearance(workspace,face,profile,shared)
    write(shared/'face.json',face['report']);write(shared/'bright_skin.json',skin)
    frame=([0,0,0],profile['head_neck_mm']*1.4)
    for name in ['source_reference','bright_reference']:
        preview(shared/(name+'.glb'),shared/(name+'.png'),frame,profile['preview_size'])
    report={'version':'Memento3D V2.1','status':'RUNNING','face':face['report'],
            'rejected_v20_skin_estimate_for_comparison':original_skin,'input_assessment':audit_input(face,profile),
            'candidates':[],'selected':None,'ready_for_machine':False,'physical_print_verified':False,
            'scope':'short close-fitting template; source face/eye geometry and protected feature texture retained'}
    assets=read_library(ROOT/profile['library'],ROOT)
    assets=[a for a in assets if a['id'] in profile['asset_ids']]
    if len(assets)!=len(profile['asset_ids']):raise ValueError('Unknown asset ID')
    if any(a['id']!='21778' for a in assets):raise ValueError('V2.1 compact prototype is validated only for short asset 21778')
    head=face['head'];eye_solids=[weld(face['points'][np.repeat(face['components']==i,3)]) for i in [1,2]]
    for asset in assets:
        fitted,fit=fit_hair(asset,ROOT,head,float(face['eye_centers'][:,1].mean()),profile['fit'],'cage')
        correspondence={}
        for ratio in profile['compact_fit']['ratios']:
            key='compact_'+asset['id']+'_'+str(ratio).replace('.','p');folder=out/key;folder.mkdir();tick=time.perf_counter();print('V2.1 candidate:',key,flush=True)
            try:
                hair,compact=compact_fit(fitted,head,float(face['eye_centers'][:,1].mean()),ratio,profile['compact_fit'],correspondence)
                attachment,views=hair_assessment(hair,head,face,profile)
                if not attachment['attachment_and_front_proxy_pass']:raise ValueError('Protected face/contact gate failed: '+str(attachment))
                if not compact['crown_proxy_pass']:raise ValueError('Hair crown protrudes beyond close-fit budget')
                if not compact['sampled_contact_proxy_pass']:raise ValueError('Sampled inner-wall gap exceeds close-fit budget')
                repair={'used':False}
                try:solid=solid_union([head,hair]+eye_solids)
                except ValueError:
                    hair,repair=repair_hair(hair,.08);repair['used']=True
                    attachment,views=hair_assessment(hair,head,face,profile)
                    if not attachment['attachment_and_front_proxy_pass']:raise ValueError('Repaired hair failed core-face gate')
                    solid=solid_union([head,hair]+eye_solids)
                guard=front_surface_guard(solid,head,face)
                base_y=float(head.bounds[0,1]-.1);export_print_stl(solid,folder/'head_mm.stl',base_y)
                export_print_stl(hair,folder/'hair_mm.stl',base_y)
                surface=mesh_audit(trimesh.load(folder/'head_mm.stl',force='mesh',process=True),'mm Z-up')
                if not surface['closed_surface_manifold']:raise ValueError('Final STL failed manifold gate')
                write(folder/'surface_audit.json',surface)
                position_error=assemble_appearance(native,folder/'appearance.glb',face,hair,profile.get('hair_rgb',[35,28,24]))
                glb(folder/'geometry.glb',[('PrintShape',solid,[.7,.7,.7,1.])])
                for row in views:row.pop('overlay').save(folder/(row['view']+'.fit.png'))
                checks=diagnose(folder/'head_mm.stl',folder,profile['printer'])
                preview(folder/'appearance.glb',folder/'appearance.png',frame,profile['preview_size'])
                preview(folder/'geometry.glb',folder/'geometry.png',frame,profile['preview_size'])
                # Rank only after geometry/contact/crown guards; this is not a beauty score.
                score=float(1-views[0]['iou']+.25*(2-views[1]['iou']-views[2]['iou'])+.15*compact['crown_added_height_mm'])
                c={'id':key,'status':'EXPORTED','fit':fit,'compact':compact,'repair':repair,'attachment':attachment,
                   'source_face_surface_guard':guard,'source_face_position_export_error_mm':position_error,'photo_consistency':views,
                   'proposal_score':score,'geometry':checks,'seconds':time.perf_counter()-tick,'ready_for_machine':False}
                write(folder/'candidate.json',c)
            except Exception as exc:
                c={'id':key,'status':'FAILED','error':str(exc),'ready_for_machine':False};write(folder/'failure.json',c);(folder/'traceback.txt').write_text(traceback.format_exc())
                print('V2.1 rejected:',str(exc),flush=True)
            report['candidates'].append(c);write(out/'report.json',report);review(out,report)
    valid=[c for c in report['candidates'] if c['status']=='EXPORTED']
    if valid:
        chosen=min(valid,key=lambda c:c['proposal_score']);dest=out/'selected';dest.mkdir()
        for name in ['appearance.glb','geometry.glb','head_mm.stl','hair_mm.stl','appearance.png','geometry.png','candidate.json']:
            shutil.copy2(out/chosen['id']/name,dest/name)
        report['selected']={'candidate':chosen['id'],'reason':'close-fit guards plus silhouette/low-crown proposal score; not independent aesthetic validation','ready_for_machine':False}
        if profile['support_preparation']['enabled']:
            try:report['selected']['experimental_supports']=prepare_supports(dest/'head_mm.stl',dest/'experimental_supports',profile['support_preparation'])
            except Exception as exc:report['selected']['support_error']=str(exc)
        write(dest/'selection.json',report['selected'])
    report['status']=('DEMO_EXPORTED' if len(valid)==len(report['candidates']) else 'PARTIAL_DEMO_EXPORTED') if valid else 'NO_VALID_CANDIDATE'
    report['seconds']=time.perf_counter()-started
    write(out/'report.json',report);review(out,report)
    write(out/'checksums.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='checksums.json'})
    print(json.dumps({'status':report['status'],'selected':report['selected']['candidate'] if valid else None,'seconds':report['seconds'],'output':str(out)},indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,default=ROOT/'configs/pipeline-v21-04.json');p.add_argument('--workspace',type=Path);p.add_argument('--parser-model',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();c=json.loads(a.profile.read_text(encoding='utf-8-sig'))
    r=run(a.workspace or ROOT/c['workspace'],a.parser_model or Path(c['parser_model']),a.output or ROOT/'runs'/('pipeline_v21_'+time.strftime('%Y%m%d_%H%M%S')),c)
    return 0 if r['status'] in ['DEMO_EXPORTED','PARTIAL_DEMO_EXPORTED'] else 2
if __name__=='__main__':raise SystemExit(main())
