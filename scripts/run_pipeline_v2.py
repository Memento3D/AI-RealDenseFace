"""Memento3D Pipeline V2: reusable hair library + personalized head demo.

One default cavity-fit method, one optional affine control. STL is millimeters
Z-up; appearance GLB meters Y-up. Resin color/support calibration is separate.
"""
import argparse,csv,html,json,platform,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import trimesh
from PIL import Image,ImageDraw
from figurine2_face import load_face,pose_head,painted_head,eyes,sha
from figurine2_hair import read_library,fit_hair
from figurine2_repair import repair_hair
from figurine_geometry import weld
from figurine2_boolean import solid_union
from figurine_preview import add_mesh
from add_graduation_cap import save
from headlab_raster import surface_depth_fast
from figurine2_render import render_fast,srgb_to_linear
from figurine2_evaluation import (audit_assets,audit_input,mesh_audit,chord_screen,
    candidate_ledger,evaluation_summary,HUMAN_REVIEW_SCHEMA,vertex_fans)
from headlab_diagnostics5 import diagnose
from headlab_support5 import prepare_supports
from run_figurine_v3 import export_print_stl

def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def validate(c):
    if c.get('schema_version')!=2:raise ValueError('Unsupported Pipeline V2 profile')
    if not np.isfinite(c['head_neck_mm']) or not 15<=c['head_neck_mm']<=35:raise ValueError('Head+neck size must be 15..35 mm')
    if not c['asset_ids'] or len(c['asset_ids'])>12 or len(set(c['asset_ids']))!=len(c['asset_ids']):raise ValueError('Choose 1..12 unique hair assets')
    if not c['expressions'] or not set(c['expressions'])<= {'source','soft_smile'}:raise ValueError('Unsupported expression')
    if c.get('teeth','auto') not in ['auto','hidden','show']:raise ValueError('Unsupported teeth mode')
    for key,bound in [('corner_lift_mm',.25),('cheek_lift_mm',.12),('max_displacement_mm',.4)]:
        if not np.isfinite(c['expression'][key]) or not 0<=c['expression'][key]<=bound:raise ValueError('Invalid expression control')
    e=c['eyes']
    if not .94<=e['size_factor']<=1.02 or not .35<=e['iris_ratio']<=.75:raise ValueError('Eye proportions outside conservative bounds')
    for key in ['gaze_yaw_deg','gaze_pitch_deg']:
        if not np.isfinite(e[key]) or abs(e[key])>15:raise ValueError('Gaze outside bound')
    for key in ['per_eye_yaw_deg','per_eye_pitch_deg']:
        if len(e[key])!=2 or not all(np.isfinite(x) and abs(x)<=10 for x in e[key]):raise ValueError('Invalid per-eye offset')
    colors=[p['rgb'] for p in c['skin_palette']]+list(c['face_colors'].values())+[e['sclera_rgb'],e['iris_rgb']]
    if not colors or any(len(v)!=3 or any(not np.isfinite(x) or not 0<=x<=255 for x in v) for v in colors):raise ValueError('Invalid RGB color')
    for k in ['anchor_overlap_mm','roof_overlap_mm']:
        if not .05<=c['fit'][k]<=.5:raise ValueError('Invalid hair attachment offset')
    if not .5<=c['fit']['cage_bandwidth_mm']<=3:raise ValueError('Invalid cage smoothness')

def glb(path,parts):
    doc={'asset':{'version':'2.0','extras':{'length_unit':'meter','pipeline':'Memento3D V2 demo'}},'scenes':[{'nodes':[]}],'scene':0,'nodes':[],'meshes':[],'materials':[],'accessors':[],'bufferViews':[]};blob=bytearray()
    for name,mesh,color in parts:
        if np.asarray(color).ndim==1:color=np.tile(color,(len(mesh.vertices),1))
        linear=np.asarray(color,float).copy();linear[:,:3]=srgb_to_linear(linear[:,:3])
        add_mesh(doc,blob,name,mesh,linear)
    save(path,doc,blob)

def preview(folder,frame,size):
    for kind in ['appearance','geometry']:
        sheet=Image.new('RGB',(size*4,size+24),(23,28,35));draw=ImageDraw.Draw(sheet)
        for i,yaw in enumerate([0,65,110,180]):
            sheet.paste(render_fast(folder/(kind+'.glb'),yaw,size,frame),(i*size,24))
            draw.text((i*size+8,5),str(yaw)+' deg',fill='white')
        sheet.save(folder/(kind+'.png'))

def hair_assessment(hair,head,face,profile):
    intersection=trimesh.boolean.intersection([head,hair],engine='manifold',check_volume=True)
    overlap=float(abs(intersection.volume))
    # Protect visible facial source points below brows; not a full anatomical classifier.
    p=head.vertices;n=head.vertex_normals;eye_y=float(face['eye_centers'][:,1].mean())
    protected=(n[:,2]>.45)&(p[:,1]<eye_y+.25)&(p[:,1]>face['anchor'][1]-.5)&(abs(p[:,0])<head.extents[0]*.35)
    sample=p[protected][::max(1,int(protected.sum())//500)]
    colliding=hair.contains(sample) if len(sample) else np.empty(0,bool)
    collision=float(colliding.mean()) if len(sample) else 1.
    passed=overlap>=profile['fit']['min_overlap_mm3'] and collision<=profile['fit']['max_front_collision_fraction']
    rows=[]
    for index,(camera,image,mask) in enumerate(zip(face['cameras'],face['images'],face['masks'])):
        size=(384,round(384*image.height/image.width));c=camera.copy();c[0]*=size[0]/image.width;c[1]*=size[1]/image.height
        m=np.asarray(Image.fromarray(np.uint8(mask)*255).resize(size,Image.Resampling.NEAREST))>0
        z=surface_depth_fast(hair.vertices,hair.faces,c,size);h=surface_depth_fast(head.vertices,head.faces,c,size)
        pred=np.isfinite(z)&((z<=h+.02*np.linalg.norm(c[2,:3]))|~np.isfinite(h))
        iou=float((pred&m).sum()/max((pred|m).sum(),1))
        rgb=np.asarray(image.resize(size)).copy();rgb[m]=np.uint8(rgb[m]*.55+np.array([35,210,75])*.45)
        from scipy import ndimage as ndi
        rgb[pred&~ndi.binary_erosion(pred)]=[230,70,210]
        rows.append({'view':['front','left','right'][index],'iou':iou,'overlay':Image.fromarray(rgb)})
    return {'head_hair_overlap_mm3':overlap,'protected_front_sample_count':len(sample),
        'protected_front_collision_fraction':collision,'attachment_and_front_proxy_pass':bool(passed),
        'source_face_fully_preserved_verified':False,'scope':'contact/intersection and source-face sampling; not exact hair clearance or mechanical strength'},rows

def front_surface_guard(solid,head,face):
    p=head.vertices;n=head.vertex_normals;eye_y=face['eye_centers'][:,1].mean()
    keep=(n[:,2]>.5)&(p[:,1]<eye_y+.25)&(p[:,1]>face['anchor'][1]-.8)&(abs(p[:,0])<head.extents[0]*.35)
    for c in face['eye_centers']:
        keep&=np.linalg.norm(p-c,axis=1)>1.35
    points=p[keep][::max(1,int(keep.sum())//160)]
    if not len(points):raise ValueError('No protected facial surface samples')
    _,distance,_=trimesh.proximity.closest_point(solid,points)
    if distance.max()>.02:raise ValueError('Boolean altered protected face beyond .02 mm sampled budget')
    return {'sample_count':len(points),'max_distance_to_posed_source_mm':float(distance.max()),
            'exact_hausdorff_verified':False,'eye_socket_and_lateral_temple_regions_excluded':True}

def html_report(output,report):
    esc=html.escape;cards=[]
    for c in report['candidates']:
        if c['status']=='FAILED':cards.append('<section><h2>'+esc(c['id'])+'</h2><pre>'+esc(c['error'])+'</pre></section>');continue
        folder=c['id']+'/'
        cards.append('<section class="candidate" data-hair="'+esc(c['asset_id'])+'" data-expression="'+esc(c['expression'])+'"><h2>'+esc(c['asset_name'])+' · '+esc(c['expression'])+' · '+esc(c['method'])+'</h2><img src="'+folder+'appearance.png"><details><summary>Hình học in và số đo</summary><img src="'+folder+'geometry.png"><pre>'+esc(json.dumps(c,indent=2))+'</pre></details><p><a href="'+folder+'appearance.glb">GLB màu</a> · <a href="'+folder+'head_mm.stl">STL mm</a> · <a href="'+folder+'hair_fitted_mm.stl">Tóc đã khớp</a></p></section>')
    palette=''.join('<span title="'+esc(str(p['id']))+'" style="display:inline-block;width:34px;height:34px;border-radius:50%;margin:4px;background:rgb('+','.join(map(str,p['rgb']))+')"></span>' for p in report['profile']['skin_palette'])
    title='Memento3D · Pipeline V2'
    text='<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{background:#171c24;color:#e9edf3;font:16px system-ui;max-width:1480px;margin:28px auto;padding:0 20px}section{background:#232d39;padding:18px;border-radius:12px;margin:20px 0}img{max-width:100%;height:auto}a{color:#8bc5ff}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.note{padding:16px;background:#42371d;border-radius:10px}select{padding:8px;margin:8px;background:#273341;color:white;border:1px solid #687689}</style><h1>'+title+'</h1><p>Đầu cá nhân hóa · tóc có thể thay thế trong phần mềm · màu da thống nhất · biểu cảm nhẹ.</p><p class="note">Đây là demo phần đầu. Màu là bản xem trước/hướng dẫn hoàn thiện; Saturn không in các màu này từ GLB. STL và support là đầu ra để đánh giá chế tạo, chưa có resin hiệu chuẩn, kiểm chứng lớp cắt thực hoặc in vật lý. Chưa có thân và đế tượng 8–10 cm.</p>'
    selected=report.get('selected')
    if selected:text+='<h2>Đầu ra được đề xuất</h2><p>'+esc(selected['candidate'])+' — '+esc(selected['reason'])+'</p><img src="selected/appearance.png"><p><a href="selected/head_mm.stl">STL được chọn</a> · <a href="selected/appearance.glb">GLB được chọn</a> · <a href="selected/selection.json">Trạng thái kiểm tra</a></p>'
    if report.get('evaluation'):
        text+='<h2>Trạng thái bằng chứng</h2><p>Hình học đo được, phép đo lấy mẫu và phần chưa kiểm chứng được ghi riêng. Chưa có đánh giá của chuyên gia hoặc kiểm chứng in thực tế.</p><p><a href="evaluation.json">Sổ bằng chứng A–B–C</a> · <a href="input_assessment.json">Ảnh và camera</a> · <a href="human_review.schema.json">Mẫu thu thập đánh giá nghiên cứu</a></p><table><tr><th>Kiểm tra</th><th>Trạng thái</th></tr><tr><td>Xuất STL và đọc lại</td><td>Đo theo từng ứng viên</td></tr><tr><td>Độ dày tối thiểu</td><td>Chưa kiểm chứng; chỉ lấy mẫu tia</td></tr><tr><td>Giống người / hài hòa / cảm xúc</td><td>Chưa có đánh giá độc lập</td></tr><tr><td>Support</td><td>Thử nghiệm voxel; chưa phải lớp cắt máy</td></tr><tr><td>Sẵn sàng chạy máy</td><td>Chưa đạt</td></tr></table><h2>Hai STL nguồn</h2><p>'+''.join('<a href="asset_audits/'+esc(a['asset_id'])+'.json">'+esc(a['asset_id'])+' · số đo và tia kiểm tra</a> &nbsp; ' for a in report['asset_audits'])+'</p>'
    text+='<h2>Màu da nền</h2>'+palette+'<p>'+esc(str(report['face']['skin']))+'</p><h2>So sánh mẫu tóc và biểu cảm</h2><label>Tóc <select id="hair"><option value="">Tất cả</option>'+''.join('<option>'+esc(k)+'</option>' for k in report['profile']['asset_ids'])+'</select></label><label>Biểu cảm <select id="expression"><option value="">Tất cả</option><option>source</option><option>soft_smile</option></select></label>'+''.join(cards)+'<h2>Báo cáo chung</h2><pre>'+esc(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2))+'</pre><script>const hair=document.getElementById("hair"),expression=document.getElementById("expression");for(const e of document.querySelectorAll("select"))e.onchange=()=>{for(const c of document.querySelectorAll(".candidate"))c.hidden=(hair.value&&c.dataset.hair!==hair.value)||(expression.value&&c.dataset.expression!==expression.value)};</script>'
    (output/'review.html').write_text(text,encoding='utf-8')

def run(workspace,model,output,profile,root=ROOT):
    validate(profile)
    if output.exists():raise ValueError('Output already exists; previous results are never overwritten')
    assets=read_library(root/profile['library'],root);selected_assets=[a for a in assets if a['id'] in profile['asset_ids']]
    if len(selected_assets)!=len(profile['asset_ids']):raise ValueError('Unknown hair asset ID')
    for a in selected_assets:
        a['custom'].update(profile.get('hair_custom',{}))
        from figurine2_hair import validate_custom
        validate_custom(a['custom'])
    output.mkdir(parents=True);shared=output/'shared';shared.mkdir();started=time.perf_counter()
    asset_audits=audit_assets(selected_assets,root,output/'asset_audits',profile['printer'])
    write(output/'profile.used.json',profile);face=load_face(workspace,model,profile,shared);write(shared/'face.json',face['report'])
    input_assessment=audit_input(face,profile);write(output/'input_assessment.json',input_assessment)
    if input_assessment['blocking_reasons']:raise ValueError('Input rejected: '+str(input_assessment['blocking_reasons']))
    print('V2 face ready; fitting reusable styles',flush=True)
    report={'version':'Memento3D Pipeline V2.0','status':'RUNNING','platform':platform.platform(),'python':platform.python_version(),
            'profile':profile,'face':face['report'],'input_assessment':input_assessment,
            'asset_audits':[{k:v for k,v in a.items() if k!='thickness_screen'}|{'thickness_screen':{k:v for k,v in a['thickness_screen'].items() if k!='samples'}} for a in asset_audits],
            'assets':selected_assets,'candidates':[],'selected':None,
            'ready_for_machine':False,'physical_print_verified':False,'body_and_figurine_base_generated':False,
            'scope':'head-only asset-assisted demo; existing source head required; software hair replacement, not detachable hardware'}
    eye_parts=eyes(face,profile);eye_solids=[weld(m.vertices[m.faces].reshape(-1,3)) for _,m,_ in eye_parts]
    frame=([0,0,0],profile['head_neck_mm']*1.5)
    for asset in selected_assets:
        for expression in profile['expressions']:
            methods=['cage']
            if profile['compare_affine'] and asset==selected_assets[0] and expression==profile['expressions'][0]:methods=['affine','cage']
            for method in methods:
                key='hair_'+asset['id']+'_'+expression+'_'+method;folder=output/key;folder.mkdir();tick=time.perf_counter();print('V2 candidate:',key,flush=True)
                try:
                    head,expression_report=pose_head(face,expression,profile)
                    hair,fit=fit_hair(asset,root,head,float(face['eye_centers'][:,1].mean()),profile['fit'],method)
                    attachment,views=hair_assessment(hair,head,face,profile)
                    if not attachment['attachment_and_front_proxy_pass']:raise ValueError('Hair attachment/front-face proxy rejected: '+str(attachment))
                    initial_links=vertex_fans(hair)
                    repair={'used':False}
                    if initial_links['nonmanifold_vertex_count']:
                        print('V2 repair pinched hair vertices:',initial_links['nonmanifold_vertex_count'],flush=True)
                        hair,repair=repair_hair(hair,profile['fit'].get('repair_pitch_mm',.10))
                        repair.update(used=True,reason='nonmanifold_vertex_links',input_nonmanifold_vertices=initial_links['nonmanifold_vertex_count'])
                        attachment,views=hair_assessment(hair,head,face,profile)
                        if not attachment['attachment_and_front_proxy_pass']:raise ValueError('Repaired hair failed attachment/front gate')
                    try:solid=solid_union([head,hair]+eye_solids)
                    except ValueError as exc:
                        if 'STL roundtrip topology failed' not in str(exc):raise
                        if repair['used']:raise ValueError('Hair-only repair budget exhausted: '+str(exc))
                        print('V2 bounded hair-only manufacturing repair:',asset['id'],flush=True)
                        hair,repair=repair_hair(hair,profile['fit'].get('repair_pitch_mm',.10));repair['used']=True
                        attachment,views=hair_assessment(hair,head,face,profile)
                        if not attachment['attachment_and_front_proxy_pass']:raise ValueError('Repaired hair failed attachment/front gate')
                        solid=solid_union([head,hair]+eye_solids)
                    # Exact surface boolean; source head is never voxel-remeshed.
                    solid.metadata['surface_processing']['hair_surface']='user STL template, bounded anisotropic/cavity-cage deformation'
                    face_guard=front_surface_guard(solid,head,face)
                    base_y=float(face['head'].bounds[0,1]-.1)
                    export_print_stl(solid,folder/'head_mm.stl',base_y);export_print_stl(hair,folder/'hair_fitted_mm.stl',base_y)
                    painted,colors=painted_head(face,head,profile)
                    hair_rgb=np.asarray(profile.get('hair_rgb',[35,28,24]))/255
                    glb(folder/'appearance.glb',[('Personalized_Head',painted,colors),('Hair_'+asset['id'],hair,np.r_[hair_rgb,1.])]+eye_parts)
                    glb(folder/'geometry.glb',[('Print_Shape',solid,[.7,.7,.7,1.])])
                    for row in views:row.pop('overlay').save(folder/(row['view']+'.fit.png'))
                    checks=diagnose(folder/'head_mm.stl',folder,profile['printer'])
                    if not checks['topology_pass']:raise ValueError('Exported STL failed topology gate')
                    final_surface=mesh_audit(trimesh.load(folder/'head_mm.stl',force='mesh',process=True),'millimeter, Z-up STL')
                    write(folder/'final_surface_audit.json',final_surface)
                    if not final_surface['closed_surface_manifold']:raise ValueError('Exported STL failed vertex-link manifold gate')
                    checks['vertex_link_manifold']=True;checks['euler_characteristic']=final_surface['euler_characteristic']
                    hair_audit={'mesh':mesh_audit(hair,'millimeter, Y-up before STL axis conversion'),
                                'thickness_screen':chord_screen(hair,min(profile['printer']['ray_samples'],128),profile['printer']['trial_feature_mm'])}
                    write(folder/'fitted_hair_audit.json',hair_audit)
                    score=float(1-views[0]['iou']+.35*(1-np.mean([v['iou'] for v in views[1:]]))+attachment['protected_front_collision_fraction']*10)
                    preview(folder,frame,profile['preview_size'])
                    row={'id':key,'status':'EXPORTED','asset_id':asset['id'],'asset_name':asset['name'],'expression':expression,'method':method,
                         'expression_report':expression_report,'fit':fit,'hair_manufacturing_repair':repair,'attachment':attachment,'photo_consistency':views,'proposal_score':score,
                         'geometry':checks,'face_surface_guard':face_guard,'surface_processing':solid.metadata.get('surface_processing'),
                         'fitted_hair_thickness_proxy':{k:v for k,v in hair_audit['thickness_screen'].items() if k!='samples'},
                         'seconds':time.perf_counter()-tick,'ready_for_machine':False,'physical_print_verified':False}
                    row['evaluation']=candidate_ledger(row);write(folder/'candidate.json',row)
                except Exception as exc:
                    row={'id':key,'status':'FAILED','error':str(exc),'ready_for_machine':False}
                    (folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');write(folder/'failure.json',row);print('V2 candidate rejected:',str(exc),flush=True)
                report['candidates'].append(row);write(output/'report.json',report);html_report(output,report)
    eligible=[r for r in report['candidates'] if r['status']=='EXPORTED' and r['method']=='cage' and r['geometry']['topology_pass']]
    preferred=profile.get('preferred_expression','soft_smile');pool=[r for r in eligible if r['expression']==preferred] or eligible
    if pool:
        chosen=min(pool,key=lambda r:r['proposal_score']);dest=output/'selected';dest.mkdir()
        for name in ['head_mm.stl','hair_fitted_mm.stl','appearance.glb','geometry.glb','appearance.png','geometry.png','candidate.json']:
            shutil.copy2(output/chosen['id']/name,dest/name)
        report['selected']={'candidate':chosen['id'],'reason':'lowest silhouette-consistency proposal score among cage fits, preferred expression; user may choose another style',
                            'topology_pass':True,'minimum_thickness_verified':False,'resin_profile_calibrated':False,'slicer_verified':False,'ready_for_machine':False}
        if profile['support_preparation']['enabled']:
            try:report['selected']['experimental_supports']=prepare_supports(dest/'head_mm.stl',dest/'experimental_supports',profile['support_preparation'])
            except Exception as exc:report['selected']['experimental_support_error']=str(exc)
        write(dest/'selection.json',report['selected'])
    support_failed=bool(report.get('selected') and report['selected'].get('experimental_support_error'))
    report['status']='DEMO_EXPORTED' if pool and all(r['status']=='EXPORTED' for r in report['candidates']) and not support_failed else 'PARTIAL_FAILURE' if pool else 'NO_VALID_CANDIDATE'
    report['evaluation']=evaluation_summary(report)
    write(output/'evaluation.json',report['evaluation']);write(output/'human_review.schema.json',HUMAN_REVIEW_SCHEMA)
    report['seconds']=time.perf_counter()-started;write(output/'report.json',report);html_report(output,report)
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['candidate','status','method','front_iou','left_iou','right_iou','overlap_mm3','front_collision_fraction','topology_pass','seconds','ready_for_machine'])
        for r in report['candidates']:
            if r['status']=='FAILED':w.writerow([r['id'],'FAILED']);continue
            w.writerow([r['id'],r['status'],r['method'],*[v['iou'] for v in r['photo_consistency']],r['attachment']['head_hair_overlap_mm3'],r['attachment']['protected_front_collision_fraction'],r['geometry']['topology_pass'],r['seconds'],False])
    write(output/'checksums.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.name!='checksums.json'})
    print(json.dumps({'status':report['status'],'candidates':len(report['candidates']),'selected':report['selected']['candidate'] if report['selected'] else None,'seconds':report['seconds'],'output':str(output),'ready_for_machine':False},indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,default=ROOT/'configs/pipeline-v2-04.json');p.add_argument('--workspace',type=Path);p.add_argument('--parser-model',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--hair-id',nargs='+');p.add_argument('--expression',choices=['source','soft_smile']);p.add_argument('--skin-id');p.add_argument('--head-mm',type=float);p.add_argument('--teeth',choices=['auto','hidden','show'],default='auto');p.add_argument('--quick',action='store_true')
    p.add_argument('--volume',type=float,default=0.);p.add_argument('--fringe-lift-mm',type=float,default=0.);p.add_argument('--nape-lift-mm',type=float,default=0.);p.add_argument('--hair-yaw-deg',type=float,default=0.)
    a=p.parse_args();c=json.loads(a.profile.read_text(encoding='utf-8-sig'))
    if a.hair_id:c['asset_ids']=a.hair_id
    if a.expression:c['expressions']=[a.expression];c['preferred_expression']=a.expression
    if a.skin_id:c['skin_id']=a.skin_id
    if a.head_mm is not None:c['head_neck_mm']=a.head_mm
    c['teeth']=a.teeth
    if a.quick:
        if not a.hair_id:c['asset_ids']=[c['asset_ids'][0]]
        if not a.expression:c['expressions']=['soft_smile']
        c['compare_affine']=False
    c['hair_custom']={'volume':a.volume,'fringe_lift_mm':a.fringe_lift_mm,'nape_lift_mm':a.nape_lift_mm,'yaw_deg':a.hair_yaw_deg}
    report=run(a.workspace or ROOT/c['workspace'],a.parser_model or Path(c['parser_model']),a.output or ROOT/'runs'/('pipeline_v2_'+time.strftime('%Y%m%d_%H%M%S')),c)
    return 0 if report['status']=='DEMO_EXPORTED' else 2
if __name__=='__main__':raise SystemExit(main())
