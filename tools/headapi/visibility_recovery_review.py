"""Audit recovered observations with an independent unbinned 3D ray test."""
import html
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import trimesh
from tools.headapi.face_texture_trial import ROOT,sha,preview


def ray_depth(vertices,faces,direction):
    """Moller-Trumbore all triangles; ray direction z=1 makes parameter = depth."""
    tri=vertices[faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h)
    inv=np.divide(1,det,out=np.zeros_like(det),where=np.abs(det)>1e-12)
    s=-tri[:,0];u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1)
    v=q@direction*inv;t=np.einsum('ij,ij->i',e2,q)*inv
    valid=(np.abs(det)>1e-12)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>0)
    return float(t[valid].min()) if valid.any() else float('inf')


def run(output_root=None):
    directory=Path(output_root).resolve() if output_root else ROOT/'runs/experiments/texture-visibility-01'
    stages=json.loads((directory/'stages.json').read_text());summaries=[];cards=[]
    policy=ROOT/'tools/headapi/continuous_visibility.py'
    for stage in stages:
        name=stage['case'];raw=directory/(name+'-raw');final=directory/(name+'-finished')
        old_raw=ROOT/'runs/experiments/texture-completion-01'/(name+'-raw')
        old_final=ROOT/'runs/experiments/texture-completion-02'/(name+'-finished')
        new_report=json.loads((raw/'report.json').read_text());old_report=json.loads((old_raw/'report.json').read_text())
        bundle=next(Path(p) for p in new_report['input_sha256'] if p.endswith('projection_bundle.npz'))
        if sha(bundle)!=new_report['input_sha256'][str(bundle)]:raise ValueError('Geometry changed')
        for slot in ['front','left','right']:
            if sha(raw/(slot+'.accepted-mask.png'))!=sha(old_raw/(slot+'.accepted-mask.png')):raise ValueError('Mask changed between policies')
        if new_report['color_harmonization']!=old_report['color_harmonization']:raise ValueError('Color policy differs')
        old=np.asarray(Image.open(old_raw/'coverage.png'))>0;owners=np.asarray(Image.open(raw/'coverage.png'))//80;new=owners>0
        recovered=new&~old;lost=old&~new
        with np.load(raw/'quality_maps.npz',allow_pickle=False) as q:maps={k:q[k] for k in q.files}
        with np.load(bundle,allow_pickle=False) as b:vertices=b['vertices'];faces=b['faces'];views=b['view_vertices'];cameras=b['cameras']
        assets=ROOT/'vendor/RealDenseFace/weights/flame/flame_assets.npz'
        if sha(assets)!=new_report['input_sha256'][str(assets)]:raise ValueError('UV assets changed')
        with np.load(assets,allow_pickle=False) as b:uv=b['uvs'][b['uv_faces']]
        final_report=json.loads((final/'report.json').read_text())
        mesh=trimesh.load(final/'face_textured.glb',force='mesh',process=False)
        if not np.array_equal(mesh.vertices[mesh.faces].astype(np.float32),vertices[faces].astype(np.float32)):raise ValueError('Geometry export changed')
        for file,digest in final_report['artifact_sha256'].items():
            if sha(final/file)!=digest:raise ValueError('Final artifact changed')
        counts=np.bincount(maps['rejection_reason'][maps['interior']>0],minlength=6)
        row={'case':name,'old_observed_texels':int(old.sum()),'new_observed_texels':int(new.sum()),'recovered_texels':int(recovered.sum()),'lost_texels':int(lost.sum()),'net_observation_gain':int(new.sum()-old.sum()),
             'new_rejection_counts':counts.tolist(),'reason_labels':new_report['rejection_reason_labels'],
             'recovered_skin_texels':int((recovered&(maps['skin']>0)).sum()),'recovered_feature_texels':int((recovered&(maps['protected']>0)).sum()),
             'same_masks_and_color_policy':True,'geometry_roundtrip_equal':True,'source_bundle_sha256':sha(bundle),'continuous_policy_sha256':sha(policy),'brute_ray_samples':[]}
        tolerance=new_report.get('visibility_tolerance_native',.001);row['visibility_tolerance_native']=tolerance
        # Spread selected points across source ownership and UV grid; deterministic and bounded.
        chosen=[]
        for source in [1,2,3]:
            for y in range(0,2048,128):
                for x in range(0,2048,128):
                    yy,xx=np.where(recovered[y:y+128,x:x+128]&(owners[y:y+128,x:x+128]==source))
                    if len(yy):chosen.append((int(y+yy[0]),int(x+xx[0])))
                    if len(chosen)>=16*source:break
                if len(chosen)>=16*source:break
        camera_data=[]
        for i in range(3):
            k,r,c,*_=cv2.decomposeProjectionMatrix(cameras[i]);k/=k[2,2];center=(c[:3]/c[3]).reshape(3)
            camera_data.append(((views[i]-center)@r.T,k))
        for y,x in chosen[:48]:
            face=int(maps['face_index'][y,x]);tri=uv[face];target=np.array([(x+.5)/2048,1-(y+.5)/2048])
            coeff=np.linalg.solve(np.column_stack((tri[0]-tri[2],tri[1]-tri[2])),target-tri[2]);bary=np.r_[coeff,1-coeff.sum()]
            vi=int(owners[y,x])-1;cam,k=camera_data[vi];point=bary@cam[faces[face]]
            nearest=ray_depth(cam,faces,point/point[2]);error=abs(nearest-point[2])
            row['brute_ray_samples'].append({'uv_pixel':[x,y],'face':face,'view':vi,'point_depth':float(point[2]),'nearest_3d_ray_depth':nearest,'depth_error':float(error),'passes_same_tolerance':bool(error<tolerance)})
        row['independent_ray_test_passed']=all(p['passes_same_tolerance'] for p in row['brute_ray_samples'])
        if not row['independent_ray_test_passed']:raise ValueError('Recovered visibility failed independent ray audit')
        colors=np.array([[90,90,90],[65,160,210],[140,100,60],[155,60,190],[235,160,50],[230,60,60]],np.uint8)
        reason_atlas=colors[maps['rejection_reason']]
        Image.fromarray(reason_atlas).save(directory/(name+'.reasons.png'));preview(vertices,faces,uv,reason_atlas,directory/(name+'.reason-preview.png'))
        changes=np.full((2048,2048,3),90,np.uint8);changes[old&new]=[65,110,160];changes[recovered]=[40,210,100];changes[lost]=[230,70,70]
        Image.fromarray(changes).save(directory/(name+'.observations.png'));preview(vertices,faces,uv,changes,directory/(name+'.observation-preview.png'))
        sheet=Image.new('RGB',(1440,4*414),'#20252c');draw=ImageDraw.Draw(sheet)
        for i,(label,path) in enumerate([('Previous candidate',old_final/'face_textured.png'),('Continuous visibility + finishing',final/'face_textured.png'),('Green recovered / red removed / blue retained',directory/(name+'.observation-preview.png')),('Mask purple / angle orange / depth red / observed blue',directory/(name+'.reason-preview.png'))]):
            draw.text((10,i*414+6),name+' / '+label,fill='white');sheet.paste(Image.open(path),(0,i*414+30))
        sheet.save(directory/(name+'.visibility-comparison.png'))
        summaries.append(row)
        cards.append('<section><h2>'+html.escape(name)+'</h2><p>Recovered '+str(row['recovered_texels'])+' / removed '+str(row['lost_texels'])+' / net '+str(row['net_observation_gain'])+' UV texels. Same geometry/masks/gains. Independent ray samples '+str(len(row['brute_ray_samples']))+' passed.</p><a href="../texture-completion-02/'+name+'-finished/face_textured.glb">Previous GLB</a> | <a href="'+name+'-finished/face_textured.glb">New candidate GLB</a> | <a href="'+name+'.observations.png">Observation changes</a> | <a href="'+name+'.reasons.png">Rejection map</a><p><img src="'+name+'.visibility-comparison.png"></p></section>')
    result={'status':'EXPLORATORY_CONTINUOUS_VISIBILITY_REVIEW','baseline_promoted':False,'cases':summaries,'limitations':['Visibility is relative to existing mesh/cameras, not photo occluder ground truth','Same capture repeatedly used; no independent likeness benchmark','Net texel counts are not anatomical facial coverage','Reason map uses rejection precedence and last-write UV ownership']}
    with (directory/'visibility-audit.json').open('x',encoding='utf-8') as file:json.dump(result,file,indent=2)
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><title>Continuous visibility review</title><style>body{font:16px system-ui;background:#20252c;color:#eee;margin:24px}a{color:#8cf}img{width:100%;max-width:1440px}section{border-top:1px solid #777}</style><h1>Visibility: trước / sau / nguyên nhân thiếu màu</h1><p>Candidate thử nghiệm. Màu mới lấy từ ảnh, không inpaint. Kiểm tra nhìn thấy dựa vào mesh/camera hiện có, không xác nhận độ giống hay khả năng in. Baseline giữ nguyên.</p><p><a href="visibility-audit.json">Audit / sample rays</a> | <a href="README.md">Report</a> | <a href="stages.json">Resources</a></p>'+''.join(cards)+'</html>'
    with (directory/'review.html').open('x',encoding='utf-8') as file:file.write(page)
    print(json.dumps([{k:v for k,v in row.items() if k!='brute_ray_samples'} for row in summaries],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');arguments=parser.parse_args();run(arguments.output)
