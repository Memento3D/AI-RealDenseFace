"""Conservative best-view color selection; no shape fitting or invented pixels."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from memento3d_head.adapters.photo_projection import project_photo, write_photo_viewer
from memento3d_head.adapters.smooth_preview import vertex_normals

SLOTS=('front','left','right')


def select_sources(vertices,faces,views,coverage,protected=None):
    """Keep frontal coverage intact; side photos only supplement outside protection."""
    import cv2
    scores=[]
    for slot,covered in zip(SLOTS,coverage):
        view=views[slot]
        camera=vertices@cv2.Rodrigues(np.asarray(view['rotation_vector'],float))[0].T+view['translation_mm']
        tri=camera[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);direction=-tri.mean(1)
        score=np.einsum('ij,ij->i',normal,direction)/np.maximum(np.linalg.norm(normal,axis=1)*np.linalg.norm(direction,axis=1),1e-12)
        scores.append(np.where(covered,score,-np.inf))
    scores=np.asarray(scores);chosen=np.argmax(scores,axis=0)
    chosen[~np.isfinite(scores).any(axis=0)]=-1
    chosen[np.asarray(coverage[0],bool)]=0
    if protected is not None:
        chosen[np.asarray(protected,bool)&~np.asarray(coverage[0],bool)]=-1
    return chosen



def protected_features(faces,coordinates,landmarks):
    """Image-space feature rectangles, deliberately not a semantic face mask."""
    points=np.asarray(landmarks,float)
    if len(points)<478: return np.ones(len(faces),bool)
    eyes=points[[33,263],:2];mouth=points[[61,291],:2]
    span=abs(eyes[1,0]-eyes[0,0])
    nose=points[1,:2]
    rectangles=[
        ([eyes[:,0].min()-.06*span,min(points[70,1],points[300,1])-.04*span],
         [eyes[:,0].max()+.06*span,eyes[:,1].max()+.13*span]),
        ([nose[0]-.18*span,eyes[:,1].min()],
         [nose[0]+.18*span,nose[1]+.12*span]),
        ([mouth[:,0].min()-.12*span,mouth[:,1].min()-.16*span],
         [mouth[:,0].max()+.12*span,mouth[:,1].max()+.18*span])]
    q=coordinates[faces];xy=q[:,:,:2]/q[:,:,2,None];xy[:,:,1]=1-xy[:,:,1]
    protected=np.zeros(len(faces),bool)
    for low,high in rectangles:
        protected|=np.all(xy.max(1)>=low,axis=1)&np.all(xy.min(1)<=high,axis=1)
    return protected


def balance_side_colors(images,coordinates,coverage,protected):
    """Bounded median RGB offset from shared outer-face samples; no relighting."""
    corrected=[images[0]];diagnostics={}
    arrays=[np.asarray(im,dtype=float) for im in images]
    for index,slot in enumerate(SLOTS[1:],1):
        shared=np.asarray(coverage[0])&np.asarray(coverage[index])&~protected
        samples=[]
        for j in (0,index):
            q=coordinates[j];uv=q[:,:2]/q[:,2,None]
            # Coordinates here have already been reduced to triangle centers.
            x=np.clip((uv[:,0]*images[j].width).astype(int),0,images[j].width-1)
            y=np.clip(((1-uv[:,1])*images[j].height).astype(int),0,images[j].height-1)
            samples.append(arrays[j][y[shared],x[shared]])
        count=int(shared.sum());offset=np.zeros(3)
        if count>=30:offset=np.clip(np.median(samples[0]-samples[1],axis=0),-15.,15.)
        corrected.append(Image.fromarray(np.clip(arrays[index]+offset,0,255).astype('uint8')))
        diagnostics[slot]={'shared_outer_face_samples':count,'rgb_offset':offset.tolist(),'applied':count>=30,'offset_limit':15}
    return corrected,diagnostics

def boundary_alpha(vertex_count,faces,painted,rings=1):
    """One edge ring of gray-to-photo transition; no new texture observations."""
    painted=np.asarray(painted,bool)
    inside=np.zeros(vertex_count,bool);outside=np.zeros(vertex_count,bool)
    inside[faces[painted].ravel()]=True;outside[faces[~painted].ravel()]=True
    boundary=inside&outside
    distance=np.full(vertex_count,rings+1,dtype=int);distance[boundary]=0
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    for _ in range(rings):
        previous=distance.copy()
        np.minimum.at(distance,edges[:,0],previous[edges[:,1]]+1)
        np.minimum.at(distance,edges[:,1],previous[edges[:,0]]+1)
    return np.where(inside,np.minimum(distance/rings,1.),0.)


def build_preview(output,vertices,faces,registration,photos,provenance):
    """Write four offline viewer modes and a reproducible projection archive."""
    if len(faces)*3>65535:raise ValueError("Preview exceeds WebGL1 render-index limit")
    coordinates=[];coverage=[];information={};images=[]
    for slot in SLOTS:
        with Image.open(photos[slot]) as image:images.append(image.convert('RGB').copy())
        assessment=next(p for p in registration['assessment']['photos'] if p['slot']==slot)
        if images[-1].size!=(assessment['oriented_width'],assessment['oriented_height']):
            raise ValueError('Registered photo dimensions differ: '+slot)
        coords,covered,info=project_photo(vertices,faces,registration['views'][slot],images[-1].size,assessment['face']['landmarks'])
        coordinates.append(coords);coverage.append(covered);information[slot]=info
    front_assessment=next(p for p in registration['assessment']['photos'] if p['slot']=='front')
    protected=protected_features(faces,coordinates[0],front_assessment['face']['landmarks'])
    chosen=select_sources(vertices,faces,registration['views'],coverage,protected)
    corrected,balance=balance_side_colors(images,[q[faces].mean(1) for q in coordinates],coverage,protected)
    original_images=images
    images=corrected
    if not (chosen>=0).any():raise ValueError('No triangles passed photo visibility gates')
    widths=[im.width for im in images];height=max(im.height for im in images);width=sum(widths)
    atlas=Image.new('RGB',(width,height));offset=0;atlas_coords=[]
    for image,coords in zip(images,coordinates):
        atlas.paste(image,(offset,0));q=coords.copy()
        q[:,0]=(coords[:,0]*image.width+offset*coords[:,2])/width
        q[:,1]=(coords[:,1]*image.height+(height-image.height)*coords[:,2])/height
        atlas_coords.append(q);offset+=image.width
    atlas.save(output/'photos.atlas.png')
    raw_atlas=Image.new('RGB',(width,height));offset=0
    for image in original_images:raw_atlas.paste(image,(offset,0));offset+=image.width
    raw_atlas.save(output/'photos.raw.atlas.png')
    # Duplicate only render vertices to allow different UVs at source seams.
    # Positions and triangle connectivity in physical space are unchanged.
    expanded=vertices[faces].reshape(-1,3);render_faces=np.arange(len(expanded)).reshape(-1,3)
    normals=np.round(vertex_normals(vertices,faces)[faces].reshape(-1,3),5).tolist()
    pages={}
    for mode in (*SLOTS,'combined'):
        assignment=chosen if mode=='combined' else np.where(coverage[SLOTS.index(mode)],SLOTS.index(mode),-1)
        coords=np.zeros((len(expanded),3));coords[:,2]=1
        for index in range(3):
            selected=np.flatnonzero(assignment==index)
            coords.reshape(-1,3,3)[selected]=atlas_coords[index][faces[selected]]
        alpha_vertices=boundary_alpha(len(vertices),faces,assignment>=0)
        protected_vertices=np.unique(faces[protected])
        alpha_vertices[protected_vertices]=1.
        alpha=alpha_vertices[faces].reshape(-1)
        temp=output/('_'+mode+'.html')
        write_photo_viewer(temp,expanded,render_faces,coords,assignment>=0,output/('photos.atlas.png' if mode=='combined' else 'photos.raw.atlas.png'),provenance,photo_alpha=alpha)
        html=temp.read_text(encoding='utf-8');start=html.index('>',html.index('<script id="model-data"'))+1;end=html.index('</script>',start)
        payload=json.loads(html[start:end]);payload['baseline']['normals']=normals;payload['candidate']['normals']=normals
        payload['photo_coordinates']=np.round(coords,6).tolist()
        html=html[:start]+json.dumps(payload,separators=(',',':')).replace('<','\\u003c')+html[end:]
        html=html.replace('Màu ảnh chính diện','Màu ảnh đã chọn').replace('Màu được chiếu từ ảnh chính diện','Màu được chiếu từ ảnh nguồn đã chọn')
        pages[mode]=html;temp.unlink()
    encoded=json.dumps(pages,separators=(',',':')).replace('<','\\u003c')
    shell='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Memento3D · Màu ba ảnh</title><style>body{margin:0;background:#10151e;color:#eee;font:15px system-ui}nav{padding:12px}select{padding:8px}iframe{width:100%;height:90vh;border:0}</style><nav>Nguồn màu: <select id="mode"><option value="combined">Ghép ba ảnh</option><option value="front">Chính diện</option><option value="left">Trái</option><option value="right">Phải</option></select> · Ưu tiên ảnh chính diện; ảnh bên bổ sung ngoài vùng mắt–mũi–miệng. Cân màu giới hạn; mép màu–xám có chuyển tiếp mềm. Chưa sửa hình học mắt–mũi–miệng. Đổi nguồn sẽ đặt lại góc xem.</nav><iframe id="viewer" title="Xem mesh và màu ảnh" sandbox="allow-scripts"></iframe><script id="pages" type="application/json">__PAGES__</script><script>const pages=JSON.parse(document.getElementById('pages').textContent),mode=document.getElementById('mode'),viewer=document.getElementById('viewer');function show(){viewer.srcdoc=pages[mode.value];}mode.addEventListener('change',show);show();</script></html>'''
    (output/'multiview_preview.html').write_text(shell.replace('__PAGES__',encoded),encoding='utf-8')
    np.savez_compressed(output/'multiview_projection.npz',vertices=vertices,faces=faces,coordinates=np.asarray(coordinates),covered_faces=np.asarray(coverage),chosen_source=chosen)
    return {'views':information,'selected_triangles':{slot:int((chosen==i).sum()) for i,slot in enumerate(SLOTS)},'unknown_triangles':int((chosen<0).sum()),'covered_triangles':int((chosen>=0).sum()),'total_triangles':len(faces),'selection':'front_priority; side_supplement_outside_protected_feature_rectangles', 'protected_triangles':int(protected.sum()),'color_balance':balance,'coverage_boundary_feather':{'method':'one_mesh_edge_ring_to_original_gray_outside_protected_features','rings':1,'fills_unknown_regions':False},'color_blending':False,'seam_color_correction':False,'geometry_modified':False,'cameras_modified':False,'albedo_recovered':False,'unobserved_regions_filled':False,'likeness_verified':False}
