"""Frontal photographic color projection, not albedo recovery or full-head texturing."""
import base64
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from memento3d_head.adapters.smooth_preview import vertex_normals
from memento3d_head.domain.contracts import ValidationError

POLICY={'version':2,'mask_erosion_px':4,'minimum_facing_cosine':.4,
        'visibility_tolerance_mm':.75,'visibility_samples_per_face':4,'calibrated':False}


def depth_buffer(camera_vertices,faces,k,size):
    w,h=size;z=camera_vertices[:,2]
    xy=(camera_vertices@k.T)[:,:2]/z[:,None]
    depth=np.full((h,w),np.inf)
    for face in faces:
        p=xy[face];low=np.maximum(np.floor(p.min(0)).astype(int),0);high=np.minimum(np.ceil(p.max(0)).astype(int),[w-1,h-1])
        if np.any(high<low):continue
        a,b,c=p;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-9:continue
        yy,xx=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1];xx=xx+.5;yy=yy+.5
        wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;wc=1-wa-wb
        inside=(wa>=-1e-8)&(wb>=-1e-8)&(wc>=-1e-8)
        inverse=wa/z[face[0]]+wb/z[face[1]]+wc/z[face[2]]
        values=np.where(inside,1/np.maximum(inverse,1e-12),np.inf)
        section=depth[low[1]:high[1]+1,low[0]:high[0]+1];np.minimum(section,values,out=section)
    return xy,depth


def face_hull_mask(landmarks,size,erosion=4):
    # A conservative image-space region, not semantic skin/hair segmentation.
    w,h=size;p=np.asarray(landmarks,float)
    if p.ndim!=2 or p.shape[0]<10 or p.shape[1]<2 or not np.isfinite(p).all():
        raise ValidationError('INVALID_LANDMARKS','Expected finite normalized face landmarks')
    xy=p[:,:2]*[w,h];ordered=sorted(set(map(tuple,xy.tolist())))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for point in ordered:
        while len(lower)>=2 and cross(lower[-2],lower[-1],point)<=0:lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper)>=2 and cross(upper[-2],upper[-1],point)<=0:upper.pop()
        upper.append(point)
    polygon=lower[:-1]+upper[:-1]
    if len(polygon)<3:raise ValidationError('INVALID_LANDMARKS','Degenerate face hull')
    image=Image.new('L',(w,h));ImageDraw.Draw(image).polygon(polygon,fill=255)
    if erosion:image=image.filter(ImageFilter.MinFilter(erosion*2+1))
    return np.asarray(image)>0


def project_photo(vertices,faces,view,size,landmarks):
    import cv2
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int);w,h=size
    r=np.asarray(view['rotation_vector'],float);t=np.asarray(view['translation_mm'],float);k=np.asarray(view['camera_matrix'],float)
    camera=vertices@cv2.Rodrigues(r)[0].T+t
    if not np.isfinite(camera).all() or np.any(camera[:,2]<=0):raise ValidationError('INVALID_DEPTH','Mesh crosses camera plane')
    xy,depth=depth_buffer(camera,faces,k,size);mask=face_hull_mask(landmarks,size,POLICY['mask_erosion_px'])
    triangles=camera[faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    to_camera=-triangles.mean(1)
    facing=np.einsum('ij,ij->i',normals,to_camera)/np.maximum(np.linalg.norm(normals,axis=1)*np.linalg.norm(to_camera,axis=1),1e-12)
    covered=facing>=POLICY['minimum_facing_cosine'];reasons={'facing_rejected':int((~covered).sum()),'mask_or_frame_rejected':0,'depth_rejected':0}
    inverse_k=np.linalg.inv(k)
    bary=np.array([[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]])
    for index,tri in enumerate(triangles):
        if not covered[index]:continue
        points=np.vstack((tri,bary@tri));pix=(points@k.T)[:,:2]/points[:,2,None]
        ij=np.floor(pix).astype(int)
        if (np.any(ij<0) or np.any(ij[:,0]>=w) or np.any(ij[:,1]>=h)
            or not mask[ij[:,1],ij[:,0]].all()):
            covered[index]=False;reasons['mask_or_frame_rejected']+=1;continue
        samples=points[3:];sample_ij=ij[3:];visible_depth=depth[sample_ij[:,1],sample_ij[:,0]]
        # Compare both depths on the SAME pixel-center ray. Comparing the original
        # subpixel sample depth to floor(pixel)'s depth wrongly rejects sloped faces.
        rays=np.column_stack((sample_ij[:,0]+.5,sample_ij[:,1]+.5,np.ones(len(sample_ij))))@inverse_k.T
        plane_normal=normals[index];denominator=rays@plane_normal
        expected_depth=np.divide(np.dot(plane_normal,tri[0])*rays[:,2],denominator,
                                 out=np.full(len(rays),np.inf),where=np.abs(denominator)>1e-12)
        if not np.isfinite(visible_depth).all() or np.any(np.abs(expected_depth-visible_depth)>POLICY['visibility_tolerance_mm']):
            covered[index]=False;reasons['depth_rejected']+=1
    # Homogeneous source-photo coordinates for perspective-correct projective lookup.
    coordinates=np.column_stack((xy[:,0]/w*camera[:,2],(1-xy[:,1]/h)*camera[:,2],camera[:,2]))
    info={'policy':POLICY,'covered_triangles':int(covered.sum()),'total_triangles':len(faces),
          'covered_fraction_of_entire_mesh':float(covered.mean()),'rejections':reasons,
          'mask_type':'eroded_convex_hull_of_registered_photo_face_landmarks',
          'skin_segmentation_performed':False,'visibility':'full_mesh_pixel_center_ray_plane_depth_four_interior_samples',
          'lighting_removed':False,'unobserved_regions_filled':False,'likeness_verified':False,
          'warning':'Finite visibility samples and approximate camera cannot guarantee no bleed; photo includes shadows and any occlusions'}
    return coordinates,covered,info


def export_photo_obj(destination,vertices,faces,coordinates,covered):
    obj=destination/'head.photo.obj';mtl=destination/'head.photo.mtl'
    mtl.write_text('newmtl unknown\nKd 0.65 0.68 0.72\n\nnewmtl photo\nKd 1 1 1\nmap_Kd front.photo.png\n',encoding='utf-8')
    with obj.open('w',encoding='utf-8') as stream:
        stream.write('mtllib head.photo.mtl\n')
        for v in vertices:stream.write('v %.8f %.8f %.8f\n'%tuple(v))
        for p in coordinates:stream.write('vt %.8f %.8f\n'%(p[0]/p[2],p[1]/p[2]))
        previous=None
        for face,painted in zip(faces,covered):
            material='photo' if painted else 'unknown'
            if material!=previous:stream.write('usemtl '+material+'\n');previous=material
            stream.write('f '+' '.join('%d/%d'%(i+1,i+1) for i in face)+'\n')


def write_photo_viewer(path,vertices,faces,coordinates,covered,photo,provenance,photo_alpha=None):
    center=(vertices.min(0)+vertices.max(0))*.5;radius=float(np.linalg.norm(vertices-center,axis=1).max())
    normals=vertex_normals(vertices,faces).tolist();model={'vertices':vertices.tolist(),'normals':normals}
    payload={'schema_version':1,'units':'millimeters','center_mm':center.tolist(),'radius_mm':radius,
        'faces':faces.tolist(),'baseline':model,'candidate':model,'displacement_mm':[0.]*len(vertices),'max_displacement_mm':0.,
        'photo_coordinates':coordinates.tolist(),'photo_faces':faces[covered].tolist(),
        'photo_alpha':np.ones(len(vertices)).tolist() if photo_alpha is None else np.asarray(photo_alpha,float).tolist(),
        'photo_image':'data:image/png;base64,'+base64.b64encode(photo.read_bytes()).decode('ascii'),
        'provenance':provenance,'geometry_modified':False,'photo_projection_only':True}
    encoded=json.dumps(payload,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=Path(__file__).with_name('photo_preview.html').read_text(encoding='utf-8')
    path.write_text(template.replace('__HEAD_DATA__',encoded),encoding='utf-8')
