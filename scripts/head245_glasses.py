"""Observed front frame bounds + conservative 3D frame design; hidden eyes are design estimates."""
import numpy as np
import trimesh
from PIL import Image
from figurine24_appearance import append_array,append_png
from head24_features import front_observation
from figurine22_hair import ear_anchors

def tube(points,radius,closed=False,sides=10):
    p=np.asarray(points,float)
    if len(p)<3 or not np.isfinite(p).all() or not np.isfinite(radius) or radius<=0:raise ValueError('Invalid frame tube')
    tangent=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0)
    if not closed:tangent[0]=p[1]-p[0];tangent[-1]=p[-1]-p[-2]
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-10)
    ref=np.tile([0.,0.,1.],(len(p),1));parallel=abs(tangent[:,2])>.9;ref[parallel]=[0,1,0]
    a=np.cross(tangent,ref);a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-10);b=np.cross(tangent,a)
    t=np.arange(sides)*2*np.pi/sides;v=(p[:,None,:]+radius*(a[:,None,:]*np.cos(t)[None,:,None]+b[:,None,:]*np.sin(t)[None,:,None])).reshape(-1,3)
    faces=[]
    for i in range(len(p) if closed else len(p)-1):
        j=(i+1)%len(p)
        for k in range(sides):
            q=(k+1)%sides;faces.extend([[i*sides+k,j*sides+k,j*sides+q],[i*sides+k,j*sides+q,i*sides+q]])
    if not closed:
        for k in range(1,sides-1):faces.extend([[0,k+1,k],[(len(p)-1)*sides,(len(p)-1)*sides+k,(len(p)-1)*sides+k+1]])
    mesh=trimesh.Trimesh(v,faces,process=True);mesh.fix_normals()
    if not mesh.is_volume:raise ValueError('Frame tube not closed volume')
    return mesh

def design_eye_material(doc,blob,face,settings):
    """Keep every eye position. Replace occluded photo/reflection with labelled neutral design."""
    reports=[];q=np.asarray(face['points']);side=128;yy,xx=np.mgrid[:side,:side];u=(xx+.5)/side;v=(yy+.5)/side
    r=np.sqrt((u-.5)**2+(v-.5)**2);rgb=np.tile(np.array([218,211,194],np.uint8),(side,side,1));rgb[r<.29]=[73,54,36];rgb[r<.14]=[25,22,20]
    for i,word in [(1,'Negative'),(2,'Positive')]:
        mask=np.repeat(face['components']==i,3);p=q[mask];lo=p.min(0);extent=np.maximum(np.ptp(p,axis=0),1e-6)
        uv=np.zeros((len(q),2));uv[mask]=np.c_[(p[:,0]-lo[0])/extent[0],1-(p[:,1]-lo[1])/extent[1]]
        idx=append_array(doc,blob,uv,'VEC2');tex=append_png(doc,blob,Image.fromarray(rgb));found=0
        for node in doc['nodes']:
            if 'Eye_ModelX'+word in node.get('name',''):
                for primitive in doc['meshes'][node['mesh']]['primitives']:
                    primitive['attributes']['TEXCOORD_0']=idx;m=doc['materials'][primitive['material']];m['pbrMetallicRoughness'].update(baseColorTexture={'index':tex},metallicFactor=0,roughnessFactor=.25);m.setdefault('extensions',{})['KHR_materials_unlit']={};found+=1
        if found!=1:raise ValueError('Unknown eye primitive layout')
        reports.append({'component':i,'method':'neutral stylized iris on unchanged source eye; design estimate under occlusion','geometry_changed':False,'hidden_eye_details_recovered':False,'gaze_estimated':False})
    return reports

def frame_mesh(face,settings):
    if not .12<=settings['radius_mm']<=.35 or not .15<=settings['stand_off_mm']<=1.:raise ValueError('Frame design dimensions outside conservative bounds')
    label=face['labels'][0];observed=label==6
    if not settings['enabled'] or observed.sum()<20:return None,{'status':'NO_OBSERVED_GLASSES'}
    obs=front_observation(face['data'],face['images'][0])
    if obs is None:raise ValueError('Frame requires image-bound front landmarks')
    camera=np.asarray(face['cameras'][0]);A=camera[:,:3];inv=np.linalg.inv(A);origin=-inv@camera[:,3]
    eyecenters=np.asarray(face['eye_centers']);eye_mid=eyecenters.mean(0)
    pixels=np.argwhere(observed)[:,::-1];candidates=[]
    # The detected silhouette includes lenses/reflections. Bounds guide a frame design only.
    p3=np.c_[eyecenters,np.ones(2)]@camera.T;centers=p3[:,:2]/p3[:,2,None];order=np.argsort(centers[:,0]);split=centers[:,0].mean()
    zplane=float(eye_mid[2]+settings['stand_off_mm'])
    from figurine_geometry import weld
    surface=trimesh.util.concatenate([face['head']]+[weld(face['points'][np.repeat(face['components']==i,3)]) for i in [1,2]])
    # Preserve a clean manufactured frame instead of warping the rim over cheeks.
    # Use observed rays only to choose a shared foreground plane for both lenses.
    probe=pixels[::max(1,len(pixels)//256)]
    dirs=np.c_[probe,np.ones(len(probe))]@inv.T;dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    hit,hit_ids,_=surface.ray.intersects_location(np.tile(origin,(len(dirs),1)),dirs,multiple_hits=False)
    if len(hit)<len(probe)*.6:raise ValueError('Insufficient source-surface frame support')
    zplane=float(np.quantile(hit[:,2],.99)+settings['radius_mm']+.06)
    ray_support=[{'rays':len(probe),'source_hits':len(hit),'frame_plane_z_mm':zplane,'method':'shared foreground plane from observed surface-depth proxy; not per-vertex face warping'}]
    def unproject(xy):
        d=np.c_[xy,np.ones(len(xy))]@inv.T;t=(zplane-origin[2])/d[:,2]
        return origin+d*t[:,None]
    loops=[];parts=[];bounds=[]
    for k in order:
        center=centers[k];gap=abs(centers[0,0]-centers[1,0]);half=(pixels[:,0]<split) if center[0]<split else (pixels[:,0]>=split)
        safe=half&(abs(pixels[:,0]-center[0])<gap*.85)&(abs(pixels[:,1]-center[1])<gap*.65)
        if safe.sum()<50:raise ValueError('Insufficient observed per-lens bounds')
        lo,hi=np.quantile(pixels[safe],[.03,.97],axis=0);c=(lo+hi)/2;size=(hi-lo)/2
        if (size<5).any():raise ValueError('Degenerate frame bounds')
        t=np.arange(64)*2*np.pi/64;xy=c+size*np.c_[np.sign(np.cos(t))*abs(np.cos(t))**.62,np.sign(np.sin(t))*abs(np.sin(t))**.62]
        loop=unproject(xy);loops.append(loop);parts.append(tube(loop,settings['radius_mm'],True));bounds.append([*lo.tolist(),*hi.tolist()])
    left,right=loops;li=left[np.argmax(left[:,0])];ri=right[np.argmin(right[:,0])]
    mid=(li+ri)/2;mid[1]+=.15;mid[2]=max(mid[2],float(face['head'].bounds[1,2])+.04)
    # Quadratic bridge joins both lenses and contacts the source nose.
    t=np.linspace(0,1,16)[:,None];bridge=(1-t)**2*li+2*t*(1-t)*mid+t*t*ri;parts.append(tube(bridge,settings['radius_mm']))
    anchors=ear_anchors(face);contacts=[]
    for loop in loops:
        side=-1 if loop[:,0].mean()<0 else 1;start=loop[np.argmax(loop[:,0]*side)];anchor=next(a for a in anchors if a['side']==side)
        end=np.asarray(anchor['center']);end[1]=np.asarray(anchor['high'])[1]-.2;end[0]-=side*.12
        control=(start+end)/2;control[0]=side*max(abs(start[0]),abs(end[0]));control[1]+=.15
        arm=(1-t)**2*start+2*t*(1-t)*control+t*t*end;parts.append(tube(arm,settings['radius_mm']));contacts.append(end.tolist())
    mesh=trimesh.util.concatenate(parts)
    report={'status':'EXPERIMENTAL_FRAME_DESIGN','method':'front semantic lens bounds, shared face-depth plane, regular rounded tubes, source-ear anchored temples; no optical lens surface',
        'bounds_front_px':bounds,'source_surface_rays':ray_support,'tube_diameter_mm':2*settings['radius_mm'],'stand_off_mm':settings['stand_off_mm'],'ear_contact_design_points_mm':contacts,
        'hidden_eyes':'neutral design estimate; not recovered anatomy or gaze','source_face_geometry_modified':False,'lens_reflections_copied':False,
        'minimum_feature_calibrated':False,'physical_print_verified':False}
    return mesh,report

def visible_ear_coverage(hair,face,limit=.10):
    """Absolute observed-ear sample coverage, independent of the starting hair template."""
    p=face['points'];weights=np.asarray(face['data']['weights']).reshape(-1,3);ear=np.zeros(len(p),bool)
    for j,camera in enumerate(face['cameras']):
        q=np.c_[p,np.ones(len(p))]@camera.T;xy=q[:,:2]/q[:,2,None];x,y=np.floor(xy.T).astype(int);h,w=face['labels'][j].shape
        valid=(q[:,2]>0)&(x>=0)&(x<w)&(y>=0)&(y<h)&(weights[:,j]>.05)&face['head_corners']
        ids=np.flatnonzero(valid);ear[ids[np.isin(face['labels'][j][y[ids],x[ids]],[7,8])]]=True
    rows=[]
    for side in [-1,1]:
        samples=np.unique(p[ear&(p[:,0]*side>0)],axis=0)
        if len(samples)<20:raise ValueError('Insufficient absolute ear surface support')
        fraction=float(hair.contains(samples).mean());rows.append({'side':side,'unique_samples':len(samples),'covered_fraction':fraction})
    return {'pass':all(r['covered_fraction']<=limit for r in rows),'limit':limit,'sides':rows,'scope':'visible semantic source-ear samples, not exact clearance or mechanical certification'}
