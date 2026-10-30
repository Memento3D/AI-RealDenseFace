"""V3.1 continuous scalp-fitted hair and photo-guided painted eyes.

Original procedural geometry. Back hair is a bounded design prior, not a
reconstruction from an unseen back camera. No independent hair strands.
"""
import io
import numpy as np
from scipy import ndimage as ndi
import trimesh
from PIL import Image


def ray_distances(mesh, origin, directions):
    """Nearest positive triangle intersection from an interior scalp origin."""
    tri=np.asarray(mesh.triangles);a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a
    s=np.asarray(origin)-a;q=np.cross(s,e1);num=np.einsum('ij,ij->i',e2,q)
    result=[]
    for first in range(0,len(directions),96):
        rays=directions[first:first+96]
        cross=np.cross(rays[:,None,:],e2[None,:,:]);det=np.einsum('tj,rtj->rt',e1,cross)
        inv=np.divide(1.,det,out=np.zeros_like(det),where=abs(det)>1e-10)
        u=np.einsum('tj,rtj->rt',s,cross)*inv
        v=(rays@q.T)*inv;t=num[None,:]*inv
        valid=(abs(det)>1e-10)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>1e-6)
        distances=np.where(valid,t,np.inf).min(1)
        if not np.isfinite(distances).all():raise ValueError('Scalp rays missed the source head; center/topology needs review')
        result.extend(distances)
    return np.asarray(result)


def directions_for(theta, phi):
    return np.stack((np.sin(phi)*np.sin(theta),np.cos(phi),np.sin(phi)*np.cos(theta)),axis=-1)


def cap_mesh(outer,inner):
    """Two radial surfaces joined at a continuous closed hairline, with one pole each."""
    rings,columns=outer.shape[:2]
    vertices=np.vstack((outer[0].mean(0)[None],outer[1:].reshape(-1,3),inner[0].mean(0)[None],inner[1:].reshape(-1,3)))
    count=1+(rings-1)*columns;faces=[]
    for offset,reverse in [(0,False),(count,True)]:
        layer=[]
        for j in range(columns):layer.append([offset,offset+1+j,offset+1+(j+1)%columns])
        for r in range(rings-2):
            for j in range(columns):
                a=offset+1+r*columns+j;b=offset+1+r*columns+(j+1)%columns;c=b+columns;d=a+columns
                layer.extend([[a,d,c],[a,c,b]])
        faces.extend([f[::-1] if reverse else f for f in layer])
    start=1+(rings-2)*columns
    for j in range(columns):
        a=start+j;b=start+(j+1)%columns;c=b+count;d=a+count
        faces.extend([[a,b,c],[a,c,d]])
    mesh=trimesh.Trimesh(vertices,faces,process=False);mesh.fix_normals()
    if not mesh.is_volume:raise ValueError('Continuous hair cap is not a closed positive volume')
    return mesh


def continuous_hair(head,eye_y,cameras,masks,depths,config):
    """Fit a continuous hairline, then add gentle longitudinal lock ridges.

    Masks adjust the boundary only; they never punch through the cap interior.
    Out-of-frame/occluded/unsupported samples do not count as negative evidence.
    """
    nt=128;np_=64
    low,high=head.bounds;height=high[1]-low[1]
    upper=head.vertices[head.vertices[:,1]>eye_y+.075*height]
    center=(upper.min(0)+upper.max(0))/2;center[1]=eye_y+.015*height
    theta=np.linspace(-np.pi,np.pi,nt,endpoint=False);phi=np.linspace(0,np.deg2rad(147),np_)
    directions=directions_for(theta[None,:]+np.zeros((np_,nt)),phi[:,None]+np.zeros((np_,nt)))
    radial=ray_distances(head,center,directions.reshape(-1,3)).reshape(np_,nt)
    smooth=ndi.gaussian_filter(radial,sigma=(1.1,1.1),mode=('nearest','wrap'))
    points=center+directions*radial[:,:,None]
    support=np.zeros((np_,nt),int);observations=np.zeros_like(support)
    for camera,mask,depth in zip(cameras,masks,depths):
        q=np.column_stack((points.reshape(-1,3),np.ones(points.size//3)))@camera.T
        xy=np.floor(q[:,:2]/q[:,2,None]).astype(int)
        valid=(q[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0])
        ids=np.flatnonzero(valid);observed=depth[xy[ids,1],xy[ids,0]]
        tolerance=.35*np.linalg.norm(camera[2,:3])
        visible=np.isfinite(observed)&(q[ids,2]<=observed+tolerance)
        ids=ids[visible]
        observations.reshape(-1)[ids]+=1
        support.reshape(-1)[ids]+=mask[xy[ids,1],xy[ids,0]].astype(int)
    rear=(1-np.cos(theta))/2
    prior=np.deg2rad(81+18*np.sin(theta)**2+49*rear**3)
    boundary=prior.copy();fitted=0
    for j in range(nt):
        positive=(support[:,j]>0)&(phi>np.deg2rad(40))&(phi<np.deg2rad(142))
        # Require a run of supported samples, rejecting isolated parsing noise.
        positive=ndi.binary_opening(positive,structure=np.ones(3))
        if positive.sum()>=4:
            estimate=phi[np.flatnonzero(positive)[-1]]
            weight=.82*(1-.8*rear[j]**2)
            boundary[j]=(1-weight)*prior[j]+weight*np.clip(estimate,prior[j]-.28,prior[j]+.28)
            fitted+=1
    boundary=ndi.gaussian_filter1d(boundary,2.5,mode='wrap')
    # Gentle attached tips break the helmet-like edge without carving the base.
    boundary+=np.deg2rad(1.1)*np.cos(theta*int(config['clumps'])-config['part'])*(1-.85*rear)
    # Boundaries near the ears should remain above the upper side of the face.
    boundary=np.clip(boundary,np.deg2rad(63),np.deg2rad(140))
    progress=np.linspace(0,1,49)[:,None]
    sample_phi=progress*boundary[None,:]
    grid_index=np.vstack(((sample_phi/phi[-1]*(np_-1)).ravel(),np.broadcast_to(np.arange(nt),sample_phi.shape).ravel()))
    source_r=ndi.map_coordinates(radial,grid_index,order=1,mode='nearest').reshape(sample_phi.shape)
    smooth_r=ndi.map_coordinates(smooth,grid_index,order=1,mode='nearest').reshape(sample_phi.shape)
    theta_grid=np.broadcast_to(theta,sample_phi.shape)
    directions=directions_for(theta_grid,sample_phi)
    # Curved longitudinal guides; ridges fade out at crown and boundary.
    phase=theta_grid-config['sweep']*.28*progress**1.6-config['part']*.22
    guides=np.linspace(-np.pi,np.pi,int(config['clumps']),endpoint=False)
    distance=np.abs(np.angle(np.exp(1j*(phase[:,:,None]-guides))))
    ridge=np.exp(-.5*(distance/.16)**2).max(-1)
    envelope=np.sin(np.minimum(sample_phi,np.pi/2))**1.5
    edge=np.clip((1-progress)/.13,0,1);edge=edge**2*(3-2*edge)
    back_fade=1-.75*rear[None,:]
    relief=config['ridge_mm']*ridge*envelope*edge*back_fade
    # Shallow channels separate broad attached locks, with base thickness retained.
    middle=guides+np.pi/int(config['clumps'])
    valley_distance=np.abs(np.angle(np.exp(1j*(phase[:,:,None]-middle))))
    valley=np.exp(-.5*(valley_distance/.07)**2).max(-1)
    groove=config.get('groove_mm',.15)*valley*envelope*edge*back_fade
    outer_r=np.maximum(smooth_r+config['base_mm']+relief-groove,source_r+.12)
    inner_r=source_r-config['anchor_mm']
    outer=center+directions*outer_r[:,:,None];inner=center+directions*inner_r[:,:,None]
    mesh=cap_mesh(outer,inner)
    return mesh,{'method':'continuous source-scalp ray fit; masks adjust hairline only; longitudinal curved ridges',
        'independent_strands':False,'closed_cap':True,'hairline_columns':nt,'mask_fitted_columns':fitted,
        'unknown_views':'ignored as negative evidence; back boundary is a smoothed design prior',
        'hidden_regions':'designed completion, not observed reconstruction','nape_continuity':'continuous periodic rings; no voxel mask carving',
        'min_sampled_radial_thickness_mm':float((outer_r-inner_r).min()),'max_added_relief_mm':float(relief.max()),'max_designed_groove_mm':float(groove.max()),
        'parameters':config,'hairline_phi_degrees':np.rad2deg(boundary).tolist(),
        'source_sampling':'ray intersections with closed source head; no voxelized scalp',
        'quality_verified':False}


def photo_eye_direction(image,labels,camera,center,radii):
    """Conservative dark-region candidate inside an eroded semantic eye region.

    This is not an iris detector or a calibrated gaze estimate. Rejection falls
    back to the camera-facing direction; per-eye manual offsets remain explicit.
    """
    rgb=np.asarray(image,float);scale=np.array([rgb.shape[1]/512,rgb.shape[1]/512])
    full=camera.copy();full[0]*=scale[0];full[1]*=scale[1]
    camera_center=-np.linalg.solve(full[:,:3],full[:,3])
    default=(camera_center-center)/radii;default/=np.linalg.norm(default)
    p=full@np.r_[center,1];uv=p[:2]/p[2]
    eye=ndi.binary_erosion(np.isin(labels,[4,5]),iterations=1)
    yy,xx=np.nonzero(eye)
    if not len(xx):return default,{'method':'camera-facing fallback','reason':'no semantic eye region'}
    distances=(xx-uv[0])**2+(yy-uv[1])**2
    samples=center[None,:]+np.vstack((np.diag(radii),-np.diag(radii)))
    projected=np.column_stack((samples,np.ones(6)))@full.T
    pixel_radius=float(np.linalg.norm(projected[:,:2]/projected[:,2,None]-uv,axis=1).max())
    keep=distances<(pixel_radius*1.6)**2;xx=xx[keep];yy=yy[keep]
    if len(xx)<12:return default,{'method':'camera-facing fallback','reason':'insufficient eye pixels'}
    lo=np.array([xx.min(),yy.min()]);hi=np.array([xx.max(),yy.max()]);span=hi-lo+1
    central=(xx>lo[0]+span[0]*.18)&(xx<hi[0]-span[0]*.18)&(yy>lo[1]+span[1]*.25)&(yy<hi[1]-span[1]*.08)
    xx=xx[central];yy=yy[central]
    if len(xx)<8:return default,{'method':'camera-facing fallback','reason':'insufficient central eye support'}
    lum=rgb[yy,xx]@np.array([.2126,.7152,.0722]);contrast=float(np.quantile(lum,.9)-np.quantile(lum,.1))
    dark=lum<=np.quantile(lum,.28);weights=np.maximum(np.quantile(lum,.65)-lum[dark],1)
    target=np.array([np.average(xx[dark],weights=weights),np.average(yy[dark],weights=weights)])
    direction=np.linalg.solve(full[:,:3],np.r_[target,1]);origin=(camera_center-center)/radii;ray=direction/radii
    a=ray@ray;b=2*(origin@ray);c=origin@origin-1;disc=b*b-4*a*c
    if disc<=0 or contrast<10:return default,{'method':'camera-facing fallback','reason':'ambiguous contrast or candidate misses eyeball','contrast':contrast}
    t=(-b-np.sqrt(disc))/(2*a);gaze=origin+t*ray;gaze/=np.linalg.norm(gaze)
    angle=float(np.rad2deg(np.arccos(np.clip(default@gaze,-1,1))))
    if t<=0 or angle>28:return default,{'method':'camera-facing fallback','reason':'candidate outside conservative angular bound','angle_deg':angle}
    # Blend uncertain photo evidence toward a common viewing target.
    gaze=.25*gaze+.75*default;gaze/=np.linalg.norm(gaze)
    return gaze,{'method':'front-photo dark-region candidate blended with camera-facing prior','candidate_pixel':target.tolist(),
        'contrast':contrast,'candidate_angle_deg':angle,'gaze_ground_truth_verified':False,'iris_detector':False,'photo_weight':.25,'shared_camera_target_weight':.75}


def painted_eye(center,radii,config,direction,eye_index):
    mesh=trimesh.creation.uv_sphere(radius=1,count=[40,80]);unit=mesh.vertices.copy();mesh.vertices=center+unit*radii
    yaw=np.deg2rad(config.get('per_eye_yaw_deg',[0,0])[eye_index]+config['gaze_yaw_deg'])
    pitch=np.deg2rad(config.get('per_eye_pitch_deg',[0,0])[eye_index]+config['gaze_pitch_deg'])
    ry=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    rx=np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
    gaze=ry@rx@direction;gaze/=np.linalg.norm(gaze)
    right=np.cross([0.,1,0],gaze);right/=np.linalg.norm(right);up=np.cross(gaze,right)
    width,height=1024,512
    u,v=np.meshgrid((np.arange(width)+.5)/width,(np.arange(height)+.5)/height)
    theta=(u-.5)*2*np.pi;phi=v*np.pi
    normals=directions_for(theta,phi)
    x=normals@right;y=normals@up;z=normals@gaze;r=np.hypot(x,y)
    colors=np.broadcast_to(np.asarray(config['sclera_rgb'],float)/255,(*r.shape,3)).copy()
    def blend(color,weight):
        nonlocal colors
        colors=colors*(1-weight[:,:,None])+np.asarray(color)[None,None,:]*weight[:,:,None]
    def edge(radius):return np.clip((radius-r)/.018+.5,0,1)*(z>0)
    iris=np.asarray(config['iris_rgb'])/255
    iris_color=iris[None,None,:]*(.85+.18*np.clip(r/config['iris_ratio'],0,1))[:,:,None]
    w=edge(config['iris_ratio']);colors=colors*(1-w[:,:,None])+iris_color*w[:,:,None]
    blend([.045,.035,.03],edge(config['iris_ratio']*config['pupil_ratio']))
    h=np.clip((config['highlight_ratio']-np.hypot(x+.12,y-.13))/.018+.5,0,1)*(z>0)*w
    blend([.85,.82,.76],h*.85)
    png=io.BytesIO();Image.fromarray(np.uint8(np.clip(colors,0,1)*255)).save(png,format='PNG')
    uv=np.column_stack((np.arctan2(unit[:,0],unit[:,2])/(2*np.pi)+.5,np.arccos(np.clip(unit[:,1],-1,1))/np.pi))
    # Split seam-crossing triangles, duplicating vertices with wrapped u coordinates.
    vertices=[];uvs=[];faces=[]
    for face in mesh.faces:
        st=uv[face].copy()
        if np.ptp(st[:,0])>.5:st[st[:,0]<.5,0]+=1
        first=len(vertices);vertices.extend(mesh.vertices[face]);uvs.extend(st);faces.append([first,first+1,first+2])
    result=trimesh.Trimesh(vertices,faces,process=False)
    # Assembly uses this expanded mesh for UVs; print boolean welds it first.
    result.metadata['eye_texture_png']=png.getvalue();result.metadata['eye_uv']=np.asarray(uvs,dtype=np.float32)
    # Preserve smooth sphere normals rather than flat normals of the split triangles.
    result.metadata['eye_normals']=np.concatenate([unit[f]/radii for f in mesh.faces])
    nn=result.metadata['eye_normals'];result.metadata['eye_normals']=nn/np.linalg.norm(nn,axis=1,keepdims=True)
    return result,np.ones((len(result.vertices),4))
