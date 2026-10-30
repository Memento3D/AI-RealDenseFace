"""Head-owned hair scaffold: observed meridian boundaries, periodic regularization.

The cropped single-image mesh is never treated as a standalone scalp cap.
Unobserved posterior boundaries stay a documented bounded prior.
"""
import numpy as np
from scipy import ndimage as ndi,sparse
from scipy.optimize import lsq_linear,least_squares
import trimesh
from headlab_raster import surface_depth_fast,fragments


def clean_mask(image,labels):
    mask=labels==17
    if mask.sum()<100:raise ValueError('Insufficient hair pixels')
    inside=ndi.binary_erosion(mask,iterations=3)
    rgb=np.asarray(image.convert('RGB') if hasattr(image,'convert') else image,dtype=float)
    if rgb.max(initial=0)>1:rgb=rgb/255
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    y=linear@np.array([.2126729,.7151522,.0721750])
    delta=6/29
    f=np.where(y>delta**3,np.cbrt(y),y/(3*delta**2)+4/29)
    L=116*f-16
    threshold=float(np.quantile(L[inside],.95)+8) if inside.sum()>100 else 101.
    suspect=mask&(ndi.distance_transform_edt(mask)<3)&(L>threshold)
    cleaned=mask&~suspect
    cleaned=ndi.binary_opening(cleaned,structure=ndi.generate_binary_structure(2,1),iterations=1)
    groups,n=ndi.label(cleaned);counts=np.bincount(groups.ravel());counts[0]=0
    if n==0:raise ValueError('Hair cleaning removed all support')
    cleaned=ndi.binary_fill_holes(groups==counts.argmax())
    return cleaned,{'input_hair_pixels':int(mask.sum()),'cleaned_hair_pixels':int(cleaned.sum()),'bright_boundary_pixels_removed':int(suspect.sum()),'method':'conservative boundary-color rejection, one-pixel opening, main connected component'}


def radial_samples(head,origin,directions,require_all=True):
    radius=np.full(len(directions),np.nan);normals=np.zeros((len(directions),3))
    for start in range(0,len(directions),128):
        d=directions[start:start+128];o=np.tile(origin,(len(d),1))
        loc,ids,tri=head.ray.intersects_location(o,d,multiple_hits=False)
        radius[start+ids]=np.linalg.norm(loc-origin,axis=1);normals[start+ids]=head.face_normals[tri]
    pole=np.flatnonzero(~np.isfinite(radius)&(directions[:,1]>.99))
    if len(pole):
        offsets=np.array([[.035,0,0],[-.035,0,0],[0,0,.035],[0,0,-.035]])
        for index in pole:
            nearby=directions[index]+offsets
            nearby/=np.linalg.norm(nearby,axis=1,keepdims=True)
            loc,ids,tri=head.ray.intersects_location(np.tile(origin,(len(nearby),1)),nearby,multiple_hits=False)
            if len(ids)<3:continue
            radius[index]=float(np.median(np.linalg.norm(loc-origin,axis=1)))
            normal=head.face_normals[tri].mean(0)
            normals[index]=normal/max(np.linalg.norm(normal),1e-12)
    if require_all and not np.isfinite(radius).all():raise ValueError('Source head does not cover scalp directions')
    return radius,normals


def directions(theta,phi):return np.stack([np.sin(phi)*np.sin(theta),np.cos(phi),np.sin(phi)*np.cos(theta)],axis=-1)


def scalp_origin(head,eye_centers):
    eye=float(np.asarray(eye_centers)[:,1].mean())
    upper=head.vertices[head.vertices[:,1]>eye+.07*head.extents[1]]
    if len(upper)<8:raise ValueError('Insufficient upper-head geometry for cap origin')
    lo,hi=np.quantile(upper,[.005,.995],axis=0)
    center=np.array([(lo[0]+hi[0])/2,(lo[2]+hi[2])/2])
    theta=np.linspace(-np.pi,np.pi,32,endpoint=False)
    phi=np.deg2rad([1.5,20,45,75,100,124])
    th,ph=np.meshgrid(theta,phi,indexing='ij')
    cover=directions(th,ph).reshape(-1,3)
    for offset in np.linspace(.22,.38,17):
        origin=np.array([center[0],eye+offset*head.extents[1],center[1]])
        if not head.contains(origin[None])[0]:
            continue
        try:
            radial_samples(head,origin,cover)
        except ValueError:
            continue
        return origin
    raise ValueError('Source head does not cover a closed scalp-cap origin')


def infer_boundary(face,masks,settings):
    head=face['head'];view_heads=face.get('view_heads',[head]*len(masks))
    view_eyes=face.get('view_eye_centers',[face['eye_centers']]*len(masks))
    if len(view_heads)!=len(masks) or len(view_eyes)!=len(masks):raise ValueError('View geometry does not match hair masks')
    origin=scalp_origin(head,face['eye_centers'])
    n=settings['theta_samples'];m=settings['phi_samples']
    theta=np.linspace(-np.pi,np.pi,n,endpoint=False)
    phi=np.linspace(.025,np.deg2rad(125),m)
    surface_th,surface_ph=np.meshgrid(theta,phi,indexing='ij')
    surface_d=directions(surface_th,surface_ph).reshape(-1,3)
    radius,normals=radial_samples(head,origin,surface_d)
    observation_phi=np.linspace(.025,np.deg2rad(150 if len(masks)>3 else 125),m)
    observation_th,observation_ph=np.meshgrid(theta,observation_phi,indexing='ij')
    observation_d=directions(observation_th,observation_ph).reshape(-1,3)
    score=np.zeros(len(observation_d));weight=np.zeros(len(observation_d));records=[]
    hair_view_valid=face.get('hair_view_valid',[True]*len(masks))
    if len(hair_view_valid)!=len(masks):raise ValueError('Hair-mask reliability does not match the view set')
    for j,(camera,mask,label,image) in enumerate(zip(face['cameras'],masks,face['labels'],face['images'])):
        if not hair_view_valid[j]:
            records.append({'view':j,'visible_supported_samples':0,'used_for_hair_fit':False,
                'reason':'parser returned insufficient hair pixels'})
            continue
        view_head=view_heads[j];view_origin=scalp_origin(view_head,view_eyes[j])
        view_radius,view_normals=radial_samples(
            view_head,view_origin,observation_d,require_all=len(masks)<4)
        p=view_origin+observation_d*view_radius[:,None]
        depth=surface_depth_fast(view_head.vertices,view_head.faces,camera,image.size)
        center=-np.linalg.solve(camera[:,:3],camera[:,3])
        geometry_finite=np.isfinite(p).all(axis=1)
        safe_p=np.where(geometry_finite[:,None],p,center)
        view=center-safe_p
        view_norm=np.linalg.norm(view,axis=1,keepdims=True)
        view=np.divide(view,view_norm,out=np.zeros_like(view),where=view_norm>0)
        facing=np.einsum('ij,ij->i',view_normals,view)
        q=np.c_[safe_p,np.ones(len(safe_p))]@camera.T
        xy=q[:,:2]/q[:,2,None]
        ix,iy=np.floor(xy.T).astype(int);h,w=mask.shape
        valid=geometry_finite&(q[:,2]>0)&(ix>=0)&(ix<w)&(iy>=0)&(iy<h)&(facing>settings['min_facing'])
        # Rear ownership: oblique front photos cannot establish a posterior root.
        if j<3:valid&=abs(observation_th.ravel())<np.deg2rad(settings['front_views_max_theta_deg'])
        else:valid&=abs(observation_th.ravel())>np.deg2rad(95)
        ids=np.flatnonzero(valid);obsdepth=depth[iy[ids],ix[ids]]
        seen=np.isfinite(obsdepth)&(q[ids,2]<=obsdepth+.25*np.linalg.norm(camera[2,:3]));ids=ids[seen]
        lab=label[iy[ids],ix[ids]];accepted=mask[iy[ids],ix[ids]]|np.isin(lab,[1,2,3,4,5,7,8,10,11,12,13,14,16]);ids=ids[accepted]
        signed=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
        value=np.clip(.5+ndi.map_coordinates(signed,[xy[ids,1],xy[ids,0]],order=1)/3,0,1)
        wgt=facing[ids]**4;score[ids]+=value*wgt;weight[ids]+=wgt
        records.append({'view':j,'visible_supported_samples':len(ids),'used_for_hair_fit':True,'camera_center':center.tolist(),
            'geometry_bounds':view_head.bounds.tolist()})
    probability=np.divide(score,weight,out=np.zeros_like(score),where=weight>0).reshape(n,m);weights=weight.reshape(n,m)
    # Hairline prior expressed relative to the skull, only a regularizer where observed.
    front=np.maximum(np.cos(theta),0);back=np.maximum(-np.cos(theta),0)
    prior=np.deg2rad(92-31*front**2+17*back**2)
    low=np.deg2rad(35+35*(1-front));high=np.deg2rad(83+37*(1-front))
    raw=prior.copy();confidence=np.zeros(n)
    for k in range(n):
        pos=weights[k]*probability[k];neg=weights[k]*(1-probability[k])
        cost=np.cumsum(neg)+np.cumsum(pos[::-1])[::-1]-pos
        allowed=(observation_phi>=low[k])&(observation_phi<=high[k]);minimum=cost[allowed].min();candidates=np.flatnonzero(allowed&(cost<=minimum+.02))
        raw[k]=np.mean(observation_phi[candidates])
        # Both hair and nonhair are necessary to claim a measured boundary.
        if pos.sum()>.3 and neg.sum()>.3:confidence[k]=min(pos.sum(),neg.sum(),5.)
    if len(masks)<4:confidence[abs(theta)>np.deg2rad(settings['front_views_max_theta_deg'])]=0
    # Periodic second differences penalize kinks; bounded posterior cannot grow a side-view tail.
    i=np.arange(n);L=sparse.coo_matrix((np.r_[np.ones(n),-2*np.ones(n),np.ones(n)],(np.r_[i,i,i],np.r_[(i-1)%n,i,(i+1)%n])),shape=(n,n)).tocsr()
    prior_weight=np.where(confidence>0,.04,settings['unknown_prior_weight'])
    A=sparse.vstack([sparse.diags(np.sqrt(confidence)),np.sqrt(settings['boundary_smoothness'])*L,sparse.diags(np.sqrt(prior_weight))]).tocsr()
    b=np.r_[np.sqrt(confidence)*raw,np.zeros(n),np.sqrt(prior_weight)*prior]
    fit=lsq_linear(A,b,bounds=(low,high),tol=1e-7,max_iter=200)
    root=fit.x
    rear_observed=int(((confidence>0)&(abs(theta)>np.deg2rad(120))).sum())
    required_rear=max(4,n//20)
    if len(masks)>3 and rear_observed<required_rear:
        raise ValueError(f'Back photo has insufficient posterior root support ({rear_observed}/{n} meridians; need {required_rear}); refuse to claim a measured nape')
    observed=int((confidence>0).sum())
    if not np.isfinite(root).all() or observed<n*.15:
        raise ValueError(f'Insufficient observed hairline support: {observed}/{n} meridians')
    return {'origin':origin,'theta':theta,'phi':phi,'radius':radius.reshape(n,m),'normal':normals.reshape(n,m,3),
        'root':root,'confidence':confidence,'probability':probability,'weights':weights,
        'observation_phi':observation_phi,
        'report':{'method':'visibility-owned meridian classification with bounded periodic curvature regularization',
            'observed_meridians':int((confidence>0).sum()),'total_meridians':n,'posterior_observed':bool(rear_observed>0),'posterior_supported_meridians':rear_observed,
            'posterior_probe_max_deg':float(np.rad2deg(observation_phi[-1])),
            'posterior_policy':('posterior photo owns rear; oblique photos excluded' if len(masks)>3 else 'bounded source-skull prior; front/oblique images have no posterior-root ownership'),
            'source_views':records,'source_face_changed':False,'raw_root_deg':np.rad2deg(raw).tolist(),'root_deg':np.rad2deg(root).tolist(),
            'root_confidence':confidence.tolist(),'independent_camera_calibration_verified':False}}


def cap_patch(face,field,settings):
    n=len(field['theta']);rings=settings['cap_rings'];root=field['root'];theta=field['theta'];o=field['origin']
    tt=np.tile(theta,rings);frac=np.repeat(np.arange(1,rings+1)/rings,n);pp=np.tile(root,rings)*frac
    ds=np.vstack([[0.,1.,0.],directions(tt,pp)]);r,normals=radial_samples(face['head'],o,ds)
    # Remove narrow ear-surface bumps from the scaffold without moving the source head.
    table=r[1:].reshape(rings,n);smooth=ndi.median_filter(np.pad(table,((0,0),(3,3)),mode='wrap'),size=(1,7))[:,3:-3]
    ear_band=(pp>np.deg2rad(70))&(abs(np.cos(tt))<.6)
    radial=r.copy();radial[1:][ear_band]=np.minimum(table.ravel()[ear_band],smooth.ravel()[ear_band])
    p=o+ds*radial[:,None];faces=[]
    for k in range(n):faces.append([0,1+k,1+(k+1)%n])
    for j in range(rings-1):
        a=1+j*n;b=a+n
        for k in range(n):
            kn=(k+1)%n;faces.extend([[a+k,b+k,b+kn],[a+k,b+kn,a+kn]])
    patch=trimesh.Trimesh(p,faces,process=False)
    # Ensure the patch normal points away from the internal origin.
    if np.median(np.einsum('ij,ij->i',patch.face_normals,patch.triangles_center-o))<0:patch.faces=patch.faces[:,::-1]
    boundary=np.arange(1+(rings-1)*n,1+rings*n)
    return patch,ds,radial,boundary,np.r_[0,frac]


def volume_basis(ds):
    theta=np.arctan2(ds[:,0],ds[:,2]);phi=np.arccos(np.clip(ds[:,1],-1,1))
    angular=np.sin(phi)**2
    B=np.column_stack([np.exp(-(phi/.65)**2),angular*np.maximum(np.cos(theta),0)**2,
        angular*np.maximum(-np.sin(theta),0)**2,angular*np.maximum(np.sin(theta),0)**2,
        angular*np.maximum(-np.cos(theta),0)**2])
    return B/np.maximum(B.sum(1,keepdims=True),1e-9)


def fit_thickness(face,masks,patch,ds,radial,fraction,settings,view_surface_points=None):
    points=patch.vertices;o=points[0]-ds[0]*radial[0];theta=np.arctan2(ds[:,0],ds[:,2]);phi=np.arccos(np.clip(ds[:,1],-1,1))
    B=volume_basis(ds)
    taper=np.clip((1-fraction)/.14,0,1);taper=taper*taper*(3-2*taper);records=[]
    normals=patch.vertex_normals
    if view_surface_points is not None and (len(view_surface_points)!=len(masks) or any(np.asarray(p).shape!=points.shape for p in view_surface_points)):
        raise ValueError('Per-view cap surface samples do not match the cap scaffold')
    hair_view_valid=face.get('hair_view_valid',[True]*len(masks))
    if len(hair_view_valid)!=len(masks):raise ValueError('Hair-mask reliability does not match the view set')
    for i,(camera,mask) in enumerate(zip(face['cameras'],masks)):
        if not hair_view_valid[i]:continue
        surface=points if view_surface_points is None else np.asarray(view_surface_points[i])
        center=-np.linalg.solve(camera[:,:3],camera[:,3]);view=center-surface;view/=np.linalg.norm(view,axis=1,keepdims=True)
        facing=np.einsum('ij,ij->i',normals,view)
        ids=np.flatnonzero((facing>-.08)&(facing<.4)&(fraction<.80)&(abs(theta)<np.deg2rad(115)))
        ids=ids[::max(1,len(ids)//250)]
        signed=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
        records.append((camera,signed,ids,surface))
    if not records:raise ValueError('No reliable hair masks are available for volume fitting')
    def offsets(x):
        x=x.copy()
        if len(masks)<4:x[-1]=settings['base_mm']
        return settings['edge_mm']+(B@x-settings['edge_mm'])*taper
    def residual(x):
        outer=points+ds*offsets(x)[:,None];result=[]
        for camera,sd,ids,surface in records:
            candidate=surface+ds*offsets(x)[:,None]
            q=np.c_[candidate[ids],np.ones(len(ids))]@camera.T;xy=q[:,:2]/q[:,2,None]
            values=ndi.map_coordinates(sd,[xy[:,1],xy[:,0]],order=1,mode='nearest');result.extend(values/6.)
        result.extend((x-settings['base_mm'])/.55)
        return np.asarray(result)
    initial=np.full(5,settings['base_mm']);fit=least_squares(residual,initial,bounds=(settings['min_mm'],settings['max_mm']),loss='soft_l1',max_nfev=45)
    tolerance=max(.001,(settings['max_mm']-settings['min_mm'])*.01)
    names=['crown','front','left','right','rear']
    at_bounds=[name for name,value in zip(names,fit.x)
        if value-settings['min_mm']<=tolerance or settings['max_mm']-value<=tolerance]
    return offsets(fit.x),{'method':'five-field bounded radial offset fit to mask tangent proxies', 'parameters_mm':fit.x.tolist(),
        'posterior_volume_fixed_prior':len(masks)<4,'converged':bool(fit.success),'residual_rms_before':float(np.sqrt(np.mean(residual(initial)**2))),
        'residual_rms_after':float(np.sqrt(np.mean(residual(fit.x)**2))),
        'parameters_at_bounds':at_bounds,'scope':'image-consistency proxy, not true hair thickness or print calibration'}


def close_cap(patch,ds,offset,boundary,inset):
    p=patch.vertices;outer=p+ds*offset[:,None];inner=p-ds*inset;n=len(p)
    faces=np.vstack([patch.faces,patch.faces[:,::-1]+n]);walls=[]
    for a,b in zip(boundary,np.roll(boundary,-1)):walls.extend([[b,a,a+n],[b,a+n,b+n]])
    cap=trimesh.Trimesh(np.vstack([outer,inner]),np.vstack([faces,walls]),process=False);cap.fix_normals()
    if not cap.is_volume:raise ValueError('Scalp-owned cap is not closed positive volume')
    return cap
