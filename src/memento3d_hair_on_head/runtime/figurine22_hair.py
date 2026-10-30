"""Close-fit shell with independent radial detail; local constrained Laplacian sideburn editing.

Not ARAP, no strand reconstruction. Full face geometry is never a solve variable.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from figurine21_fit import compact_fit
from figurine2_hair import nearest_ray


def adjacency(mesh):
    edges=mesh.edges_unique;n=len(mesh.vertices)
    a=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    return sparse.diags(1/np.maximum(np.asarray(a.sum(1)).ravel(),1))@a


def shell_fit(hair,head,eye_y,settings,correspondence=None):
    if not .2<=settings['body_mm']<=1.0 or not 0<=settings['detail_strength']<=1.2 or not 0<settings['detail_limit_mm']<=.3 or not .1<=settings['edge_taper_mm']<=.8:
        raise ValueError('Invalid shell/detail design controls')
    if not isinstance(settings['detail_smoothing_iterations'],int) or not 0<=settings['detail_smoothing_iterations']<=400:raise ValueError('Invalid detail smoothing iterations')
    if correspondence is None:correspondence={}
    old,report=compact_fit(hair,head,eye_y,.25,settings,correspondence)
    upper=head.vertices[head.vertices[:,1]>eye_y+.07*head.extents[1]];lo,hi=np.quantile(upper,[.005,.995],axis=0)
    origin=np.array([(lo[0]+hi[0])/2,eye_y-.04*head.extents[1],(lo[2]+hi[2])/2])
    vec=hair.vertices-origin;r=np.linalg.norm(vec,axis=1);directions=vec/r[:,None]
    smooth=r.copy();a=adjacency(hair)
    for _ in range(settings['detail_smoothing_iterations']):smooth=.5*smooth+.5*(a@smooth)
    detail=np.clip(r-smooth,-settings['detail_limit_mm'],settings['detail_limit_mm'])*settings['detail_strength']
    wall=np.maximum(r-correspondence['inner'],0)
    fraction=1-np.exp(-wall/settings['edge_taper_mm'])
    radius=correspondence['scalp']-settings['inset_mm']+fraction*(settings['body_mm']+settings['inset_mm']+detail)
    output=hair.copy();output.vertices=origin+directions*radius[:,None]
    if not output.is_volume:raise ValueError('Independent-detail shell lost closed positive volume')
    report.update(method='source-scalp shell; body thickness independent of source radial detail',
        body_mm=settings['body_mm'],detail_strength=settings['detail_strength'],detail_limit_mm=settings['detail_limit_mm'],
        detail_radial_p95_mm=float(np.quantile(abs(detail),.95)),
        crown_added_height_mm=float(output.bounds[1,1]-head.bounds[1,1]),
        crown_proxy_pass=bool(output.bounds[1,1]-head.bounds[1,1]<=settings['max_crown_added_mm']),source_face_modified=False)
    return output,report


def ear_anchors(face):
    pts=face['points'];weights=np.asarray(face['data']['weights']).reshape(-1,3);head=face['head_corners']
    results=[]
    for side in [-1,1]:
        ear=np.zeros(len(pts),bool);observed_hair=np.zeros(len(pts),bool)
        for j,camera in enumerate(face['cameras']):
            q=np.c_[pts,np.ones(len(pts))]@camera.T;xy=q[:,:2]/q[:,2,None];x,y=np.floor(xy.T).astype(int);h,w=face['labels'][j].shape
            valid=(x>=0)&(x<w)&(y>=0)&(y<h)&(weights[:,j]>.05)&head&(pts[:,0]*side>0)
            ids=np.flatnonzero(valid);lab=face['labels'][j][y[ids],x[ids]]
            ear[ids[np.isin(lab,[7,8])]]=True;observed_hair[ids[lab==17]]=True
        if ear.sum()<30:raise ValueError('Insufficient visible semantic-ear support for automatic sideburn anchor')
        p=pts[ear];lo,center,hi=np.quantile(p,[.10,.50,.90],axis=0)
        local=observed_hair&(pts[:,0]*side>abs(center[0])-.4)&(abs(pts[:,2]-(hi[2]+.45))<1.4)&(pts[:,1]<hi[1]+1.2)&(pts[:,1]>lo[1])
        results.append({'side':side,'center':center.tolist(),'low':lo.tolist(),'high':hi.tolist(),
            'semantic_ear_corners':int(ear.sum()),'observed_hair_corners':int(local.sum()),
            'observed_hair_low_y_mm':float(np.quantile(pts[local,1],.15)) if local.sum()>=10 else None,
            'scope':'visible semantic projections; not independently calibrated anatomical landmarks'})
    return results


def laplacian_displacement(mesh,region,handles,target,weight=100.):
    """Minimize ||L d||^2 + w^2||d_handle-target||^2, with outside vertices fixed."""
    a=adjacency(mesh);n=len(mesh.vertices);L=sparse.eye(n,format='csr')-a
    ids=np.flatnonzero(region);handles=np.asarray(handles,int)
    if len(ids)<20 or len(handles)<3:raise ValueError('Insufficient Laplacian region or handles')
    # Include every Laplacian row touching the ROI, so its outer boundary is anchored smoothly.
    touching=np.flatnonzero(np.asarray(abs(L[:,ids]).sum(1)).ravel()>0)
    B=L[touching][:,ids];local=np.full(n,-1);local[ids]=np.arange(len(ids))
    if (local[handles]<0).any():raise ValueError('Laplacian handles outside region')
    C=sparse.coo_matrix((np.full(len(handles),weight),(np.arange(len(handles)),local[handles])),shape=(len(handles),len(ids))).tocsr()
    matrix=B.T@B+C.T@C+sparse.eye(len(ids))*1e-8
    rhs=C.T@(np.asarray(target)*weight)
    delta=np.zeros((n,3));delta[ids]=spsolve(matrix.tocsc(),rhs)
    if not np.isfinite(delta).all():raise ValueError('Nonfinite local deformation')
    return delta


def sideburn_targets(points,tip,z,extension,settings):
    if not .1<=settings['tip_taper_factor']<=1 or not .1<=settings['tip_rounding_sigma_mm']<=1 or not 0<=settings['max_fore_aft_adjust_mm']<=1.2:
        raise ValueError('Invalid sideburn taper controls')
    target=np.zeros((len(points),3));offsets=points[:,2]-tip[2]
    rounded=.25+.75*np.exp(-(offsets/settings['tip_rounding_sigma_mm'])**2)
    target[:,1]=-extension*rounded
    target[:,2]=np.clip(z+offsets*settings['tip_taper_factor']-points[:,2],-settings['max_fore_aft_adjust_mm'],settings['max_fore_aft_adjust_mm'])
    return target


def extend_sideburns(hair,head,face,extra_mm,settings):
    if not np.isfinite(extra_mm) or not 0<=extra_mm<=settings['max_extra_mm']:raise ValueError('Sideburn extension outside budget')
    anchors=ear_anchors(face);result=hair.copy();measurements=[]
    for anchor in anchors:
        side=anchor['side'];center=np.asarray(anchor['center']);hi=np.asarray(anchor['high']);lo=np.asarray(anchor['low']);p=result.vertices.copy()
        z=hi[2]+settings['front_of_ear_mm']
        candidates=(p[:,0]*side>abs(center[0])-.3)&(abs(p[:,2]-z)<settings['tip_band_mm'])&(p[:,1]<face['eye_centers'][:,1].mean()+3.)&(p[:,1]>lo[1])
        if candidates.sum()<10:raise ValueError('No supported template sideburn tip in front of ear')
        ids=np.flatnonzero(candidates);lowest=np.quantile(p[ids,1],.03)
        tipids=ids[p[ids,1]<lowest+.12];tip=np.median(p[tipids],axis=0)
        distance=np.linalg.norm(p-tip,axis=1)
        region=(distance<settings['region_radius_mm'])&(p[:,0]*side>0)&(p[:,2]>hi[2]+.10)
        handles=np.flatnonzero(candidates&(p[:,1]<lowest+.20)&(distance<.75))
        base_extension=max(0.,tip[1]-(hi[1]+settings['ear_top_clearance_mm']))
        extension=min(base_extension+extra_mm,settings['max_total_extension_mm'])
        target=sideburn_targets(p[handles],tip,z,extension,settings)
        delta=np.zeros_like(p) if extension==0 else laplacian_displacement(result,region,handles,target)
        magnitude=np.linalg.norm(delta,axis=1)
        if magnitude.max()>np.linalg.norm(target,axis=1).max()+.05:raise ValueError('Local sideburn solve exceeds movement budget')
        moved=p+delta
        active=np.flatnonzero(magnitude>1e-6)
        eye_y=float(face['eye_centers'][:,1].mean());upper=head.vertices[head.vertices[:,1]>eye_y+.07*head.extents[1]]
        scalp_lo,scalp_hi=np.quantile(upper,[.005,.995],axis=0)
        origin=np.array([(scalp_lo[0]+scalp_hi[0])/2,eye_y-.04*head.extents[1],(scalp_lo[2]+scalp_hi[2])/2])
        for first in range(0,len(active),256):
            ids=active[first:first+256];before=p[ids]-origin;after=moved[ids]-origin
            rb=np.linalg.norm(before,axis=1);ra=np.linalg.norm(after,axis=1);db=before/rb[:,None];da=after/ra[:,None];o=np.tile(origin,(len(ids),1))
            sb=nearest_ray(head,o,db);sa=nearest_ray(head,o,da)
            if not (np.isfinite(sb).all() and np.isfinite(sa).all()):raise ValueError('Sideburn scalp reattachment ray incomplete')
            moved[ids]=origin+da*(sa+(rb-sb))[:,None]
        result.vertices=moved
        magnitude=np.linalg.norm(moved-p,axis=1)
        if magnitude.max()>settings['max_total_extension_mm']+.5:raise ValueError('Reattached sideburn exceeded total design budget')
        if not result.is_volume:raise ValueError('Sideburn deformation lost closed positive volume')
        measurements.append(dict(anchor,tip_before_mm=tip.tolist(),tip_after_mm=np.median(result.vertices[handles],axis=0).tolist(),
            edited_vertices=int((magnitude>1e-5).sum()),max_displacement_mm=float(magnitude.max()),
            tip_taper_factor=settings['tip_taper_factor'],rounded_tip=True,requested_extension_mm=extra_mm,ear_alignment_extension_mm=float(base_extension),total_design_extension_mm=float(extension),handle_count=len(handles),outside_region_max_displacement_mm=float(magnitude[~region].max()) if (~region).any() else 0.))
    return result,{'method':'local normalized-Laplacian displacement solve; outside region fixed',
        'is_arap':False,'automatic_ear_anchors':True,'extension_mode':'ear-relative fit plus bounded design customization; not measured hair length',
        'sides':measurements,'source_face_modified':False,'ear_collision_certified':False}


def contact_screen(hair,head,eye_y):
    upper=head.vertices[head.vertices[:,1]>eye_y+.07*head.extents[1]];lo,hi=np.quantile(upper,[.005,.995],axis=0)
    origin=np.array([(lo[0]+hi[0])/2,eye_y-.04*head.extents[1],(lo[2]+hi[2])/2])
    theta=np.tile(np.linspace(-np.pi,np.pi,32,endpoint=False),5);phi=np.repeat(np.deg2rad([10,30,50,70,85]),32)
    d=np.c_[np.sin(phi)*np.sin(theta),np.cos(phi),np.sin(phi)*np.cos(theta)]
    o=np.tile(origin,(len(d),1));a=nearest_ray(head,o,d);b=nearest_ray(hair,o,d);valid=np.isfinite(a)&np.isfinite(b)
    gaps=np.maximum(b[valid]-a[valid],0)
    return {'rays':int(valid.sum()),'gap_p95_mm':float(np.quantile(gaps,.95)), 'gap_max_mm':float(gaps.max()),
        'pass':bool(valid.sum()>=60 and np.quantile(gaps,.95)<=.08),'scope':'sampled upper-head radial contact; not full clearance/thickness certification'}


def reference_correspondence(run,face,fitted,asset,settings):
    """Recover an exact prior radial correspondence from a verified V2.1 GLB.

Optional speed path. Recompute scalp rays, invert the recorded monotone map,
then cross-check a deterministic subset against fresh source-template rays.
No reference run is needed for normal fresh operation.
"""
    import json
    from pathlib import Path
    from figurine2_face import sha
    from add_graduation_cap import load
    from figurine_preview import accessor
    run=Path(run).resolve();report=json.loads((run/'report.json').read_text())
    if report['face']['workspace_sha256']!=face['report']['workspace_sha256']:raise ValueError('Reference run workspace mismatch')
    if report['face']['head_neck_mm']!=face['report']['head_neck_mm']:raise ValueError('Reference scale mismatch')
    candidates=[c for c in report['candidates'] if c['status']=='EXPORTED' and abs(c['compact']['ratio']-.25)<1e-12 and c['fit']['asset_id']==asset['id']]
    if not candidates:raise ValueError('Reference run has no compatible 0.25 candidate')
    c=candidates[0]
    if c['fit']['asset_sha256']!=asset['sha256'] or c['repair']['used']:raise ValueError('Reference asset or topology mismatch')
    name=c['id']+'/appearance.glb';path=(run/name).resolve()
    if not path.is_relative_to(run):raise ValueError('Unsafe reference candidate path')
    checks=json.loads((run/'checksums.json').read_text())
    if sha(path)!=checks[name]:raise ValueError('Reference GLB checksum mismatch')
    doc,blob=load(path);matches=[]
    for node in doc['nodes']:
        if node.get('name')=='Compact_Hair':
            prim=doc['meshes'][node['mesh']]['primitives'][0];matches.append(accessor(doc,blob,prim['attributes']['POSITION'])*1000)
    if len(matches)!=1 or matches[0].shape!=fitted.vertices.shape:raise ValueError('Reference vertex correspondence missing')
    old=matches[0];eye_y=float(face['eye_centers'][:,1].mean());head=face['head']
    upper=head.vertices[head.vertices[:,1]>eye_y+.07*head.extents[1]];lo,hi=np.quantile(upper,[.005,.995],axis=0)
    origin=np.array([(lo[0]+hi[0])/2,eye_y-.04*head.extents[1],(lo[2]+hi[2])/2])
    vec=fitted.vertices-origin;r=np.linalg.norm(vec,axis=1);d=vec/r[:,None];oldvec=old-origin;oldr=np.linalg.norm(oldvec,axis=1)
    direction_error=float(np.linalg.norm(oldvec/oldr[:,None]-d,axis=1).max())
    if direction_error>2e-5:raise ValueError('Reference vertex order or fitted directions differ')
    scalp=[]
    for first in range(0,len(r),256):
        rays=d[first:first+256];scalp.append(nearest_ray(head,np.tile(origin,(len(rays),1)),rays))
    scalp=np.concatenate(scalp)
    if not np.isfinite(scalp).all():raise ValueError('Reference scalp ray incomplete')
    cp=c['compact'];value=oldr-scalp+cp['inset_mm'];low=np.zeros(len(r));high=np.full(len(r),20.)
    for _ in range(45):
        mid=(low+high)/2;mapped=cp['ratio']*mid+cp['floor_mm']*(1-np.exp(-mid/.3))
        under=mapped<value;low[under]=mid[under];high[~under]=mid[~under]
    wall=(low+high)/2;inner=r-wall
    # Audit the inference with actual first-entry queries, not only algebraic roundtrip.
    ids=np.linspace(0,len(r)-1,min(128,len(r)),dtype=int);hits=[]
    for first in range(0,len(ids),16):
        rays=d[ids[first:first+16]];hits.append(nearest_ray(fitted,np.tile(origin,(len(rays),1)),rays))
    hits=np.concatenate(hits);valid=np.isfinite(hits);error=float(np.max(abs(hits[valid]-inner[ids[valid]]))) if valid.any() else float('inf')
    if valid.sum()<100 or error>.002:raise ValueError('Reference cavity correspondence cross-check failed')
    cache={'scalp':scalp,'inner':inner,'inner_repaired':True,'entry_missed_count':0,'endpoint_fallback_count':0}
    provenance={'mode':'verified V2.1 map inversion with fresh scalp rays and independent cavity-ray subset',
        'reference_glb_sha256':sha(path),'reference_workspace_sha256':report['face']['workspace_sha256'],
        'direction_error_max':direction_error,'independent_cavity_rays':int(valid.sum()),'independent_cavity_error_max_mm':error}
    return cache,provenance


def ear_intrusion_screen(hair,baseline,face):
    pts=face['points'];weights=np.asarray(face['data']['weights']).reshape(-1,3);ear=np.zeros(len(pts),bool)
    for j,camera in enumerate(face['cameras']):
        q=np.c_[pts,np.ones(len(pts))]@camera.T;xy=q[:,:2]/q[:,2,None];x,y=np.floor(xy.T).astype(int);h,w=face['labels'][j].shape
        valid=(x>=0)&(x<w)&(y>=0)&(y<h)&(weights[:,j]>.05)&face['head_corners'];ids=np.flatnonzero(valid)
        ear[ids[np.isin(face['labels'][j][y[ids],x[ids]],[7,8])]]=True
    result=[]
    for side in [-1,1]:
        p=np.unique(pts[ear&(pts[:,0]*side>0)],axis=0)
        if len(p)<20:raise ValueError('Insufficient source ear surface for intrusion screen')
        p=p[np.linspace(0,len(p)-1,min(256,len(p)),dtype=int)]
        old=baseline.contains(p);new=hair.contains(p);added=float((new&~old).mean())
        result.append({'side':side,'samples':len(p),'baseline_covered_fraction':float(old.mean()),'edited_covered_fraction':float(new.mean()),'newly_covered_fraction':added})
    return {'sides':result,'pass':all(r['newly_covered_fraction']<=.03 for r in result),
        'scope':'visible semantic ear corners, sampled source-mesh coverage; not a full anatomical collision certificate'}
