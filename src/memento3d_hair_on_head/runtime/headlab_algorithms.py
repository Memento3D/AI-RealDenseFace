"""Reproducible HeadLab algorithms; research implementations, not print certificates.

Uniform-weight Laplacian displacement editing and local/global ARAP follow
published principles. They are not copies of the authors' software. Hair volume
parameterization and normal-ray screening are Memento experimental adaptations.
"""
import time
import numpy as np
from scipy import sparse, ndimage
from scipy.sparse.linalg import factorized
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import least_squares
import trimesh

SOURCES=[
 {'id':'wendland1995','title':'Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree','year':1995,'url':'https://doi.org/10.1007/BF02123482','use':'V5 uses compact kernel (1-r)^4_+(4r+1), normalized into a custom partition-of-unity hair offset basis; not a published hairstyle reconstruction method.'},
 {'id':'moller1997','title':'Fast, Minimum Storage Ray-Triangle Intersection','year':1997,'url':'https://doi.org/10.1080/10867651.1997.10487468','use':'Implemented ray/triangle intersection for sampled inward-normal chord screening; not a wall thickness certificate.'},
 {'id':'laplacian2004','title':'Laplacian Surface Editing','year':2004,'url':'https://igl.ethz.ch/projects/Laplacian-mesh-processing/Laplacian-mesh-editing/','use':'Implemented uniform-weight Laplacian displacement regularization with automatic positional targets; not full rotation-invariant original LSE.'},
 {'id':'arap2007','title':'As-Rigid-As-Possible Surface Modeling','year':2007,'url':'https://igl.ethz.ch/projects/ARAP/','use':'Implemented positive uniform-weight spokes ARAP, local SVD rotations and global sparse solve; automatic anchors.'},
 {'id':'scipy_ls','title':'SciPy bounded nonlinear least squares','url':'https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html','use':'Implemented bounded TRF, linear versus soft_l1 loss; same residuals and initialization.'},
 {'id':'catmullrom','title':'On the Parameterization of Catmull-Rom Curves','year':2009,'url':'https://www.cemyuksel.com/research/catmullrom_param/','use':'Curve evaluation utility and tests only; not used to infer hairstyle topology or orientation in this benchmark.'},
 {'id':'hairmeshes','title':'Hair Meshes','year':2009,'url':'https://www.cemyuksel.com/research/hairmeshes/','use':'Reference only; no Hair Meshes implementation or licensed asset bundled.'},
 {'id':'shape_diameter','title':'CGAL Shape Diameter Function','url':'https://doc.cgal.org/latest/Surface_mesh_segmentation/index.html','use':'Reference only. Implemented normal-ray chords are a screening proxy, NOT CGAL SDF or a certified minimum wall thickness.'},
 {'id':'strength2016','title':'Direct shape optimization for strengthening 3D printable objects','year':2016,'url':'https://kalo-ai.github.io/papers/directshapeoptimization/index.html','use':'Reference only; no FEA, loads or material strength simulation implemented.'},
 {'id':'uvtools','title':'UVtools','url':'https://github.com/sn4k3/UVtools','use':'Reference only in default setup. Sliced-layer verification requires a configured external driver; no UVtools execution claimed.'}
]


def graph(mesh):
    edges=np.asarray(mesh.edges_unique)
    n=len(mesh.vertices);i,j=edges.T
    a=sparse.coo_matrix((np.ones(2*len(i)),(np.r_[i,j],np.r_[j,i])),shape=(n,n)).tocsr()
    return edges,(sparse.diags(np.asarray(a.sum(1)).ravel())-a).tocsc()


def solve_fixed(matrix,rhs,fixed,positions):
    n=len(positions);free=np.setdiff1d(np.arange(n),fixed)
    result=positions.copy()
    if len(free):
        solve=factorized(matrix[free][:,free].tocsc())
        b=rhs[free]-matrix[free][:,fixed]@positions[fixed]
        result[free]=np.column_stack([solve(b[:,k]) for k in range(3)])
    return result


def laplacian_edit(mesh,target,fixed,weight=.35):
    rest=np.asarray(mesh.vertices);_,lap=graph(mesh)
    matrix=(lap.T@lap+weight*sparse.eye(len(rest))).tocsc()
    rhs=lap.T@(lap@rest)+weight*target
    constrained=rest.copy();constrained[fixed]=target[fixed]
    result=solve_fixed(matrix,rhs,fixed,constrained)
    return result,{'method':'uniform Laplacian displacement regularization','linear_solves':3,'target_weight':weight,'source_ids':['laplacian2004']}


def arap_energy(rest,points,edges,rotations):
    i,j=edges.T;e=rest[i]-rest[j];d=points[i]-points[j]
    ri=np.einsum('nij,nj->ni',rotations[i],e);rj=np.einsum('nij,nj->ni',rotations[j],e)
    return float(.5*((d-ri)**2+(d-rj)**2).sum())


def arap_edit(mesh,target,fixed,weight=.35,iterations=8):
    rest=np.asarray(mesh.vertices);edges,lap=graph(mesh);i,j=edges.T;n=len(rest)
    matrix=(lap+weight*sparse.eye(n)).tocsc();free=np.setdiff1d(np.arange(n),fixed)
    solve=factorized(matrix[free][:,free].tocsc()) if len(free) else None
    coupling=matrix[free][:,fixed]
    points=target.copy();energies=[]
    for _ in range(iterations):
        e=rest[i]-rest[j];d=points[i]-points[j]
        covariance=np.zeros((n,3,3))
        products=np.einsum('ni,nj->nij',e,d)
        np.add.at(covariance,i,products);np.add.at(covariance,j,products)
        u,_,vt=np.linalg.svd(covariance)
        rotation=vt.transpose(0,2,1)@u.transpose(0,2,1)
        negative=np.linalg.det(rotation)<0
        vt[negative,-1,:]*=-1
        rotation=vt.transpose(0,2,1)@u.transpose(0,2,1)
        b=np.zeros_like(rest)
        edge_rhs=.5*np.einsum('nij,nj->ni',rotation[i]+rotation[j],e)
        np.add.at(b,i,edge_rhs);np.add.at(b,j,-edge_rhs)
        rhs=b+weight*target
        points[fixed]=target[fixed]
        if len(free):
            reduced=rhs[free]-coupling@points[fixed]
            points[free]=np.column_stack([solve(reduced[:,k]) for k in range(3)])
        energies.append(arap_energy(rest,points,edges,rotation)+weight*float(((points-target)**2).sum()))
    return points,{'method':'uniform spokes ARAP local/global','iterations':iterations,'target_weight':weight,'objective_history':energies,'source_ids':['arap2007']}


def catmull_rom(points,samples_per_segment=12):
    """Centripetal Catmull-Rom; duplicated controls rejected, linear endpoint extension."""
    p=np.asarray(points,float)
    if p.ndim!=2 or len(p)<2 or not np.isfinite(p).all():raise ValueError('Invalid curve controls')
    if np.any(np.linalg.norm(np.diff(p,axis=0),axis=1)<1e-8):raise ValueError('Repeated curve controls')
    padded=np.vstack((2*p[0]-p[1],p,2*p[-1]-p[-2]));result=[]
    for k in range(len(p)-1):
        a,b,c,d=padded[k:k+4]
        t=np.r_[0,np.cumsum(np.sqrt(np.linalg.norm(np.diff([a,b,c,d],axis=0),axis=1)))]
        x=np.linspace(t[1],t[2],samples_per_segment,endpoint=False)[:,None]
        a1=(t[1]-x)/(t[1]-t[0])*a+(x-t[0])/(t[1]-t[0])*b
        a2=(t[2]-x)/(t[2]-t[1])*b+(x-t[1])/(t[2]-t[1])*c
        a3=(t[3]-x)/(t[3]-t[2])*c+(x-t[2])/(t[3]-t[2])*d
        b1=(t[2]-x)/(t[2]-t[0])*a1+(x-t[0])/(t[2]-t[0])*a2
        b2=(t[3]-x)/(t[3]-t[1])*a2+(x-t[1])/(t[3]-t[1])*a3
        result.extend((t[2]-x)/(t[2]-t[1])*b1+(x-t[1])/(t[2]-t[1])*b2)
    return np.vstack((result,p[-1]))


def taper(mesh,boundary,width):
    edges=mesh.edges_unique;distance=np.linalg.norm(mesh.vertices[edges[:,0]]-mesh.vertices[edges[:,1]],axis=1)
    n=len(mesh.vertices)
    g=sparse.coo_matrix((np.r_[distance,distance],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    d=dijkstra(g,directed=False,indices=np.unique(boundary),min_only=True)
    if not np.isfinite(d).all():raise ValueError('Hair component has no boundary anchor')
    t=np.clip(d/width,0,1);return t*t*(3-2*t)


def rbf_basis(points,eligible,count=10,mode="gaussian"):
    ids=np.flatnonzero(eligible)
    if len(ids)<count:raise ValueError('Not enough supported hair volume controls')
    centers=[int(ids[np.argmax(points[ids,1])])];dist=np.full(len(points),np.inf)
    for _ in range(count-1):
        dist=np.minimum(dist,np.linalg.norm(points-points[centers[-1]],axis=1))
        centers.append(int(ids[np.argmax(dist[ids])]))
    centers=np.asarray(centers);radius=max(np.ptp(points,axis=0).max()*.4,1e-6)
    distances=np.linalg.norm(points[:,None]-points[centers][None],axis=2)
    if mode=='compact':
        radius=max(float(distances.min(1).max())*1.2,1e-6)
        relative=distances/radius
        b=np.maximum(1-relative,0)**4*(4*relative+1)
    elif mode=='gaussian':b=np.exp(-(distances**2)/(2*radius**2))
    else:raise ValueError('Unknown RBF mode')
    b/=np.maximum(b.sum(1,keepdims=True),1e-12)
    return b,centers


def fit_offsets(patch,normals,t,known,cameras,masks,depths,base,seam,settings,loss):
    """Fit a low-dimensional offset volume to visible near-tangent photo boundaries.

    This fits a proxy contour, not a differentiable rasterizer. No strand direction
    or hidden hairstyle is claimed recovered. Both loss choices use identical data.
    """
    from figurine_hairline32 import project
    points=np.asarray(patch.vertices);basis,controls=rbf_basis(points,known&(t>.5),settings['controls'],settings.get('basis_mode','gaussian'))
    records=[];counts=[]
    for camera,mask,depth in zip(cameras,masks,depths):
        center=-np.linalg.solve(camera[:,:3],camera[:,3]);view=center-points
        view/=np.maximum(np.linalg.norm(view,axis=1,keepdims=True),1e-12)
        facing=np.einsum('ij,ij->i',normals,view)
        xy,z=project(points,camera);ix=np.floor(xy[:,0]).astype(int);iy=np.floor(xy[:,1]).astype(int)
        valid=(z>0)&(ix>=0)&(ix<mask.shape[1])&(iy>=0)&(iy<mask.shape[0])&(facing>-.05)&(facing<.35)&(t>.7)&known
        ids=np.flatnonzero(valid)
        ids=ids[np.isfinite(depth[iy[ids],ix[ids]])&(z[ids]<=depth[iy[ids],ix[ids]]+.4*np.linalg.norm(camera[2,:3]))]
        ids=ids[np.linspace(0,max(len(ids)-1,0),min(settings['samples_per_view'],len(ids))).astype(int)] if len(ids) else ids
        signed=ndimage.distance_transform_edt(mask)-ndimage.distance_transform_edt(~mask)
        records.append((camera,signed,ids));counts.append(len(ids))
    if sum(counts)<12:raise ValueError('Too few visible tangent samples for a photo-fitted offset model')
    evaluations=0
    def offsets(theta):return seam+(basis@theta-seam)*t
    def residual(theta):
        nonlocal evaluations
        evaluations+=1;outer=points+normals*offsets(theta)[:,None];res=[]
        for view_index,(camera,signed,ids) in enumerate(records):
            xy,z=project(outer[ids],camera)
            # Out-of-frame movement must be penalized, never clipped away silently.
            sample=ndimage.map_coordinates(signed,[xy[:,1],xy[:,0]],order=1,mode='nearest')
            outside=np.maximum(0,-xy[:,0])+np.maximum(0,xy[:,0]-(signed.shape[1]-1))+np.maximum(0,-xy[:,1])+np.maximum(0,xy[:,1]-(signed.shape[0]-1))
            view_weight=settings.get('view_weights',[1.]*len(records))[view_index]
            res.extend((sample+outside)/settings['pixel_scale']*np.sqrt(view_weight))
        res.extend((theta-base)/settings['prior_mm'])
        return np.asarray(res)
    initial=np.full(settings['controls'],base)
    before=residual(initial)
    start=time.perf_counter()
    fit=least_squares(residual,initial,bounds=(settings['min_offset_mm'],settings['max_offset_mm']),method='trf',loss=loss,f_scale=1.,max_nfev=settings['max_nfev'],ftol=1e-5,xtol=1e-5,gtol=1e-5)
    if not np.isfinite(fit.x).all():raise ValueError('Nonfinite offset fit')
    return points+normals*offsets(fit.x)[:,None],{'loss':loss,'optimizer':'bounded TRF','theta_mm':fit.x.tolist(),'control_vertex_ids':controls.tolist(),'sample_counts_per_view':counts,'converged':bool(fit.success),'message':fit.message,'residual_calls':evaluations,'max_nfev':settings['max_nfev'],'fit_seconds':time.perf_counter()-start,'initial_residual_rms':float(np.sqrt(np.mean(before**2))),'final_residual_rms':float(np.sqrt(np.mean(residual(fit.x)**2))),'scope':'photo-mask tangent-contour proxy plus offset prior; robust loss applies to the entire residual vector','source_ids':['scipy_ls']}


def normal_ray_screen(mesh,samples=128,threshold_mm=.4):
    """Deterministic inward-normal ray chords. NOT minimum local thickness or CGAL SDF.

    Moller-Trumbore ray/triangle intersections, chunked to bound memory. Thin
    features can be missed; no pass/fail printability certification is emitted.
    """
    if not mesh.is_watertight:return {'status':'SKIPPED_OPEN_MESH','minimum_wall_thickness_verified':False}
    ids=np.unique(np.linspace(0,len(mesh.faces)-1,min(samples,len(mesh.faces))).astype(int))
    starts=mesh.triangles_center[ids];directions=-mesh.face_normals[ids]
    triangles=np.asarray(mesh.triangles);a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    chords=[];eps=max(mesh.extents.max()*1e-7,1e-8)
    for start,direction in zip(starts,directions):
        origin=start+direction*eps
        h=np.cross(direction,e2);det=np.einsum('ij,ij->i',e1,h)
        valid=abs(det)>1e-12;inv=np.divide(1,det,out=np.zeros_like(det),where=valid)
        s=origin-a;u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1)
        v=q@direction*inv;distance=np.einsum('ij,ij->i',e2,q)*inv
        hit=valid&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(distance>eps)
        chords.append(float(distance[hit].min()+eps) if hit.any() else None)
    finite=np.asarray([x for x in chords if x is not None])
    return {'status':'SCREEN_ONLY','method':'inward-normal first-exit chords, deterministic face samples','sampled_faces':len(ids),'unresolved_rays':sum(x is None for x in chords),'min_sampled_chord_mm':float(finite.min()) if len(finite) else None,'p05_sampled_chord_mm':float(np.quantile(finite,.05)) if len(finite) else None,'sampled_chords_below_trial_threshold':int((finite<threshold_mm).sum()),'trial_threshold_mm':threshold_mm,'minimum_wall_thickness_verified':False,'strength_verified':False,'scope':'sampling proxy; may miss thin features, no physical printability implication'}
