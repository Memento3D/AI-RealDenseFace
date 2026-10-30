"""Bounded ARAP on mesh edges or a layer-separated control graph.

The control graph is a numerical approximation, not a printable remesh or an
injectivity/volume certificate. Original dense faces and vertex order remain.
"""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree


def mesh_edges(faces):
 f=np.asarray(faces,int)
 if f.ndim!=2 or f.shape[1]!=3 or not len(f):raise ValueError('Invalid ARAP faces')
 return np.unique(np.sort(np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0)


def arap_edges(vertices,edges,handles,targets,iterations=8,max_displacement=2.):
 p=np.asarray(vertices,float);edges=np.asarray(edges,int);h=np.asarray(handles,int);target=np.asarray(targets,float)
 if p.ndim!=2 or p.shape[1]!=3 or not len(p) or len(p)>60000 or target.shape!=(len(h),3) or len(h)<3 or not np.isfinite(p).all() or not np.isfinite(target).all():raise ValueError('Invalid ARAP geometry/handles or control budget exceeded')
 if edges.ndim!=2 or edges.shape[1]!=2 or not len(edges) or edges.min()<0 or edges.max()>=len(p):raise ValueError('Invalid ARAP edges')
 if len(np.unique(h))!=len(h) or h.min()<0 or h.max()>=len(p) or not 1<=iterations<=30 or not np.isfinite(max_displacement) or max_displacement<=0:raise ValueError('Invalid ARAP indices/budget')
 edges=np.unique(np.sort(edges,axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]];i=np.r_[edges[:,0],edges[:,1]];j=np.r_[edges[:,1],edges[:,0]]
 W=sparse.coo_matrix((np.ones(len(i)),(i,j)),shape=(len(p),len(p))).tocsr();L=sparse.diags(np.asarray(W.sum(1)).ravel())-W
 count,components=connected_components(W,directed=False)
 if len(np.unique(components[h]))!=count:raise ValueError('ARAP component has no handle; refusing singular unconstrained solve')
 free=np.setdiff1d(np.arange(len(p)),h);q=p.copy();q[h]=target;history=[]
 if len(free):
  solve=factorized(L[free][:,free].tocsc());boundary=L[free][:,h]@target;delta=p[i]-p[j]
  for iteration in range(iterations):
   S=np.zeros((len(p),3,3));np.add.at(S,i,delta[:,:,None]*(q[i]-q[j])[:,None,:]);u,s,vt=np.linalg.svd(S)
   R=vt.transpose(0,2,1)@u.transpose(0,2,1);neg=np.linalg.det(R)<0;vt[neg,-1]*=-1;R=vt.transpose(0,2,1)@u.transpose(0,2,1)
   rhs=np.zeros_like(p);np.add.at(rhs,i,np.einsum('nij,nj->ni',.5*(R[i]+R[j]),delta))
   for axis in range(3):q[free,axis]=solve(rhs[free,axis]-boundary[:,axis])
   q[h]=target
   error=(q[i]-q[j])-np.einsum('nij,nj->ni',R[i],delta);history.append(float(np.mean(np.sum(error**2,axis=1))))
 distance=np.linalg.norm(q-p,axis=1)
 if not np.isfinite(q).all() or distance.max()>max_displacement:raise ValueError('ARAP solve exceeded bounded movement or became nonfinite')
 return q,{'method':'uniform-edge local/global ARAP with hard Dirichlet handles','iterations':iterations if len(free) else 0,'energy_trace':history,'max_displacement_mm':float(distance.max()),'handle_count':len(h),'volume_preservation_verified':False,'self_intersections_verified':False,'printability_verified':False}


def arap(vertices,faces,handles,targets,iterations=8,max_displacement=2.):
 return arap_edges(vertices,mesh_edges(faces),handles,targets,iterations,max_displacement)


def control_graph(vertices,faces,layers,cell_mm=.55,max_nodes=6000):
 """Cluster by position AND shell layer; graph edges derive from actual topology."""
 p=np.asarray(vertices,float);layers=np.asarray(layers,int)
 if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or layers.shape!=(len(p),) or not .1<=cell_mm<=2 or not 32<=max_nodes<=12000:raise ValueError('Invalid control graph inputs or budget')
 edges=mesh_edges(faces)
 if edges.min()<0 or edges.max()>=len(p):raise ValueError('Invalid dense mesh indices')
 for attempt in range(12):
  keys=np.column_stack([np.floor((p-p.min(0))/cell_mm).astype(np.int64),layers]);unique,inverse=np.unique(keys,axis=0,return_inverse=True)
  if len(unique)<=max_nodes:break
  cell_mm*=1.2
 else:raise ValueError('Unable to meet control graph budget')
 n=len(unique);counts=np.bincount(inverse,minlength=n);nodes=np.column_stack([np.bincount(inverse,weights=p[:,axis],minlength=n)/counts for axis in range(3)])
 links=np.unique(np.sort(inverse[edges],axis=1),axis=0);links=links[links[:,0]!=links[:,1]]
 return {'vertices':nodes,'edges':links,'inverse':inverse,'counts':counts,'layers':unique[:,-1],'cell_mm':cell_mm,'source_vertex_count':len(p)}


def graph_targets(graph,targets):
 target=np.asarray(targets,float);ids=graph['inverse'];n=len(graph['vertices'])
 if target.shape!=(len(ids),3) or not np.isfinite(target).all():raise ValueError('Invalid dense control targets')
 return np.column_stack([np.bincount(ids,weights=target[:,axis],minlength=n)/graph['counts'] for axis in range(3)])


def transfer_displacement(vertices,layers,graph,moved):
 p=np.asarray(vertices,float);moved=np.asarray(moved,float);layers=np.asarray(layers,int);delta=moved-graph['vertices'];out=p.copy()
 if moved.shape!=graph['vertices'].shape or not np.isfinite(moved).all() or layers.shape!=(len(p),):raise ValueError('Invalid graph displacement')
 for layer in np.unique(layers):
  ids=np.flatnonzero(layers==layer);nodes=np.flatnonzero(graph['layers']==layer)
  if not len(nodes):raise ValueError('Missing corresponding control layer')
  distance,nearest=cKDTree(graph['vertices'][nodes]).query(p[ids],k=min(4,len(nodes)))
  if distance.ndim==1:distance=distance[:,None];nearest=nearest[:,None]
  weights=1/np.maximum(distance,1e-6)**2;weights/=weights.sum(1,keepdims=True)
  out[ids]+=np.einsum('nk,nkj->nj',weights,delta[nodes[nearest]])
 return out


def orientation_blend(reference,candidate,faces,min_fraction=.125):
 """Backtrack the full residual, retaining a spatially continuous deformation."""
 ref=np.asarray(reference,float);candidate=np.asarray(candidate,float);f=np.asarray(faces,int)
 if ref.shape!=candidate.shape or not np.isfinite(ref).all() or not np.isfinite(candidate).all() or not 0<min_fraction<=1:raise ValueError('Invalid orientation blend inputs')
 def normals(v):
  t=v[f];return np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
 a=normals(ref);valid=np.linalg.norm(a,axis=1)>1e-12;history=[];fraction=1.
 while fraction>=min_fraction-1e-12:
  q=ref+fraction*(candidate-ref);b=normals(q);flipped=int(np.sum(np.einsum('ij,ij->i',a[valid],b[valid])<=0));collapsed=int(np.sum(np.linalg.norm(b[valid],axis=1)<=1e-12));history.append({'fraction':fraction,'flipped':flipped,'collapsed':collapsed})
  if not flipped and not collapsed:return q,{'retained_arap_fraction':fraction,'backtracking':history,'scope':'local triangle orientation relative to radial reference; not self-intersection certification'}
  fraction*=.5
 raise ValueError('ARAP residual cannot preserve local radial-reference orientation within blend budget: '+str(history))


def smooth_residual(reference,candidate,faces,locked,iterations=24):
 ref=np.asarray(reference,float);delta=np.asarray(candidate,float)-ref;locked=np.asarray(locked,int);edges=mesh_edges(faces);i=np.r_[edges[:,0],edges[:,1]];j=np.r_[edges[:,1],edges[:,0]]
 W=sparse.coo_matrix((np.ones(len(i)),(i,j)),shape=(len(ref),len(ref))).tocsr();W=sparse.diags(1/np.maximum(np.asarray(W.sum(1)).ravel(),1))@W
 delta[locked]=0
 for step in range(iterations):delta=.5*delta+.5*(W@delta);delta[locked]=0
 return ref+delta


def fit_style_arap(fitted,target,correspondence,settings):
 started=time.perf_counter();p=fitted.vertices;center=np.asarray(settings['origin']);r=np.linalg.norm(p-center,axis=1);wall=np.maximum(r-correspondence['inner'],0)
 # Inner/outer vertices that are geometrically close must not share a control.
 # Separate intermediate shell depths too; no resampling of dense mesh faces.
 layers=np.where(wall<.06,0,np.where(wall>.4,2,1));inner=np.flatnonzero(layers==0)
 graph=control_graph(p,fitted.faces,layers,settings.get('cell_mm',.55),settings.get('max_nodes',6000));dest=graph_targets(graph,target.vertices)
 inner_nodes=np.flatnonzero(graph['layers']==0);outer=np.flatnonzero(graph['layers']==2);outer=outer[::max(1,len(outer)//160)];handles=np.unique(np.r_[inner_nodes,outer])
 if len(handles)<3:raise ValueError('Insufficient observed template contact/control handles')
 cq,note=arap_edges(graph['vertices'],graph['edges'],handles,dest[handles],settings['iterations'],settings['max_displacement_mm'])
 q=transfer_displacement(p,layers,graph,cq)
 # Keep the contact/rim band and sharp moulded creases on the proven radial
 # reference; use ARAP only for the remaining coarse form residual.
 crease_faces=target.face_adjacency[target.face_adjacency_angles>np.pi/3]
 crease=np.unique(fitted.faces[crease_faces].ravel()) if len(crease_faces) else np.array([],int)
 locked=np.unique(np.r_[np.flatnonzero(layers!=2),crease]);q[locked]=target.vertices[locked]
 q=smooth_residual(target.vertices,q,fitted.faces,locked)
 q,blend=orientation_blend(target.vertices,q,fitted.faces,settings.get('min_fraction',.125))
 distance=np.linalg.norm(q-p,axis=1)
 if not np.isfinite(q).all() or distance.max()>settings['max_displacement_mm']:raise ValueError('Dense ARAP transfer exceeded movement budget')
 result=fitted.copy();result.vertices=q
 if not result.is_volume:raise ValueError('ARAP candidate lost closed positive volume')
 note.update(control_nodes=len(graph['vertices']),control_edges=len(graph['edges']),source_vertices=len(p),cell_mm=graph['cell_mm'],control_layer_counts={str(i):int((graph['layers']==i).sum()) for i in np.unique(layers)},transfer='4-nearest inverse-distance displacement within shell layer; inner dense vertices exactly reattached to radial target',dense_max_displacement_mm=float(distance.max()),dense_topology_preserved=True,triangle_flips_vs_radial=0,radial_locked_vertices=len(locked),crease_anchor_angle_degrees=60,residual_smoothing_iterations=24,orientation_blend=blend,radial_residual_p95_mm=float(np.quantile(np.linalg.norm(q-target.vertices,axis=1),.95)),seconds=time.perf_counter()-started)
 return result,note
