"""Conservative coverage fade and approximate hidden skin fill; never alters geometry."""
from collections import deque
import numpy as np


def prepare(vertices, weights, coordinates, images, gains, skin, sample, linear, corner_gains=None, source_bias=None):
    # Weld exact duplicates introduced by the triangle-expanded workspace.
    points, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    edges = np.unique(np.sort(edges,axis=1),axis=0)
    src = np.concatenate((edges[:,0],edges[:,1]))
    dst = np.concatenate((edges[:,1],edges[:,0]))
    degree = np.bincount(src,minlength=len(points))
    active = weights.sum((1,2)) > 1e-8
    yes = np.zeros(len(points),bool); no = yes.copy()
    yes[faces[active].ravel()] = True
    no[faces[~active].ravel()] = True
    boundary = yes & no
    adjacency = [[] for _ in points]
    for a,b in edges:
        adjacency[a].append(b); adjacency[b].append(a)
    distance = np.full(len(points),1000.,dtype=float)
    distance[boundary] = 0
    queue = deque(np.flatnonzero(boundary).tolist())
    while queue:
        a=queue.popleft()
        for b in adjacency[a]:
            if yes[b] and distance[b]>distance[a]+1:
                distance[b]=distance[a]+1;queue.append(b)
    alpha = np.clip(distance[inverse].reshape(-1,3)/2.,0,1)*active[:,None]
    # Average observed corner colors at welded vertices, then smoothly extend
    # only plausible skin colors into unobserved regions. This is a material
    # approximation, not reconstruction of hidden photographs or albedo.
    flat_weights=weights.reshape(-1,3)
    total=flat_weights.sum(1)
    color=np.zeros((len(vertices),3))
    for j in range(3):
        q=coordinates[j].reshape(-1,3)
        rgb=sample(images[j],q[:,:2]/q[:,2,None])
        corrected=linear(rgb)*(gains[j] if corner_gains is None else corner_gains[j])
        if source_bias is not None:corrected=np.maximum(corrected+source_bias[j],0)
        color+=corrected*flat_weights[:,j,None]
    color/=np.maximum(total[:,None],1e-12)
    color=np.where(color<=.0031308,color*12.92,1.055*np.maximum(color,0)**(1/2.4)-.055)
    counts=np.bincount(inverse,weights=(total>1e-8),minlength=len(points))
    observed=np.stack([np.bincount(inverse,weights=color[:,k]*(total>1e-8),minlength=len(points)) for k in range(3)],axis=1)/np.maximum(counts[:,None],1)
    plausible=(counts>0)&(np.linalg.norm(observed-skin,axis=1)<.35)
    anchor=np.clip(observed, np.maximum(skin-.16,0),np.minimum(skin+.16,1))
    fill=np.tile(skin,(len(points),1))
    strength=plausible.astype(float)*4
    for _ in range(120):
        neighbor=np.stack([np.bincount(src,weights=fill[dst,k],minlength=len(points)) for k in range(3)],axis=1)/np.maximum(degree[:,None],1)
        fill=(neighbor+.02*skin+strength[:,None]*anchor)/(1.02+strength[:,None])
    return alpha, fill[inverse].reshape(-1,3,3), dict(
        method='two_edge_ring_inward_fade_and_screened_skin_fill',
        boundary_vertices=int(boundary.sum()), textured_triangles=int(active.sum()),
        geometry_modified=False, new_photo_projection=False,
        hidden_skin_fill='approximate smooth material extension; not recovered detail')
