"""Recognize only the expected FLAME three-component head/eyeball topology."""
import numpy as np


def identify(vertices):
    points, inverse = np.unique(vertices, axis=0, return_inverse=True)
    parent = np.arange(len(points))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a,b,c in inverse.reshape(-1,3):
        parent[find(b)] = find(a)
        parent[find(c)] = find(a)
    roots=np.array([find(i) for i in range(len(points))])
    groups=sorted((np.flatnonzero(roots==r) for r in np.unique(roots)),key=len,reverse=True)
    if [len(x) for x in groups] != [3931,546,546]:
        raise ValueError('Eye split requires verified FLAME component sizes 3931/546/546')
    head_extent=np.ptp(points[groups[0]],axis=0)
    eyes=sorted(groups[1:],key=lambda g:points[g,0].mean())
    centers=[points[g].mean(0) for g in eyes]
    for g in eyes:
        size=np.ptp(points[g],axis=0)
        if size.max()/size.min()>1.5 or not .05<size.max()/head_extent.max()<.2:
            raise ValueError('Unexpected eyeball proportions; refusing automatic split')
    if not .1<(centers[1][0]-centers[0][0])/head_extent[0]<.5:
        raise ValueError('Unexpected eye separation')
    labels=np.zeros(len(points),dtype=int)
    for i,g in enumerate(eyes,1):labels[g]=i
    face_labels=labels[inverse].reshape(-1,3)
    if not (face_labels==face_labels[:,:1]).all():raise ValueError('Mixed component triangle')
    return face_labels[:,0],dict(component_vertices=[3931,546,546],
        object_names=['Head','Eye_ModelXNegative','Eye_ModelXPositive'],
        eye_centers_native=[x.tolist() for x in centers],
        eye_positions_modified=False,gaze_estimated=False,
        anatomical_left_right_verified=False)
