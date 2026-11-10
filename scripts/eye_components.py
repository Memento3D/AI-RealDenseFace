"""Recognize only the expected FLAME three-component head/eyeball topology."""
import numpy as np


def identify(vertices, faces=None):
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError('Expected a non-empty Nx3 vertex array')
    if not np.isfinite(vertices).all():
        raise ValueError('Non-finite mesh vertices')

    if faces is None:
        points, inverse = np.unique(vertices, axis=0, return_inverse=True)
        face_vertices = inverse.reshape(-1, 3)
        vertex_ids = np.arange(len(points))
    else:
        faces = np.asarray(faces)
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu':
            raise ValueError('Expected a non-empty Mx3 integer face array')
        face_vertices = faces.astype(np.intp, copy=False)
        if face_vertices.min() < 0 or face_vertices.max() >= len(vertices):
            raise ValueError('Face index outside vertex array')
        points = vertices
        vertex_ids = np.unique(face_vertices)

    parent = np.arange(len(points))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in face_vertices:
        parent[find(b)] = find(a)
        parent[find(c)] = find(a)

    roots = np.array([find(i) for i in vertex_ids])
    groups = sorted(
        (vertex_ids[roots == root] for root in np.unique(roots)),
        key=len,
        reverse=True,
    )
    if [len(x) for x in groups] != [3931, 546, 546]:
        raise ValueError(
            'Eye split mismatch: expected component sizes [3931, 546, 546], '
            f'actual {[len(group) for group in groups]}; unique_vertices={len(vertex_ids)}'
        )

    head_extent = np.ptp(points[groups[0]], axis=0)
    if not np.isfinite(head_extent).all() or head_extent.min() <= 0:
        raise ValueError('Degenerate head component')
    eyes = sorted(groups[1:], key=lambda group: points[group, 0].mean())
    centers = [points[group].mean(0) for group in eyes]
    for group in eyes:
        size = np.ptp(points[group], axis=0)
        if size.min() <= 0 or size.max()/size.min() > 1.5 or not .05 < size.max()/head_extent.max() < .2:
            raise ValueError('Unexpected eyeball proportions; refusing automatic split: '
                f'extent={size.tolist()}, axis_ratio={size.max()/max(size.min(), 1e-12):.6f}, '
                f'eye_to_head={size.max()/head_extent.max():.6f}')
    if not .1 < (centers[1][0]-centers[0][0])/head_extent[0] < .5:
        raise ValueError('Unexpected eye separation')

    labels = np.zeros(len(points), dtype=int)
    for index, group in enumerate(eyes, 1):
        labels[group] = index
    face_labels = labels[face_vertices]
    if not (face_labels == face_labels[:, :1]).all():
        raise ValueError('Mixed component triangle')
    return face_labels[:, 0], dict(
        component_vertices=[3931, 546, 546],
        object_names=['Head', 'Eye_ModelXNegative', 'Eye_ModelXPositive'],
        eye_centers_native=[center.tolist() for center in centers],
        eye_positions_modified=False,
        gaze_estimated=False,
        anatomical_left_right_verified=False,
    )
