"""Geometry-only operations for the versioned graduation preview preset."""
import numpy as np


AXES = np.array([[1., 0., 0., 0.], [0., 0., -1., 0.],
                 [0., 1., 0., 0.], [0., 0., 0., 1.]])


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def eye_alignment(vertices, faces, embedding, reference_vertices, reference_matrix):
    """Uniform scale and eye anchoring, preserving the new person's head shape."""
    ids = [36, 39, 42, 45]
    triangles = np.asarray(embedding['full_faces'])[ids]
    weights = np.asarray(embedding['full_weights'])[ids]
    faces = np.asarray(faces)
    if not np.isfinite(weights).all() or not np.allclose(weights.sum(1), 1):
        raise ValueError('Invalid landmark binding')
    def landmarks(v):
        return np.einsum('ij,ijk->ik', weights, np.asarray(v)[faces[triangles]])
    source = transform_points(landmarks(vertices), AXES)
    reference = transform_points(landmarks(reference_vertices), AXES)
    span = np.linalg.norm(source[:2].mean(0) - source[2:].mean(0))
    ref_span = np.linalg.norm(reference[:2].mean(0) - reference[2:].mean(0))
    if not np.isfinite([span, ref_span]).all() or min(span, ref_span) < 1e-6:
        raise ValueError('Degenerate eye landmarks')
    ratio = ref_span / span
    if not .7 <= ratio <= 1.4:
        raise ValueError('Head is outside this preset size range')
    reference_matrix = np.asarray(reference_matrix, float)
    matrix = reference_matrix.copy()
    matrix[:3, :3] *= ratio
    target = transform_points(reference.mean(0, keepdims=True), reference_matrix)[0]
    matrix[:3, 3] = target - matrix[:3, :3] @ source.mean(0)
    return matrix @ AXES


def upper_bounds(vertices):
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid head geometry')
    # RDF/glTF Y-up: fit the scalp, excluding the flared neck base.
    upper = vertices[vertices[:, 1] >= np.quantile(vertices[:, 1], .45)]
    bounds = np.array([upper.min(0), upper.max(0)])
    if np.any(np.ptp(bounds, axis=0) < 1e-6):
        raise ValueError('Degenerate scalp bounds')
    return bounds


def fit_library_hair(hair, reference_head, new_head):
    """Fit a designed hairstyle; never deform personalized head geometry."""
    source = upper_bounds(reference_head)
    target = upper_bounds(new_head)
    scale = np.ptp(target, axis=0) / np.ptp(source, axis=0)
    if np.any((scale < .65) | (scale > 1.5)):
        raise ValueError('Hair preset is outside the supported head range')
    result = (np.asarray(hair) - source.mean(0)) * scale + target.mean(0)
    return result, scale
