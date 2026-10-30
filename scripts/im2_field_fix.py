"""Transform encoded orientation fields without averaging scalar angles across wrap boundaries."""
import numpy as np
import cv2
from PIL import Image, ImageDraw

from im2_diagnostics import project


def check_affine(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 2x3 affine')
    linear = matrix[:, :2]
    if abs(np.linalg.det(linear)) < 1e-10:
        raise ValueError('Singular affine')
    # The current recipe is audited for positive-scale similarity transforms only.
    gram = linear.T @ linear
    if np.linalg.det(linear) <= 0 or not np.allclose(gram, np.eye(2)*np.trace(gram)/2, rtol=1e-5, atol=1e-8):
        raise ValueError('Expected a non-reflecting similarity alignment')
    return matrix


def warp_field(theta, source_mask, matrix, output_size, axial):
    """Angles use x-right/y-down coordinates. Axial fields have period pi; directed fields 2pi."""
    matrix = check_affine(matrix)
    theta, weight = np.asarray(theta, dtype=float), np.asarray(source_mask, dtype=float)
    if theta.shape != weight.shape or not np.isfinite(theta).all():
        raise ValueError('Invalid angle map or source mask')
    vec = np.stack((np.cos(theta), np.sin(theta)), -1) @ matrix[:, :2].T
    transformed = np.arctan2(vec[..., 1], vec[..., 0])
    period_factor = 2 if axial else 1
    def warp(array):
        return cv2.warpAffine(np.asarray(array, dtype=np.float32), matrix, output_size,
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    den = warp(weight)
    c = warp(np.cos(period_factor*transformed)*weight)
    s = warp(np.sin(period_factor*transformed)*weight)
    angle = np.arctan2(s, c)/period_factor
    strength = np.divide(np.hypot(c, s), den, out=np.zeros_like(den), where=den > 1e-6)
    # A cancelled circular mean has no defined direction. Retain a transformed
    # nearest source angle rather than silently inventing atan2(0, 0) = 0.
    fallback = cv2.warpAffine(transformed.astype(np.float32), matrix, output_size,
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    angle = np.where((strength < 1e-4) & (den > 1e-6), fallback, angle)
    return angle, strength, den > 1e-6


def correct_fields(raw_gabor, raw_strand, raw_mask, aligned_strand, aligned_mask, matrix):
    """Preserve camera/masks and saved R channel. Change only direction encodings inside hair."""
    h, w = aligned_mask.shape
    if raw_gabor.shape != raw_mask.shape or raw_strand.shape != raw_mask.shape + (3,):
        raise ValueError('Mismatched raw feature shapes')
    if aligned_strand.shape != (h, w, 3):
        raise ValueError('Mismatched aligned strand shape')
    # Upstream converts encoded E to image-plane direction phi = pi-E in its Gabor branch.
    phi = np.pi - raw_gabor.astype(float)/255*np.pi
    gtheta, gstrength, gvalid = warp_field(phi, raw_mask, matrix, (w, h), axial=True)
    encoded = np.mod(-gtheta, np.pi)/np.pi*255
    gabor = np.zeros((h, w), np.uint8)
    usable = aligned_mask & gvalid
    gabor[usable] = np.uint8(np.round(encoded[usable]))
    # Testable convention hypothesis: GB channels describe x-right/y-down directed vectors.
    # It is recorded explicitly; not asserted as independently calibrated by this fix.
    theta = np.arctan2(2*raw_strand[..., 2].astype(float)/255-1,
                       2*raw_strand[..., 1].astype(float)/255-1)
    htheta, hstrength, hvalid = warp_field(theta, raw_mask, matrix, (w, h), axial=False)
    strand = aligned_strand.copy()
    usable_hair = aligned_mask & hvalid
    strand[usable_hair, 1] = np.uint8(np.round((np.cos(htheta[usable_hair])*.5+.5)*255))
    strand[usable_hair, 2] = np.uint8(np.round((np.sin(htheta[usable_hair])*.5+.5)*255))
    if np.any(aligned_mask & ~gvalid) or np.any(aligned_mask & ~hvalid):
        raise ValueError('Aligned hair contains pixels without source field support; stop before GPU')
    note = {
        'gabor_method': 'masked double-angle resampling, similarity vector transform, re-encode pi-angle',
        'hairstep_method': 'masked directed-vector resampling, similarity vector transform, unit re-encode',
        'hairstep_coordinate_convention_hypothesis': 'x right, y down; consistent with upstream Gabor conversion target, not independently calibrated',
        'gabor_support_missing_inside_mask': int((aligned_mask & ~gvalid).sum()),
        'hairstep_support_missing_inside_mask': int((aligned_mask & ~hvalid).sum()),
        'gabor_resultant_length_quantiles': np.quantile(gstrength[aligned_mask], [.1, .5, .9]).tolist(),
        'hairstep_resultant_length_quantiles': np.quantile(hstrength[aligned_mask], [.1, .5, .9]).tolist(),
        'mask_modified': False, 'camera_modified': False, 'depth_modified': False,
        'native_gabor_noise_recovered': False, 'quality_verified': False,
        'ambiguous_mean_fallback': 'transformed nearest source angle when resultant length < 1e-4',
    }
    return gabor, strand, note


def root_review(strands, camera, image, mask, head_depth, output):
    """Visible projected roots vs early segments vs full strands; absence is not a baldness diagnosis."""
    from im2_diagnostics import hair_support, overlay, support_metrics
    uv, z, valid = project(strands[:, 0], camera)
    h, w = mask.shape
    in_frame = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids = np.flatnonzero(in_frame)
    xy = np.floor(uv[ids]).astype(int)
    finite_head = head_depth[np.isfinite(head_depth)]
    tolerance = np.median(finite_head)*.002 if len(finite_head) else 0
    visible = z[ids] <= head_depth[xy[:, 1], xy[:, 0]]+tolerance
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    inside = mask[xy[:, 1], xy[:, 0]]
    for (x, y), front, hair in zip(xy, visible, inside):
        if front:
            draw.ellipse((x-1, y-1, x+1, y+1), fill=(0, 255, 0) if hair else (255, 70, 40))
    canvas.save(output / 'roots.png')
    early_count = max(2, int(np.ceil(strands.shape[1]*.05)))
    early, _ = hair_support(strands[:, :early_count], camera, image.size, head_depth)
    early_raster, early_metric = support_metrics(early, mask, 1)
    overlay(image, mask, early_raster).save(output / 'early_segments.png')
    return {
        'strand_count': len(strands), 'roots_in_frame': int(in_frame.sum()),
        'visible_roots': int(visible.sum()),
        'visible_roots_outside_photo_hair_mask': int((visible & ~inside).sum()),
        'early_segment_point_count': early_count, 'early_segment_proxy': early_metric,
        'interpretation': 'Roots may lie under or behind the visible hair surface; outside mask alone is not proof of wrong scalp. No roots moved or pruned.',
    }


def field_comparison(old_gabor, new_gabor, old_strand, new_strand, mask, output):
    def axial_delta(a, b):
        return np.abs((a-b+np.pi/2) % np.pi-np.pi/2)
    old_theta = np.pi-old_gabor.astype(float)/255*np.pi
    new_theta = np.pi-new_gabor.astype(float)/255*np.pi
    gabor_delta = np.rad2deg(axial_delta(old_theta, new_theta))[mask]
    def angle(rgb):
        return np.arctan2(2*rgb[..., 2].astype(float)/255-1, 2*rgb[..., 1].astype(float)/255-1)
    h_delta = np.rad2deg(np.abs((angle(old_strand)-angle(new_strand)+np.pi) % (2*np.pi)-np.pi))[mask]
    def view(theta):
        hsv = np.zeros(mask.shape+(3,), np.uint8)
        hsv[..., 0] = np.uint8(np.mod(theta, np.pi)/np.pi*255)
        hsv[..., 1] = 255; hsv[..., 2] = np.uint8(mask)*255
        return Image.fromarray(hsv, mode='HSV').convert('RGB')
    canvas = Image.new('RGB', (mask.shape[1]*2, mask.shape[0]*2+40))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), 'Gabor before / after (top); HairStep before / after (bottom)', fill='white')
    for i, theta in enumerate((old_theta, new_theta, angle(old_strand), angle(new_strand))):
        canvas.paste(view(theta), ((i % 2)*mask.shape[1], 40+(i//2)*mask.shape[0]))
    canvas.save(output / 'fields_before_after.png')
    return {'gabor_change_deg_quantiles': np.quantile(gabor_delta, [.1, .5, .9]).tolist(),
            'hairstep_change_deg_quantiles': np.quantile(h_delta, [.1, .5, .9]).tolist(),
            'scope': 'Change caused by transformation, not direction accuracy'}
