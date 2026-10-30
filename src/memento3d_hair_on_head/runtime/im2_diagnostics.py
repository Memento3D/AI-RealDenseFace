"""CPU projection diagnostics. Pixel support is a sparse proxy, not rendered hair quality."""
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt


def project(points, camera):
    points = np.asarray(points, dtype=np.float64)
    camera = np.asarray(camera, dtype=np.float64)
    if points.shape[-1] != 3 or camera.shape != (3, 4):
        raise ValueError('Expected XYZ points and a 3x4 camera')
    if not np.isfinite(points).all() or not np.isfinite(camera).all():
        raise ValueError('Non-finite camera or geometry')
    homogeneous = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), axis=-1)
    q = homogeneous @ camera.T
    z = q[..., 2]
    valid = z > 1e-8
    uv = np.full(q.shape[:-1] + (2,), np.nan)
    np.divide(q[..., :2], z[..., None], out=uv, where=valid[..., None])
    return uv, z, valid


def read_obj(path):
    """Read geometry without applying scene transforms or reordering vertices."""
    vertices, faces = [], []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'v':
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == 'f':
            ids = [int(v.split('/')[0]) for v in fields[1:]]
            ids = [i-1 if i > 0 else len(vertices)+i for i in ids]
            faces.extend((ids[0], ids[i], ids[i+1]) for i in range(1, len(ids)-1))
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2:
        raise ValueError('OBJ has no triangle surface')
    if faces.min() < 0 or faces.max() >= len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Invalid OBJ indices/vertices')
    return vertices, faces


def surface_depth(vertices, faces, camera, size):
    """Two-sided triangle depth in the camera's homogeneous Z units."""
    w, h = size
    uv, z, valid = project(vertices, camera)
    depth = np.full((h, w), np.inf)
    for ids in faces:
        if not valid[ids].all():
            continue
        tri, tz = uv[ids], z[ids]
        lo = np.maximum(np.floor(tri.min(0)).astype(int), [0, 0])
        hi = np.minimum(np.ceil(tri.max(0)).astype(int), [w-1, h-1])
        if np.any(lo > hi):
            continue
        a, b, c = tri
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-10:
            continue
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        aa = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        bb = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        cc = 1-aa-bb
        inside = (aa >= -1e-8) & (bb >= -1e-8) & (cc >= -1e-8)
        invz = aa/tz[0]+bb/tz[1]+cc/tz[2]
        candidate = np.full_like(invz, np.inf)
        np.divide(1, invz, out=candidate, where=inside & (invz > 0))
        area = depth[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        np.minimum(area, candidate, out=area)
    return depth


def hair_support(strands, camera, size, head_depth=None):
    """Eight samples per segment; head-occluded samples omitted, no hair self-occlusion."""
    strands = np.asarray(strands)
    if strands.ndim != 3 or strands.shape[-1] != 3 or strands.shape[1] < 2:
        raise ValueError('Expected S x P x 3 strands')
    w, h = size
    flat = strands.reshape(-1, 3)
    _, _, positive = project(flat, camera)
    depth = np.full((h, w), np.inf)
    for t in np.linspace(0, 1, 8):
        samples = strands[:, :-1] * (1-t) + strands[:, 1:] * t
        uv, z, valid = project(samples.reshape(-1, 3), camera)
        in_frame = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        xy = np.floor(uv[in_frame]).astype(int)
        np.minimum.at(depth, (xy[:, 1], xy[:, 0]), z[in_frame])
    support = np.isfinite(depth)
    occluded = np.zeros_like(support)
    if head_depth is not None:
        finite_head = head_depth[np.isfinite(head_depth)]
        tolerance = float(np.median(finite_head) * .002) if finite_head.size else 0
        occluded = support & np.isfinite(head_depth) & (depth > head_depth + tolerance)
        support &= ~occluded
    return support, {
        'positive_depth_point_fraction': float(positive.mean()),
        'projected_sample_pixels': int(np.isfinite(depth).sum()),
        'head_occluded_sample_pixels': int(occluded.sum()),
        'samples_per_segment': 8,
        'hair_self_occlusion': False,
        'head_depth_tolerance_fraction': .002,
    }


def support_metrics(support, mask, radius):
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    raster = binary_dilation(support, structure=xx*xx+yy*yy <= radius*radius)
    intersection = (raster & mask).sum()
    union = (raster | mask).sum()
    outside_dist = distance_transform_edt(~mask)
    missing_dist = distance_transform_edt(~support) if support.any() else None
    return raster, {
        'radius_px': radius,
        'sparse_support_iou_proxy': float(intersection / union) if union else None,
        'mask_covered_fraction_proxy': float(intersection / mask.sum()) if mask.any() else None,
        'support_inside_mask_fraction_proxy': float(intersection / raster.sum()) if raster.any() else None,
        'mean_centerline_distance_to_mask_px': float(outside_dist[support].mean()) if support.any() else None,
        'mean_mask_distance_to_centerline_px': float(missing_dist[mask].mean()) if missing_dist is not None and mask.any() else None,
    }


def overlay(image, mask, support):
    rgb = np.asarray(image.convert('RGB'), dtype=float).copy()
    rgb[mask] = .55 * rgb[mask] + .45 * np.array([0, 255, 0])
    rgb[support] = .35 * rgb[support] + .65 * np.array([255, 40, 180])
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))


def direction_review(dataset, mask, output):
    """Visualize the saved features; disagreement is not an accuracy measurement."""
    strand = np.asarray(Image.open(dataset / 'strand_map_aligned/front.png').convert('RGB')) / 255.
    gabor = np.asarray(Image.open(dataset / 'orientation_maps_aligned/front.png').convert('L')) / 255.
    gx, gy = 2*strand[..., 1]-1, 2*strand[..., 2]-1
    hair_angle = np.arctan2(gy, gx)
    # Same undirected angle convention used by the upstream conversion branch.
    gabor_angle = np.deg2rad((180-gabor*180) % 180)
    disagreement = np.abs((hair_angle-gabor_angle+np.pi/2) % np.pi - np.pi/2)
    def color(theta):
        hsv = np.zeros(mask.shape + (3,), dtype=np.uint8)
        hsv[..., 0] = np.uint8((theta % np.pi) / np.pi * 255)
        hsv[..., 1] = 255
        hsv[..., 2] = np.uint8(mask)*255
        return Image.fromarray(hsv, mode='HSV').convert('RGB')
    views = [Image.fromarray(np.uint8(strand*255)), color(hair_angle), color(gabor_angle)]
    canvas = Image.new('RGB', (mask.shape[1]*3, mask.shape[0]+28))
    draw = ImageDraw.Draw(canvas)
    for i, (view, title) in enumerate(zip(views, ['Saved HairStep RGB encoding', 'HairStep undirected angle', 'Gabor undirected angle'])):
        canvas.paste(view, (i*mask.shape[1], 28))
        draw.text((i*mask.shape[1]+5, 5), title, fill='white')
    canvas.save(output / 'direction_review.png')
    return {'hairstep_gabor_median_undirected_disagreement_deg': float(np.median(np.rad2deg(disagreement[mask]))),
            'hairstep_encoded_vector_length_quantiles': np.quantile(np.hypot(gx, gy)[mask], [.1, .5, .9]).tolist(),
            'scope': 'Agreement of saved maps, not ground truth accuracy; affine direction handling remains unverified',
            'maps_modified': False}


def diagnose(npz, camera, image, mask, head_depth, output):
    output.mkdir(parents=True, exist_ok=False)
    with np.load(npz, allow_pickle=False) as data:
        strands = data['strands']
    if not np.isfinite(strands).all() or len(strands) == 0:
        raise ValueError('Empty or non-finite strands')
    support, notes = hair_support(strands, camera, image.size, head_depth)
    Image.fromarray(np.uint8(support)*255).save(output / 'centerlines.png')
    sensitivity = []
    for radius in (1, 2, 3):
        raster, metrics = support_metrics(support, mask, radius)
        sensitivity.append(metrics)
        overlay(image, mask, raster).save(output / f'overlay_r{radius}.png')
    result = {
        'strand_count': int(len(strands)), 'points_per_strand': int(strands.shape[1]),
        'bounds_native': [strands.min((0, 1)).tolist(), strands.max((0, 1)).tolist()],
        'projection': notes, 'radius_sensitivity': sensitivity,
        'metric_scope': 'Sparse centerline pixel-support proxy; not Gaussian render IoU or likeness',
        'geometry_modified': False, 'quality_verified': False,
    }
    (output / 'diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def write_review(output, report):
    esc = lambda v: html.escape(str(v))
    cards = []
    for item in report['results']:
        diagnostic = item.get('diagnostic')
        images = ''
        if diagnostic:
            images = ''.join(f'<figure><img src="{esc(item["variant"])}/overlay_r{r}.png"><figcaption>Radius {r} px</figcaption></figure>' for r in (1, 2, 3))
        cards.append('<section><h2>'+esc(item['variant'])+'</h2><p>'+esc(item['status'])+'</p>'
                     +'<p>'+esc(item.get('run', ''))+'</p><div class="row">'+images+'</div>'
                     +'<details><summary>Measurements / errors</summary><pre>'
                     +esc(json.dumps(item, indent=2))+'</pre></details></section>')
    document = '''<!doctype html><meta charset="utf-8"><title>Im2Haircut review</title>
<style>body{font:16px system-ui;background:#161b22;color:#eee;max-width:1600px;margin:auto;padding:24px}section{background:#222b35;padding:18px;margin:20px 0}.row{display:flex;flex-wrap:wrap}figure{margin:5px;flex:1;min-width:240px}img{width:100%;max-width:512px}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#9df}</style>
<h1>Personal hair reconstruction: diagnostics and ablations</h1>
<p>Green = proposed photo hair mask. Magenta = projected sparse hair support after reference-head occlusion.
Compare all three pixel radii. These are diagnostic proxies, not final hair renders or automatic quality scores.</p>
<p>Head preview: blue = projected reference head, orange = aligned BFM face vertices. Neither is an independent calibration ground truth.</p>
<img src="input_review.png"><img src="head_projection.png">
<p>Direction inputs: saved encoding, HairStep undirected angle, Gabor undirected angle (same color convention).</p><img style="max-width:1500px" src="direction_review.png">
<p>No automatic winner. Back/side photos are not optimized by this single-view experiment. No FLAME attachment or recovered hair color.</p>'''
    status = '<p><strong>Suite status: ' + esc(report['status']) + '</strong></p>'
    if report.get('error'):
        status += '<p>' + esc(report['error']) + '</p>'
    (output / 'review.html').write_text(document + status + ''.join(cards), encoding='utf-8')
