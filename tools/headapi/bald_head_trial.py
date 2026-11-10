"""Versioned bald-head appearance experiment; never changes reconstruction geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import trimesh

from scripts.eye_components import identify
from tools.headapi.demo_appearance import atlas_positions
from tools.headapi.service import sha, write
from tools.headapi.texture_completion_trial import export_texture

POLICY = 'bald_head_surface_v3'


def surface_colors(vertices, faces, samples, colors, confidence, prior):
    """Screened graph Laplacian, positive weights, separate mesh components."""
    count = len(vertices)
    ids = cKDTree(vertices).query(samples)[1]
    weight = np.bincount(ids, weights=confidence, minlength=count)
    sums = np.stack([np.bincount(ids, weights=confidence*colors[:, k], minlength=count)
                     for k in range(3)], axis=1)
    targets = np.divide(sums, weight[:, None], out=np.zeros_like(sums), where=weight[:, None]>0)
    anchors = np.minimum(weight, 1.) * 30
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    i, j = edges.T
    adjacency = coo_matrix((np.ones(len(i)*2), (np.r_[i, j], np.r_[j, i])), shape=(count, count)).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    # Constant prior stabilizes unobserved components without crossing eye/head topology.
    system = laplacian + diags(anchors + .015)
    result = spsolve(system.tocsc(), anchors[:, None]*targets + .015*np.asarray(prior))
    if not np.isfinite(result).all():
        raise ValueError('Non-finite surface color solution')
    return np.clip(result, 0, 255)


def fill_regions(positions, observed, skin, protected, head, eye, height):
    valid = np.isfinite(positions).all(-1) & head
    x, y, z = np.moveaxis(positions, -1, 0)
    # A head template prior is used above the forehead and behind the ears.
    scalp = valid & ((y > eye[1]+.075*height) |
                     ((z < eye[2]-.12*height) & (y > eye[1]-.05*height)))
    # Eye, mouth and accessory semantics remain excluded below scalp.
    missing = valid & ~observed & ((protected == 0) | scalp)
    # Continue forehead skin into the scalp, including dark hair misclassified as skin.
    replace_hair = scalp & (~skin.astype(bool) | (y > eye[1]+.075*height))
    return missing | replace_hair, scalp


def topology_report(vertices, faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    return dict(vertices=len(vertices), triangles=len(faces), boundary_edges=int((counts==1).sum()),
                nonmanifold_edges=int((counts>2).sum()), watertight=bool(mesh.is_watertight),
                winding_consistent=bool(mesh.is_winding_consistent),
                self_intersections_checked=False, minimum_wall_thickness_checked=False,
                manufacturing_ready=False)


def run(job, output):
    job, output = Path(job).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', policy=POLICY, geometry_changed=False,
                  hair_included=False, glasses_processed=False, ready_for_machine=False,
                  likeness_verified=False, skin_realism_verified=False)
    try:
        uvroot = job/'rdf_uv'
        raw = json.loads((uvroot/'raw/report.json').read_text(encoding='utf-8'))
        binding = json.loads((uvroot/'report.json').read_text(encoding='utf-8'))
        bundle = job/'rdf/projection_bundle.npz'
        assets = next(Path(p) for p in raw['input_sha256'] if Path(p).name=='flame_assets.npz')
        atlas_path = uvroot/'final/face_atlas.png'
        quality = uvroot/'raw/quality_maps.npz'
        sources = [bundle, assets, atlas_path, quality, uvroot/'report.json', uvroot/'raw/report.json',
                   uvroot/'raw/coverage.png', uvroot/'final/provenance-labels.png', uvroot/'final/confidence.png',
                   uvroot/'final/face_textured.glb']
        pins = {str(p): sha(p) for p in sources}
        if binding.get('status')!='UV_FACE_EXPORTED' or binding['input_sha256'].get(str(bundle))!=sha(bundle):
            raise ValueError('Unbound or unsuccessful UV reconstruction')
        if sha(assets)!=raw['input_sha256'][str(assets)] or sha(bundle)!=raw['input_sha256'][str(bundle)]:
            raise ValueError('Raw UV inputs changed')
        if sha(uvroot/'final/face_textured.glb')!=binding['head_sha256']:
            raise ValueError('Source head checksum mismatch')
        with np.load(bundle, allow_pickle=False) as b:
            v, faces = b['vertices'], b['faces']
        with np.load(assets, allow_pickle=False) as b:
            if not np.array_equal(faces, b['faces']):
                raise ValueError('UV topology mismatch')
            uv = b['uvs'][b['uv_faces']]
        labels, anatomy = identify(v, faces)
        with np.load(quality, allow_pickle=False) as b:
            maps = {k: b[k] for k in b.files}
        atlas = np.array(Image.open(atlas_path).convert('RGB'))
        original = atlas.copy()
        observed = np.asarray(Image.open(uvroot/'raw/coverage.png'))>0
        positions = atlas_positions(maps['face_index'], v, faces, uv)
        valid = maps['face_index']>=0
        head = valid & (labels[np.maximum(maps['face_index'], 0)]==0)
        height = float(np.ptp(v[:, 1]))
        eye = np.mean(anatomy['eye_centers_native'], axis=0)
        fill, scalp = fill_regions(positions, observed, maps['skin'], maps['protected'], head, eye, height)
        # Skin semantics plus confidence/brightness reject obvious mask/shadow contamination.
        mean = atlas.mean(-1)
        donors = head & observed & (maps['skin']>0) & (maps['protected']==0) & (maps['confidence']>.35) & (mean>45) & (mean<225)
        # Old hairline can be mislabeled skin. Never use the region being replaced as its own donor.
        donors &= ~fill & (positions[..., 1] < eye[1]+.075*height)
        yy, xx = np.where(donors)
        if len(yy)<128:
            raise ValueError('Insufficient observed skin donors')
        step = max(1, int(np.ceil(len(yy)/50000)))
        yy, xx = yy[::step], xx[::step]
        color = original[yy, xx].astype(float)
        # Robust prior favors upper-face skin near the unseen scalp.
        upper = positions[yy, xx, 1]>eye[1]+.025*height
        prior = np.median(color[upper] if upper.sum()>=32 else color, axis=0)
        field = surface_colors(v, faces, positions[yy, xx], color, maps['confidence'][yy, xx], prior)
        donor_tree = cKDTree(positions[yy, xx])
        nearest_vertex = cKDTree(v).query(positions[yy, xx])[1]
        residual = np.clip(color-field[nearest_vertex], -25, 25)
        fy, fx = np.where(fill)
        # Interpolate on the original triangles; identical values at UV chart cuts.
        for start in range(0, len(fy), 65536):
            y, x = fy[start:start+65536], fx[start:start+65536]
            fi = maps['face_index'][y, x]
            triangle = uv[fi]
            p = np.c_[(x+.5)/atlas.shape[1], 1-(y+.5)/atlas.shape[0]]
            a, b, c = triangle.transpose(1, 0, 2)
            den = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0])+(c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
            w0 = ((b[:, 1]-c[:, 1])*(p[:, 0]-c[:, 0])+(c[:, 0]-b[:, 0])*(p[:, 1]-c[:, 1]))/den
            w1 = ((c[:, 1]-a[:, 1])*(p[:, 0]-c[:, 0])+(a[:, 0]-c[:, 0])*(p[:, 1]-c[:, 1]))/den
            weights = np.maximum(np.c_[w0, w1, 1-w0-w1], 0)
            weights /= weights.sum(1, keepdims=True)
            inferred = np.einsum('ni,nij->nj', weights, field[faces[fi]])
            distance, index = donor_tree.query(positions[y, x], k=8)
            local_weight = 1/np.maximum(distance, .001*height)**2
            local_weight /= local_weight.sum(1, keepdims=True)
            correction = (residual[index]*local_weight[..., None]).sum(1)
            # Only local detail near observed skin; no invented pores on distant scalp.
            inferred += correction*np.exp(-(distance[:, :1]/(.035*height))**2)*(~scalp[y, x])[:, None]
            # Soft transition across observed upper forehead, below fully inferred scalp.
            blend = np.clip((positions[y, x, 1]-eye[1]-.075*height)/(.025*height), 0, 1)
            feather = observed[y, x] & (maps['skin'][y, x]>0) & scalp[y, x]
            alpha = np.where(feather, blend, 1.)[:, None]
            atlas[y, x] = np.clip(alpha*inferred+(1-alpha)*original[y, x], 0, 255).astype(np.uint8)
        if not np.array_equal(atlas[~fill], original[~fill]):
            raise ValueError('Changed protected source texture')
        provenance = np.array(Image.open(uvroot/'final/provenance-labels.png'))
        provenance[fill] = 5
        confidence = np.array(Image.open(uvroot/'final/confidence.png'))
        confidence[fill] = 13
        Image.fromarray(atlas).save(output/'head-atlas.png')
        Image.fromarray(provenance).save(output/'provenance-labels.png')
        Image.fromarray(confidence).save(output/'confidence.png')
        export_texture(v, faces, uv, atlas, output/'head.glb')
        export_texture(v, faces, uv, np.full((8, 8, 3), 160, np.uint8), output/'head-gray.glb')
        for p, digest in pins.items():
            if sha(p)!=digest:
                raise ValueError('Source changed during export')
        report.update(status='BALD_HEAD_EXPORTED_FOR_REVIEW', input_sha256=pins,
                      code_sha256=sha(Path(__file__)), head_sha256=sha(output/'head.glb'),
                      triangle_coordinates_float32_roundtrip_equal=True,
                      untouched_texels_byte_equal=True, filled_texels=int(fill.sum()),
                      replaced_observed_nonskin_scalp_texels=int((fill & observed & scalp).sum()),
                      donor_samples=len(yy), prior_rgb=prior.tolist(), topology=topology_report(v, faces),
                      provenance_labels={'5':'inferred skin continuation; not observed scalp'},
                      confidence='13/255 for inferred appearance; heuristic, not probability',
                      limitations=['Hidden scalp geometry is template-derived, not measured',
                                   'Photographic illumination remains; not recovered albedo',
                                   'Coarse geometry ROI and semantic masks can be wrong',
                                   'Glasses/eye occlusions remain unchanged',
                                   'No pores or invisible identity details recovered',
                                   'No physical print validation or self-intersection check'])
    except Exception as exc:
        report.update(status='FAILED', error=str(exc))
        raise
    finally:
        write(output/'report.json', report)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.job, args.output)
