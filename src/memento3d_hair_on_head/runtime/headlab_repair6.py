"""Evidence-constrained scalp footprint repair. Own experimental integration.

No symmetry, fixed nape height or fixed hair length. This representation is
limited to attached short hair; silhouettes do not reveal hidden hair roots.
"""
import json
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi, sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra
from figurine_hairline32 import project, clip_surface, stabilize_labels
from headlab_raster import surface_depth_fast

NAMES = ['front', 'left', 'right', 'back']


def validate_settings(settings):
    for key,low,high in [('min_facing',.05,.8),('mask_uncertainty_px',.5,8),
                         ('rear_weight',.1,5),('silhouette_negative_weight',.05,1),
                         ('max_view_iou_loss',0,.05),('max_region_excess_increase',0,.05)]:
        value=settings[key]
        if not np.isfinite(value) or not low<=value<=high:raise ValueError('Invalid repair setting: '+key)
    a=settings['regularization_trials'];b=settings['veto_trials']
    if not a or not b or len(a)*len(b)>12:raise ValueError('Repair budget must be 1..12 trials')
    if not all(np.isfinite(x) and 0<x<=4 for x in a):raise ValueError('Invalid regularization trial')
    if not all(np.isfinite(x) and 0<=x<=8 for x in b):raise ValueError('Invalid veto trial')


def classify_edges(mask, label, rear=False):
    """Separate visible skin contact from silhouette and ambiguous boundaries.

    Rear label 1 in V5 was synthetic background, so NEVER treat it as skin.
    """
    small = np.asarray(Image.fromarray(np.uint8(label)).resize(
        (mask.shape[1], mask.shape[0]), Image.Resampling.NEAREST))
    border = mask & ~ndi.binary_erosion(mask)
    skin = np.isin(small, [1, 14]) if not rear else np.zeros_like(mask)
    ear = np.isin(small, [7, 8]) if not rear else np.zeros_like(mask)
    contact = border & ndi.binary_dilation(skin & ~mask, iterations=2)
    silhouette = border & ndi.binary_dilation((small == 0) & ~mask, iterations=2) & ~contact
    if rear:
        # Rear source only supplies an outer contour, never an anatomical root.
        silhouette = border.copy()
    return dict(contact=contact, silhouette=silhouette,
                ambiguous=border & ~(contact | silhouette), skin=skin, ear=ear)


def sample_evidence(work, shared, settings):
    n = len(work.vertices); probabilities = []; weights = []; negatives = []; diagnostics = []
    for index, (camera, mask, depth, label) in enumerate(zip(
            shared['cameras'], shared['masks'], shared['depths'], shared['labels'])):
        edges = classify_edges(mask, label, index == 3)
        signed = ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
        xy, z = project(work.vertices, camera)
        ix = np.floor(xy[:, 0]).astype(int); iy = np.floor(xy[:, 1]).astype(int)
        valid = (z > 0) & (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
        center = -np.linalg.solve(camera[:, :3], camera[:, 3])
        toward = center - work.vertices
        toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-12)
        facing = np.einsum('ij,ij->i', work.vertex_normals, toward)
        ids = np.flatnonzero(valid)
        front_depth = depth[iy[ids], ix[ids]]
        seen = np.isfinite(front_depth) & (z[ids] <= front_depth + .12 * np.linalg.norm(camera[2, :3]))
        seen &= facing[ids] > settings['min_facing']
        ids = ids[seen]
        d = ndi.map_coordinates(signed, [xy[ids, 1], xy[ids, 0]], order=1, mode='nearest')
        skin = edges['skin'][iy[ids], ix[ids]]; ear = edges['ear'][iy[ids], ix[ids]]
        inside = mask[iy[ids], ix[ids]]
        # Background is a soft silhouette veto, not a measured skin/root label.
        outside = d < -settings['mask_uncertainty_px']
        applicable = inside | skin | ear | outside
        ids, d, skin, ear, inside, outside = [a[applicable] for a in (ids, d, skin, ear, inside, outside)]
        p = np.clip(.5 + d / (2 * settings['mask_uncertainty_px']), 0, 1)
        w = np.maximum(facing[ids], 0) ** 4
        # A positive silhouette pixel near a grazing surface is weak root evidence.
        interior = np.clip(np.abs(d) / settings['mask_uncertainty_px'], .12, 1)
        w *= interior
        # Near the projected scalp silhouette, a visible hair pixel may belong
        # to hair behind that scalp point. Keep it as volume evidence, but weaken
        # its interpretation as a root footprint on this surface.
        model_interior = ndi.distance_transform_edt(np.isfinite(depth))
        scalp_margin = ndi.map_coordinates(model_interior,[xy[ids,1],xy[ids,0]],order=1,mode='nearest')
        silhouette_root = inside & (scalp_margin < settings.get('scalp_interior_px', 2.5))
        w[silhouette_root] *= settings.get('silhouette_root_weight', .08)
        if index == 3: w *= settings['rear_weight']
        negative = (skin | ear | outside) & ~inside
        # A visible contradiction at an oblique angle must not vanish as fast
        # as a positive footprint guess. Still exclude back-facing/occluded points.
        w[negative] = np.maximum(facing[ids[negative]], 0) ** 2 * interior[negative]
        # Skin is direct evidence. A background veto has pose uncertainty.
        w *= np.where(outside & ~(skin | ear), settings['silhouette_negative_weight'], 1.)
        prob = np.zeros(n); weight = np.zeros(n); neg = np.zeros(n)
        prob[ids] = p; weight[ids] = w; neg[ids[negative]] = w[negative]
        probabilities.append(prob); weights.append(weight); negatives.append(neg)
        diagnostics.append({'view': NAMES[index], 'visible_usable_vertices': int(len(ids)),
            'contact_edge_pixels': int(edges['contact'].sum()),
            'silhouette_edge_pixels': int(edges['silhouette'].sum()),
            'ambiguous_edge_pixels': int(edges['ambiguous'].sum()),
            'rear_contact_is_unobserved': index == 3})
    return np.asarray(probabilities), np.asarray(weights), np.asarray(negatives), diagnostics


def graph_laplacian(work):
    e = work.edges_unique; length = np.linalg.norm(work.vertices[e[:, 0]] - work.vertices[e[:, 1]], axis=1)
    w = 1 / np.maximum(length, 1e-5)
    adjacency = sparse.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                                 shape=(len(work.vertices), len(work.vertices))).tocsr()
    # Normalize global scale; retains a symmetric positive semidefinite system.
    adjacency /= np.median(np.asarray(adjacency.sum(1)).ravel())
    return sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency


def solve_field(prior, probabilities, weights, negatives, lap, strength, veto, fixed):
    w = weights + veto * negatives
    target = np.divide((probabilities * weights).sum(0), w.sum(0),
                       out=np.zeros(len(prior)), where=w.sum(0) > 1e-9)
    confidence = w.sum(0)
    # Local prior: unobserved regions stay close to the measured V5 footprint.
    system = sparse.diags(confidence + .015) + strength * lap
    rhs = confidence * target + .015 * prior
    free = np.flatnonzero(~fixed); held = np.flatnonzero(fixed)
    result = prior.copy(); result[held] = 0
    result[free] = spsolve(system[free][:, free].tocsc(), rhs[free])
    return np.clip(result, 0, 1)


def patch_group(shared, prior_group, field, known):
    result = prior_group.copy(); work = prior_group['work']
    patch, normals, pk, boundary = clip_surface(work, field, known)
    skin, sn, _, sb = clip_surface(work, 1 - field, known)
    result.update(field=field, patch=patch, normals=normals, known=pk,
                  boundary=boundary, skin=skin, skin_normals=sn, skin_boundary=sb)
    return result


def region_mask(mask, kind):
    """Photo-relative monitoring zones, NOT anatomical landmarks or fixed lengths."""
    y, x = np.nonzero(mask)
    if not len(x): raise ValueError('Empty hair mask')
    yy, xx = np.indices(mask.shape)
    low, high = np.min(y), np.max(y); left, right = np.min(x), np.max(x)
    margin = max(3., (right-left)*.12)
    if kind == 'lower': return yy >= low + .60*(high-low)
    if kind == 'lateral': return (yy >= low+.45*(high-low)) & ((xx < left+.30*(right-left)) | (xx > right-.30*(right-left)))
    return (xx >= left-margin) & (xx <= right+margin) & (yy >= low-margin) & (yy <= high+margin)


def evaluate(group, shared, outer=None, seam=.08, base=.45):
    from headlab_algorithms import taper
    patch = group['patch']
    if outer is None:
        blend = taper(patch, group['boundary'], .65)
        outer = patch.vertices + group['normals']*(seam+(base-seam)*blend)[:, None]
    rows=[]
    for i, (camera, mask, depth, label) in enumerate(zip(shared['cameras'], shared['masks'], shared['depths'], shared['labels'])):
        z = surface_depth_fast(outer, patch.faces, camera, (mask.shape[1], mask.shape[0]))
        pred = np.isfinite(z) & ((z <= depth+.02*np.linalg.norm(camera[2, :3])) | ~np.isfinite(depth))
        area = max(int(mask.sum()), 1); fp = pred & ~mask; fn = mask & ~pred
        lower = region_mask(mask, 'lower'); lateral = region_mask(mask, 'lateral')
        boundary = pred & ~ndi.binary_erosion(pred)
        contact = classify_edges(mask, label, i == 3)['contact']
        # Normalize by width so different image resolutions remain comparable.
        bx = np.nonzero(mask)[1]; width = max(int(np.ptp(bx)), 1)
        distance = ndi.distance_transform_edt(~boundary)
        contact_p95 = float(np.quantile(distance[contact], .95)/width) if contact.any() else None
        rows.append({'view':NAMES[i], 'iou':float((pred & mask).sum()/max((pred | mask).sum(), 1)),
            'excess_fraction':float(fp.sum()/area), 'missing_fraction':float(fn.sum()/area),
            'lower_excess_fraction':float((fp & lower).sum()/area),
            'lateral_excess_fraction':float((fp & lateral).sum()/area),
            'contact_p95_head_width_fraction':contact_p95})
    edges = group['work'].edges_unique; f = group['field']
    roughness = float(np.mean((f[edges[:,0]]-f[edges[:,1]])**2))
    score = np.mean([r['excess_fraction']+r['missing_fraction'] for r in rows])
    score += .6*np.mean([r['lower_excess_fraction']+r['lateral_excess_fraction'] for r in rows])
    contact = [r['contact_p95_head_width_fraction'] for r in rows if r['contact_p95_head_width_fraction'] is not None]
    score += .5*(np.mean(contact) if contact else 0)+.25*roughness
    return {'score':float(score), 'views':rows, 'field_edge_variation':roughness,
            'scope':'training-mask consistency and photo-relative zones; not likeness, anatomical truth or print certification'}


def accepts(candidate, baseline, limits):
    if not np.isfinite(candidate['score']) or candidate['score'] >= baseline['score']-1e-5:
        return False, 'no_objective_improvement'
    for a,b in zip(candidate['views'], baseline['views']):
        if a['iou'] < b['iou']-limits['max_view_iou_loss']:
            return False, 'view_regression_'+a['view']
        if a['lower_excess_fraction'] > b['lower_excess_fraction']+limits['max_region_excess_increase']:
            return False, 'lower_region_regression_'+a['view']
    return True, 'measured_proxy_improved'


def repair_surface(shared, prior, output, settings):
    validate_settings(settings)
    output.mkdir(); work=prior['work']; initial=prior['field'].copy()
    p,w,neg,diagnostics=sample_evidence(work, shared, settings)
    known=w.sum(0)>1e-6; lap=graph_laplacian(work)
    # Synthetic closures identified by the same source topology as V5.
    from figurine_geometry import weld
    from figurine_hairline32 import refine_source_surface
    source=weld(shared['source_mm'][np.repeat(shared['components']==0,3)])
    _,closure=refine_source_surface(shared['closed'],len(source.faces))
    fixed=np.zeros(len(initial),bool);fixed[closure]=True
    edges=work.edges_unique
    lengths=np.linalg.norm(work.vertices[edges[:,0]]-work.vertices[edges[:,1]],axis=1)
    graph=sparse.coo_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(initial),len(initial))).tocsr()
    border=np.unique(edges[(initial[edges[:,0]]>.5)!=(initial[edges[:,1]]>.5)])
    boundary_distance=dijkstra(graph,directed=False,indices=border,min_only=True)
    best=prior; baseline=evaluate(prior,shared); best_metrics=baseline; history=[]
    rounds=0
    for strength in settings['regularization_trials']:
        for veto in settings['veto_trials']:
            rounds+=1
            field=solve_field(initial,p,w,neg,lap,strength,veto,fixed)
            outside_budget=boundary_distance>settings.get('max_root_change_mm',1.5)
            clamped=int(np.sum(outside_budget & ((field>.5)!=(initial>.5))))
            field[outside_budget]=initial[outside_budget]
            field,_=stabilize_labels(work,field,field,known)
            candidate=patch_group(shared,prior,field,known)
            metrics=evaluate(candidate,shared)
            accepted,reason=accepts(metrics,best_metrics,settings)
            changed=(field>.5)!=(initial>.5)
            radius=float(boundary_distance[changed].max()) if changed.any() else 0.
            if radius>settings.get('max_root_change_mm',1.5):accepted,reason=False,'root_change_budget_exceeded'
            history.append({'trial':rounds,'regularization':strength,'veto':veto,'accepted':accepted,'reason':reason,'changes_outside_local_budget_prevented':clamped,'max_change_distance_from_initial_boundary_mm':radius,'metrics':metrics})
            print('Repair trial',rounds,reason,'score',round(metrics['score'],5),flush=True)
            if accepted: best,best_metrics=candidate,metrics
    changed=(best['field']>.5)!=(initial>.5)
    removed=(initial>.5)&(best['field']<=.5); added=(initial<=.5)&(best['field']>.5)
    report={'method':'visibility and edge-type weighted field solve; bounded deterministic trial loop',
        'evaluated_trials':rounds,'accepted_trials':sum(r['accepted'] for r in history),
        'changed_vertices':int(changed.sum()),'removed_hair_vertices':int(removed.sum()),'added_hair_vertices':int(added.sum()),
        'baseline':baseline,'selected':best_metrics,'trials':history,'evidence':diagnostics,
        'symmetry_forced':False,'fixed_nape_length':False,'source_face_geometry_changed':False,
        'anatomical_hairline_verified':False,'hidden_hair_roots_recovered':False,
        'representation_scope':'short attached hair; long/tied hair requires a separate volume/attachment representation',
        'all_views_used_for_optimization':True,'independent_validation':False,'ready_for_machine':False}
    np.savez_compressed(output/'repair_arrays.npz',vertices=work.vertices,faces=work.faces,initial_field=initial,
                        selected_field=best['field'],probabilities=p,weights=w,negative_weights=neg,changed=changed)
    (output/'repair.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for i,(mask,label,image) in enumerate(zip(shared['masks'],shared['labels'],shared['images'])):
        types=classify_edges(mask,label,i==3);rgb=np.asarray(image.resize((mask.shape[1],mask.shape[0]))).copy()
        rgb[types['contact']]=[30,230,240];rgb[types['silhouette']]=[255,190,40];rgb[types['ambiguous']]=[240,80,210]
        Image.fromarray(rgb).save(output/(NAMES[i]+'.edge_types.png'))
    # Source-coordinate map, never part of print STL.
    from figurine_preview import add_mesh
    from add_graduation_cap import save
    colors=np.tile([.3,.3,.3,1.],(len(work.vertices),1));colors[removed]=[1,.15,.1,1];colors[added]=[.1,.9,.3,1]
    doc={'asset':{'version':'2.0'},'scenes':[{'nodes':[]}],'scene':0,'nodes':[],'meshes':[],'materials':[],'accessors':[],'bufferViews':[]};blob=bytearray()
    add_mesh(doc,blob,'RepairEvidenceMap',work,colors);save(output/'repair_map.glb',doc,blob)
    best=best.copy();best['fusion']=prior['fusion'].copy();best['fusion']['repair_v6']=report
    return best
