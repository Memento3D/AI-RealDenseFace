"""View-dependent surface occluding edges restricted to the pinned face region.

No cut boundary of a cropped mask is used: edge adjacency is always computed
on the original full mesh, then restricted by semantic vertex membership.
"""
import numpy as np
from memento3d_head.adapters.contour_audit import mesh_visibility
from memento3d_head.domain.contracts import ValidationError


def visible_face_chains(vertices,faces,face_ids,excluded_ids,view):
    import cv2
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=int)
    allowed=np.zeros(len(vertices),dtype=bool)
    for ids in (face_ids,excluded_ids):
        ids=np.asarray(ids)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or (len(ids) and (ids.min()<0 or ids.max()>=len(vertices))):
            raise ValidationError('INVALID_FACE_MASK','Semantic mask indices outside mesh')
    allowed[face_ids]=True;allowed[excluded_ids]=False
    rotation=cv2.Rodrigues(np.asarray(view['rotation_vector'],dtype=float))[0]
    translation=np.asarray(view['translation_mm'],dtype=float)
    camera=vertices@rotation.T+translation
    if not np.isfinite(camera).all() or np.any(camera[:,2]<=0):
        raise ValidationError('INVALID_DEPTH','Mesh crosses camera plane')
    triangles=camera[faces]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    facing=np.einsum('ij,ij->i',normals,triangles.mean(axis=1))
    adjacency={}
    for index,face in enumerate(faces):
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            adjacency.setdefault(tuple(sorted((int(a),int(b)))),[]).append(index)
    candidates=[]
    for edge,neighbors in adjacency.items():
        # Original open boundaries and nonmanifold edges are not accepted as jaw.
        if len(neighbors)!=2 or not allowed[list(edge)].all():continue
        a,b=neighbors
        if facing[a]*facing[b]<0:candidates.append(edge)
    retained=[]
    for a,b in candidates:
        points=np.array([vertices[a]*.75+vertices[b]*.25,(vertices[a]+vertices[b])*.5,
                         vertices[a]*.25+vertices[b]*.75])
        visible=mesh_visibility(vertices,faces,points,view['rotation_vector'],translation,tolerance_mm=.25)
        if all(item['mesh_visible'] for item in visible):retained.append((a,b))
    graph={}
    for a,b in retained:graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    unused={tuple(sorted(edge)) for edge in retained};chains=[]
    # Stop at branching vertices instead of fabricating a connection across them.
    starts=[v for v,n in graph.items() if len(n)!=2]+list(graph)
    for start in starts:
        for neighbor in sorted(graph[start]):
            edge=tuple(sorted((start,neighbor)))
            if edge not in unused:continue
            unused.remove(edge);chain=[start,neighbor];previous=start;current=neighbor
            while len(graph[current])==2:
                nxt=next(v for v in graph[current] if v!=previous)
                edge=tuple(sorted((current,nxt)))
                if edge not in unused:break
                unused.remove(edge);chain.append(nxt);previous,current=current,nxt
            if len(chain)>=2:
                xy=cv2.projectPoints(vertices[chain],np.asarray(view['rotation_vector'],float),translation,
                                     np.asarray(view['camera_matrix'],float),None)[0].reshape(-1,2)
                chains.append(xy.tolist())
    return chains,{'candidate_edges':len(candidates),'visible_edges':len(retained),'chains':len(chains),
                   'method':'full_mesh_facing_transition_edges_semantic_face_filter_full_mesh_ray_test',
                   'mask_cut_edges_used':False,'visibility_tolerance_mm':.25,'anatomical_jaw_verified':False}


# Conservative engineering guards, not thresholds calibrated on human faces.
ARC_POLICY={'version':1,'samples':64,'endpoint_fraction_of_reference_length':.08,
            'minimum_tolerance_px':6.,'minimum_reference_coverage':.85,
            'minimum_length_ratio':.65,'maximum_length_ratio':1.6,
            'maximum_ordered_distance_factor':2.}


def arc_support(reference,arc):
    reference=np.asarray(reference,float);arc=np.asarray(arc,float)
    length=float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())
    arc_length=float(np.linalg.norm(np.diff(arc,axis=0),axis=1).sum())
    tolerance=max(ARC_POLICY['minimum_tolerance_px'],length*ARC_POLICY['endpoint_fraction_of_reference_length'])
    endpoints=np.linalg.norm(reference[[0,-1]]-arc[[0,-1]],axis=1)
    distances=np.linalg.norm(reference[:,None,:]-arc[None,:,:],axis=2).min(axis=1)
    coverage=float(np.mean(distances<=tolerance))
    ratio=arc_length/max(length,1e-12)
    ordered=np.linalg.norm(reference-arc,axis=1)
    reasons=[]
    if np.max(endpoints)>tolerance:reasons.append('ENDPOINTS_NOT_SPANNED')
    if coverage<ARC_POLICY['minimum_reference_coverage']:reasons.append('INSUFFICIENT_REFERENCE_COVERAGE')
    if not ARC_POLICY['minimum_length_ratio']<=ratio<=ARC_POLICY['maximum_length_ratio']:reasons.append('LENGTH_RATIO_OUT_OF_RANGE')
    if ordered.max()>tolerance*ARC_POLICY['maximum_ordered_distance_factor']:reasons.append('ORDERED_PATH_OUTSIDE_CORRIDOR')
    return {'supported':not reasons,'rejection_reasons':reasons,'reference_coverage_fraction':coverage,
            'arc_to_reference_length_ratio':ratio,'endpoint_distances_px':endpoints.tolist(),
            'tolerance_px':tolerance,'policy_calibrated':False}


def compare_open_arc(points,chains):
    from memento3d_head.adapters.manual_contour_comparison import resample
    reference=resample(points,ARC_POLICY['samples']);options=[];rejected=[]
    for chain in chains:
        if len(chain)<2:continue
        dense=resample(chain,512)
        a=int(np.argmin(np.linalg.norm(dense-reference[0],axis=1)))
        b=int(np.argmin(np.linalg.norm(dense-reference[-1],axis=1)))
        if a==b:
            rejected.append({'supported':False,'rejection_reasons':['COLLAPSED_ENDPOINTS']});continue
        path=dense[min(a,b):max(a,b)+1]
        if a>b:path=path[::-1]
        arc=resample(path,ARC_POLICY['samples']);support=arc_support(reference,arc)
        if not support['supported']:
            rejected.append(support);continue
        errors=np.linalg.norm(arc-reference,axis=1)
        options.append((float(errors.mean()),arc,errors,support))
    if not options:return {'status':'NO_SUPPORTED_ARC','correspondence_verified':False,
                            'candidate_arc_count':len(chains),'supported_arc_count':0,
                            'rejected_arcs':rejected,'support_policy':ARC_POLICY,
                            'warning':'No individual continuous chain passes endpoint/coverage guards; no distance or fabricated bridge'}
    options.sort(key=lambda item:item[0]);mean,arc,errors,support=options[0]
    return {'status':'UNVERIFIED_ARC_PROPOSED','mean_ordered_distance_px':mean,
            'max_ordered_distance_px':float(errors.max()),'p95_ordered_distance_px':float(np.percentile(errors,95)),
            'endpoint_distances_px':[float(errors[0]),float(errors[-1])],
            'reference_samples_px':reference.tolist(),'mesh_samples_px':arc.tolist(),
            'candidate_arc_count':len(chains),'supported_arc_count':len(options),
            'support':support,'support_policy':ARC_POLICY,'rejected_arcs':rejected,
            'correspondence_verified':False,
            'method':'coverage_guarded_continuous_open_surface_chain_64_arclength_samples',
            'warning':'Passing engineering coverage guards is not proof of anatomical jaw correspondence'}
