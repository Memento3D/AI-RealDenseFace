"""Lower boundary of the pinned FLAME face patch: an anatomical proxy, not a silhouette."""
import numpy as np
from memento3d_head.domain.contracts import ValidationError
from memento3d_head.adapters.jaw_surface import arc_support,ARC_POLICY
from memento3d_head.adapters.manual_contour_comparison import resample


def lower_face_boundary(vertices,faces,face_ids):
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int);ids=np.asarray(face_ids)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or ids.min()<0 or ids.max()>=len(vertices):
        raise ValidationError('INVALID_FACE_MASK','Invalid semantic face indices')
    allowed=np.zeros(len(vertices),bool);allowed[ids]=True
    selected=faces[allowed[faces].all(axis=1)];counts={}
    for a,b,c in selected:
        for edge in ((a,b),(b,c),(c,a)):
            key=tuple(sorted(map(int,edge)));counts[key]=counts.get(key,0)+1
    if any(n>2 for n in counts.values()):raise ValidationError('INVALID_FACE_PATCH','Nonmanifold face patch')
    remaining={edge for edge,count in counts.items() if count==1};graph={}
    for a,b in remaining:graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    if not graph or any(len(n)!=2 for n in graph.values()):
        raise ValidationError('INVALID_FACE_PATCH','Expected closed disjoint semantic boundary loops')
    loops=[]
    while remaining:
        start,b=min(remaining);remaining.remove((start,b));loop=[start,b]
        while loop[-1]!=start:
            choices=sorted(n for n in graph[loop[-1]] if tuple(sorted((n,loop[-1]))) in remaining)
            if len(choices)!=1:raise ValidationError('INVALID_FACE_PATCH','Ambiguous boundary traversal')
            n=choices[0];remaining.remove(tuple(sorted((n,loop[-1]))));loop.append(n)
        loops.append(loop[:-1])
    # The exterior face boundary spans the greatest canonical vertical range;
    # eye/nostril/mouth holes are not used. Only pinned neutral FLAME axes apply.
    outer=max(loops,key=lambda loop:float(np.ptp(vertices[loop,1])))
    cutoff=float((vertices[ids,1].min()+vertices[ids,1].max())*.5)
    starts=[i for i,k in enumerate(outer) if vertices[k,1]>cutoff]
    if not starts:raise ValidationError('INVALID_FACE_PATCH','No upper/lower face separation')
    i=starts[0];ordered=outer[i:]+outer[:i];parts=[];part=[]
    for k in ordered+[ordered[0]]:
        if vertices[k,1]<=cutoff:part.append(k)
        elif part:
            if len(part)>=2:parts.append(part)
            part=[]
    if len(parts)!=1:raise ValidationError('INVALID_FACE_PATCH','Expected one continuous lower face boundary')
    return parts[0],{'reference_type':'lower_semantic_face_patch_boundary_proxy',
        'boundary_loops':len(loops),'outer_loop_vertices':len(outer),'lower_boundary_vertices':len(parts[0]),
        'canonical_y_cutoff_mm':cutoff,'canonical_axes':'neutral_pinned_FLAME2020',
        'is_occluding_silhouette':False,'anatomical_correspondence_verified':False,
        'boundary_vertices':parts[0]}


def project_anatomical_jaw(vertices,faces,face_ids,view,boundary=None):
    import cv2
    ids,details=lower_face_boundary(vertices,faces,face_ids) if boundary is None else boundary
    details=dict(details)
    details["boundary_frozen_from_baseline"]=boundary is not None
    points=np.asarray(vertices)[ids]
    rotation=np.asarray(view['rotation_vector'],float);translation=np.asarray(view['translation_mm'],float)
    camera=points@cv2.Rodrigues(rotation)[0].T+translation
    if not np.isfinite(camera).all() or np.any(camera[:,2]<=0):
        raise ValidationError('INVALID_DEPTH','Jaw crosses camera plane')
    xy=cv2.projectPoints(points,rotation,translation,np.asarray(view['camera_matrix'],float),None)[0].reshape(-1,2)
    # Keep the anatomical path intact: far-side portions can be visible in projection
    # inside the face. They must not be treated as visible silhouette evidence.
    details.update({'projection_visibility_filtered':False,'hidden_side_may_be_projected':True,
                    'method':'topological_face_patch_outer_boundary_lower_half',
                    'chains':1,'jaw_region_verified':False})
    return [xy.tolist()],details


def compare_anatomical_arc(points,chains):
    reference=resample(points,ARC_POLICY['samples']);proposals=[]
    for chain in chains:
        if len(chain)<2:continue
        dense=resample(chain,1024)
        a=int(np.argmin(np.linalg.norm(dense-reference[0],axis=1)))
        b=int(np.argmin(np.linalg.norm(dense-reference[-1],axis=1)))
        if a==b:continue
        path=dense[min(a,b):max(a,b)+1]
        if a>b:path=path[::-1]
        arc=resample(path,ARC_POLICY['samples']);support=arc_support(reference,arc)
        errors=np.linalg.norm(arc-reference,axis=1)
        proposals.append((not support['supported'],float(errors.mean()),arc,errors,support))
    if not proposals:return {'status':'NO_SUPPORTED_ARC','correspondence_verified':False,
                             'warning':'Anatomical proxy could not span distinct annotation endpoints'}
    proposals.sort(key=lambda p:(p[0],p[1]));rejected,mean,arc,errors,support=proposals[0]
    result={'status':'ANATOMICAL_PROXY_SUPPORTED' if not rejected else 'ANATOMICAL_PROXY_INSUFFICIENT_SUPPORT',
            'correspondence_verified':False,'reference_type':'lower_semantic_face_patch_boundary_proxy',
            'support':support,'support_policy':ARC_POLICY,
            'projection_visibility_verified':False,
            'warning':'Anatomical proxy; not an occluding silhouette or verified visible jaw. No fitting.'}
    if rejected:
        result['diagnostic_mesh_samples_px']=arc.tolist()
        result['diagnostic_reference_samples_px']=reference.tolist()
        # Unsupported proposals must never expose a fitting metric or sample pairs.
    else:
        result.update({'mean_ordered_distance_px':mean,'max_ordered_distance_px':float(errors.max()),
                       'p95_ordered_distance_px':float(np.percentile(errors,95)),
                       'reference_samples_px':reference.tolist(),'mesh_samples_px':arc.tolist(),
                       'endpoint_distances_px':support['endpoint_distances_px']})
    return result
