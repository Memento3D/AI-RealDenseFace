"""Compact short-template adaptation: exterior relief relative to source scalp.

Separates original template wall thickness from the desired external volume.
Only short attached hair is supported; it is not long-hair reconstruction.
"""
import numpy as np
from figurine2_hair import nearest_ray

def compact_fit(hair,head,eye_y,ratio,settings,correspondence=None):
    if not .08<=ratio<=.60:raise ValueError('Compact ratio out of range')
    values=np.array([settings['inset_mm'],settings['floor_mm'],settings['max_crown_added_mm']],float)
    if not np.isfinite(values).all() or not 0<=values[0]<=.4 or not 0<=values[1]<=.5 or not 0<values[2]<=2:
        raise ValueError('Invalid compact-fit dimensions')
    upper=head.vertices[head.vertices[:,1]>eye_y+.07*head.extents[1]]
    lo,hi=np.quantile(upper,[.005,.995],axis=0)
    origin=np.array([(lo[0]+hi[0])/2,eye_y-.04*head.extents[1],(lo[2]+hi[2])/2])
    if not head.contains(origin[None])[0]:raise ValueError('Compact-fit scalp origin is not inside the source head')
    points=hair.vertices.copy();vector=points-origin;radius=np.linalg.norm(vector,axis=1)
    directions=vector/np.maximum(radius[:,None],1e-8)
    # Dense supplied STL ray queries must be bounded to avoid OOM. Reuse across ratios.
    if correspondence is None:correspondence={}
    def query(mesh,key):
        if key not in correspondence:
            chunks=[]
            for start in range(0,len(points),256):
                rays=directions[start:start+256]
                chunks.append(nearest_ray(mesh,np.tile(origin,(len(rays),1)),rays))
            correspondence[key]=np.concatenate(chunks)
        values=correspondence[key]
        if values.shape!=(len(points),):raise ValueError('Invalid cached ray correspondence')
        return values
    scalp=query(head,'scalp')
    if not np.isfinite(scalp).all():raise ValueError('Scalp ray incomplete; refuse invented compact completion')
    if hair.contains(origin[None])[0]:
        raise ValueError('Scalp origin lies in template material; cavity fitting is invalid')
    # The first entry from a point outside a closed volume must have n·(p-o)<0.
    # Excluding outward-facing triangles preserves that first entry exactly and
    # avoids querying dense decorative exterior triangles unnecessarily.
    cavity=hair.copy()
    entry=np.einsum('ij,ij->i',cavity.face_normals,cavity.triangles_center-origin)<1e-8
    cavity.update_faces(entry);cavity.remove_unreferenced_vertices()
    inner=query(cavity,'inner')
    if 'inner_repaired' not in correspondence:
        missing=~np.isfinite(inner)
        correspondence['entry_missed_count']=int(missing.sum())
        if missing.any():
            ids=np.flatnonzero(missing);recovered=[]
            for start in range(0,len(ids),128):
                block=ids[start:start+128]
                recovered.append(nearest_ray(hair,np.tile(origin,(len(block),1)),directions[block]))
            inner[ids]=np.concatenate(recovered)
        tangent=~np.isfinite(inner)
        if tangent.mean()>.01:raise ValueError('Too many unresolved cavity rays; refuse endpoint fallback')
        # The target vertex itself is a known mesh intersection on this ray.
        # This bounded fallback is limited to grazing numerical misses, reported.
        inner[tangent]=radius[tangent]
        correspondence['endpoint_fallback_count']=int(tangent.sum())
        correspondence['inner_repaired']=True
    if not np.isfinite(inner).all():raise ValueError('Template cavity ray incomplete')
    thickness=np.maximum(radius-inner,0)
    mapped=ratio*thickness+settings['floor_mm']*(1-np.exp(-thickness/.3))-settings['inset_mm']
    result=hair.copy();result.vertices=origin+directions*(scalp+mapped)[:,None]
    if not result.is_volume:raise ValueError('Compact adaptation lost positive closed volume')
    crown=float(result.bounds[1,1]-head.bounds[1,1])
    theta=np.tile(np.linspace(-np.pi,np.pi,24,endpoint=False),4)
    phi=np.repeat(np.deg2rad([10,30,50,70]),24)
    ray=np.column_stack((np.sin(phi)*np.sin(theta),np.cos(phi),np.sin(phi)*np.cos(theta)))
    starts=np.tile(origin,(len(ray),1));rhead=nearest_ray(head,starts,ray);rhair=nearest_ray(result,starts,ray)
    valid=np.isfinite(rhead)&np.isfinite(rhair);gap=rhair[valid]-rhead[valid]
    report={'method':'radial template wall compression; each sampled inner wall mapped into source scalp',
            'entry_missed_count':correspondence['entry_missed_count'],
            'endpoint_fallback_count':correspondence['endpoint_fallback_count'],
            'cavity_entry_triangles':int(entry.sum()),'template_triangles':len(hair.faces),
            'ratio':ratio,'inset_mm':settings['inset_mm'],'floor_mm':settings['floor_mm'],
            'crown_added_height_mm':crown,'maximum_allowed_crown_mm':settings['max_crown_added_mm'],
            'crown_proxy_pass':bool(crown<=settings['max_crown_added_mm']),
            'sampled_inner_gap_p95_mm':float(np.quantile(np.maximum(gap,0),.95)) if len(gap) else None,
            'sampled_contact_proxy_pass':bool(len(gap)>=48 and np.quantile(np.maximum(gap,0),.95)<=.08),
            'attachment_rays':int(valid.sum()),'gap_scope':'upper-head radial samples; not full clearance certification',
            'source_face_modified':False,'scope':'short close-fitting supplied template, not observed true hair roots'}
    return result,report
