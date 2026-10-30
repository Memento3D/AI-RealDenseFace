"""Recompute observation labels before clipping; no frozen V4 nape contour."""
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components,dijkstra
from figurine_hairline32 import refine_source_surface,collect_observations,propagate_unknown,stabilize_labels,clip_surface
from figurine_geometry import weld
from headlab_backview import prepare_back
from headlab_raster import surface_depth_fast


def fused_surface(shared,view_count,output,settings):
    source=weld(shared['source_mm'][np.repeat(shared['components']==0,3)])
    work,closure=refine_source_surface(shared['closed'],len(source.faces))
    n=len(work.vertices);values=[];weights=[];records=[]
    for i in range(view_count):
        label=shared['labels'][i]
        if i==3:
            # At a visible projected model point, background outside the accepted
            # rear hair silhouette is evidence AGAINST hair, not evidence of skin.
            label=np.where(shared['masks'][i],17,1).astype(np.uint8)
        v,k,w,r=collect_observations(work,[shared['cameras'][i]],[shared['masks'][i]],[shared['depths'][i]],[label])
        # Original weights contain cos^2; square again for local view preference.
        w=w*w
        if i==3:w*=settings['back_weight']
        values.append(v);weights.append(w);records.extend(r)
    values=np.asarray(values);weights=np.asarray(weights)
    total=weights.sum(0);known=total>1e-6
    observed=np.divide((values*weights).sum(0),total,out=np.zeros(n),where=known)
    positive=((values>.5)&(weights>.02)).any(0);negative=((values<.5)&(weights>.02)).any(0)
    conflict=positive&negative
    fixed=known.copy();fixed[closure]=True;seed=observed.copy();seed[closure]=0
    field=propagate_unknown(work,seed,fixed);field[closure]=0
    # Feature-preserving graph regularization of uncertain boundary labels.
    # High-confidence observed hair/skin are fixed; uncertainty is smoothed only.
    if settings['boundary_regularization']>0:
        e=work.edges_unique;i,j=e.T
        a=sparse.coo_matrix((np.ones(len(i)*2),(np.r_[i,j],np.r_[j,i])),shape=(n,n)).tocsr()
        lap=sparse.diags(np.asarray(a.sum(1)).ravel())-a
        confidence=np.clip(total,0,1)
        anchor=(known&((observed<.15)|(observed>.85))&~conflict);anchor[closure]=True
        free=np.flatnonzero(~anchor);held=np.flatnonzero(anchor)
        strength=settings['boundary_regularization'];system=sparse.diags(confidence+.015)+strength*lap
        rhs=confidence*observed+.015*field
        result=field.copy();result[held]=seed[held]
        result[free]=spsolve(system[free][:,free].tocsc(),rhs[free]-system[free][:,held]@result[held])
        field=np.clip(result,0,1);field[closure]=0
    opening_removed=0
    radius=settings.get('min_patch_width_mm',0)/2
    if radius>0:
        edges=work.edges_unique;lengths=np.linalg.norm(work.vertices[edges[:,0]]-work.vertices[edges[:,1]],axis=1)
        graph=sparse.coo_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
        positive=field>.5
        interior=dijkstra(graph,directed=False,indices=np.flatnonzero(~positive),min_only=True)
        core=positive&(interior>=radius)
        if core.any():
            spread=dijkstra(graph,directed=False,indices=np.flatnonzero(core),min_only=True)
            rejected=positive&(spread>radius+1e-8)
            opening_removed=int(rejected.sum());field[rejected]=0
    field,removed=stabilize_labels(work,field,observed,known)
    patch,normals,patch_known,boundary=clip_surface(work,field,known)
    skin,skin_normals,_,skin_boundary=clip_surface(work,1-field,known)
    edge_lengths=np.linalg.norm(patch.vertices[boundary[:,0]]-patch.vertices[boundary[:,1]],axis=1)
    unsupported=~(patch_known[boundary[:,0]]&patch_known[boundary[:,1]])
    report={'view_count':view_count,'observed_vertices':int(known.sum()),'unknown_vertices':int((~known).sum()),'conflicting_observed_vertices':int(conflict.sum()),
        'unknown_boundary_length_fraction':float(edge_lengths[unsupported].sum()/edge_lengths.sum()),'unsupported_components_removed_vertices':removed,
        'fusion':'cosine^4 visibility weights; rear-view weight only where visible, no forced symmetry','boundary_regularization':settings['boundary_regularization'],
        'rear_negative_silhouette_constraints':view_count==4,'negative_background_is_not_called_skin':True,'narrow_patch_removed_vertices':opening_removed,'min_hair_patch_width_trial_mm':settings.get('min_patch_width_mm',0),'observation_geometry_simplified':opening_removed>0,'hairline_recomputed':True,'rear_view_is_approximate':view_count==4,'source_face_geometry_changed':False}
    output.mkdir();np.savez_compressed(output/'surface.npz',field=field,known=known,conflict=conflict,vertices=work.vertices,faces=work.faces,
        patch_vertices=patch.vertices,patch_faces=patch.faces,boundary=boundary,patch_known=patch_known)
    (output/'fusion.json').write_text(json.dumps(report,indent=2))
    result=shared.copy();result.update(field=field,work=work,patch=patch,normals=normals,known=patch_known,boundary=boundary,skin=skin,skin_normals=skin_normals,skin_boundary=skin_boundary,
        cameras=shared['cameras'][:view_count],masks=shared['masks'][:view_count],depths=shared['depths'][:view_count],labels=shared['labels'][:view_count],images=shared['images'][:view_count],records=records,fusion=report)
    return result


def add_back(shared,path,output):
    image,mask,label,camera,report=prepare_back(path,shared['closed'],output)
    for key,value in [('images',image),('masks',mask),('labels',label),('cameras',camera)]:shared[key]=list(shared[key])+[value]
    shared['depths']=list(shared['depths'])+[surface_depth_fast(shared['closed'].vertices,shared['closed'].faces,camera,(mask.shape[1],mask.shape[0]))]
    shared['back_report']=report
    return shared
